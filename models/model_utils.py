import torch
from collections import OrderedDict
from detectron2.config import LazyCall, instantiate
import inspect
from omegaconf import DictConfig

import cv2
import supervision as sv
from torchvision.ops import box_convert
from dotenv import load_dotenv
load_dotenv()


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


def load_model(model, ckpt_path, strict=False):
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=strict)
        print(f"[Notice] [LOAD] {model.__class__.__name__} use checkpoint /{ckpt_path} ({len(missing)} missing, {len(unexpected)} unexpected)")
    return model


def safe_init(cls, args: dict):
    sig = inspect.signature(cls.__init__)
    valid_keys = sig.parameters.keys() - {'self'}

    filtered_args = {}
    for k, v in args.items():
        if k in valid_keys:
            if isinstance(v, (LazyCall, DictConfig)) and '_target_' in v:
                v = instantiate(v)
            filtered_args[k] = v

    missing_keys = valid_keys - filtered_args.keys()
    if missing_keys:
        print(f"[Notice] [INIT] {cls.__name__} Missing keys (set default): {sorted(missing_keys)}")

    return cls(**filtered_args)


def check_frozen(model, max_depth=5):
    def collect_status(module, name_path=""):
        own_params = list(module.named_parameters(recurse=False))

        node = {
            'Trainable Params': 0,
            'Frozen Params': 0,
            'Total Params': 0,
            'children': {}
        }

        for param_name, param in own_params:
            param_node = {
                'Trainable Params': int(param.requires_grad),
                'Frozen Params': int(not param.requires_grad),
                'Total Params': param.numel()
            }
            node['children'][param_name] = param_node
            node['Trainable Params'] += param_node['Trainable Params']
            node['Frozen Params'] += param_node['Frozen Params']
            node['Total Params'] += param_node['Total Params']

        for child_name, child_module in module.named_children():
            full_child_name = f"{name_path}.{child_name}" if name_path else child_name
            child_node = collect_status(child_module, full_child_name)
            node['children'][child_name] = child_node
            node['Trainable Params'] += child_node['Trainable Params']
            node['Frozen Params'] += child_node['Frozen Params']
            node['Total Params'] += child_node['Total Params']

        return node

    def print_tree(tree, depth=0, max_depth=1, prefix=""):
        if depth > max_depth:
            return
        if depth == 0:
            module_width = 40
            print("-" * (module_width + 34))
            print(f"{'Module'.ljust(module_width)} | {'Trainable':>9} | {'Frozen':>6} |   Size  ")
            print("-" * (module_width + 34))

        module_width = 40
        total = len(tree)
        for idx, (module_name, node) in enumerate(tree.items()):
            trainable = node['Trainable Params']
            frozen = node['Frozen Params']
            total_params = format_params(node['Total Params'])

            if idx == total - 1:
                branch = "└── "
            else:
                branch = "├── "

            name_field = prefix + branch + module_name
            print(f"{name_field.ljust(module_width)} | {trainable:9} | {frozen:6} | {str(total_params).rjust(9)}")

            if 'children' in node and depth < max_depth:
                if idx == total - 1:
                    extension = "    "
                else:
                    extension = "│   "
                print_tree(node['children'], depth=depth + 1, max_depth=max_depth, prefix=prefix + extension)

    tree = {'__root__': collect_status(model)}
    print_tree(tree, max_depth=max_depth)
    return tree


def format_params(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def check_grad(model):
    def collect(module):
        tree = {
            'Param Count': 0,
            'children': {},
            'Param Norm': 0.0,
            'Grad Norm': 0.0,
        }
        for name, param in module.named_parameters(recurse=False):
            if param.grad is not None:
                try:
                    param_norm = param.data.norm(dim=-1).mean().item()
                    grad_norm = param.grad.norm(dim=-1).mean().item()
                except RuntimeError:
                    # fallback
                    param_norm = param.data.norm().item()
                    grad_norm = param.grad.norm().item()
                if param_norm < 1e-4:
                    ratio = 0
                else:
                    ratio = grad_norm / param_norm
                count = param.numel()
                tree['children'][name] = {
                    'Param Count': count,
                    'Param Norm': param_norm,
                    'Grad Norm': grad_norm,
                    'Ratio': ratio,
                    'is_leaf': True
                }
                tree['Param Count'] += count
                tree['Param Norm'] += param_norm
                tree['Grad Norm'] += grad_norm

        for child_name, child_module in module.named_children():
            child_tree = collect(child_module)
            if child_tree['Param Count'] > 0:
                tree['children'][child_name] = child_tree
                tree['Param Count'] += child_tree['Param Count']
                tree['Param Norm'] += child_tree['Param Norm']
                tree['Grad Norm'] += child_tree['Grad Norm']
        return tree

    def print_node(name, node, prefix="", is_last=True):
        branch = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        full_name = prefix + branch + name

        param_count = node.get("Param Count", 0)

        if node.get("is_leaf", False):
            param_norm = node.get("Param Norm", 0.0)
            grad_norm = node.get("Grad Norm", 0.0)
            ratio = node.get("Ratio", 0.0)
            print(f"{full_name:<50} | {format_params(param_count):>8} | {param_norm:10.4f} | {grad_norm:10.4f} | {ratio:8.4f}")
        else:
            print(f"{full_name:<50} | {format_params(param_count):>8} | {'':>10} | {'':>10} | {'':>8}")

        children = list(node.get("children", {}).items())
        for i, (child_name, child_node) in enumerate(children):
            print_node(child_name, child_node, prefix + extension, i == len(children) - 1)

    # Header
    print("-" * 100)
    print(f"{'Parameter':<50} | {'Param#':>8} | {'‖θ‖':>10} | {'‖∇L‖':>10} | {'ratio':>8}")
    print("-" * 100)

    tree = collect(model)
    print_node("__root__", tree)



def visualize(mode, input, image=None, batch=0, idx =-1, threshold=0.1, labels=None, vocab=None, caption=None, save_name=None):
    """
    Args:
        pred_logit: (N, C_prompt) logits before sigmoid
        pred_boxes: (N, 4) boxes in cxcywh format, normalized (0~1)
        caption: prompt string (e.g. "dog . cat . zebra .")
        image: torch Tensor [3, H, W], pixel image
        threshold: detection confidence threshold
    """
    is_logit = True
    is_cxcy = True
    is_norm = True
    save_name = mode if save_name is None else save_name
    if mode == "gt":
        input = input[batch]
        if not hasattr(input, "gt_classes"):
            gt = input["instances"]
            image = input["image"]
        else:
            gt = input
        class_index = gt.gt_classes
        pred_boxes = gt.gt_boxes.tensor
        if hasattr(gt, "gt_names"):
            labels = gt.gt_names
        image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        h, w, _ = image.shape
        class_score = class_index + 100
        save_name = "gt"
    else:
        if mode == "nms":
            pred_logits = input['nms_prob'][batch][:idx]
            pred_boxes = input['nms_boxes'][batch][:idx]
            is_logit = False
            is_cxcy = False
        elif mode == "model":
            pred_logits = input['pred_logits'][batch][:idx]
            pred_boxes = input['pred_boxes'][batch][:idx]
            K = min(len(pred_logits),len(pred_boxes))
            pred_logits = pred_logits[:K]
            pred_boxes = pred_boxes[:K]
            is_logit = False
        else:
            pred_logits = input['pred_logits'][batch][:idx]
            pred_boxes = input['pred_boxes'][batch][:idx]

        # 1. Prepare Image
        image = image.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        h, w, _ = image.shape

        # 2. Process predictions
        if is_logit:
            pred_logits = pred_logits.sigmoid()
        if pred_logits.dim() == 1:
            pred_logits.unsqueeze_(1)
        pred_logits = pred_logits.detach().cpu()  # [N, C]
        pred_boxes = pred_boxes.detach().cpu()  # [N, 4], cxcywh (0~1)

        class_score, class_index = pred_logits.max(dim=1) # [N]

        # 3. Filter by confidence
        mask = class_score > threshold
        class_index = class_index[mask]
        class_score = class_score[mask]
        pred_boxes = pred_boxes[mask]
        if labels:
            labels = [l for l, m in zip(labels, mask.tolist()) if m]
        if pred_boxes.shape[0] == 0:
            print("No boxes passed threshold.")
            return None

        # 4. Box conversion and scaling
        if is_cxcy:
            pred_boxes = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")  # [M, 4]
        if is_norm:
            pred_boxes *= torch.tensor([w, h, w, h])  # scale to absolute coords
    pred_boxes = pred_boxes.numpy()
    if not labels:
        labels = []
        if not vocab:
            if not caption:
                vocab = coco_name_list
            else:
                vocab = [c.strip() for c in caption.strip(" .").split(".") if c.strip()]
        for i, s in zip(class_index, class_score):
            if s >= 100:
                labels.append("GT-" + vocab[i])
            else:
                labels.append(vocab[i])

    # 6. Prepare visualization
    class_index = class_index.detach().cpu().numpy()
    detections = sv.Detections(xyxy=pred_boxes, class_id=class_index)

    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_box = box_annotator.annotate(
        scene=annotated_img,
        detections=detections
    )
    annotated = label_annotator.annotate(
        scene=annotated_box,
        detections=detections,
        labels=labels
    )

    num_gt = (class_score >= 100).sum().item()
    print(f"predict {len(class_index)} instances (GT: {num_gt})")
    cv2.imwrite(f"outputs/{save_name}.jpg", annotated)

    return annotated

def visualize_pca(features: torch.Tensor, gt_labels: list, title="PCA by Class (Batched)"):
    """
    features: (B, Q, D)
    gt_labels: list of B tensors, each (Qi,) with class IDs (int), may contain -100
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np
    from collections import Counter

    if len(features.shape) == 2:
        features = features.unsqueeze(0)
        gt_labels = [gt_labels]

    B, Q, D = features.shape
    all_feats = []
    all_labels = []

    for b in range(B):
        feat_b = features[b]
        label_b = gt_labels[b]
        n = min(feat_b.shape[0], label_b.shape[0])
        feat_b = feat_b[:n]
        label_b = label_b[:n]

        all_feats.append(feat_b)
        all_labels.append(label_b)

    feats_cat = torch.cat(all_feats, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)

    # Prepare label counts
    label_list = labels_cat.cpu().tolist()
    class_counts = Counter(label_list)

    # PCA
    features_np = feats_cat.detach().cpu().numpy()
    labels_np = labels_cat.detach().cpu().numpy()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features_np)

    # Plot
    unique_labels = np.unique(labels_np)
    num_classes = len(unique_labels)
    cmap = plt.cm.get_cmap('tab20', num_classes)

    plt.figure(figsize=(8, 7))
    for idx, label in enumerate(unique_labels):
        mask = labels_np == label
        count = class_counts[label]
        plt.scatter(reduced[mask, 0], reduced[mask, 1],
                    label=f"{label} ({count})", s=30, alpha=0.7, color=cmap(idx))

    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Class ID (count)")
    plt.tight_layout()
    plt.show()


coco_name_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
def get_coco_cat_name(id_list):
    return [coco_name_list[i] for i in id_list]