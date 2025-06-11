import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HectoCriterion(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, outputs, progress=0):
        pred_logits = outputs['pred_logits']  # (B, C)
        labels = [x['label'] for x in outputs['batched_input']]  # list of int
        labels = torch.tensor(labels, dtype=torch.long, device=pred_logits.device)  # (B,)

        # Cross Entropy
        log_probs = F.log_softmax(pred_logits, dim=-1)  # (B, C)
        one_hot = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1.0)  # (B, C)
        ce_loss = -torch.sum(one_hot * log_probs, dim=1)  # (B,)
        loss = ce_loss.mean()  # reduction='mean'

        return {
            'ce_loss': loss * self.loss_weight
        }


class HectoCriterionEmb(nn.Module):
    def __init__(self,
                 class_weight=0.25,
                 mse_weight = 1.0,
                 align_weight = 0.25
                 ):
        super().__init__()
        self.class_weight = class_weight
        self.mse_weight = mse_weight
        self.align_weight = align_weight

    def forward(self, outputs):
        total_loss = {}

        # Cross Entropy Loss
        pred_logits = outputs['pred_logits']  # (B, C)
        labels = [x['label'] for x in outputs['batched_input']]  # list of int
        labels = torch.tensor(labels, dtype=torch.long, device=pred_logits.device)  # (B,)
        log_probs = F.log_softmax(pred_logits, dim=-1)  # (B, C)
        one_hot = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1.0)  # (B, C)
        ce_loss = -torch.sum(one_hot * log_probs, dim=1)  # (B,)
        class_loss = ce_loss.mean()
        total_loss['class_loss'] = class_loss * self.class_weight

        # Embedding Loss
        query_pred = outputs['query_pred']  # [B, Q, D]
        targets = outputs['targets']  # [B, D]
        query_target = targets['query_target']
        target_labels = targets['target_labels']
        if query_pred.dim() != query_target.dim():
            B, Q, D = query_pred.shape
            query_target = query_target.unsqueeze(1).expand(B, Q, D)
            target_labels = target_labels.unsqueeze(1).expand(B, Q)

        mse_loss = F.mse_loss(query_pred, query_target)
        total_loss['mse_loss'] = mse_loss * self.mse_weight

        align_loss = multiple_info_nce(query_pred, query_target, target_labels, temperature=0.2)
        total_loss['align_loss'] = align_loss * self.align_weight

        return total_loss

def multiple_info_nce(pred, target, labels, temperature=0.2, fg_only=True):
    """
    - pred: (N, D) predicted region embeddings
    - target: (N, D) target GT embeddings (0 if non-GT)
    - labels: (N,) int label or -1 for non-GT

    Goal:
    1. FG same class: pred >-< target
    2. FG diff class: pred ↔ target
    3. pred FG ↔ pred BG
    """
    device = pred.device
    fg_mask = labels != -1
    bg_mask = ~fg_mask

    if fg_mask.sum() == 0:
        loss_nce_fg = torch.tensor(0.0, device=device)
    else:
        # -------- Part 1: pred ↔ target InfoNCE--------
        pred_FG = F.normalize(pred[fg_mask], dim=-1)      # (F, D)
        target_FG = F.normalize(target[fg_mask], dim=-1)  # (F, D)
        labels_FG = labels[fg_mask]                       # (F,)

        sim_matrix = torch.matmul(pred_FG, target_FG.T) / temperature  # (F, F)
        sim_exp = sim_matrix.exp()

        # positive: same class (exclude self if needed)
        pos_mask = labels_FG.unsqueeze(1) == labels_FG.unsqueeze(0)
        pos_mask = pos_mask.float()
        numerator = (sim_exp * pos_mask).sum(dim=-1)
        denominator = sim_exp.sum(dim=-1)

        loss_nce_fg = -torch.log((numerator + 1e-6) / (denominator + 1e-6)).mean()

    if fg_only:
        return loss_nce_fg

    # -------- Part 2: pred FG ↔ pred BG --------
    if fg_mask.sum() == 0 or bg_mask.sum() == 0:
        loss_nce_bg = torch.tensor(0.0, device=device)
    else:
        pred_BG = F.normalize(pred[bg_mask], dim=-1)  # (B, D)
        sim_neg = torch.matmul(pred_FG, pred_BG.T)     # (F, B)
        loss_nce_bg = sim_neg.exp().mean()

    return loss_nce_fg, loss_nce_bg


def get_temperature(progress, tau_min=0.05, tau_max=0.2):
    return tau_max - (tau_max - tau_min) * (1 - math.cos(math.pi/2 * progress))