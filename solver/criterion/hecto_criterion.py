import torch
import torch.nn as nn
import torch.nn.functional as F

class HectoCriterion(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, outputs, progress=0):
        pred_logits = outputs['pred_logits']           # (B, num_classes)
        labels = [x['label'] for x in outputs['batched_input']]  # list of int
        labels = torch.tensor(labels, dtype=torch.long, device=pred_logits.device)

        target = F.one_hot(labels, num_classes=pred_logits.shape[1]).float()  # (B, C)

        loss = self.loss_fn(pred_logits, target)

        return {
            'bce_loss': loss * self.loss_weight
        }
