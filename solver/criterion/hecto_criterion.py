import torch
import torch.nn as nn
import torch.nn.functional as F


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
