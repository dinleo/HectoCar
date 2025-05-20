import torch
import torch.nn as nn

class HectoCriterion(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, progress=0):
        pred_logits = outputs['pred_logits']           # (B, num_classes), raw logits
        labels = [x['label'] for x in outputs['batched_input']]  # list of int
        labels = torch.tensor(labels, dtype=torch.long, device=pred_logits.device)  # (B,)

        loss = self.loss_fn(pred_logits, labels)

        return {
            'ce_loss': loss * self.loss_weight
        }
