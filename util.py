import torch
from torchmetrics import Metric


class Delta1(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('nbatch', default=torch.tensor(0.0), dist_reduce_fx='sum')

    def update(self, pred, gt):
        pred, gt = pred.detach(), gt.detach()
        self.correct += (torch.max(pred / gt, gt / pred) < 1.25).float().mean()
        self.nbatch += 1

    def compute(self):
        return self.correct / self.nbatch
