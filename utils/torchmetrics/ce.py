import torch

class CE():
    def __init__(self): 
        self.reset()
        
    def reset(self): 
        self.sum = 0
        self.counts = 0

    @torch.no_grad()
    def accumulate(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ):
        ## assume equal batch size
        ## coutns - num prev steps
        num_quantiles = predictions.shape[-1]
        self.sum = (self.counts / (self.counts + 1)) * self.sum + (torch.nn.functional.cross_entropy(predictions.view(-1, num_quantiles), targets.view(-1), reduction='mean')) / (self.counts + 1)
        self.counts += 1
    
    @torch.no_grad()
    def value(self):
        return self.sum