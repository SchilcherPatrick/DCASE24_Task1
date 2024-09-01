import torch

class MeanEnsemble:
    def __init__(self, cprestnets=[], passts=[]):
        self.cprestnets = cprestnets
        self.passts = passts

        self.num_models = len(self.cprestnets) + len(self.passts)
    
    def forward(self, x, x_p=None):
        logits = [model(x) for model in self.cprestnets]
        logits.extend(
            [model(x_p)[0] for model in self.passts] # only get output logits from passt model
        )
        
        return torch.stack(logits, dim=0).mean(dim=0)
    
    def __call__(self, x, x_p=None):
        return self.forward(x, x_p)

class BEAEnsemble:
    def __init__(self, cprestnets=[], passts=[]):
        self.cprestnets = cprestnets
        self.passts = passts

        self.num_models = len(self.cprestnets) + len(self.passts)
    
    def forward(self, x, x_p=None):
        logits = [model(x) for model in self.cprestnets]
        logits.extend(
            [model(x_p)[0] for model in self.passts] # only get output logits from passt model
        )
        
        mu = torch.stack(logits, dim=0).mean(dim=0)
        kappa = torch.stack([logit.var(dim=0) for logit in logits], dim=0).mean(dim=0)
        combined_logits = mu + kappa / self.num_models
        return combined_logits
    
    def __call__(self, x, x_p=None):
        return self.forward(x, x_p)