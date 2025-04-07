import numpy as np
import torch
import skorch.dataset as skd

class TabDataset(skd.Dataset):
    def __init__(self, X, y, num_features, cat_features):
        super().__init__(X, y)
        
        self.num_feat = num_features
        self.cat_feat = cat_features
        
        self.num_X = X[self.num_feat].values.astype(np.float32)
        self.cat_X = [X[categs].values.astype(np.float32) for categs in self.cat_feat]

    def __getitem__(self, idx):
        y = self.y[idx] if self.y is not None else torch.Tensor([0])            
        return {"xnum": self.num_X[idx], "xcat": [arr[idx] for arr in self.cat_X]}, y
    