from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


def emb_sz_rule(n_cat):
    "Rule of thumb to pick embedding size corresponding to `n_cat`"
    return min(600, round(1.6 * n_cat**0.56))


class MLPblock(nn.Module):
    def __init__(self, nfeat_in, nfeat_out, do_activate=False, drop_rate=None, batch_norm=False):
        super().__init__()
        ## HParams
        self.nfeat_in = nfeat_in
        self.nfeat_out = nfeat_out
        self.do_activate = do_activate
        self.drop_rate = drop_rate
        self.batch_norm = batch_norm
        self._init_model()

    def _init_model(self):
        ## Create layers
        self.linear = nn.Linear(in_features=self.nfeat_in, out_features=self.nfeat_out)
        if self.do_activate:
            self.relu = nn.ReLU()
        else:
            self.relu = None
        ## Reg layers
        if self.drop_rate:
            self.dropout = nn.Dropout(p=self.drop_rate)
        else:
            self.dropout = None
        ## Batch norm layers
        if self.batch_norm:
            self.bnlayer = nn.BatchNorm1d(num_features=self.nfeat_out, eps=1e-5, momentum=0.1)
        else:
            self.bnlayer = None

    def forward(self, x):
        hidden = self.linear(x)
        if self.bnlayer:
            hidden = self.bnlayer(hidden)
        if self.relu:
            hidden = self.relu(hidden)
        if self.dropout:
            hidden = self.dropout(hidden)
        return hidden
    
    
class TabularMultiClass(nn.Module):
    def __init__(self, num_feats, cat_feats, n_classes, archi_hparams):
        super().__init__()
        
        ## Dimensions
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        
        self.pnum = len(num_feats)
        self.pcats = [len(f) for f in cat_feats]
        self.n_classes = n_classes
        
        ## Archi params
        self.emb_size = [emb_sz_rule(pc) for pc in self.pcats]
        self.archi_hparams = archi_hparams
        self._init_model()

    def _init_model(self):
        if len(self.cat_feats)>0:
            self.emb_layer = nn.ModuleList([nn.Linear(in_features=pc, out_features=es, 
                                                      bias=False) for pc,es in zip(self.pcats,self.emb_size)])
            
            tot_emb = np.sum(self.emb_size)
        else:
            self.emb_layer = None
            tot_emb = 0
            
        ## Feature extraction layers
        self.layers = OrderedDict()
        in_nn = self.pnum + tot_emb
        for il, nl in enumerate(self.archi_hparams["archi"]):
            block_il = MLPblock(nfeat_in=in_nn,
                                nfeat_out=nl,
                                do_activate=True,
                                batch_norm=self.archi_hparams["bn"],
                                drop_rate=self.archi_hparams["dropout"])
            
            self.layers["mlp_block%d" % il] = block_il
            in_nn = nl
            
        self.feat_ext = nn.Sequential(OrderedDict(self.layers))
        
        ## Classification
        self.classif = nn.Linear(in_features=in_nn, out_features=self.n_classes)
        self.apply(initialize_weights)

    def forward(self, xnum, xcat):
        cat_emb = [self.emb_layer[i](arr) for i, arr in enumerate(xcat)]
        hidden = torch.cat([xnum] + cat_emb, dim=1)
        hidden = self.feat_ext(hidden)
        logit = self.classif(hidden)
        return logit