from torchvision.models import resnet50
from torchinfo import summary
import torchmetrics.functional as tmf
import pytorch_lightning as pl

from .losses import *
from .mlp import *
from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImageTabFusionMul(pl.LightningModule):
    def __init__(self, n_channels, pnum, pcats, n_classes, archi_hparams, learning_hparams):
        super().__init__()
        
        ### Learning hyperaparams
        self.batch_size = learning_hparams['batch_size']
        self.learning_rate = learning_hparams['learning_rate']
        self.save_hyperparameters()
        
        ## Dimensions
        self.n_channels = n_channels
        self.pnum = pnum
        self.pcats = pcats
        self.n_classes = n_classes
        
        ## Archi params
        self.archi_hparams = archi_hparams
        
        ## Learning params
        self.learning_hparams = learning_hparams
        
        ## Create architecture
        #### Image feature extraction 
        self.img_feat_ext = torch.load('deep_models/so2sat_state_dict.pt')
        self.img_emb_size = self.img_feat_ext.fc.in_features
        self.img_feat_ext.fc = torch.nn.Identity(in_features=self.img_emb_size)
        
        self.fusion = TabularMultiClass(self.img_emb_size + self.pnum, self.pcats, self.n_classes, self.archi_hparams)
        
        ## Biases
        if self.learning_hparams['init_bias'] is not None:
            self.init_bias = torch.FloatTensor(self.learning_hparams['init_bias'])
        else:
            self.init_bias = None
        
        ## Losses
        if self.learning_hparams['class_weights'] is not None:
            self.class_weights = torch.FloatTensor(self.learning_hparams['class_weights'])
        else:
            self.class_weights = None
        
        if self.learning_hparams['loss']=='fl':
            self.objective = FocalLoss(weight=self.class_weights,gamma=self.learning_hparams['gamma'])
            
        elif self.learning_hparams['loss']=='ldam':
            self.objective = LDAMLoss(cls_num_list=self.learning_hparams['cls_num'],
                                      max_m=self.learning_hparams['margin'],
                                      weight=self.class_weights)
            
        else:
            self.objective = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            
        
        self.initialize_model()
            
    def initialize_model(self):
        ### Initialize weights for added layers
        self.apply(initialize_weights)
        self.set_freeze_mode(self.archi_hparams['unfreeze'])
        
        clm = self.fusion.classif
        if self.init_bias is not None:
            clm.bias.data = self.init_bias
            
    def set_freeze_mode(self, mode=0):
        ### Set some layers from resnet as frozen
        for param in self.img_feat_ext.parameters():
            param.requires_grad = True if mode>2 else False
        
        if mode==2:
            print('Training ResNet block 3 and 4 + fusion MLP')
            for param in self.img_feat_ext.layer3.parameters():
                param.requires_grad = True  
                    
            for param in self.img_feat_ext.layer4.parameters():
                param.requires_grad = True
                     
        elif mode==1:
            print('Training ResNet block 4 + fusion MLP')
            for param in self.img_feat_ext.layer4.parameters():
                param.requires_grad = True
        else:
            print('Training fusion MLP only !' if mode==0 else 'Full training')
                
        
    def forward(self, ximg, xnum, xcat):
        hidden = self.img_feat_ext(ximg)
        
        if xnum is not None:
            hidden = torch.cat((hidden, xnum), dim=1)
        
        if xcat is not None:
            logit = self.fusion(hidden, xcat)
            
        else:
            logit = self.fusion(hidden, [])
        
        return logit
    
    
    def predict_step(self, batch, batch_idx):
        img_tensor = batch['img']
        if self.pnum>0:
            num_tensor = batch['tabnum']
        else:
            num_tensor = None
        
        if len(self.pcats)>0:
            cat_tensor = batch['tabcat']
        else:
            cat_tensor = None
            
        label_tensor = batch['labels']
        
        output_tensor = self(img_tensor, num_tensor, cat_tensor)
        #output_pred = torch.softmax(output_tensor)
        
        return output_tensor
        
    def compute_loss(self, batch, batch_idx, do_eval=False):
        label_tensor =  batch['labels']
        output_tensor = self.predict_step(batch, batch_idx)
        loss = self.objective(output_tensor,label_tensor)            
        
        if do_eval:
            perfs = self.compute_accuracies(output_tensor, label_tensor)
            perfs.update({'loss':loss})
            return perfs
        else:
            return loss   
    
    def compute_accuracies(self, output_tensor, label_tensor):
        ## Eval
        perfs = dict(
            rec = tmf.recall(output_tensor, label_tensor.int()),
            top1 = tmf.accuracy(output_tensor, label_tensor.int(),top_k=1)
        )
        
        for k in [3,5,10]:
            if self.n_classes>k:
                perfs.update({'top%d'%k: tmf.accuracy(output_tensor, label_tensor.int(),top_k=k)})
        
        return perfs
        
    
    def training_step(self, batch, batch_idx):
        perfs = self.compute_loss(batch, batch_idx, do_eval=True)
        self.log('train_loss', perfs['loss'], prog_bar=True)
        for key, val in perfs.items():
            self.log('train_%s'%key,val,prog_bar=True)
            
        return perfs['loss']

    def validation_step(self, batch, batch_idx):
        perfs = self.compute_loss(batch, batch_idx, do_eval=True)
        for key, val in perfs.items():
            self.log('val_%s'%key,val,prog_bar=True)
        
        return perfs['loss']

    def test_step(self, batch, batch_idx):
        perfs  = self.compute_loss(batch, batch_idx, do_eval=True)
        for key, val in perfs.items():
            self.log('test_%s'%key,val,prog_bar=True)
        
        return perfs['loss']

    def configure_optimizers(self):
        if self.learning_hparams['optimizer']=='adam':
            optim = torch.optim.Adam(self.parameters(), 
                                    lr=self.learning_rate,
                                    weight_decay=self.learning_hparams['weight_decay'])   
        
        elif self.learning_hparams['optimizer']=='radam':
            optim = torch.optim.RAdam(self.parameters(), 
                                    lr=self.learning_rate,
                                    weight_decay=self.learning_hparams['weight_decay'])  
            
        elif self.learning_hparams['optimizer']=='adamW':
            optim = torch.optim.AdamW(self.parameters(), 
                                    lr=self.learning_rate,
                                    weight_decay=self.learning_hparams['weight_decay'])              
        else:
            optim = torch.optim.SGD(self.parameters(), 
                                    lr=self.learning_rate,
                                    momentum=0.9, nesterov=True,
                                    weight_decay=self.learning_hparams['weight_decay'])
    
    
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0 = 2, eta_min = 1e-5)
        self.optimizer = optim
        self.scheduler = scheduler
        
        return [self.optimizer], [self.scheduler]    