import torch 
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import kornia.augmentation as KA
from torchvision import transforms as ttransforms

import tifffile
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

from ..preprocessing import *

class SentinelEUNISDataset(Dataset):
    '''
        Name: name of the dataset
        Mode: tab, img, tab1img
    '''
    def __init__(self, obs_df, feat_df, num_feat, cat_feat, path_col, target_col, img_dir, transform=None):

        super().__init__()

        ## Input type
        self.num_feat = num_feat
        self.cat_feat = cat_feat
        self.img_dir = img_dir
        self.mode = 'img' if (feat_df is None) else 'tab1img'

        ## Observations / labels
        self.target_col = target_col

        self.obs_paths = obs_df[path_col].values.astype(str)
        self.obs_data = obs_df[target_col].values.astype(np.int64)
        
        self.num_feat_data = feat_df[num_feat].values.astype(np.float32) if feat_df is not None else None
        self.cat_feat_data = [feat_df[categs].values.astype(np.float32) for _,categs in self.cat_feat.items()] if feat_df is not None else None

        ### Image transforms
        self.transform = transform

    def __len__(self):
        return self.obs_data.shape[0]

    def __getitem__(self, idx):
        labels = self.obs_data[idx]
        out_dict = {'labels':labels}

         ## Path
        img_path = self.obs_paths[idx]

        ## Read image
        img = tifffile.imread(img_path)

        ## Transform (eventually)
        if self.transform:
            img = self.transform(img.T).squeeze()

        out_dict.update({'img': img})

        if len(self.num_feat)>0:
            in_tab_num = self.num_feat_data[idx]
            out_dict.update({'tabnum':in_tab_num})

        if len(self.cat_feat)>0:
            in_tab_cat = [arr[idx] for arr in self.cat_feat_data]
            out_dict.update({'tabcat':in_tab_cat})

        return out_dict
    

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = [0] * len(np.unique(dataset.obs_data))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1
            
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        
    def _get_label(self, dataset, idx):
        return dataset.obs_data[idx]
                
    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples
    
    
class EUNISDataModule(pl.LightningDataModule):
    def __init__(self, 
                 ### Observation data, labels
                 obs_df, idx_col='PlotObservationID', target_col='target', split_col='fold',
                 ### Images
                 path_col='fpath',img_dir='./',mean_vals=None, std_vals=None,
                 ### Other features
                 feature_metadata = None, cat_names=None, categories = None,
                 num_workers=0, pin=False,balanced_sampler=False, 
                 train_batch_size=64, val_batch_size=64):
        
        super().__init__()

        ### Metadata
        self.idx_col = idx_col
        self.target_col = target_col
        self.path_col = path_col
        self.split_col = split_col
        self.img_dir = img_dir     
        
        ### Classes
        self.obs_df = obs_df[[idx_col, split_col, path_col, target_col]]
        self.label_encoder = None
        self.classes = None
        self.nclasses = None
        
        ### Features
        self.feature_metadata = feature_metadata
        self.cat_names = cat_names
        self.categories = categories
        self.feat_df = obs_df[[split_col]+self.feature_metadata['feature_code'].tolist()] if feature_metadata is not None else None
            
        self.feat_prep = None
        self.num_vars = []
        self.cat_vars = []
        
        ### Preprocessing params
        self.mean_vals = mean_vals
        self.std_vals = std_vals
        
        ### Loader settings
        self.num_workers = num_workers
        self.balanced_sampler = balanced_sampler,
        self.pin = pin
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
    
    def setup(self, stage=None):  
        ## Create preprocessing pipelines for labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.obs_df[self.target_col])
        self.classes = self.label_encoder.classes_
        self.nclasses = len(self.classes)
        
        Y_train = self.obs_df.query('%s=="train"'%self.split_col).copy()
        Y_train['target'] = self.label_encoder.transform(Y_train[self.target_col])
        
        Y_valid = self.obs_df.query('%s!="train"'%self.split_col).copy()
        Y_valid['target'] = self.label_encoder.transform(Y_valid[self.target_col])

        ## Create preprocessing pipelines for images
        self.augmentations = KA.AugmentationSequential(
            KA.RandomHorizontalFlip(p=0.5), 
            KA.RandomVerticalFlip(p=0.5)#,
            #K.RandomRotation(degrees=30,p=0.5)
        )

        norm = ttransforms.Normalize(mean=self.mean_vals,std=self.std_vals) 

        self.viz_transform = ttransforms.Compose([
            ttransforms.ToTensor(),
            self.augmentations
        ])
        
        self.train_transform = ttransforms.Compose([
            ttransforms.ToTensor(),
            norm,
            self.augmentations
        ])

        self.valid_transform = ttransforms.Compose([
            ttransforms.ToTensor(),
            norm
        ])  
        
        ## Create preprocessing pipelines for tabular features
        if self.feat_df is not None:
            train_feat_df = self.feat_df.query('%s=="train"'%self.split_col).drop([self.split_col],axis=1)
            valid_feat_df = self.feat_df.query('%s!="train"'%self.split_col).drop([self.split_col],axis=1)
        
            self.feat_prep, in_covars, _, _ = create_prep_pipeline(
                feature_metadata=self.feature_metadata,
                pc_groups=None, 
                categories=self.categories,  
                std=True, 
                onehot=1
            )
            
            self.cat_vars = {cv:categs for cv, categs in zip(self.cat_names,self.categories)}
            self.num_vars = list(set(in_covars).difference(list(itertools.chain(*self.categories))))
            
            self.feat_prep.fit(train_feat_df)
            X_train = self.feat_prep.transform(train_feat_df)
            X_valid = self.feat_prep.transform(valid_feat_df)
            
        else:
            X_train = X_valid = None
        
        
        ###### Creating dataset objects   
        self.viz_dataset = SentinelEUNISDataset(obs_df = Y_train, feat_df = X_train, 
                                                  num_feat = self.num_vars,cat_feat = self.cat_vars, 
                                                  path_col = self.path_col, 
                                                  target_col = 'target',  
                                                  img_dir = self.img_dir, 
                                                  transform=None)
        
        self.train_dataset = SentinelEUNISDataset(obs_df = Y_train, feat_df = X_train, 
                                                  num_feat = self.num_vars,cat_feat = self.cat_vars, 
                                                  path_col = self.path_col, 
                                                  target_col = 'target',  
                                                  img_dir = self.img_dir, 
                                                  transform=self.train_transform)
        
        self.valid_dataset = SentinelEUNISDataset(obs_df = Y_valid, feat_df = X_valid, 
                                                  num_feat = self.num_vars,
                                                  cat_feat = self.cat_vars, 
                                                  path_col = self.path_col, 
                                                  target_col = 'target',  
                                                  img_dir = self.img_dir, 
                                                  transform=self.valid_transform)      
        
      
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.train_batch_size,
            pin_memory=self.pin,
            sampler= ImbalancedDatasetSampler(dataset=self.train_dataset,num_samples=len(self.train_dataset)) if self.balanced_sampler else None,
            shuffle= (self.balanced_sampler==False)
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            num_workers=self.num_workers,
            batch_size=self.val_batch_size,
            pin_memory=self.pin,
            shuffle=False
        )      