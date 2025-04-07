import skorch
from skorch.net import NeuralNet
from skorch.helper import predefined_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.special import softmax

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from .neural_networks.mlp import *
from .neural_networks.losses import *
from .neural_networks.tabular_dataset import *

from .habitat_model import HabitatModel

class NeuralHabitatModel(HabitatModel):
    def __init__(self, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.criterion = None
        
    def prefit(self, X, y):
        super().prefit(X, y)
        
        ### Recomputing num and cat feature sets as previous computation does not suit MLP processing
        self.cat_vars = self.param_dict['inputs']['categories']
        
        flat_cat = list(itertools.chain(*self.cat_vars))
        self.num_vars = list(set(self.in_covars).difference(flat_cat)) 
        
    def prepare_dataset(self, X, y):
        ### Process as usual
        X_prep, y_prep = super().prepare_dataset(X,y)
        
        ### Torch specific preps
        tab_ds = TabDataset(X=X_prep, y=y_prep, num_features=self.num_vars, cat_features=self.cat_vars)
        
        return tab_ds
        
    def fit(self, X, y, X_val, y_val):
        self.prefit(X, y)
        train_ds = self.prepare_dataset(X,y)
        valid_ds = self.prepare_dataset(X_val,y_val)
        
        train_params = self.param_dict['train_params']
        
        # Create loss function
        if train_params['loss']['weighted']:
            loss_w = self.class_w
        else:
            loss_w = np.ones_like(self.class_w)
        
        
        if train_params['loss']['lossname']=="ce":
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(loss_w))
            
        elif train_params['loss']['lossname']=="fl":
            self.criterion = FocalLoss(weight=torch.FloatTensor(loss_w),gamma=train_params['loss']['gamma']) 
            
        elif train_params['loss']['lossname']=="ldam":
            self.criterion = LDAMLoss(cls_num_list=np.bincount(train_ds.y),
                                      max_m=train_params['loss']['margin'],
                                      weight=torch.FloatTensor(loss_w))
            
        else:
            self.criterion = None
            print('Unrecognized loss')

        ## Create callbacks
        callbacks = [
            skorch.callbacks.EarlyStopping(threshold=train_params['threshold'],patience=train_params['patience'],load_best=True),
#             skorch.callbacks.EpochScoring(scoring=ds_accuracy, lower_is_better=False,use_caching=False,name='val_accuracy'),
             skorch.callbacks.LRScheduler(policy=ReduceLROnPlateau,monitor='valid_loss')
        ]
        
        self.neural_module = NeuralNet(
            ### Parameters to build the network architecture
            module=TabularMultiClass,
            module__num_feats=self.num_vars,
            module__cat_feats=self.cat_vars,
            module__n_classes=len(self.labels),
            module__archi_hparams=self.param_dict['hyperparams'],
            predict_nonlinearity=None,
            ### Parameters to train the network
            criterion=self.criterion,
            train_split=predefined_split(valid_ds),
            iterator_train__shuffle=train_params['shuffle'],
            iterator_train__num_workers=train_params['num_workers'],
            iterator_train__pin_memory=train_params['pin'],
            max_epochs=train_params['max_epochs'],
            batch_size=train_params['batch_size'],            
            ## Parameters for the optimizer
            optimizer=train_params['optimizer']['optim'],
            optimizer__lr=train_params['optimizer']['lr'],
            optimizer__weight_decay=train_params['optimizer']['wdecay'],
            ## callbackd and device
            callbacks=callbacks,
            device=self.device,
        )
        
        self.neural_module.fit(train_ds)
           
    def predict_logit(self, X):
        predict_ds = self.prepare_dataset(X,y=None)
        y_pred = self.neural_module.predict(predict_ds)
        
        return pd.DataFrame(data=y_pred,columns=self.labels, index=X.index)

    def predict_proba(self, X):
        ### Evaluation metrics are not affected by temperature but probabilities hence confidences are. 
        logit = self.predict_logit(X) * self.temperature
        y_hat = softmax(logit,axis=1)
        return y_hat
    
    def plot_learning(self):
        hist_df = pd.DataFrame(data=np.array([(hist['epoch'],
                                               hist['train_loss'],
                                               hist['valid_loss'])  for hist in self.neural_module.history_[:-1]]),
             columns=['epoch','train_loss','valid_loss'])

        fig, ax = plt.subplots(1,1)
        sns.lineplot(data=hist_df, x='epoch', y='train_loss', label='train',ax=ax,color='blue')
        sns.lineplot(data=hist_df, x='epoch', y='valid_loss', label='valid',ax=ax,color='red')
        fig.suptitle('Learning history')
        
        return fig
