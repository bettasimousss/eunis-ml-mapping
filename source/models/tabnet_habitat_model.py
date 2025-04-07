from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from .neural_networks.losses import FocalLoss, LDAMLoss
from .habitat_model import HabitatModel
import numpy as np

class TabNetHabitatModel(HabitatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(self, X, y, X_val, y_val):
        self.prefit(X, y)
        
        X_prep, y_prep = self.prepare_dataset(X,y)
        X_val_prep, y_val_prep = self.prepare_dataset(X_val,y_val)
        
        ### Categorical features setting
        self.cat_vars = self.param_dict['inputs']['cat_names']
        self.cat_idxs = [self.in_covars.index(cv) for cv in self.cat_vars]
        self.cat_dims = [len(pc) for pc in self.param_dict['inputs']['categories']]
        self.cat_emb_dim = [min(emb_sz_rule(len(pc)),3) for pc in self.param_dict['inputs']['categories']]
        
        ### Feature groups
        if self.param_dict['train_params']['attention_level']=='group':
            fgroups = self.param_dict['inputs']['metadata'].set_index('feature_code').loc[self.original_feats,'feature_group']
            fgroups.index = self.in_covars
            self.idx_groups = [np.where(fgroups==fg)[0].tolist() for fg in fgroups.unique()]
            
        else:
            self.idx_groups = None
        
        ## Create model object
        self.model = TabNetClassifier(cat_idxs=self.cat_idxs, cat_dims=self.cat_dims, cat_emb_dim=self.cat_emb_dim,
                                      grouped_features=self.idx_groups, 
                                      optimizer_fn=self.param_dict['train_params']['optim'],
                                      optimizer_params=dict(lr=self.param_dict['train_params']['learning_rate']),
                                      **self.param_dict['hyperparams']) 
        
        ## Loss            
        loss_params = self.param_dict['train_params']['loss']
        if loss_params['loss_fn']=="fl":
            self.loss_fn = FocalLoss(gamma=loss_params['gamma']) 
        
        elif loss_params['loss_fn']=="ldam":
            self.loss_fn = LDAMLoss(cls_num_list=np.bincount(y_prep),max_m=loss_params['margin'])
            
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        
        
        ## Training
        if self.param_dict['train_params']['pretraining']:
            print('Self-supervised pretraining')
            
            # TabNetPretrainer
            self.unsupervised_model = TabNetPretrainer(
                cat_idxs=self.cat_idxs, cat_dims=self.cat_dims, cat_emb_dim=self.cat_emb_dim,
                grouped_features=self.idx_groups, **self.param_dict['hyperparams']
            )

            self.unsupervised_model.fit(
                X_train=X_prep.values,
                eval_set=[X_val_prep.values],
                pretraining_ratio=0.8
            )
            
        else:
            self.unsupervised_model = None
            
        self.model.fit(X_train=X_prep.values, y_train=y_prep, 
                       eval_set=[(X_val_prep.values, y_val_prep)],
                       loss_fn=self.loss_fn,
                       num_workers=self.param_dict['train_params']['num_workers'],
                       weights=self.param_dict['train_params']['weights'],
                       eval_metric=self.param_dict['train_params']['eval_metric'],
                       max_epochs=self.param_dict['train_params']['max_epochs'],                    
                       from_unsupervised=self.unsupervised_model)

        
    def predict_proba(self, X):
        X_prep, _ = self.prepare_dataset(X)
        Y_hat = pd.DataFrame(data=self.model.predict_proba(X_prep.values),columns=self.labels,index=X.index)
        
        return Y_hat
    
    def plot_learning(self):
        history = self.model.history
        epochs = len(history['loss'])
        
        fig, ax = plt.subplots(1,1)
        sns.lineplot(ax=ax,x=np.arange(epochs),y=history['loss'], label='train',color='blue')
        sns.scatterplot(ax=ax,x=np.arange(epochs),y=history['loss'], color='blue')

        sns.lineplot(ax=ax,x=np.arange(epochs),y=np.array(history['val_0_logloss']), label='valid', color='red')
        sns.scatterplot(ax=ax,x=np.arange(epochs),y=np.array(history['val_0_logloss']), color='red') 
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')    
        
        return fig