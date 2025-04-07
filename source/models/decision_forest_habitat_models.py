from .habitat_model import HabitatModel
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgbm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool


class RandomForestHabitatModel(HabitatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(self, X, y):
        self.prefit(X, y)
        X_prep, y_prep = self.prepare_dataset(X,y)
        self.model = RandomForestClassifier(**self.param_dict['hyperparams'])
        self.model.fit(X_prep, y_prep)
    
    def predict_proba(self, X):
        X_prep, _ = self.prepare_dataset(X, y=None)
        Y_hat = pd.DataFrame(data=self.model.predict_proba(X_prep),columns=self.labels,index=X.index)
        
        return Y_hat
    
    
class XGBoostHabitatModel(HabitatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def prepare_dataset(self, X=None, y=None):
        X_prep, y_prep = super().prepare_dataset(X,y)
        if X_prep is not None:
            if len(self.cat_vars)>0:
                X_prep[self.cat_vars]=X_prep[self.cat_vars].astype('category')

            X_prep[self.num_vars]=X_prep[self.num_vars].astype(float)            
            
        return X_prep, y_prep
        
    def fit(self, X, y, X_val, y_val):
        self.prefit(X, y)
        X_prep, y_prep = self.prepare_dataset(X,y)
        X_val_prep, y_val_prep = self.prepare_dataset(X_val, y_val)
        
        train_params = self.param_dict['train_params']
        es_cbk = xgb.callback.EarlyStopping(rounds=train_params['early_stopping_rounds'], 
                                   metric_name=train_params['eval_metric'], 
                                   data_name='validation_1', 
                                   maximize=train_params['maximize'], 
                                   save_best=True, 
                                   min_delta=train_params['min_delta'])
        
        self.model = xgb.XGBClassifier(callbacks = [es_cbk], **self.param_dict['hyperparams'])
        
        
        if self.param_dict['train_params']['sample_weight']:
            print('using sample weights')
            sweight = self.sample_w
        else:
            sweight = None
            
        self.model.fit(X_prep, y_prep, eval_set=[(X_prep, y_prep), (X_val_prep, y_val_prep)], sample_weight=sweight)
    
    def predict_proba(self, X):
        X_prep, _ = self.prepare_dataset(X,y=None)
        Y_hat = pd.DataFrame(data=self.model.predict_proba(X_prep),columns=self.labels,index=X.index)
        
        return Y_hat
    
    def plot_learning(self, met_name='mlogloss'):
        history = self.model.evals_result_
        epochs = len(history['validation_0'][met_name])
        
        fig, ax = plt.subplots(1,1)
        sns.lineplot(ax=ax,x=np.arange(epochs),y=history['validation_0'][met_name], label='train',color='blue')
        sns.scatterplot(ax=ax,x=np.arange(epochs),y=history['validation_0'][met_name], color='blue')

        sns.lineplot(ax=ax,x=np.arange(epochs),y=np.array(history['validation_1'][met_name]), label='valid', color='red')
        sns.scatterplot(ax=ax,x=np.arange(epochs),y=np.array(history['validation_1'][met_name]), color='red') 
        ax.set_xlabel('Boosting rounds')
        ax.set_ylabel(met_name)   
        
        return fig
        
        
class LightGBMHabitatModel(HabitatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(self, X, y, X_val, y_val):
        self.prefit(X, y)
        
        X_prep, y_prep = self.prepare_dataset(X,y)
        X_val_prep, y_val_prep = self.prepare_dataset(X_val,y_val)
        
        train_params = self.param_dict['train_params']
        es_cbk = lgbm.early_stopping(stopping_rounds=train_params['early_stopping_rounds'])
        
        self.model = lgbm.LGBMClassifier(**self.param_dict['hyperparams'])    
        self.model.fit(X_prep, y_prep, 
                       eval_set=[(X_prep, y_prep), (X_val_prep, y_val_prep)], 
                       categorical_feature = self.cat_vars,
                       callbacks = [es_cbk],
                       eval_names = ['training','validation'])
    
    def predict_proba(self, X):
        X_prep, _ = self.prepare_dataset(X)
        Y_hat = pd.DataFrame(data=self.model.predict_proba(X_prep),columns=self.labels,index=X.index)
        
        return Y_hat
    
    def plot_learning(self, met_name='multi_logloss'):
        history = self.model.evals_result_
        epochs = len(history['training'][met_name])
        
        fig, ax = plt.subplots(1,1)
        sns.lineplot(ax=ax,x=np.arange(epochs),y=history['training'][met_name], label='train',color='blue')
        sns.scatterplot(ax=ax,x=np.arange(epochs),y=history['training'][met_name], color='blue')

        sns.lineplot(ax=ax,x=np.arange(epochs),y=np.array(history['validation'][met_name]), label='valid', color='red')
        sns.scatterplot(ax=ax,x=np.arange(epochs),y=np.array(history['validation'][met_name]), color='red') 
        ax.set_xlabel('Boosting rounds')
        ax.set_ylabel(met_name)      
        

class CatBoostHabitatModel(HabitatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def prepare_dataset(self, X=None, y=None):
        X_prep, y_prep = super().prepare_dataset(X,y)
        if X_prep is not None:
            if len(self.cat_vars)>0:
                X_prep[self.cat_vars]=X_prep[self.cat_vars].astype('category')

            X_prep[self.num_vars]=X_prep[self.num_vars].astype(float)            
            
        return X_prep, y_prep        
        
    def fit(self, X, y, X_val, y_val):
        ### Prepare 
        self.prefit(X, y)
        X_prep, y_prep = self.prepare_dataset(X,y)
        X_val_prep, y_val_prep = self.prepare_dataset(X_val,y_val)
        
        train_params = self.param_dict['train_params']
        
        self.model = CatBoostClassifier(classes_count=len(self.labels),
                                        cat_features=self.cat_vars, 
                                        name=self.model_name,
                                        train_dir=train_params['train_dir'],
                                        **self.param_dict['hyperparams'])  
        
        self.model.fit(X=X_prep, y=y_prep, eval_set=[(X_val_prep, y_val_prep)],
                       verbose_eval=True,early_stopping_rounds=train_params['early_stopping_rounds'],use_best_model=True)
        
    def predict_proba(self, X):
        X_prep, _ = self.prepare_dataset(X,y=None)
        Y_hat = pd.DataFrame(data=self.model.predict_proba(X_prep),columns=self.labels,index=X.index)
        
        return Y_hat
    
    def plot_learning(self, met_name='MultiClass'):
        history = self.model.evals_result_
        epochs = len(history['learn'][met_name])
        
        fig, ax = plt.subplots(1,1)
        sns.lineplot(ax=ax,x=np.arange(epochs),y=history['learn'][met_name], label='train',color='blue')
        sns.scatterplot(ax=ax,x=np.arange(epochs),y=history['learn'][met_name], color='blue')

        sns.lineplot(ax=ax,x=np.arange(epochs),y=np.array(history['validation'][met_name]), label='valid', color='red')
        sns.scatterplot(ax=ax,x=np.arange(epochs),y=np.array(history['validation'][met_name]), color='red') 
        ax.set_xlabel('Boosting rounds')
        ax.set_ylabel(met_name)          