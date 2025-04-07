import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from .preprocessing import *
from .evaluation import *

class HabitatModel(object):
    def __init__(self, model_name=None, problem=None, param_dict=None, k_list=[3,5,10],output_logit=False, *args, **kwargs):
        self.model_name=model_name
        self.problem=problem
        self.param_dict=param_dict
        self.label_encoder = None
        
        ### problem setting
        self.labels=None
        self.covariates = None
        self.in_covars = None
        self.original_feats = None
        
        ### Evaluation params
        self.k_list=k_list
        
        ### Calibration params
        self.temperature = 1.0
        self.output_logit = output_logit
    
    def prefit(self, X, y):
        ## Fit label encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y)
        self.labels = self.label_encoder.classes_ 
        
        ## Compute class weights
        self.class_w = compute_class_weight(y=y,classes=self.labels,class_weight='balanced')
        self.sample_w = compute_sample_weight(y=y,class_weight='balanced')
        
        ## Fit feature preprocessor
        self.covariates = self.param_dict['inputs']['metadata']['feature_code'].unique().tolist()
        self.in_transfo, self.in_covars, self.original_feats, self.pc_cols = create_prep_pipeline(
            feature_metadata=self.param_dict['inputs']['metadata'],
            pc_groups=self.param_dict['inputs']['groups'], 
            categories=self.param_dict['inputs']['categories'],  
            std=self.param_dict['inputs']['std'], 
            onehot=self.param_dict['inputs']['onehot']
        )
        
        self.in_transfo.fit(X)
        
        if self.param_dict['inputs']['onehot']!=1:
            self.cat_vars = self.param_dict['inputs']['metadata'].query('feature_type=="cat"')['feature_code'].tolist()
        else:
            self.cat_vars = []
                                               
        self.num_vars = list(set(self.in_covars).difference(self.cat_vars)) 
        
        
    
    def prepare_dataset(self, X=None, y=None):
        if y is not None:
            y_prep = self.label_encoder.transform(y)
            
        else:
            y_prep = None
            
        if X is not None:
            X_prep = self.in_transfo.transform(X)
        else:
            X_prep = None
            
        return X_prep, y_prep
        
    def fit(self, X, y):
        pass
        
    def predict_proba(self, X):
        pass
    
    def predict(self, X):
        y_hat = self.predict_proba(X)
        
        return y_hat.idxmax(axis=1)
    
    def evaluate(self,X,y):
        y_hat = self.predict_proba(X)
        perfs, conf_mat = eval_classifier(y_score=y_hat,y_true=y, k_list=self.k_list, 
                                          model_name=self.model_name, 
                                          super_class=self.problem, classes=self.labels)
        
        return perfs, conf_mat
    
    def load_model(self,file):
        self.model = joblib.load(file)
    
    def save_model(self,file):
        joblib.dump(self.model, file)
        
        
class EnsembleHabitatModel(object):
    def __init__(self, model_names=[], models=[], k_list=[3,5,10], ensemble_name='ensemble', problem=None):
        self.model_names = model_names
        self.models = models
        self.labels = np.unique([c for mod in self.models for c in mod.labels]).tolist()
        
        self.ensemble_name = ensemble_name
        self.problem = problem
        self.k_list = k_list
        
    def predict_proba(self,X):
        n = X.shape[0]
        m = len(self.labels)
        p = len(self.models)
        
        Y_score = pd.DataFrame(data=np.zeros((n,m)),columns=self.labels,index=X.index)
        Y_committee = pd.DataFrame(data=np.zeros((n,m)),columns=self.labels,index=X.index)
        Y_raw = {}
        
        for mod_name, mod in zip(self.model_names, self.models):
            print('Predicting using %s'%mod_name)
            
            ## Sum probas
            y_hat = mod.predict_proba(X)
            Y_score[y_hat.columns]+=y_hat.values
            
            ## Sum indicators
            y_class = pd.get_dummies(y_hat.idxmax(axis=1))
            Y_committee[y_class.columns]+=y_class.values
            
            Y_raw[mod_name]=y_hat
        
        print('Voting')
        ### Soft voting
        Y_score = Y_score / p
        
        ###  Hard voting
        Y_committee = Y_committee / p
        
        return Y_raw, Y_score, Y_committee
    
    
    def evaluate(self,X,y):
        _, Y_score, Y_committee = self.predict_proba(X)
        
        soft_perfs, soft_conf_mat = eval_classifier(y_score=Y_score,y_true=y, k_list=self.k_list,
                                                    model_name='soft_%s'%self.ensemble_name, super_class=self.problem, classes=self.labels)
        
        hard_perfs, hard_conf_mat = eval_classifier(y_score=Y_committee,y_true=y, 
                                                    k_list=self.k_list,model_name='hard_%s'%self.ensemble_name, 
                                                    super_class=self.problem, classes=self.labels)
        
        return soft_perfs, hard_perfs, soft_conf_mat, hard_conf_mat        