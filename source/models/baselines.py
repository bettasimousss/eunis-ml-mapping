from sklearn.dummy import DummyClassifier
from .habitat_model import HabitatModel

dummy_params = {
    'strategy':'prior'
}

class DummyHabitatModel(HabitatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(self, X,y):
        self.model = DummyClassifier(strategy=self.param_dict['strategy'])
        self.model.fit(X,y)
        
        self.labels = self.model.classes_
        
    def predict_proba(self, X):
        y_hat = pd.DataFrame(self.model.predict_proba(X),columns=self.labels,index=X.index)
        
        return y_hat
    
    
    

biogeo_params = {
    'idx': 'grid', 
    'att' : 'biogeo'
}

class BiogeoHabitatModel(HabitatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(self, X,y):
        idx=self.param_dict['idx']
        att=self.param_dict['att']
        
        train_data = X[[idx,att]].copy()
        train_data['label'] = y
        
        self.model = train_data.pivot_table(index=att,columns='label',fill_value=0,aggfunc=len)[idx]
        self.model.loc['OTHER',:] = self.model.sum().values
        self.model = self.model.apply(lambda x: x/sum(x),axis=1)
        
        
        self.labels = self.model.columns.tolist()
        self.clusters = self.model.index.tolist()
        
    def predict_proba(self, X):
        pool = self.clusters
        att = self.param_dict['att']
        
        X_in = X[[att]].copy()
        other = X_in.query('%s not in @pool'%att).index
        X_in.loc[other,att] = 'OTHER'
        
        y_hat = self.model.loc[X_in[att].tolist(),:]
        y_hat.index = X_in.index
        
        return y_hat