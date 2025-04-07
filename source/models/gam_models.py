from .habitat_model import HabitatModel
import pandas as pd
from pygam import LogisticGAM
from pygam import LogisticGAM
from pygam.terms import SplineTerm, LinearTerm, Term, Intercept
    
class GAMHabitatModel(HabitatModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(self, X, y):
        self.prefit(X, y)
        X_prep, y_prep = self.prepare_dataset(X,y)
        
        ## Get params
        term = self.param_dict['hyperparams']['terms']
        max_iter = self.param_dict['hyperparams']['max_iter']
        
        ## Set formula
        formule = Intercept()
        for v, vname in enumerate(X_prep.columns):
            if term=="linear":
                formule += LinearTerm(feature=v, verbose=True)
            else:
                formule += SplineTerm(feature=v, verbose=True)
        
        ## Fit individual models
        self.model = []
        for cl, label in enumerate(self.labels):
            print('Fitting model for class %d : %s'%(cl,label))
            y_in = (y_prep==cl)*1
            cl_model = LogisticGAM(terms=formule,max_iter=max_iter)
            cl_model.gridsearch(X=X_prep.values,y=y_in,weights=self.sample_w if self.param_dict['hyperparams']['samplew'] else None)
            self.model.append(cl_model)
    
    def predict_proba(self, X):
        X_prep, _ = self.prepare_dataset(X, y=None)
        Y_hat = pd.DataFrame(data=np.stack([mod.predict_proba(X_prep.values) for mod in self.model],axis=1), 
                             columns=self.labels, index=X_prep.index)
        
        return Y_hat