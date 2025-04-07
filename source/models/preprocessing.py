from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, QuantileTransformer, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import itertools
import numpy as np
import pandas as pd

# Define custom transformer
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns].values
    
# Define custom transformer
class ColumnScaler(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X/self.scale_factor 
    
# Define custom transformer
class CyclicalTransformer(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, cycle):
        self.cycle = cycle
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        arr = X*(2.*np.pi/self.cycle)
        cos = np.cos(arr)
        sin = np.sin(arr)

        return np.concatenate([cos,sin],axis=1)
    
def create_pipeline_groupwise_pc(feature_metadata, pc_groups, colnames, original_names):
    pc_transformers = []
    pc_colnames = []
    
    ### Setting up PCA  objects groupwise
    for k, v in pc_groups.items():
        print(k)
        feats = feature_metadata.query('feature_group in @v')['feature_code'].tolist()
        k_feats = [colnames[idx] for idx, f in enumerate(original_names) if f in feats]
        k_pipeline = Pipeline([
                    ('selector', ColumnSelector(k_feats)),
                    ('pc', PCA(n_components=2))
                ])

        pc_colnames+=['%s_pc0'%k,'%s_pc1'%k]
        pc_transformers.append(('%s_pc'%k,k_pipeline))
        
    ### Union and aggregation in a common pipeline
    pc_feature_union = FeatureUnion(pc_transformers)

    return pc_feature_union, pc_colnames


def create_prep_pipeline(feature_metadata, pc_groups = None, categories=None, std=False, onehot=1):
    transformers = []
    colnames = []
    original_names = []
    cont_vars = feature_metadata.query('feature_type=="cont"')['feature_code'].tolist()
    if len(cont_vars)>0:
        if std:
            cont_prep = Pipeline([
                ('selector', ColumnSelector(cont_vars)),
                ('scale', StandardScaler())
            ])

        else:
            cont_prep = Pipeline([
                ('selector', ColumnSelector(cont_vars))
            ])


        colnames+=cont_vars   
        original_names+=cont_vars
        transformers.append(('cont',cont_prep))


    dcount_vars = feature_metadata.query('feature_type=="dcount"')['feature_code'].tolist()    
    if len(dcount_vars)>0:
        if std:
            dcount_prep = Pipeline([
                    ('selector', ColumnSelector(dcount_vars)),
                    ('scale', ColumnScaler(scale_factor=365))
                ])
        else:
            dcount_prep = Pipeline([
                ('selector', ColumnSelector(dcount_vars))
            ])

        colnames+=dcount_vars
        original_names+=dcount_vars
        transformers.append(('dcount',dcount_prep))

    freq_vars = feature_metadata.query('feature_type=="freq"')['feature_code'].tolist()
    freq_refs = feature_metadata.query('feature_type=="freq"')['reference'].values

    if len(freq_vars)>0:
        if std:
            freq_prep = Pipeline([
                ('selector', ColumnSelector(freq_vars)),
                ('scale', ColumnScaler(scale_factor=freq_refs))
            ])

        else:
            freq_prep = Pipeline([
                ('selector', ColumnSelector(freq_vars))
            ])

        colnames+=freq_vars
        original_names+=freq_vars
        transformers.append(('freq',freq_prep))

    cyclic_vars = feature_metadata.query('feature_type=="cycle"')['feature_code'].tolist()
    cyclic_refs = feature_metadata.query('feature_type=="cycle"')['reference'].values

    if len(cyclic_vars)>0:
        for cvar, cref in zip(cyclic_vars,cyclic_refs):
            cyclic_prep = Pipeline([
                ('selector', ColumnSelector([cvar])),
                ('scale', CyclicalTransformer(cycle=cref))
            ])

            transformers.append(('cyclic_%s'%cvar,cyclic_prep))
            original_names+=[cvar,cvar]

        colnames+=['%s_%s'%(v,f) for v in cyclic_vars for f in ['cos','sin']]

    cat_vars = feature_metadata.query('feature_type=="cat"')['feature_code'].tolist()
    if len(cat_vars)>0:
        if onehot==1:
            cat_prep = Pipeline([
                ('selector', ColumnSelector(cat_vars)),
                ('scale',OneHotEncoder(categories=categories, sparse_output=False,handle_unknown='ignore'))
            ])

            colnames+=list(itertools.chain(*categories))
            for i, cv in enumerate(cat_vars):
                original_names+=([cv]*len(categories[i]))
        
        elif onehot==2:
            cat_prep = Pipeline([
                ('selector', ColumnSelector(cat_vars)),
                ('scale',OrdinalEncoder(categories=categories, dtype=np.int32,handle_unknown='use_encoded_value',unknown_value=-1))
            ])

            colnames+=cat_vars  
            original_names+=cat_vars
            
        else:
            cat_prep = Pipeline([
                ('selector', ColumnSelector(cat_vars))
            ])

            colnames+=cat_vars
            original_names+=cat_vars

        transformers.append(('cat',cat_prep))

    feature_union = FeatureUnion(transformers)
    
    if pc_groups:
        pc_processor, pc_cols = create_pipeline_groupwise_pc(feature_metadata, pc_groups, colnames, original_names)
        preprocessor = Pipeline(steps=[
                ('union', feature_union),
                ("pandarizer1",FunctionTransformer(lambda x: pd.DataFrame(x, columns = colnames))),
                ('grouppc', pc_processor),
                ("pandarizer2",FunctionTransformer(lambda x: pd.DataFrame(x, columns = pc_cols)))
        ])
        
    else:
        preprocessor = Pipeline(steps=[
                        ('union', feature_union),
                        ("pandarizer1",FunctionTransformer(lambda x: pd.DataFrame(x, columns = colnames)))
                ])
        
        pc_cols = None
        
    return preprocessor, colnames, original_names, pc_cols