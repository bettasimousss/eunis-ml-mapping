import sklearn.metrics as skm
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import RocCurveDisplay, roc_auc_score

    
'''
Y_score: pandas DataFrame with columns = classes in alphabetical order
Y_true: pandas Series

Use class codes as EUNIS codes, not label encoder to avoid mistakes and increase interpretability
'''

def eval_classifier(y_score,y_true, k_list=[3,5,10], model_name=None, super_class=None, classes=None):
    y_hat = y_score.idxmax(axis=1)
    perfs = pd.DataFrame.from_dict(skm.classification_report(y_true=y_true,y_pred=y_hat, output_dict=True, zero_division=0))
    
    if len(classes)>2:
        for k in k_list:
            perfs['top%d'%k] = skm.top_k_accuracy_score(y_true=y_true, y_score=y_score, k=k, labels=classes)
        
    perfs['adj_balanced_accuracy'] = skm.balanced_accuracy_score(y_true=y_true, y_pred=y_hat, adjusted=True)
    perfs['balanced_accuracy'] = skm.balanced_accuracy_score(y_true=y_true, y_pred=y_hat, adjusted=False)
    
    ##### Computing coverage: does the correct class rank well on average at least ? 
    ohe = OneHotEncoder(categories=[y_score.columns.tolist()],sparse_output=False)
    ohe.fit(y_true.values.reshape(-1,1))
    y_1h = pd.DataFrame(data=ohe.transform(y_true.values.reshape(-1,1)), columns=y_score.columns.tolist())
    
    perfs['coverage'] = skm.coverage_error(y_true=y_1h, y_score=y_score)
    
    #### Confusion matrix
    cm = pd.DataFrame(data=skm.confusion_matrix(y_pred=y_hat,y_true=y_true,labels=classes),columns=classes,index=classes)
    
    ##### Add model metadata
    perfs['nb_classes'] = y_score.shape[1]
    
    if model_name is not None:
        perfs['model_name']=model_name
        
    if super_class is not None:
        perfs['super_class']=super_class
        
    return perfs, cm


def plot_confusion_matrix(conf_mat, title='',gs=10):
    fig, ax = plt.subplots(1,1,figsize=(2*gs,gs))
    conf_norm = conf_mat.apply(lambda x: x/sum(x), axis=1)
    sns.heatmap(data=conf_norm,cmap='Reds',vmin=0,vmax=1,ax=ax)
    fig.suptitle(title)
    plt.close()
    
    return fig


def evaluate_ensemble(ens_model, cl, eval_dataset,att='EUNIS3',dname='dataset',covars=['biogeo']):
    print('Evaluation on %s dataset'%dname)
    eval_data = eval_dataset.query('EUNIS1==@cl').dropna(subset=[att])
    eval_data[att]=eval_data[att].astype(str)
    calib_pool = ens_model.labels
    test_pool = eval_data[att].unique()
    
    diff = set(test_pool).difference(calib_pool) 
    print('Evaluation labels not evaluated : ')
    print(diff)
    
    eval_data = eval_data.query('%s in @calib_pool'%att)
    if eval_data.shape[0]>0:
        eval_perfs = ens_model.evaluate(X=eval_data[['grid']+covars], y=eval_data[att])
    else:
        print('No observations from class %s in %s'%(cl,dname))
        eval_perfs = None
        
    del eval_data
    
    return eval_perfs
    
    
def mcroc_eval(Y_hat,y_true, title):
    Y_true = pd.get_dummies(y_true)
    fig, ax = plt.subplots(1,1)
    for class_id in Y_true.columns.tolist():
        RocCurveDisplay.from_predictions(
            Y_true[class_id].values,
            Y_hat[class_id].values,
            name=f"ROC curve for : %s"%class_id,
            ax=ax
        )

    fig.suptitle(title)
    
    auc_scores = {}
    for k in Y_hat.columns:
        try:
            score = roc_auc_score(y_score=Y_hat[k], y_true=Y_true[k])
            auc_scores[k] = score
        except:
            auc_scores[k] = np.nan
    
    return fig, auc_scores