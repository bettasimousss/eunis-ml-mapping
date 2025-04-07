import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax

eps = 1e-138
def compute_eta(mu):
    eta = np.log(mu+eps)
    c = -np.mean(eta)
    
    return eta+c

def invert_softmax(pred):
    logits = pred.apply(compute_eta,axis=1)
    return logits


class TemperatureScaling(object):
    def __init__(self,predictions,labels,as_logit=True):          
        ## Getting true labels and encoding them
        self.labels = labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
        
        self.targets = self.label_encoder.transform(self.labels)
        
        ## Getting model predictions
        self.as_logit = as_logit
        self.predictions = predictions[self.label_encoder.classes_]   
        
        if self.as_logit:
            self.logits = self.predictions.values
        else:
            self.logits = invert_softmax(self.predictions).values
                
        ## Default values
        self.temperature = 1.0
          
    def get_proba(self):
        probas = softmax(self.logits * self.temperature, axis=1)
        return probas
    
    def compute_objective(self, temperature):
        nll = F.cross_entropy(input=torch.from_numpy(self.logits * temperature),
                              target=torch.from_numpy(self.targets)).numpy()
        
        return nll
    
    def optimize_temperature(self):
        optim = minimize_scalar(fun=self.compute_objective)
        self.temperature = optim.x
        
        return optim
    
    def plot_calibration(self, title=''):
        fig = plot_calibration_curve(probas=pd.DataFrame(data=self.get_proba(),columns=self.label_encoder.classes_),
                               labels=self.labels,nbins=10, title='%s - Temp: %.2f'%(title,self.temperature))
        
        return fig
    
    
def plot_calibration_curve(probas,labels,nbins=10,title=''):
    confidences = probas.max(axis=1)
    predictions = probas.idxmax(axis=1)    
    
    plot_data = pd.DataFrame(data=np.stack([predictions,confidences,labels],axis=1),columns=['yhat','confidence','ytrue'])
    plot_data['accuracy']=(plot_data['ytrue']==plot_data['yhat'])*1
    plot_data['bins'] = np.digitize(plot_data['confidence'],bins=[x/nbins for x in range(nbins)])
    plot_data['centroid']=plot_data['bins']/10
    plot_data['confidence']=plot_data['confidence'].astype(float)
    reliability_data = plot_data[['bins','accuracy','confidence','centroid']].groupby('bins').agg(np.mean)
    reliability_data['gap']=reliability_data['confidence']-reliability_data['accuracy']
    
    fig, ax = plt.subplots(1,1)
    ax.bar(reliability_data['centroid'], reliability_data['centroid'] , width=1.0/float(nbins), alpha=0.75, ec='#000000', fc='pink', label='Gap')
    ax.bar(reliability_data['centroid'], reliability_data['accuracy'], width=1.0/float(nbins), align='center', lw=1, ec='#000000', fc='#2233aa', alpha=1, label='Outputs', zorder=1)
    #plt.bar(reliability_data['centroid'], reliability_data['gap'] , width=1.0/float(nbins), alpha=1, label='Gap')
    ax.plot(np.linspace(0, 1.0, 20), np.linspace(0, 1.0, 20), '--', lw=2, alpha=.7, color='gray', label='Perfectly calibrated', zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('\nconfidence')
    ax.set_ylabel('accuracy\n')
    ax.set_title(title)
    ax.set_xticklabels(np.around(reliability_data['centroid'],decimals=1), rotation=-45)
    ax.legend(loc='upper left')
    fig.tight_layout()
    
    return fig    