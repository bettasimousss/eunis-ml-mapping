import numpy as np
import pandas as pd

from skmultilearn.model_selection import IterativeStratification

from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def my_palplot(pal,labels, size=1, ax=None):
    """Plot the values in a color palette as a horizontal array.
    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by seaborn.color_palette()
    size :
        scaling factor for size of plot
    ax :
        an existing axes to use
    """

    n = len(pal)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mpl.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(n) - .5)
    ax.set_yticks([-.5, .5])
    # Ensure nice border between colors
    ax.set_xticklabels(labels,rotation=90)
    # The proper way to set no ticks
    ax.yaxis.set_major_locator(ticker.NullLocator())
    #plt.xticks(rotation=90)

def set_eea_geom(lon,lat, grid_size=1e5):
    xmin = lon//grid_size
    ymin = lat//grid_size
    xcode='W' if xmin<0 else 'E'
    ycode='N' if ymin>0 else 'S'

    cell_code = 'EEA_%d%s%d%s'%(xmin,xcode,ymin,ycode)
    
    return cell_code

def mlsplit(global_data,target='T', n_splits=5,level=3,gs=100):
    target_data = global_data.query('EUNIS1==@target').dropna(subset=['EUNIS%d'%level],axis=0).copy()
    comm_data = (target_data[['grid','eea_%dkm'%gs,'EUNIS%d'%level]].pivot_table(index='eea_%dkm'%gs,columns='EUNIS%d'%level,fill_value=0,aggfunc=len)>0)*1
    target_data['fold']=-1
    
    indices = {}
    k_fold = IterativeStratification(n_splits=n_splits, order=1)
    X = comm_data.reset_index()[['eea_%dkm'%gs]].values
    Y = comm_data.values
    
    for f, (train, test) in enumerate(k_fold.split(X, Y)):
        print('Fold %d'%f)
        cells=comm_data.index[test].tolist()
        indices['fold%d'%f]=cells
        target_data.loc[target_data.query('eea_%dkm in @cells'%gs).index,'fold']=f
        
    return target_data, indices


def plot_occurrence(df, lon_var='decimalLongitude', lat_var='decimalLatitude',att=None,xlim=[-15,50],ylim=[30,90],title='',msize=5,bgd='black',palette=None, cmap=None,categorical=False):
    
    geometry = [Point(xy) for xy in zip(df[lon_var], df[lat_var])]

    gdf = GeoDataFrame(df, geometry=geometry)

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(figsize=(10, 6), color='black' if bgd is None else bgd)
    gdf.plot(ax=ax, marker='o', column=att if palette is None else None, color=gdf[att].map(palette) if palette is not None else None, markersize=5, legend=True,categorical=categorical, cmap=cmap if palette is None else None)
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.set_title(title)


def plot_partition(data, level=3):
    fig, ax = plt.subplots(1,1)
    data['fold'].hist(ax=ax)
    fig.suptitle('Fold statistics')
    
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    split_stats = data[['grid','fold','EUNIS%d'%level]].groupby(['fold','EUNIS%d'%level]).agg(len).reset_index()
    split_stats.pivot_table(index='EUNIS%d'%level,columns='fold',fill_value=0).plot.bar(stacked=True, ax=ax)
    fig.suptitle('Class distribution across folds')
    
    plot_occurrence(data, lon_var='Longitude', lat_var='Latitude',att='fold',xlim=[-15,50],ylim=[30,90],title='Fold distribution')