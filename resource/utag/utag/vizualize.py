import os

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata
import holoviews as hv
from holoviews import opts, dim

from utag.types import Path, Array, AnnData, DataFrame


def add_spatial_image(
    adata: AnnData,
    image_path: Path,
    rgb_channels = [19, 9, 14],
    log_transform: bool = False,
    median_filter: bool = False,
    scale_method: str = 'adjust_gamma',
    contrast_percentile = (0, 90),
    gamma: float = 0.2,
    gain: float = 0.5
):
      
    adata.obsm['spatial'] = adata.obs[['Y_centroid', 'X_centroid']].to_numpy()
    adata.uns["spatial"] = {'image': {}}
    adata.uns["spatial"]['image']["images"] = {}
    
    img = rgbfy_multiplexed_image(
        image_path = image_path,
        rgb_channels = rgb_channels,
        contrast_percentile = contrast_percentile,
        log_transform = log_transform,
        median_filter = median_filter,
        scale_method = scale_method,
        gamma = gamma,
        gain = gain
    )
    
    
    adata.uns["spatial"]['image']["images"] = {"hires": img}
    adata.uns["spatial"]['image']["scalefactors"] = {"tissue_hires_scalef": 1, "spot_diameter_fullres": 1}
    return adata

def add_scale_box_to_fig(
    img: Array,
    ax,
    box_width: int = 100,
    box_height: float = 3,
    color: str = 'white'
) -> Array:    
    import matplotlib.patches as patches
    x = img.shape[1]
    y = img.shape[0]
    
    # Create a Rectangle patch
    rect = patches.Rectangle((x - box_width, y * (1-box_height/100)), box_width, y * (box_height/100), linewidth=0.1, edgecolor='black', facecolor=color)

    # Add the patch to the Axes
    ax.add_patch(rect)
    return ax

def rgbfy_multiplexed_image(
    image_path: Path,
    rgb_channels = [19, 9, 14],
    log_transform: bool = True,
    median_filter: bool = True,
    scale_method: str = 'adjust_gamma',
    contrast_percentile = (10, 90),
    gamma: float = 0.4,
    gain: float = 1
) -> Array:
    from skimage.exposure import rescale_intensity, adjust_gamma, equalize_hist
    from scipy.ndimage import median_filter as mf
    import tifffile
    
    def rescale(img, contrast_percentile):
        r1, r2 = np.percentile(img, contrast_percentile)
        img = rescale_intensity(img, in_range = (r1, r2), out_range = (0,1))
        return img
    #assert(len(rgb_channels) == 3 or len(rgb_channels) == 1)
    
    img = tifffile.imread(image_path)
    img = img.astype(np.float32)
    if median_filter == True:
        img = mf(img, size = 3)
        
    image_to_save = np.stack([img[x] for x in rgb_channels], axis = 2)
    
    for i in range(len(rgb_channels)):
        if log_transform == True:
            image_to_save[:,:,i] = np.log(image_to_save[:,:,i] + 1)
        else:
            image_to_save[:,:,i] = image_to_save[:,:,i]
    
    output_img = image_to_save
        
    for i in range(3):
        if scale_method == 'contrast_stretch':
            output_img[:,:,i] = rescale(output_img[:,:,i], contrast_percentile)
        elif scale_method == 'adjust_gamma':
            output_img[:,:,i] = adjust_gamma(output_img[:,:,i], gamma=gamma, gain=gain)
            #output_img[:,:,i] = rescale(output_img[:,:,i], contrast_percentile)
        elif scale_method == 'equalize_hist':
            output_img[:,:,i] = equalize_hist(output_img[:,:,i])
            
        output_img[:,:,i] = np.clip(output_img[:,:,i], 0, 1)
    return output_img


def draw_network(
    adata: AnnData,
    node_key: str = 'UTAG Label',
    adjacency_matrix_key: str = 'UTAG Label_domain_adjacency_matrix',
    figsize: tuple = (11,11),
    dpi: int = 200,
    font_size: int = 12,
    node_size_min: int = 1000,
    node_size_max: int = 3000,
    edge_weight: float = 5,
    edge_weight_baseline: float = 1,
    log_transform: bool = True,
    ax = None
):
    import networkx as nx
    s1 = adata.obs.groupby(node_key).count()
    s1 = s1[s1.columns[0]]
    node_size = s1.values
    node_size = (node_size - node_size.min()) / (node_size.max() - node_size.min()) * (node_size_max - node_size_min) + node_size_min
    
    if ax == None:
        fig = plt.figure(figsize = figsize, dpi = dpi)
    G = nx.from_numpy_matrix(np.matrix(adata.uns[adjacency_matrix_key]), create_using=nx.Graph)
    G = nx.relabel.relabel_nodes(G, {i: label for i, label in enumerate(adata.uns[adjacency_matrix_key].index)})
    
    edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
    
    if log_transform:
        weights = np.log(np.array(list(weights))+1)
    else:
        weights = np.array(list(weights))
        
    weights = (weights - weights.min()) / (weights.max() - weights.min()) * edge_weight + edge_weight_baseline
    weights = tuple(weights.tolist())
    
    #pos = nx.spectral_layout(G, weight = 'weight')
    pos = nx.spring_layout(G, weight = 'weight', seed = 42, k = 1)
    
    if ax:
        nx.draw(G, pos, node_color='w', edgelist=edges, edge_color=weights, width=weights, edge_cmap=plt.cm.YlOrRd, with_labels=True, font_size = font_size, node_size = node_size, ax = ax)
    else:
        nx.draw(G, pos, node_color='w', edgelist=edges, edge_color=weights, width=weights, edge_cmap=plt.cm.YlOrRd, with_labels=True, font_size = font_size, node_size = node_size)

    if ax == None:
        ax = plt.gca()
    
    color_key = node_key + '_colors'
    if color_key in adata.uns:
        ax.collections[0].set_edgecolor(adata.uns[color_key])
        ax.collections[0].set_facecolor(adata.uns[color_key])
    else:
        ax.collections[0].set_edgecolor('lightgray')
    ax.collections[0].set_linewidth(3)
    ax.set_xlim([1.3*x for x in ax.get_xlim()])
    ax.set_ylim([1*y for y in ax.get_ylim()])
    
    if ax == None:
        return fig

def adj2chord(
    adjacency_matrix: Array,
    size:int = 300
):
    hv.output(fig='svg', size=size)
    
    links = adjacency_matrix.stack().reset_index().rename(columns = {'level_0': 'source', 'level_1': 'target', 0: 'value'}).dropna()
    order2ind = {k:i for i, k in enumerate(adjacency_matrix.index.tolist())}
    
    links['source'] = links['source'].replace(order2ind)
    links['target'] = links['target'].replace(order2ind)
    links['value'] = links['value'].astype(int)

    nodes = pd.DataFrame(order2ind.keys(), index = order2ind.values(), columns = ['name']).reset_index()
    nodes['group'] = nodes['index']
    del nodes['index']
    nodes = hv.Dataset(nodes, 'index')

    chord = hv.Chord((links, nodes)).select(value=(5, None))
    chord.opts(
        opts.Chord(
            cmap='tab10',
            edge_cmap='tab10',
            edge_color=dim('source').str(), 
            labels='name',
            node_color=dim('index').str()
        )
    )
    
    return chord