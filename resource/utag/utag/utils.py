#!/usr/bin/env python

"""
Helper functions used throughout the package.
"""

import typing as tp

import numpy as np
import scipy
import pandas as pd
import networkx as nx

from utag.types import Array, Graph, DataFrame, Path, AnnData


def domain_connectivity(
    adata: AnnData,
    slide_key: str = 'Slide',
    domain_key: str = 'UTAG Label',
) -> AnnData:
    import squidpy as sq
    import numpy as np
    from tqdm import tqdm

    order = sorted(adata.obs[domain_key].unique().tolist())

    global_pairwise_connection = pd.DataFrame(np.zeros(shape = (len(order),len(order))), index = order, columns = order)
    for slide in tqdm(adata.obs[slide_key].unique()):
        adata_batch = adata[adata.obs[slide_key] == slide].copy()

        sq.gr.spatial_neighbors(adata_batch, radius = 40, coord_type = 'generic')

        pairwise_connection = pd.DataFrame(index = order, columns = order)
        for label in adata_batch.obs[domain_key].unique():
            self_connection = adata_batch[adata_batch.obs[domain_key] == label].obsp['spatial_connectivities'].todense().sum()/2
            self_connection = self_connection.round()

            pairwise_connection.loc[label, label] = self_connection

        for label in adata_batch.obs[domain_key].unique():
            for label2 in adata_batch.obs[domain_key].unique():
                if label != label2:
                    pairwise = adata_batch[adata_batch.obs[domain_key].isin([label, label2])].obsp['spatial_connectivities'].todense().sum()/2
                    pairwise = pairwise.round()
                    pairwise_connection.loc[label, label2] = pairwise - pairwise_connection.loc[label, label] - pairwise_connection.loc[label2, label2]
                    pairwise_connection.loc[label2, label] = pairwise_connection.loc[label, label2]

        pairwise_connection = pairwise_connection.fillna(0)
        global_pairwise_connection = global_pairwise_connection + pairwise_connection
    adata.uns[f'{domain_key}_domain_adjacency_matrix'] =  global_pairwise_connection
    return adata

def celltype_connectivity(
    adata: AnnData,
    slide_key: str = 'Slide',
    domain_key: str = 'UTAG Label',
    celltype_key: str = 'cluster_0.5_label',
) -> AnnData:
    import squidpy as sq
    import numpy as np
    from tqdm import tqdm

    global_pairwise_utag = dict()
    for label in adata.obs[domain_key].unique():
        cell_types = adata.obs[celltype_key].unique().tolist()
        global_pairwise_utag[label] = pd.DataFrame(np.zeros(shape = (len(cell_types),len(cell_types))), index = cell_types, columns = cell_types)

    for slide in tqdm(adata.obs[slide_key].unique()):
        adata_batch = adata[adata.obs[slide_key] == slide].copy()    
        sq.gr.spatial_neighbors(adata_batch, radius = 40, coord_type = 'generic')

        for label in adata.obs[domain_key].unique():
            adata_batch2 = adata_batch[adata_batch.obs[domain_key] == label].copy()
            pairwise_connection = pd.DataFrame(index = cell_types, columns = cell_types)

            for cell_type1 in adata_batch2.obs[celltype_key].unique():
                self_connection = adata_batch2[adata_batch2.obs[celltype_key] == cell_type1].obsp['spatial_connectivities'].todense().sum()/2
                self_connection = self_connection.round()

                pairwise_connection.loc[cell_type1, cell_type1] = self_connection

            for cell_type1 in adata_batch.obs[celltype_key].unique():
                for cell_type2 in adata_batch2.obs[celltype_key].unique():
                    if cell_type1 != cell_type2:
                        pairwise = adata_batch2[adata_batch2.obs[celltype_key].isin([cell_type1, cell_type2])].obsp['spatial_connectivities'].todense().sum()/2
                        pairwise = pairwise.round()
                        pairwise_connection.loc[cell_type1, cell_type2] = pairwise - pairwise_connection.loc[cell_type1, cell_type1] - pairwise_connection.loc[cell_type2, cell_type2]
                        pairwise_connection.loc[cell_type2, cell_type1] = pairwise_connection.loc[cell_type1, cell_type2]

            pairwise_connection = pairwise_connection.fillna(0)
            global_pairwise_utag[label] = global_pairwise_utag[label] + pairwise_connection
            
    adata.uns[f'{domain_key}_celltype_adjacency_matrix'] = global_pairwise_utag
    return adata


def slide_connectivity(
    adata: AnnData,
    slide_key: str = 'roi',
    domain_key: str = 'UTAG Label',
) -> dict():
    import squidpy as sq
    import numpy as np
    from tqdm import tqdm

    order = sorted(adata.obs[domain_key].unique().tolist())
    slide_connection = dict()
    
    for slide in tqdm(adata.obs[slide_key].unique()):
        adata_batch = adata[adata.obs[slide_key] == slide].copy()

        sq.gr.spatial_neighbors(adata_batch, radius = 40, coord_type = 'generic')

        pairwise_connection = pd.DataFrame(index = order, columns = order)
        for label in adata_batch.obs[domain_key].unique():
            self_connection = adata_batch[adata_batch.obs[domain_key] == label].obsp['spatial_connectivities'].todense().sum()/2
            self_connection = self_connection.round()

            pairwise_connection.loc[label, label] = self_connection

        for label in adata_batch.obs[domain_key].unique():
            for label2 in adata_batch.obs[domain_key].unique():
                if label != label2:
                    pairwise = adata_batch[adata_batch.obs[domain_key].isin([label, label2])].obsp['spatial_connectivities'].todense().sum()/2
                    pairwise = pairwise.round()
                    pairwise_connection.loc[label, label2] = pairwise - pairwise_connection.loc[label, label] - pairwise_connection.loc[label2, label2]
                    pairwise_connection.loc[label2, label] = pairwise_connection.loc[label, label2]

        pairwise_connection = pairwise_connection.fillna(0)
        pairwise_connection = pairwise_connection.loc[(pairwise_connection!=0).any(1), (pairwise_connection!=0).any(0)]
        #pairwise_connection = pairwise_connection.dropna(axis = 1)
        slide_connection[slide] = pairwise_connection
        
    return slide_connection

def measure_per_domain_cell_type_colocalization(
    adata: AnnData,
    utag_key: str = "UTAG Label",
    max_dist: int = 40,
    n_iterations: int = 100,
):
    import squidpy as sq
    a_ = adata.copy()
    sq.gr.spatial_neighbors(a_, radius=max_dist, coord_type="generic")

    G = nx.from_scipy_sparse_matrix(a_.obsp["spatial_connectivities"])

    utag_map = {i: x for i, x in enumerate(adata.obs[utag_key])}
    nx.set_node_attributes(G, utag_map, name=utag_key)

    adj, order = nx.linalg.attrmatrix.attr_matrix(G, node_attr=utag_key)
    order = pd.Series(order).astype(adata.obs[utag_key].dtype)
    freqs = pd.DataFrame(adj, order, order).fillna(0) + 1

    norm_freqs = correct_interaction_background_random(G, freqs, utag_key, n_iterations)
    return norm_freqs


def correct_interaction_background_random(
    graph: nx.Graph, freqs: pd.DataFrame, attribute: str, n_iterations: int = 100
):
    values = {x: graph.nodes[x][attribute] for x in graph.nodes}
    shuffled_freqs = list()
    for _ in range(n_iterations):
        g2 = graph.copy()
        shuffled_attr = pd.Series(values).sample(frac=1)
        shuffled_attr.index = values
        nx.set_node_attributes(g2, shuffled_attr.to_dict(), name=attribute)
        rf, rl = nx.linalg.attrmatrix.attr_matrix(g2, node_attr=attribute)
        rl = pd.Series(rl, dtype=freqs.index.dtype)
        shuffled_freqs.append(pd.DataFrame(rf, index=rl, columns=rl))#.fillna(0) + 1)
    shuffled_freq = pd.concat(shuffled_freqs)
    shuffled_freq = shuffled_freq.groupby(level=0).sum()
    shuffled_freq = shuffled_freq.fillna(0) + 1
    
    fl = np.log((freqs / freqs.values.sum()))
    sl = np.log((shuffled_freq / shuffled_freq.values.sum()))
    # make sure both contain all edges/nodes
    fl = fl.reindex(sl.index, axis=0).reindex(sl.index, axis=1)
    sl = sl.reindex(fl.index, axis=0).reindex(fl.index, axis=1)
    return fl - sl


def evaluate_clustering(
    adata: AnnData,
    cluster_keys: Array,
    celltype_label: str = 'celltype',
    slide_key: str = 'roi',
    metrics: Array = ['entropy', 'cluster_number', 'silhouette_score', 'connectivity']
) -> DataFrame:
    
    if type(cluster_keys) == str:
        cluster_keys = [cluster_keys]
    if type(metrics) == str:
        metrics = [metrics]
    
    cluster_loss = pd.DataFrame(index = metrics, columns = cluster_keys)
    from tqdm import tqdm
    
    for metric in metrics:
        print(f'Evaluating Cluster {metric}')
        for cluster in tqdm(cluster_keys):    
            assert(metric in ['entropy', 'cluster_number', 'silhouette_score', 'connectivity'])
            
            if metric == 'entropy':
                from scipy.stats import entropy
                distribution = adata.obs.groupby([celltype_label, cluster]).count()[slide_key].reset_index().pivot(index = cluster, columns = celltype_label, values = slide_key)
                cluster_entropy = distribution.apply(entropy, axis = 1).sort_values().mean()
                
                cluster_loss.loc[metric, cluster] = cluster_entropy
            elif metric == 'cluster_number':
                
                cluster_loss.loc[metric, cluster] = len(adata.obs[cluster].unique())
            elif metric == 'silhouette_score':
                
                from sklearn.metrics import silhouette_score
                cluster_loss.loc[metric, cluster] = silhouette_score(adata.X, labels = adata.obs[cluster])
            elif metric == 'connectivity':
                global_pairwise_connection = domain_connectivity(adata = adata, slide_key = slide_key, domain_key = cluster)
                inter_spatial_connectivity = np.log(np.diag(global_pairwise_connection).sum() / (global_pairwise_connection.sum().sum() - np.diag(global_pairwise_connection).sum()))

                cluster_loss.loc[metric, cluster] = inter_spatial_connectivity
    return cluster_loss

def to_uint(x: Array, base: int = 8) -> Array:
    return (x * (2 ** base - 1)).astype(f"uint{base}")


def to_float(x: Array, base: int = 32) -> Array:
    return (x / x.max()).astype(f"float{base}")


def open_image_with_tf(filename: str, file_type="png"):
    import tensorflow as tf

    img = tf.io.read_file(filename)
    return tf.io.decode_image(img, file_type)


def filter_kwargs(
    kwargs: tp.Dict[str, tp.Any], callabl: tp.Callable, exclude: bool = None
) -> tp.Dict[str, tp.Any]:
    from inspect import signature

    args = signature(callabl).parameters.keys()
    if "kwargs" in args:
        return kwargs
    return {k: v for k, v in kwargs.items() if (k in args) and k not in (exclude or [])}


def array_to_graph(
    arr: Array,
    max_dist: int = 5,
    node_attrs: tp.Mapping[int, tp.Mapping[str, tp.Union[str, int, float]]] = None,
) -> Graph:
    """
    Generate a Graph of object distance-based connectivity in euclidean space.

    Parameters
    ----------
    arr: np.ndarray
        Labeled array.
    """
    mask = arr > 0
    idx = arr[mask]
    xx, yy = np.mgrid[: arr.shape[0], : arr.shape[1]]
    arri = np.stack([xx[mask], yy[mask]]).T
    dists = pd.DataFrame(scipy.spatial.distance.cdist(arri, arri), index=idx, columns=idx)
    np.fill_diagonal(dists.values, np.nan)

    attrs = dists[dists <= max_dist].reset_index().melt(id_vars="index").dropna()
    attrs.index = attrs.iloc[:, :2].apply(tuple, axis=1).tolist()
    value = attrs["value"]
    g = nx.from_edgelist(attrs.index)
    nx.set_edge_attributes(g, value.to_dict(), "distance")
    nx.set_edge_attributes(g, (1 / value).to_dict(), "connectivity")

    if node_attrs is not None:
        nx.set_node_attributes(g, node_attrs)

    return g

def compute_and_draw_network(
    adata,
    slide_key: str = 'roi',
    node_key: str = 'UTAG Label',
    figsize: tuple = (11,11),
    dpi: int = 100,
    font_size: int = 12,
    node_size_min: int = 1000,
    node_size_max: int = 3000,
    edge_weight: float = 10,
    log_transform: bool = True,
    ax = None
) -> nx.Graph:
    from utag.utils import domain_connectivity
    import networkx as nx
    import matplotlib.pyplot as plt
    
    adjacency_matrix = domain_connectivity(adata = adata, slide_key = slide_key, domain_key = node_key)
    s1 = adata.obs.groupby(node_key).count()
    s1 = s1[s1.columns[0]]
    node_size = s1.values
    node_size = (node_size - node_size.min()) / (node_size.max() - node_size.min()) * (node_size_max - node_size_min) + node_size_min
    
    if ax == None:
        fig = plt.figure(figsize = figsize, dpi = dpi)
    G = nx.from_numpy_matrix(np.matrix(adjacency_matrix), create_using=nx.Graph)
    G = nx.relabel.relabel_nodes(G, {i: label for i, label in enumerate(adjacency_matrix.index)})
    pos = nx.circular_layout(G)

    edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())
    if log_transform:
        weights = np.log(np.array(list(weights))+1)
    else:
        weights = np.array(list(weights))
    weights = (weights - weights.min()) / (weights.max() - weights.min()) * edge_weight + 0.2
    weights = tuple(weights.tolist())
    
    if ax:
        nx.draw(G, pos, node_color='w', edgelist=edges, edge_color=weights, width=weights, edge_cmap=plt.cm.YlOrRd, with_labels=True, font_size = font_size, node_size = node_size, ax = ax)
    else:
        nx.draw(G, pos, node_color='w', edgelist=edges, edge_color=weights, width=weights, edge_cmap=plt.cm.YlOrRd, with_labels=True, font_size = font_size, node_size = node_size)
    #nx.draw(G, pos, cmap = plt.cm.tab10, node_color = range(8), edgelist=edges, edge_color=weights, width=3, edge_cmap=plt.cm.coolwarm, with_labels=True, font_size = 14, node_size = 1000)

    if ax == None:
        ax = plt.gca()
    
    color_key = node_key + '_colors'
    if color_key in adata.uns:
        ax.collections[0].set_edgecolor(adata.uns[color_key])
    else:
        ax.collections[0].set_edgecolor('lightgray')
    ax.collections[0].set_linewidth(3)
    ax.set_xlim([1.1*x for x in ax.get_xlim()])
    ax.set_ylim([1.1*y for y in ax.get_ylim()])

    return G
    
def get_adjacency_matrix(g: Graph) -> Array:
    return nx.adjacency_matrix(g, weight="connectivity").todense()


def get_feature_matrix(g: Graph) -> DataFrame:
    return pd.DataFrame({n: g.nodes[n] for n in g.nodes}).T


def message_pass_graph(adj: Array, feat: DataFrame) -> DataFrame:
    return (adj @ feat).set_index(feat.index)


def pad_feature_matrix(df: DataFrame, size: int) -> DataFrame:
    index = df.index.tolist() + (df.index.max() + np.arange(size - df.shape[0])).tolist()
    return pd.DataFrame(
        np.pad(
            df.values,
            [(0, size - df.shape[0]), (0, 0)],
        ),
        index=index,
        columns=df.columns,
    )


def pad_adjacency_matrix(mat: DataFrame, size: int) -> DataFrame:
    return np.pad(
        mat,
        [
            (0, size - mat.shape[0]),
            (0, size - mat.shape[0]),
        ],
    )


def message_pass_graphs(gs: tp.Sequence[Graph]) -> Array:
    n = max([len(g) for g in gs])
    _adjs = list()
    _feats = list()
    for g in gs:
        adj = get_adjacency_matrix(g)
        adj = pad_adjacency_matrix(adj, n)
        _adjs.append(adj)
        feat = get_feature_matrix(g)
        feat = pad_feature_matrix(feat, n)
        _feats.append(feat)
    adjs = np.stack(_adjs)
    feats = np.stack(_feats).astype(float)

    return adjs @ feats


def mask_to_labelme(
    labeled_image: Array,
    filename: Path,
    overwrite: bool = False,
    simplify: bool = True,
    simplification_threshold: float = 5.0,
) -> None:
    import io
    import base64
    import json

    import imageio
    import tifffile
    from imantics import Mask
    from shapely.geometry import Polygon

    output_file = filename.replace_(".tif", ".json")
    if overwrite or output_file.exists():
        return
    polygons = Mask(labeled_image).polygons()
    shapes = list()
    for point in polygons.points:

        if not simplify:
            poly = np.asarray(point).tolist()
        else:
            poly = np.asarray(
                Polygon(point).simplify(simplification_threshold).exterior.coords.xy
            ).T.tolist()
        shape = {
            "label": "A",
            "points": poly,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        }
        shapes.append(shape)

    f = io.BytesIO()
    imageio.imwrite(f, tifffile.imread(filename), format="PNG")
    f.seek(0)
    encoded = base64.encodebytes(f.read())

    payload = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": filename.name,
        "imageData": encoded.decode("ascii"),
        "imageHeight": labeled_image.shape[0],
        "imageWidth": labeled_image.shape[1],
    }
    with open(output_file.as_posix(), "w") as fp:
        json.dump(payload, fp, indent=2)


def z_score(x: Array) -> Array:
    """
    Scale (divide by standard deviation) and center (subtract mean) array-like objects.
    """
    return (x - x.min()) / (x.max() - x.min())


def sparse_matrix_dstack(
    matrices: tp.Sequence[scipy.sparse.csr_matrix],
) -> scipy.sparse.csr_matrix:
    """
    Diagonally stack sparse matrices.
    """
    import scipy
    from tqdm import tqdm

    n = sum([x.shape[0] for x in matrices])
    _res = list()
    i = 0
    for x in tqdm(matrices):
        v = scipy.sparse.csr_matrix((x.shape[0], n))
        v[:, i : i + x.shape[0]] = x
        _res.append(v)
        i += x.shape[0]
    return scipy.sparse.vstack(_res)
