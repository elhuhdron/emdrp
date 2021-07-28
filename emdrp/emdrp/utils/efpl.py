import numpy as np
import pandas as pd
import wknml
import h5py

KNOSSOS_BASE =  np.ones((1, 3), dtype=int)
NML_DIM_ORDER = 'xyz'
HDF5_DIM_ORDER = 'zyx'

HDF5_BACKGROUND_LABEL = 0

def calc_supervoxel_eftpl(nml_filepath, h5_filepath, dset_vals=None, dset_folder='with_background', dset_suffix='labels',
    size=None, dset_start=np.zeros((1,3), dtype=int), 
    ):

    with open(nml_filepath, "rb") as f:
        nml = wknml.parse_nml(f)

    if dset_vals is None:
        with h5py.File(h5_filepath, 'r') as h5file:
            dset = h5file[dset_folder]
            dset_vals = [str(k) for k in dset.keys()]

    eftpl = np.zeros(len(dset_vals))

    h5_dim_idcs = [HDF5_DIM_ORDER.index(ax) for ax in 'xyz']

    selection = None
    if size is not None:
        selection = tuple(slice(dset_start[ax],dset_start[ax]+size[ax]) for ax in h5_dim_idcs)

    for i, dset_val in enumerate(dset_vals):
        with h5py.File(h5_filepath, 'r') as h5file:
            dset = h5file['/'.join([dset_folder, dset_val, dset_suffix])]
            if size is None:
                Vlbls = dset[:]
            else:
                Vlbls = np.zeros(tuple(size[ax] for ax in h5_dim_idcs), dtype=dset.dtype)
                dset.read_direct(Vlbls, selection)
        eftpl[i] = calc_eftpl(nml, Vlbls, dset_start)

    return eftpl, dset_vals


def calc_eftpl(nml, Vlbls, dataset_start=np.zeros((1,3), dtype=int), verbose=False):
    
    ### Extract, clean, and transform node data from nml file

    nodes = pd.DataFrame([n for t in nml.trees for n in t.nodes])

    idx_columns = [f'idx_{ax}' for ax in NML_DIM_ORDER]
    for n ,col in enumerate(idx_columns):
        nodes[col] = nodes['position'].apply(lambda position: position[n])

    nodes = nodes.drop(columns=[col for col in nodes.columns if not col in ['id'] + idx_columns])

    nodes = nodes.astype({c:int for c in nodes.columns})

    nodes[idx_columns] -= (dataset_start + KNOSSOS_BASE)

    # Comment(erjel): That moment when you wrote an on-point pipeline and realize that you
    # have to deal with messy data ...
    hdf5_dim_idcs = np.array([NML_DIM_ORDER.index(ax) for ax in HDF5_DIM_ORDER])

    node_is_outside_volume = np.logical_or(
        np.any(nodes[idx_columns] >= np.array([[Vlbls.shape[ax_idx] for ax_idx in hdf5_dim_idcs]]), axis=1),
        np.any(nodes[idx_columns] < 0, axis=1))

    if np.any(node_is_outside_volume):
        print('Warning: Remove {} nodes outside the ROI '.format(np.sum(node_is_outside_volume)))

    node_id_outside = nodes[node_is_outside_volume].id

    nodes = nodes[np.logical_not(node_is_outside_volume)]

    nodes['label'] = Vlbls[tuple(nodes[f'idx_{ax}'] for ax in HDF5_DIM_ORDER)]

    position_colums = [f'position_{ax}' for ax in NML_DIM_ORDER]
    nodes[position_colums] = nodes[idx_columns] * np.array(nml.parameters.scale)

    ### Extract, clean, and transform edge data from nml file
    edges = pd.DataFrame(nml.trees)

    edges = edges.drop(columns=[ col for col in edges.columns if not col in ['id', 'edges'] ])
    edges = edges.rename(columns={"id": "tree_id"})

    edges = edges.explode('edges')
    edge_is_nan = edges['edges'].isna()
    if np.any(edge_is_nan):
        print(f'Warning: {np.sum(edge_is_nan)} edges have nan values!')

    edges = edges[np.logical_not(edges['edges'].isna())]

    new_col_list = ['source_node','target_node']
    for n ,col in enumerate(new_col_list):
        edges[col] = edges['edges'].apply(lambda location: location[n])

    edge_is_outside = edges.source_node.isin(node_id_outside) | edges.target_node.isin(node_id_outside)
    if np.any(edge_is_outside):
        print(f'Warning: {np.sum(edge_is_outside)} edges lead to nodes outside the ROI')

    edges = edges[np.logical_not(edge_is_outside)]    
    edges = edges.drop(columns=['edges'])

    ### Merge nodes information into edge data and calculate path lengths
    edges = edges.merge(nodes[ ['id', 'label'] + position_colums ], left_on='source_node', right_on='id')
    edges = edges.merge(nodes[ ['id', 'label'] + position_colums ], left_on='target_node', right_on='id',
                        suffixes=['_source', '_target'])

    edges = edges.drop(columns=['source_node', 'target_node', 'id_source', 'id_target'])

    edges['path_length'] = np.linalg.norm(
            edges[ [pos + '_source' for pos in position_colums] ].values
            - edges[ [pos + '_target' for pos in position_colums] ].values
        )

    edges = edges.drop(columns=[c for c in edges.columns if c.startswith('position_')])

    ### Split error occurs if label at source node and label at target node are different 
    ###  or the node label is identical to the background
    edges['has_split_error'] = (edges['label_source'] != edges['label_target'])\
                                 | (edges['label_target'] == HDF5_BACKGROUND_LABEL)\
                                 | (edges['label_source'] == HDF5_BACKGROUND_LABEL)

    verbose and print(f'split error ratio = {edges.has_split_error.sum() / len(edges):.2f}')

    ### Create third (labels) dataframe to collect information about errors

    labels = edges[['label_target', 'label_source', 'tree_id']]

    labels = labels.drop_duplicates()

    labels = labels[['tree_id', 'label_target']].rename(columns={'label_target':'label'})\
                .append(labels[['tree_id', 'label_source']].rename(columns={'label_source':'label'}))   

    labels = labels.drop_duplicates()

    labels = labels.groupby('label', as_index=False).count()

    labels['has_merge_error'] = (labels['tree_id'] > 1) & (labels['label'] != HDF5_BACKGROUND_LABEL)

    verbose and print(f'merge error ratio per label = {labels.has_merge_error.sum()/ len(labels):.2f}')

    ### Use labels information for merge error per edge

    edges = edges.merge(labels[['label', 'has_merge_error']], left_on='label_target', right_on='label')
    edges = edges.merge(labels[['label', 'has_merge_error']], left_on='label_source', right_on='label')
    edges['has_merge_error'] = edges['has_merge_error_x'] | edges['has_merge_error_y'] 

    edges = edges.drop(columns=[c for c in edges.columns if (c.endswith('_x') or c.endswith('_y'))])

    edges['is_error_free'] = np.logical_not(edges.has_merge_error | edges.has_split_error)

    eftpl = edges.path_length[edges.is_error_free].sum() / edges.path_length.sum()
    
    return eftpl