import numpy as np
import pandas as pd
import wknml

KNOSSOS_BASE =  np.ones((1, 3), dtype=int)
NML_DIM_ORDER = 'xyz'
HDF5_DIM_ORDER = 'zyx'

HDF5_BACKGROUND_LABEL = 0

def calc_eftpl(nml, Vlbls, dataset_start=np.zeros((1,3), dtype=int), verbose=False):
    
    ### Extract, clean, and transform node data from nml file

    nodes = pd.DataFrame([n for t in nml.trees for n in t.nodes])

    idx_columns = [f'idx_{ax}' for ax in NML_DIM_ORDER]
    for n ,col in enumerate(idx_columns):
        nodes[col] = nodes['position'].apply(lambda position: position[n])

    nodes = nodes.drop(columns=[col for col in nodes.columns if not col in ['id'] + idx_columns])

    nodes = nodes.astype({c:int for c in nodes.columns})

    nodes[idx_columns] -= (dataset_start + KNOSSOS_BASE)

    nodes['label'] = Vlbls[tuple(nodes[f'idx_{ax}'] for ax in HDF5_DIM_ORDER)]

    position_colums = [f'position_{ax}' for ax in NML_DIM_ORDER]
    nodes[position_colums] = nodes[idx_columns] * np.array(nml.parameters.scale)

    ### Extract, clean, and transform edge data from nml file
    edges = pd.DataFrame(nml.trees)

    edges = edges.drop(columns=[ col for col in edges.columns if not col in ['id', 'edges'] ])
    edges = edges.rename(columns={"id": "tree_id"})

    edges = edges.explode('edges')

    new_col_list = ['source_node','target_node']
    for n ,col in enumerate(new_col_list):
        edges[col] = edges['edges'].apply(lambda location: location[n])

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