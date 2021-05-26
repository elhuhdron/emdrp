from emdrp.utils.efpl import *
import pytest

def test_imports():
    pass

def test_simple_skeleton():
    import wknml
    nml = util_get_nml()
    p = util_get_params()
    o = util_get_outputs(nml)

    labelsWalkEdges(o,p,None, None, None, rand_error_rate=[1])

    return


def test_checkErrorAtEdge_function():
    import wknml
    nml = util_get_nml()
    p = util_get_params()
    o = util_get_outputs(nml)

    edge_split = [np.zeros((len(t.edges), 1), dtype=bool) for t in nml.trees]
    nodes_to_labels = [t.id*np.ones((len(t.nodes), 1), dtype=int) for t in nml.trees]
    label_merged = np.zeros((len(nml.trees), 1), dtype=bool)

    pass_ = 1

    n = 0
    n1 = 0
    n2 = 1
    e = 0

    checkErrorAtEdge(p,n,n1,n2,e,pass_, edge_split, label_merged, nodes_to_labels)

    return

def test_labelsWalkEdges_ids_continous():
    """ Test requirement that node ids can be used as index labels """
    import wknml
    nml = wknml.NML(
        parameters=wknml.NMLParameters(
            name='',
            scale=(1, 1, 1),
        ),
        trees=[
            wknml.Tree(
                id=0,
                color=(255, 255, 0, 1),
                name='',
                nodes=[
                    wknml.Node(id=4, position=(1, 0, 0), radius=1),
                    wknml.Node(id=5, position=(2, 0, 0), radius=1)
                ],
                edges=[
                    wknml.Edge(source=4, target=5),
                ],
            ),
        ],
        branchpoints=[],
        comments=[],
        groups=[],
    )
    
    p = util_get_params()
    o = util_get_outputs(nml)

    edge_split = [np.zeros((len(t.edges), 1), dtype=bool) for t in nml.trees]
    nodes_to_labels = [t.id*np.ones((len(t.nodes), 1), dtype=int) for t in nml.trees]
    label_merged = np.zeros((len(nml.trees), 1), dtype=bool)

    with pytest.raises(AssertionError):
        labelsWalkEdges(o, p, edge_split, label_merged, nodes_to_labels)



def test_simple_skeleton_efpl():
    import wknml
    nml = util_get_nml()
    p = util_get_params()
    o = util_get_outputs(nml)

    edge_split = [np.zeros((len(t.edges), 1), dtype=bool) for t in nml.trees]
    nodes_to_labels = [t.id*np.ones((len(t.nodes), 1), dtype=int) for t in nml.trees]
    label_merged = np.zeros((len(nml.trees), 1), dtype=bool)

    efpl, efpl_thing_ptr, efpl_edges = labelsWalkEdges(o,p,edge_split, label_merged, nodes_to_labels)

    assert(efpl[0][0] == 3)
    assert(efpl[0][1] == 2)
    assert(efpl[1][0] == 3)
    assert(efpl[1][1] == 2)
    assert(efpl[2][0] == 3)
    assert(efpl[2][1] == 2)

def util_get_params():
    from collections import namedtuple
    Parameters = namedtuple('Parameters', [
        'npasses_edges', 'nalloc', 'empty_label', 'count_half_error_edges', 'tol'])
    p = Parameters(
        npasses_edges=3,
        nalloc=int(1e6),
        empty_label=np.uint32(2^32-1),
        count_half_error_edges=True,
        tol = 1e-5,
    )

    return p


def util_get_outputs(nml):
    from collections import namedtuple
    Outputs = namedtuple('Outputs', [
        'nThings', 'omit_things_use', 'nedges', 'edge_length', 'edges_use', 'info', 'path_length_use'])
    Info = namedtuple('Info', ['edges'])


    edge_lengths = []
    for t in nml.trees:
        tree_edge_lengths = []
        node_ids = [n.id for n in t.nodes]
        node_positions = np.array([n.position for n in t.nodes])
        for edge in t.edges:
            p1 = node_positions[node_ids.index(edge.source), :]
            p2 = node_positions[node_ids.index(edge.target), :]
            edge_length = np.linalg.norm((p1-p2) * nml.parameters.scale)
            tree_edge_lengths.append(edge_length)
        edge_lengths.append(tree_edge_lengths)

    o = Outputs(
            nThings = len(nml.trees),
            omit_things_use =  np.zeros((len(nml.trees), 1), dtype=bool),
            nedges =  [len(t.edges) for t in nml.trees], 
            edge_length  = edge_lengths,
            edges_use =  [np.ones((len(t.edges),), dtype=bool) for t in nml.trees],
            info = [Info(
                edges =  np.array([[e.source, e.target] for e in t.edges])
                ) for t in nml.trees],
            path_length_use =  [np.sum(edge_length) for edge_length in edge_lengths],
    )
    return o

def util_get_nml():
    import wknml
    trees = [
        wknml.Tree(
            id=0,
            color=(255, 255, 0, 1),
            name="Synapse 1",
            nodes=[
                wknml.Node(id=0, position=(0, 0, 0), radius=1),
                wknml.Node(id=1, position=(0, 0, 1), radius=1),
                wknml.Node(id=2, position=(0, 0, 2), radius=1),
                wknml.Node(id=3, position=(0, 1, 1), radius=1)],
            edges=[
                wknml.Edge(source=0, target=1),
                wknml.Edge(source=1, target=2),
                wknml.Edge(source=1, target=3)
            ],
            groupId=1,
        ),
        wknml.Tree(
            id=1,
            color=(255, 0, 255, 1),
            name="Synapse 2",
            nodes=[
                wknml.Node(id=0, position=(1, 0, 0), radius=1),
                wknml.Node(id=1, position=(2, 0, 0), radius=1),
                wknml.Node(id=2, position=(3, 0, 0), radius=1)
            ],
            edges=[
                wknml.Edge(source=0, target=1),
                wknml.Edge(source=1, target=2)
            ],
            groupId=1,
        ),
    ]

    nml = wknml.NML(
        parameters=wknml.NMLParameters(
            name="Test",
            scale=(1, 1, 1),
        ),
        trees=trees,
        branchpoints=[],
        comments=[],
        groups=[],
    )
    
    return nml


