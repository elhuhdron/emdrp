from emdrp.utils.efpl import *
from numpy import dtype
import pytest

def test_imports():
    pass

def test_parameters_attributes():
    p = Parameters(
        npasses_edges=3,
        nalloc=int(1e6),
        empty_label=np.uint32(2^32-1),
        count_half_error_edges=True,
        m_ij_threshold=1,
        tol = 1e-5,)

    for attr in ['m_ij_threshold','knossos_base' ]:
        assert(hasattr(p,attr ))

def test_outputs_info():
    nml = util_get_nml()
    o = Outputs.gen_from_nml(nml)
    assert(np.all([hasattr(i, 'nodes') for i in o.info]))

def test_outputs_info_nodes_shape():
    nml = util_get_nml()
    o = Outputs.gen_from_nml(nml)
    assert(len(o.info[0].nodes.shape) == 2)
    assert(o.info[0].nodes.shape[1] == 3)

def test_outputs():
    nml = util_get_nml()
    o = Outputs.gen_from_nml(nml)
    for attr in ['nnodes', 'loadcorner']:
        assert(hasattr(o, attr))
    return


def test_labelsPassEdges():
    nml = util_get_nml()
    p = util_get_params()
    o = Outputs.gen_from_nml(nml)

    Vlbls = np.ones((5, 5, 5))
    nnodes = sum([len(t.nodes) for t in nml.trees])
    nlabels = 2
    thing_list = np.arange(len(nml.trees))

    labelsPassEdges(o, p, Vlbls, nnodes, nlabels, thing_list)


def test_Outputs_gen_from_nml():
    nml = util_get_nml()
    o = Outputs.gen_from_nml(nml)

    assert(o.nThings == 2)
    assert(o.nedges[0] == 3)
    assert(len(o.path_length_use) == o.nThings)

    return

def test_simple_skeleton():
    nml = util_get_nml()
    p = util_get_params()
    o = Outputs.gen_from_nml(nml)

    labelsWalkEdges(o,p,None, None, None, rand_error_rate=[1])

    return


def test_checkErrorAtEdge_function():
    nml = util_get_nml()
    p = util_get_params()
    o = Outputs.gen_from_nml(nml)

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
                    wknml.Node(id=4, position=(2, 1, 1), radius=1),
                    wknml.Node(id=5, position=(3, 1, 1), radius=1)
                ],
                edges=[
                    wknml.Edge(source=4, target=5),
                ],
            ),
        ],
        branchpoints=[],
        comments=[{'loadcorner':(0, 0, 0), 'loadsize':(5, 5,5)}],
        groups=[],
    )
    
    p = util_get_params()
    o = Outputs.gen_from_nml(nml)

    edge_split = [np.zeros((len(t.edges), 1), dtype=bool) for t in nml.trees]
    nodes_to_labels = [t.id*np.ones((len(t.nodes), 1), dtype=int) for t in nml.trees]
    label_merged = np.zeros((len(nml.trees), 1), dtype=bool)

    with pytest.raises(AssertionError):
        labelsWalkEdges(o, p, edge_split, label_merged, nodes_to_labels)


def test_simple_skeleton_efpl():
    import wknml
    nml = util_get_nml()
    p = util_get_params()
    o = Outputs.gen_from_nml(nml)

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
    p = Parameters(
        npasses_edges=3,
        nalloc=int(1e6),
        empty_label=np.uint32(2^32-1),
        count_half_error_edges=True,
        m_ij_threshold=1,
        tol = 1e-5,
    )

    return p

def util_get_nml():
    import wknml
    trees = [
        wknml.Tree(
            id=0,
            color=(255, 255, 0, 1),
            name="Synapse 1",
            nodes=[
                wknml.Node(id=0, position=(1, 1, 1), radius=1),
                wknml.Node(id=1, position=(1, 1, 2), radius=1),
                wknml.Node(id=2, position=(1, 1, 3), radius=1),
                wknml.Node(id=3, position=(1, 2, 2), radius=1)],
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
                wknml.Node(id=0, position=(2, 1, 1), radius=1),
                wknml.Node(id=1, position=(3, 1, 1), radius=1),
                wknml.Node(id=2, position=(4, 1, 1), radius=1)
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
        comments=[{'loadcorner':(0,0,0),
            'loadsize':(5, 5, 5),
            }],
        groups=[],
    )
    
    return nml


