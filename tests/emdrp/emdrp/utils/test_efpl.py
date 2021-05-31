from emdrp.utils.efpl import *
import pytest

def test_imports():
    pass

def test_Outputs_gen_from_nml():
    nml = util_get_nml()
    o = Outputs.gen_from_nml(nml)

    assert(o.nThings == 2)
    assert(o.nedges[0] == 3)
    assert(len(o.path_length_use) == o.nThings)

    return

def test_simple_skeleton():
    import wknml
    nml = util_get_nml()
    p = util_get_params()
    o = Outputs.gen_from_nml(nml)

    labelsWalkEdges(o,p,None, None, None, rand_error_rate=[1])

    return


def test_checkErrorAtEdge_function():
    import wknml
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


