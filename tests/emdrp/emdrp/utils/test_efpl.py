from emdrp.utils.efpl import *

# How to write 
def test_imports():
    pass

def test_simple_skeleton():
    # create simple nml graph
    import wknml
    nml = util_get_nml()
    p = util_get_params()
    o = util_get_outputs(nml)

    labelsWalkEdges(o,p,None, None, None, rand_error_rate=None)



def util_get_params():
    from collections import namedtuple
    Parameters = namedtuple('Parameters', ['npasses_edges', 'nalloc'])
    p = Parameters(npasses_edges=3, nalloc=int(1e6))

    return p


def util_get_outputs(nml):
    from collections import namedtuple
    Outputs = namedtuple('Outputs', ['nThings', 'omit_things_use', 'nedges', 'edges_use', 'info'])
    Info = namedtuple('Info', ['edges'])

    o = Outputs(
            nThings = len(nml.trees),
            omit_things_use =  np.ones((len(nml.trees), 1), dtype=bool),
            nedges =  [len(t.edges) for t in nml.trees], 
            edges_use =  [np.ones((len(t.edges), 1), dtype=bool) for t in nml.trees],
            info = [Info(
                edges =  [[e.source, e.target] for e in t.edges]
                ) for t in nml.trees]
    )
    return o

def util_get_nml():
    import wknml
    trees = [
        wknml.Tree(
            id=1,
            color=(255, 255, 0, 1),
            name="Synapse 1",
            nodes=[
                wknml.Node(id=1, position=(0, 0, 0), radius=1),
                wknml.Node(id=2, position=(0, 0, 1), radius=1),
                wknml.Node(id=3, position=(0, 0, 2), radius=1),
                wknml.Node(id=4, position=(0, 1, 1), radius=1)],
            edges=[
                wknml.Edge(source=1, target=2),
                wknml.Edge(source=2, target=3),
                wknml.Edge(source=2, target=4)
            ],
            groupId=1,
        ),
        wknml.Tree(
            id=2,
            color=(255, 0, 255, 1),
            name="Synapse 2",
            nodes=[
                wknml.Node(id=5, position=(1, 0, 0), radius=1),
                wknml.Node(id=6, position=(2, 0, 0), radius=1),
                wknml.Node(id=7, position=(3, 0, 0), radius=1)
            ],
            edges=[
                wknml.Edge(source=5, target=6),
                wknml.Edge(source=6, target=7)
            ],
            groupId=1,
        ),
    ]

    nml = wknml.NML(
        parameters=wknml.NMLParameters(
            name="Test",
            scale=(11.24, 11.24, 25),
        ),
        trees=trees,
        branchpoints=[],
        comments=[],
        groups=[],
    )
    
    return nml


