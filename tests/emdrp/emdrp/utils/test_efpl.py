from emdrp.utils.efpl import *
from wknml import *
from numpy import uint32
import numpy as np
import h5py
import wknml
import pytest

def test_imports():
    pass

def test_calc_supervoxel_eftpl(tmp_path):
    dummy_data = np.random.randint(100, size=(10, 10, 10))
    with h5py.File(tmp_path / 'test.h5', 'w') as h5file:
        h5file.create_dataset('with_background/test_data/labels', data=dummy_data)

    nml = util_get_nml()

    with open(tmp_path / "test.nml", "wb") as f:
        wknml.write_nml(f, nml)

    calc_supervoxel_eftpl(tmp_path / 'test.nml', tmp_path / 'test.h5')

    return

def test_calc_efpl_outside_roi():
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
                    wknml.Node(id=5, position=(3, 1, 1), radius=1),
                    wknml.Node(id=6, position=(20, 1, 1), radius=1)
                ],
                edges=[
                    wknml.Edge(source=4, target=5),
                    wknml.Edge(source=5, target=6),

                ],
            ),
        ],
        branchpoints=[],
        comments=[],
        groups=[],
    )
    
    Vlbls = np.ones((5, 5, 5), dtype=np.uint32)
    dataset_start = np.zeros((1, 3), dtype=int)

    calc_eftpl(nml, Vlbls, dataset_start)



def test_calc_efpl_simple():
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
        comments=[],
        groups=[],
    )
    
    Vlbls = np.ones((5, 5, 5), dtype=np.uint32)
    dataset_start = np.zeros((1, 3), dtype=int)

    calc_eftpl(nml, Vlbls, dataset_start)

def test_calc_efpl_simple2():
    nml = util_get_nml()
    Vlbls = np.ones((5, 5, 5), dtype=np.uint32)
    dataset_start = np.zeros((1, 3), dtype=int)

    calc_eftpl(nml, Vlbls, dataset_start)


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
        comments=[],
        groups=[],
    )
    
    return nml


