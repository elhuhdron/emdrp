
#import argparse
import numpy as np
import numpy.random as nr
import h5py
#from scipy import ndimage as nd
#from scipy import io as sio

#from dpLoadh5 import dpLoadh5
#from dpWriteh5 import dpWriteh5
#from typesh5 import emLabels

from scipy import linalg as sla
import vtk
from vtk.util import numpy_support as nps

from scipy.special import ellipkinc, ellipeinc
 
 

mesh_in='/Users/pwatkins/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut.0.mesh.h5'

nsurfpts = 1000
doplots = True
plot_surf = False


def vtkShow(mapper=None, renderer=None):

    if renderer is None:
        # open a window and display the data specified by mapper
        # need an actor and a renderer to display data
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)

    #renderer.SetBackground(1.0, 1.0, 1.0)
    renderer.SetBackground(0.0, 0.0, 0.0)

    # optionally setup the camera (xxx - and lighting?)
    #camera = renderer.MakeCamera()
    #camera.SetPosition(-500.0, 245.5, 122.0)
    #camera.SetFocalPoint(301.0, 245.5, 122.0)
    #camera.SetViewAngle(30.0)
    #camera.SetRoll(-90.0)
    #renderer.SetActiveCamera(camera)

    # setup the renderer window / interactor and run
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
    renderInteractor.SetRenderWindow(renderWin)
    renderWin.SetSize(400, 400)
    renderInteractor.Initialize()
    renderWin.Render()
    renderInteractor.Start()

#def ellipsoid_distance(X, pts, nsurfpts):

# https://www.johndcook.com/blog/2014/07/06/ellipsoid-surface-area/
def ellipsoid_SA(r):
    a,b,c = r[0],r[1],r[2]
    phi = np.arccos(c/a)
    m = (a**2 * (b**2 - c**2)) / (b**2 * (a**2 - c**2))
    temp = ellipeinc(phi, m)*np.sin(phi)**2 + ellipkinc(phi, m)*np.cos(phi)**2
    ellipsoid_area = 2*np.pi*(c**2 + a*b*temp/np.sin(phi))
    return ellipsoid_area

def ellipsoid_points(X, points_per_area):
    #    theta, phi = np.mgrid[0:nsurfpts+1, 0:nsurfpts+1]/nsurfpts
    #    theta = theta*np.pi - np.pi/2; phi = phi*2*np.pi - np.pi
    #
    #    x = X[0]*np.cos(theta)*np.cos(phi);
    #    y = X[1]*np.cos(theta)*np.sin(phi);
    #    z = X[2]*np.sin(theta);
    #    return np.vstack((x.reshape(-1), y.reshape(-1), z.reshape(-1))).T + np.array(X[3:]).reshape((1,3))

    SA = ellipsoid_SA(X[:3]); nsurfpts = int(SA*points_per_area)
    print(SA,nsurfpts)

    # https://math.stackexchange.com/questions/973101/how-to-generate-points-uniformly-distributed-on-the-surface-of-an-ellipsoid?answertab=oldest#tab-top
    x = nr.randn(nsurfpts,1); y = nr.randn(nsurfpts,1); z = nr.randn(nsurfpts,1);
    d = np.sqrt( x*x/X[0]/X[0] + y*y/X[1]/X[1] + z*z/X[2]/X[2] ); print(d.shape)
    return np.vstack((x.reshape(-1), y.reshape(-1), z.reshape(-1))).T/d + np.array(X[3:]).reshape((1,3))

# get the volume and surface area data from the mesh file
h5file = h5py.File(mesh_in, 'r'); dset_root = h5file['0']
str_seed = ('%08d' % 0)
scale = dset_root[str_seed]['faces'].attrs['scale']; voxel_volume = scale.prod()
vertex_divisor = dset_root[str_seed]['faces'].attrs['vertex_divisor']
nseeds = len(dset_root)-1
for i in range(nseeds):
    str_seed = ('%08d' % (i+1,))
    #soma_volumes[i] = dset_root[str_seed]['vertices'].attrs['nVoxels'] * voxel_volume
    #soma_surface_areas[i] = dset_root[str_seed]['vertices'].attrs['surface_area']

    vertices = np.empty_like(dset_root[str_seed]['vertices'])
    faces = np.empty_like(dset_root[str_seed]['faces'])
    dset_root[str_seed]['vertices'].read_direct(vertices)
    dset_root[str_seed]['faces'].read_direct(faces)
    nvertices = vertices.shape[0]; nfaces = faces.shape[0]
    
    vertices = vertices.astype(np.double) / vertex_divisor
    #vertices += dset_root[str_seed]['vertices'].attrs['bounds_beg']

    # vtk needs unstructured grid preceded with number of points in each cell
    if plot_surf:
        pfaces = np.hstack((3*np.ones((nfaces, 1),dtype=faces.dtype), faces))
    else:
        faces = np.arange(nvertices, dtype=faces.dtype)[:,None]; nfaces = nvertices
        pfaces = np.hstack((1*np.ones((nfaces, 1),dtype=faces.dtype), faces))
        
    # use vertices as points for fitting
    pts = vertices.copy(); npts = pts.shape[0]

    #    sel = (somas[svox_bnd[j-1]] == j) # binary select within bounding box
    #    pts = np.transpose(np.nonzero(sel)).astype(np.double)*sampling # pts is nx3
    #    npts = pts.shape[0]

    # svd on centered points to get principal axes.
    # NOTE IMPORTANT: from scipy, svd different from matlab:
    #   "The SVD is commonly written as a = U S V.H. The v returned by this function is V.H and u = U."
    # pts is Nx3, eigenvectors in V are along the rows
    C = pts.mean(0)[:,None];
    U, S, Vt = sla.svd(pts - C.T,overwrite_a=False,full_matrices=False)
    # the std of the points along the eigenvectors
    s = np.sqrt(S**2/(npts-1));

    # scale ellipse by some number of stds as one method of fitting ellipsoid
    s = 4**(1/3)*s
      
    # rotate the points to align on cartesian axes
    rpts = ((np.dot(Vt, pts.T - C) + C).T).copy(order='C')

    # for testing ellipse points    
    #    rpts = ellipsoid_points((s[0],s[1],s[2],C[0],C[1],C[2]),1e-5).copy(order='C'); nvertices = rpts.shape[0]
    #    faces = np.arange(nvertices, dtype=faces.dtype)[:,None]; nfaces = nvertices
    #    pfaces = np.hstack((1*np.ones((nfaces, 1),dtype=faces.dtype), faces))
    
    if doplots:
        renderer = vtk.vtkRenderer()
        allPolyData = vtk.vtkAppendPolyData()

        # create a sphere in blue
        sphSrc = vtk.vtkSphereSource();
        sphSrc.SetCenter(0,0,0);
        sphSrc.SetRadius(1.0);
        sphSrc.SetThetaResolution(100)
        sphSrc.SetPhiResolution(100)
        translation = vtk.vtkTransform();
        translation.Scale(s[0],s[1],s[2])
        translation.PostMultiply()
        translation.Translate(C[0],C[1],C[2]);
        transformFilter = vtk.vtkTransformPolyDataFilter();
        transformFilter.SetInputConnection(sphSrc.GetOutputPort());
        transformFilter.SetTransform(translation);

        sphMapper = vtk.vtkPolyDataMapper()
        sphMapper.SetInputConnection(transformFilter.GetOutputPort())
        sphActor = vtk.vtkActor()
        sphActor.SetMapper(sphMapper)
        sphActor.GetProperty().SetColor(0,0,1)
        sphActor.GetProperty().SetOpacity(0.7)
        renderer.AddActor(sphActor)

        # create vertices for polydata
        # http://www.vtk.org/Wiki/VTK/Examples/Python/GeometricObjects/Display/Polygon
        points = vtk.vtkPoints(); points.SetData(nps.numpy_to_vtk(rpts))
    
        # create cells to be used as faces or vertices
        # http://stackoverflow.com/questions/20146421/how-to-convert-a-mesh-to-vtk-format/20146620#20146620
        cells = vtk.vtkCellArray()
        cells.SetCells(nfaces, nps.numpy_to_vtk(pfaces, array_type=vtk.vtkIdTypeArray().GetDataType()))
        
        # set faces and vertices of polydata
        polyData = vtk.vtkPolyData(); polyData.SetPoints(points)
        if plot_surf:
            polyData.SetPolys(cells)
        else:
            polyData.SetVerts(cells)
    
        # use appendpolydata to render multiple supervoxels per object
        allPolyData.AddInputData(polyData)

        allMappers = vtk.vtkPolyDataMapper()
        allMappers.SetInputConnection(allPolyData.GetOutputPort())
        allActors = vtk.vtkActor()
        allActors.SetMapper(allMappers)
        allActors.GetProperty().SetColor(0,1,0)
        allActors.GetProperty().SetPointSize(2)
        ##allActors.GetProperty().SetOpacity(0.8)
        renderer.AddActor(allActors)

        vtkShow(renderer=renderer)
      
h5file.close()

