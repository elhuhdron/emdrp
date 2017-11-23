
import time
import numpy as np
import numpy.random as nr
import h5py
#from scipy import ndimage as nd
from scipy import io as sio

#from dpLoadh5 import dpLoadh5
#from dpWriteh5 import dpWriteh5
#from typesh5 import emLabels

from scipy import linalg as sla
from scipy import optimize as opt
from scipy import spatial as spt
from scipy.special import ellipkinc, ellipeinc

import vtk
from vtk.util import numpy_support as nps

 
 

mesh_in='/home/watkinspv/Downloads/K0057_soma_annotation/out/K0057-D31-somas_dsx12y12z4-clean-cut.0.mesh.h5'

points_per_area = 1e-4
doplots = True
plotsvdfit = True
plot_surf = False
opacity = 0.5
penalty = 0.


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
    renderWin.SetSize(600, 600)
    renderInteractor.Initialize()
    renderWin.Render()
    renderInteractor.Start()

def ellipsoid_SA(r):
    r = np.sort(r.reshape(-1), axis=0); a,b,c = r[2],r[1],r[0]
    if np.isclose(a,b) and np.isclose(b,c):
        #print('sphere')
        ellipsoid_area = 4*np.pi*a**2
    else:
        # https://www.johndcook.com/blog/2014/07/06/ellipsoid-surface-area/
        if np.isclose(a,b):
            #print('oblate')
            m=1
        elif np.isclose(b,c):
            #print('prolate')
            m=0
        else:
            #print('triaxial')
            m = (a**2 * (b**2 - c**2)) / (b**2 * (a**2 - c**2))
        phi = np.arccos(c/a)
        temp = ellipeinc(phi, m)*np.sin(phi)**2 + ellipkinc(phi, m)*np.cos(phi)**2
        ellipsoid_area = 2*np.pi*(c**2 + a*b*temp/np.sin(phi))
    return ellipsoid_area

def ellipsoid_points(R, C, points_per_area):
    # incorrect method
    #    theta, phi = np.mgrid[0:nsurfpts+1, 0:nsurfpts+1]/nsurfpts
    #    theta = theta*np.pi - np.pi/2; phi = phi*2*np.pi - np.pi
    #
    #    x = X[0]*np.cos(theta)*np.cos(phi);
    #    y = X[1]*np.cos(theta)*np.sin(phi);
    #    z = X[2]*np.sin(theta);
    #    return np.vstack((x.reshape(-1), y.reshape(-1), z.reshape(-1))).T + np.array(X[3:]).reshape((1,3))

    SA = ellipsoid_SA(R); nsurfpts = int(SA*points_per_area)

    # for broadcoast, use npts x 3, dimensions along axis 1
    R = R.reshape((1,3)); C = C.reshape((1,3))
    
    # https://math.stackexchange.com/questions/973101/how-to-generate-points-uniformly-distributed-on-the-surface-of-an-ellipsoid?answertab=oldest#tab-top
    pts = nr.randn(nsurfpts,3) * R; d = np.sqrt( (pts*pts/R/R).sum(1) )[:, None]
    return pts/d + C

    # brute force method
    #    npts = 0; pts = np.zeros((nsurfpts,3),dtype=np.double)
    #    while npts < nsurfpts:
    #        cpts = nr.rand(1e6,3) * 2*(R+1) - (R+1)
    #        sel = (abs(np.sqrt((cpts*cpts/R/R).sum(1)) - 1) < 1e-3)
    #        cnpts = sel.sum(); print(cnpts)
    #        if npts + cnpts < nsurfpts:
    #            pts[npts:npts+cnpts,:] = cpts[sel,:]; npts = npts + cnpts
    #        else:
    #            pts[npts:,:] = cpts[sel,:][:nsurfpts-npts,:]; npts = nsurfpts
    #    return pts + C

def ellipsoid_distance(X, surf_tree, surf_pts, points_per_area, penalty=0.):
    #print(X)
    R = X[:3].reshape((1,3)); C = X[3:].reshape((1,3))
    epts = ellipsoid_points(R,C,points_per_area)
    d,i = tree.query(epts)
    dist1 = d.mean()
    
    if penalty > 0:
        # introduce some penalty for surface points that are inside the ellipse
        sel = (((surf_pts-C)**2/R/R).sum(1) < 1)
        perc_inside = sel.sum(dtype=np.double)/surf_pts.shape[0]
        return dist1 + perc_inside*dist1*penalty
    else:
        return dist1

# get the volume and surface area data from the mesh file
h5file = h5py.File(mesh_in, 'r'); dset_root = h5file['0']
str_seed = ('%08d' % 0)
scale = dset_root[str_seed]['faces'].attrs['scale']; voxel_volume = scale.prod()
vertex_divisor = dset_root[str_seed]['faces'].attrs['vertex_divisor']
nseeds = len(dset_root)-1

# for saving fits
svd_rads = np.zeros((nseeds,3),np.double); svd_ctrs = np.zeros((nseeds,3),np.double)
min_rads = np.zeros((nseeds,3),np.double); min_ctrs = np.zeros((nseeds,3),np.double)
svd_rots = np.zeros((nseeds,3,3),np.double) 

for i in range(nseeds):
#for i in range(3):
    print('Processing soma %d' % (i,)); t = time.time()
                        
    str_seed = ('%08d' % (i+1,))
    #soma_volumes[i] = dset_root[str_seed]['vertices'].attrs['nVoxels'] * voxel_volume
    #soma_surface_areas[i] = dset_root[str_seed]['vertices'].attrs['surface_area']

    vertices = np.empty_like(dset_root[str_seed]['vertices'])
    faces = np.empty_like(dset_root[str_seed]['faces'])
    dset_root[str_seed]['vertices'].read_direct(vertices)
    dset_root[str_seed]['faces'].read_direct(faces)
    nvertices = vertices.shape[0]; nfaces = faces.shape[0]
    
    vertices = vertices.astype(np.double) / vertex_divisor
    # for global coordinates add bounding box offset
    vertices += dset_root[str_seed]['vertices'].attrs['bounds_beg'] / vertex_divisor

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
    svd_std = np.sqrt(S**2/(npts-1))[:,None];

    # rotate the points to align on cartesian axes
    rpts = ((np.dot(Vt, pts.T - C) + C).T).copy(order='C')

    # create kdtree to find closest points to mesh vertices
    tree = spt.cKDTree(rpts)

    # for optimization bounds
    minrpts = rpts.min(0); maxrpts = rpts.max(0)
    minR = 0.5**(1/3)*svd_std.copy(); maxR = 8**(1/3)*svd_std.copy()
    bounds = ((minR[0],maxR[0]),(minR[1],maxR[1]),(minR[2],maxR[2]),
              (minrpts[0],maxrpts[0]),(minrpts[1],maxrpts[1]),(minrpts[2],maxrpts[2]))

    # scale ellipse by some number of stds as one method of fitting ellipsoid
    svd_ctr = C; fit_dist = np.inf
    for s in np.arange(1,10,0.1):
        crad = s**(1/3)*svd_std
        cdist = ellipsoid_distance(np.vstack((crad,svd_ctr)), tree, rpts, points_per_area, penalty)
        if cdist < fit_dist:
            svd_rad = crad; fit_dist = cdist
    print('\tDistance %.4f with SVD rad %.4f %.4f %.4f ctr %.4f %.4f %.4f' % (fit_dist,
       svd_rad[0],svd_rad[1],svd_rad[2],svd_ctr[0],svd_ctr[1],svd_ctr[2]))
    svd_rads[i,:] = svd_rad.reshape(-1); svd_ctrs[i,:] = svd_ctr.reshape(-1)
    svd_rots[i,:,:] = Vt
    
    # default to svd "fits"
    fit_rad = svd_rad; fit_ctr = svd_ctr;

    # for testing ellipsoid_distance
    #ellipsoid_distance(np.vstack((s,C)), tree, points_per_area)

    ## for testing ellipse points
    #s = np.array((500,1000,1500),dtype=np.double); C = np.zeros((3,1), np.double)
    #rpts = ellipsoid_points(s,C,points_per_area).copy(order='C'); nvertices = rpts.shape[0]
    #faces = np.arange(nvertices, dtype=faces.dtype)[:,None]; nfaces = nvertices
    #pfaces = np.hstack((1*np.ones((nfaces, 1),dtype=faces.dtype), faces))

    # normal local minimization functions do not work, error function is not smooth at all
    #X,success = opt.leastsq(ellipsoid_distance, np.vstack((minR*1.1,C-500)), 
    #                        args=(tree, points_per_area))
    #print(X,success)
    #res = opt.minimize(ellipsoid_distance, np.vstack((minR*1.1,C-500)), 
    #                        args=(tree, points_per_area), bounds=bounds)
    #print(res)

    # global minimization methods
    #X = opt.brute(ellipsoid_distance, bounds, args=(tree, points_per_area))
    res = opt.differential_evolution(ellipsoid_distance, bounds, args=(tree, rpts, points_per_area, penalty), 
                                     maxiter=10000, strategy='best1bin', polish=False, disp=False)
    fit_rad = np.array(res.x[:3]).reshape((3,1)); fit_ctr = np.array(res.x[3:]).reshape((3,1))
    fit_dist = ellipsoid_distance(np.vstack((fit_rad,fit_ctr)), tree, rpts, points_per_area, penalty)
    print('\tDistance %.4f with min rad %.4f %.4f %.4f ctr %.4f %.4f %.4f' % (fit_dist,
       fit_rad[0],fit_rad[1],fit_rad[2],fit_ctr[0],fit_ctr[1],fit_ctr[2]))
    min_rads[i,:] = fit_rad.reshape(-1); min_ctrs[i,:] = fit_ctr.reshape(-1)

    print('\tdone in %.4f s' % (time.time() - t,))
    
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
        translation.Scale(fit_rad[0],fit_rad[1],fit_rad[2])
        translation.PostMultiply()
        translation.Translate(fit_ctr[0],fit_ctr[1],fit_ctr[2]);
        transformFilter = vtk.vtkTransformPolyDataFilter();
        transformFilter.SetInputConnection(sphSrc.GetOutputPort());
        transformFilter.SetTransform(translation);

        sphMapper = vtk.vtkPolyDataMapper()
        sphMapper.SetInputConnection(transformFilter.GetOutputPort())
        sphActor = vtk.vtkActor()
        sphActor.SetMapper(sphMapper)
        sphActor.GetProperty().SetColor(0,0,1)
        sphActor.GetProperty().SetOpacity(opacity)
        renderer.AddActor(sphActor)

        if plotsvdfit:
            # create a sphere in red
            sphSrc2 = vtk.vtkSphereSource();
            sphSrc2.SetCenter(0,0,0);
            sphSrc2.SetRadius(1.0);
            sphSrc2.SetThetaResolution(100)
            sphSrc2.SetPhiResolution(100)
            translation2 = vtk.vtkTransform();
            translation2.Scale(svd_rad[0],svd_rad[1],svd_rad[2])
            translation2.PostMultiply()
            translation2.Translate(svd_ctr[0],svd_ctr[1],svd_ctr[2]);
            transformFilter2 = vtk.vtkTransformPolyDataFilter();
            transformFilter2.SetInputConnection(sphSrc2.GetOutputPort());
            transformFilter2.SetTransform(translation2);
    
            sphMapper2 = vtk.vtkPolyDataMapper()
            sphMapper2.SetInputConnection(transformFilter2.GetOutputPort())
            sphActor2 = vtk.vtkActor()
            sphActor2.SetMapper(sphMapper2)
            sphActor2.GetProperty().SetColor(1,0,0)
            sphActor2.GetProperty().SetOpacity(opacity)
            renderer.AddActor(sphActor2)

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
        allActors.GetProperty().SetOpacity(opacity)
        renderer.AddActor(allActors)

        vtkShow(renderer=renderer)

h5file.close()
  
mat_out='/home/watkinspv/Downloads/K0057_soma_annotation/out/somas_cut_fit_surf_penalty.mat'
sio.savemat(mat_out, {'svd_rads':svd_rads, 'svd_ctrs':svd_ctrs, 'min_rads':min_rads,
                      'min_ctrs':min_ctrs, 'svd_rots':svd_rots})
