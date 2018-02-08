
import numpy as np
import scipy.spatial.distance as scidist
from czifile import CziFile
from matplotlib import pylab as pl
import matplotlib.patches as patches
from lxml import etree as etree
import time

### options

# input
#fn='/data/pwatkins/kara/sample_limi_real_tissue_20180206/iron-k0100-20 and 12-dab-serial_50nm_w sections-rois.czi'
fn='/home/pwatkins/Downloads/iron-k0100-20 and 12-dab-serial_50nm_w sections-rois.czi'
# optional dump of the meta xml (empty for no dump)
metafnout = ''
#metafnout = '/home/pwatkins/Downloads/meta.xml'
print_meta = False

# which "scene" to load, base-zero
load_scene = 3

# how many polygons to load as ROIs
npolygons = 1e6

doplots = True
doplots_ds = 10 # downsampling amount so large images can load

### "constants"

# xml paths that define the sections and rois
xml_paths = {
    'ScaleX':"/ImageDocument/Metadata/Scaling/Items/Distance[@Id = 'X']/Value",
    'ScaleY':"/ImageDocument/Metadata/Scaling/Items/Distance[@Id = 'Y']/Value",
    'Calibration':\
        '/ImageDocument/Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/SubDimensionSetups/CorrelativeSetup/'+\
            'HolderDocument/Calibration',
    'Scenes':'/ImageDocument/Metadata/Information/Image/Dimensions/S/Scenes',
    'SelectionBox':"/ImageDocument/Metadata/MetadataNodes/MetadataNode/Layers/Layer[@Name = \"Cat_Ribbon\"]"+\
        '/Elements/Rectangle/Geometry',
    'SectionPoints':\
        '/ImageDocument/Metadata/MetadataNodes/MetadataNode/Layers/Layer[2]/Elements/Polygon',
    }

# how many calibration markers to read in, this should essentially be a constant
nmarkers = 3

# xxx - likely this is a Zeiss bug, units for the scale in the xml file are not correct
scale_units = 1e6

### load czi file and extract metadata

# czi file object with image data and meta
czi = CziFile(fn)

# get the root of the metadata xml
#root = ET.fromstring(metastr) # to convert to python etree
root = czi.metadata.getroottree()

if metafnout or print_meta:
    metastr = etree.tostring(root, pretty_print=True).decode('utf-8')
    if metafnout:
        with open(metafnout, 'w') as file:
            file.write(metastr)
    if print_meta:
        print(metastr)

# how to get paths by searching for tags, for reference:
#a = root.findall('.//Polygon'); print('\n'.join([str(root.getpath(x)) for x in a]))
#a = root.findall('.//Rectangle'); print('\n'.join([str(root.getpath(x)) for x in a]))

# get the pixel size
# xxx - scale is given in meters but units say microns, this is a Zeiss bug??? convert to microns
scale = np.zeros((2,), dtype=np.double)
scale[0] = float(root.xpath(xml_paths['ScaleX'])[0].text)*scale_units
scale[1] = float(root.xpath(xml_paths['ScaleY'])[0].text)*scale_units

# get the bounding box on the scene
# Could not find bounding box around all the scenes in the xml, which is the bounding box for the images.
#   The bounding box for the images is not the same as the rectangle defined by the markers.
#   So, load all the scene position and size information for calculating the bounding box.
scenes = root.xpath(xml_paths['Scenes'])[0].findall('Scene'); nscenes = len(scenes); found = False
center_positions = np.zeros((nscenes,2), dtype=np.double)
contour_sizes = np.zeros((nscenes,2), dtype=np.double)
for scene in scenes:
    i = int(scene.attrib['Index'])
    center_positions[i,:] = np.array([float(x) for x in scene.find('CenterPosition').text.split(',')])
    contour_sizes[i,:] = np.array([float(x) for x in scene.find('ContourSize').text.split(',')])
    found = (found or (i == load_scene))
assert(found) # bad scene index
center_position = center_positions[load_scene]
contour_size = contour_sizes[load_scene]
all_scenes_offset = (center_positions - contour_sizes/2).min(axis=0)

# get the marker positions
marker_points = np.zeros((nmarkers,2),dtype=np.double) # xxx - do we need the z-position of the marker?
markers = root.xpath(xml_paths['Calibration'])[0]
for i in range(nmarkers):
    marker = markers.findall('.//Marker%d' % (i+1,))
    marker_points[i,0] = float(marker[0].findall('.//X')[0].text)
    marker_points[i,1] = float(marker[0].findall('.//Y')[0].text)

# get the bounding box on the slice and ROI polygons
box = root.xpath(xml_paths['SelectionBox'])[0]
box_corner = np.zeros((2,),dtype=np.double); box_size = np.zeros((2,),dtype=np.double)
box_corner[0] = float(box.findall('.//Left')[0].text); box_corner[1] = float(box.findall('.//Top')[0].text)
box_size[0] = float(box.findall('.//Width')[0].text); box_size[1] = float(box.findall('.//Height')[0].text)
                  
# get the first slice polygon
polygons = root.xpath(xml_paths['SectionPoints'])
npolygons = len(polygons) if npolygons > len(polygons) else npolygons
polygons_points = [None]*npolygons
polygons_rotation = np.zeros((npolygons,),dtype=np.double)
for polygon,i in zip(polygons,range(npolygons)):
    polygons_points[i] = np.array([[float(y) for y in x.split(',')] \
                   for x in polygon.findall('.//Points')[0].text.split(' ')])
    polygons_rotation[i] = float(polygon.findall('.//Rotation')[0].text)/180*np.pi # convert to radians

### calculate coordinate transformations and transform points to image coordinates

# calculate the rotation angle of the rectangle defined by the markers relative to the global coordinate frame
# get the two markers that are furthest away from each other
assert(nmarkers==3) # wrote this code assuming the markers are three corners of a rectangle
D = scidist.squareform(scidist.pdist(marker_points)); diag_dist = D.max()
other_inds = np.array(np.unravel_index(np.argmax(D), (nmarkers,nmarkers)))
corner_ind = np.setdiff1d(np.arange(3), other_inds)[0]
# get the rotation angle correct by measuring the angle to the point with the larger x-deviation, centered on the corner
a = marker_points[other_inds[0],:]-marker_points[corner_ind,:]
b = marker_points[other_inds[1],:]-marker_points[corner_ind,:]
marker_vector = a if np.abs(a[0]) > np.abs(b[0]) else b    
marker_angle = np.arctan(marker_vector[1]/marker_vector[0])
c, s = np.cos(marker_angle), np.sin(marker_angle); marker_rotation = np.array([[c, -s], [s, c]])

# get the coordinates of the other corner of the marker-defined rectangle
pts = np.dot(marker_rotation.T, marker_points[other_inds,:] - marker_points[corner_ind,:])
apts = np.abs(pts); marker_rectangle_size = apts.max(axis=0); inds = np.argmax(apts,axis=0)
pt = np.zeros((2,),dtype=np.double); pt[0] = pts[inds[0],0]; pt[1] = pts[inds[1],1]
all_marker_points = np.zeros((nmarkers+1,2),dtype=np.double)
all_marker_points[:3,:] = marker_points
all_marker_points[3,:] = np.dot(marker_rotation, pt) + marker_points[corner_ind,:]

# for the marker offset from the global coordinate frome, use the corner closest to the origin
marker_offset = all_marker_points[np.argmin(np.sqrt((all_marker_points**2).sum(1))),:]

# convert to pixel coordinates using marker offsets and pixel scale
# global coordinates to the corner of the bounding box around all the scenes in pixels
all_scenes_corner_pix_global = ((np.dot(marker_rotation.T, all_scenes_offset - marker_offset) + \
                                 marker_offset)/scale).astype(np.int64)
# coordinates to the corner of the scene bounding box relative to the bounding box around all the scenes
scene_corner_pix = ((np.dot(marker_rotation.T, center_position - contour_size/2 - marker_offset) + \
                     marker_offset)/scale).astype(np.int64) - all_scenes_corner_pix_global
# the size of the scene is rotation invariant
scene_size_pix = (contour_size/scale).astype(np.int64)

# selection box is defined relative bounding box around all the scenes but is specified in pixel space
box_corner_pix = box_corner - scene_corner_pix
box_size_pix = box_size

# http://paulbourke.net/geometry/polygonmesh/
def PolyCentroid(x,y):
    xn = np.roll(x,1); yn = np.roll(y,1)
    coeff = 1/3.0/(np.dot(x,yn)-np.dot(xn,y))
    common = x*yn - xn*y
    return coeff*np.dot( x+np.roll(x,1), common ), coeff*np.dot( y+np.roll(y,1), common )

# points are defined same as the selection bounding box, polygons are rotated about ???.
#   keep ROI points as floating point.
for i in range(npolygons):
    c, s = np.cos(polygons_rotation[i]), np.sin(polygons_rotation[i]); R = np.array([[c, -s], [s, c]])
    
    # xxx - what point is Zeiss rotating about??? 
    #ctr = polygons_points[i].mean(0)
    #ctr = np.median(polygons_points[i], axis=0)
    #ctr = polygons_points[i].max(0)
    #ctr = polygons_points[i].min(0)
    #ctr = np.array(PolyCentroid(polygons_points[i][:,0], polygons_points[i][:,1]))
    #tmp = np.vstack((polygons_points[i], polygons_points[i][0,:]))
    #ctr = np.array(PolyCentroid(tmp[:,0], tmp[:,1]))

    # fudge factor for iron-k0100-20 and 12-dab-serial_50nm_w sections-rois.czi
    ctr = polygons_points[i].mean(0) + np.array([45,70],dtype=np.double)
    
    polygons_points[i] = np.dot(R, (polygons_points[i] - ctr).T).T + ctr - scene_corner_pix

### load the image data and crop to specified scene

# xxx - is there a way to just read one "scene" without importing all the data?
# (?, scenes, ?, xdim, ydim, colors?)
print('Loading czi data for scene %d' % (load_scene,)); t = time.time()
img = np.squeeze(czi.asarray()[:,load_scene,:,:,:])
print('\tdone in %.4f s' % (time.time() - t, ))
assert( img.ndim == 2 ) # multiple colors or other dims?

# crop out the scene
# xxx - meh, xy axis swapping, whose fault is this? very not-Zen-like
img = img[scene_corner_pix[1]:scene_corner_pix[1]+scene_size_pix[1],
          scene_corner_pix[0]:scene_corner_pix[0]+scene_size_pix[0]]

if doplots:
    img_ds = img[0::doplots_ds,0::doplots_ds]
    interp_string = 'nearest'
    #pl.figure(1);
    fig,ax = pl.subplots(1)
    ax.imshow(img_ds,interpolation=interp_string, cmap='gray');
    pl.title('Scene %d' % (load_scene,))
    for i in range(npolygons):
        poly = patches.Polygon(polygons_points[i]/doplots_ds,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(poly)    
    cnr = box_corner_pix/doplots_ds; sz = box_size_pix/doplots_ds
    rect = patches.Rectangle(cnr,sz[0],sz[1],linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)
    pl.show()
