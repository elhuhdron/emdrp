# EM Data Container (EMDC) for EM Data Reconstruction Pipeline (EMDRP)

## Overview

The EM Data Container should be designed for handling [all the types of data](README.md#other-emdrp-data-formats) that are manipulated in the EMDRP.

[HDF5](https://www.hdfgroup.org/HDF5/) has been chosen as the format for storage of raw EM data.

## Requirements

- hdf5 datasize always set to size of entire datasets, taken from EM acquistion parameters
- although entire datasets stored in single hdf5 files, different types of data in the EMDRP stored in separate hdf5 files
- compression enabled
    - for hdf5 this requires chunking, refer to here as *h5chunk* (*chunksize* in hdf5)
    - h5chunk sized based on typical data access
- Back-compatibility including read write to Knossos raw data format
    - location addressing based on knossos-size raw chunks, refer to here as *kchunk*
    - size 128x128x128 essentially fixed, but not hard coded
- addressing method based on xyz kchunk location and offset plus size
    - offset in voxels, size optionally in voxels or kchunks
- supports orthogonal plane reslicing of data
- hdf5 data layout in *F-order* for compatibility with imagej, knossos and Matlab
- acquisition parameters and important aspects of the dataset stored as hdf5 *attributes*
- capability of reading portions of data out writing portions of data in to/from other formats, application being supported with other format in parentheses:
    - raw (imagej, useful for sanity checks)
    - numpy ndarray in memory (python/numpy)
    - gipl (Matlab)
    - nrrd (itksnap)
    - knossos raw directory structure (Knossos)
    - mat files (Matlab)
    - sql database reads / inserts, including some database interface like jdbc (labrainth)
- output write to other formats includes optional data manipulation features:
    - zero padding
    - type conversion
    - conversion to the output-dependent convenient matrix unraveling order (F-order or C-order)
- fixed 3D dataset shape
    - need support for 2D images or optional nD support?
- support for subgroups that might encompass some varying parameter applied to the data,
    - each of these should be written to defined subgroup locations
    - examples:
        - magnification for raw EM data
        - parameters like threshold for supervoxels
        - userid for individual consensus labels

## Current Structure

hdf5 chunksize is always set to kchunk size, which is essentially locked at 128 x 128 x 128
hdf5 files for all EMDRP data formats are always stored with chunking on and compression enabled.

Currently a [python class hierarchy](images/EM_pipeline_python_class_hierarchy.png) for manipulating hdf5 EM data containers is implemented in ``recon/python``. The superclass of the hierarchy is ``dpLoadh5.py``.

TODO: fill in history of why this is messy
TODO: fill in documentation of a better class hierarchy that fits the requirements defined in this document

### Raw EM Volumes hdf5 container

Raw EM data is shown in <span style="color: rgb(245,127,38);">orange</span> in the EMDRP schematic.

For raw EM data, hdf5 datasets are stored at the top level in a dataset name called **data_mag1**. This is the raw EM data as it comes off of the microscope. Downsamplings of the raw data may also exist at the top level and are indicated by their level of downsampling, for example, data_mag2 is decimated by 2, data_mag4 is decimated by 4, etc

All components in the current EMDRP are **loading from dataset named data_mag1**.
The data type for raw EM data is **always uint8**, the same data type as it is comes off the microscope.

At least these hdf5 attributes are defined for **all** EM datasets. Attributes are stored originally when the raw EM data container is created, and then **should be copied** to any of the other EMDRP data formats based on the raw EM data from which the data was created.

| **Attribute Name** | **data type** | **shape** | **Short Description**                           |
|:-------------------|:--------------|:----------|:------------------------------------------------|
| experiment_name    | string        | ---       | experimental name or identifier for the dataset |
| magnification      | int32         | (1,)      | downsampling level as described above           |
| scale              | float64       | (3,)      | voxel resolution                                |
| rawsize            | int32         | (3,)      | kchunk (knossos) size                           |
| nchunks            | int32         | (3,)      | number of kchunks in the entire dataset         |

Currently a raw EM data container is the default type implemented in the classes ``dpLoadh5.py`` (base class for python manipulation of data container) and ``dpWriteh5.py``.

### Label EM Data hdf5 container

Label data includes consensus training labels (<span style="color: rgb(51,102,154);">blue</span> in EMDRP schematic), supervoxels (<span style="color: rgb(178,36,37);">red</span> in EMDRP schematic) and consensus or automated complete labeled neurons (<span style="color: rgb(240,107,168);">pink</span>) in EMDRP schematic.

For label data, datasets are typically also stored at the top level in a dataset named **labels**.
Label datasets are always unsigned integers that assign each voxel to a unique integer if the voxels belong to the ICS of different neurons. ICS voxels of the same neuron will have the same integer value. Supervoxels may also contain ECS labels, supervoxels that are separate from the ICS supervoxels and identified with a voxel-parallel volume of [voxel classification types](#voxel-classification-type-em-data-hdf5-container).

Although always unsigned integers, different number of bits are used depending on the location in the EMDRP. For the frontend labrainth website, labels are expected to be **uint16**. For supervoxels created in the EMDRP pipeline that are not send to the webiste, labels may be stored as **uint32** or **uint64** to allow for greater number of supervoxels created over larger volumes.

A label value that is the max value for the data type being used (i.e., 2<sup>16</sup> for uint16) indicates **unlabeled** voxels in order to distinguish them from voxels labeled zero which is typically MEM (or other background). Manually labeled EM datasets with ECS typically either use the first label (label 1) or the last label (number of ICS labels + 1) as a single label for all ECS areas. ECS may or may not be connected in 3D, but regardless it is not typically of interest to have ECS volumes segmented into different connected regions like it is for ICS volumes.

These hdf5 attributes are defined for label EM datasets.

| **Attribute Name** | **data type** | **shape** | **Short Description**                           |
|:-------------------|:--------------|:----------|:------------------------------------------------|
| types_nlabels      | int64         | (2,)      | number of ICS and ECS supervoxels               |
| types              | string array  | (3,)      | typically 'MEM', 'ICS', 'ECS'                   |

TODO: get a better handle on types, and possibly completely standardize the enum for types. currently this is flexibly implemented so that types could take on different meanings in the backend (not frontend) but this may be unnecessary and confusing

Currently a label EM data container python implementation is defined in ``utils/typesh5.py`` in the ``emLabels`` class.

### Voxel Classification Type EM Data hdf5 container

Voxel classification type appears as <span style="color: rgb(133,81,161);">purple</span> in the EMDRP schematic.

Classification types are typically stored at the top level in a dataset named **voxel_type**. The data is always again voxel-wise parallel to the raw EM volumes, each voxel labeled with an *enum* identifier of what type of voxel it has been classified as. Most typically the types are ICS, ECS or MEM. For voxel_type each voxel must be definitively assigned to one of the categories (as opposed to the [voxel classification probabilities]()). Because the total number of types will never be very large, **uint8** is used to store the voxel types. Typcally the enum for voxel types is:

Currently voxel_type can be stored into the **same** hdf5 file as the supervoxels (for use with labrainth merging task). This is an exception to the general rule that different hdf5 data formats should be stored in separate hdf5 files, simply for simplicity of the website design.

| **Voxel Type** | **voxel_type value** |
|:---------------|:---------------------|
| MEM            | 0                    |
| ICS            | 1                    |
| ECS            | 2                    |

Currently a voxel type EM data container python implementation is defined in ``utils/typesh5.py`` in the ``emVoxelType`` class.
