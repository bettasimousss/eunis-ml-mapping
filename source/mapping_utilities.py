from rasterio.profiles import Profile
from rasterio.crs import CRS
import os

class DefaultProfile(Profile):
    """Tiled, band-interleaved, LZW-compressed, 8-bit GTiff."""

    defaults = {
        'driver': 'GTiff',
        'interleave': 'band',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'compress': 'deflate',
        'dtype': 'uint8'
    }
    
def get_blocksize(val):
    """
    Blocksize needs to be a multiple of 16
    """
    if val % 16 == 0:
        return val
    else:
        return (val // 16) * 16
    
def get_rasterio_profile(count,
                         height,
                         width,
                         bounds,
                         epsg,
                         blockxsize=None,
                         blockysize=None,
                         dtype=np.uint8,
                         params={}):
    
    base_profile = DefaultProfile()
    if blockxsize is None:
        blockxsize = get_blocksize(width)

    if blockysize is None:
        blockysize = get_blocksize(height)

    crs = CRS.from_epsg(epsg)

    base_profile.update(
        transform=rio.transform.from_bounds(*bounds,
                                                 width=width,
                                                 height=height),
        width=width,
        height=height,
        blockxsize=blockxsize,
        blockysize=blockysize,
        dtype=dtype,
        crs=crs,
        count=count)
    
    base_profile.update(params)

    return base_profile   
    
    
def write_geotiff_tags(arr,
                       profile,
                       filename,
                       colormap=None,
                       nodata=None,
                       tags=None,
                       bands_tags=None,
                       scales=None,
                       offsets=None):
    """
    Utility to write an array to a geotiff

    Parameters
    ----------
    arr : np.array

    profile : rasterio.profile

    colormap : dict
        Colormap for first band

    nodata : int

    tags : dict
        dataset metadata tags to write

    bands_tags : List[Dict]
        list of dictionaries to write to the corresponding bands

    scales : List
        List of scale values to write to the dataset. List length should
        be arr.shape[0]

    offsets : List
        List of offset values to write to the dataset. List length should
        be arr.shape[0]make
    """
    bands_tags = bands_tags if bands_tags is not None else []

    if nodata is not None:
        profile.update(nodata=nodata)

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=0)

    if os.path.isfile(filename):
        os.remove(filename)

    with rio.open(filename, 'w', **profile) as dst:
        dst.write(arr)

        if tags is not None:
            dst.update_tags(**tags)

        for i, bt in enumerate(bands_tags):
            dst.update_tags(i + 1, **bt)

        if colormap is not None:
            dst.write_colormap(
                1, colormap)

        if scales is not None:
            dst.scales = scales

        if offsets is not None:
            dst.offsets = offsets
            
            
def rasterize_shapefile(vect_obj,out_file):
    # Open the existing raster mask to get its properties
    with rio.open('final_models/export/masks/land_mask.tif') as src:
        transform = src.transform
        out_shape = src.shape
        mask_crs = src.crs
        out_profile = src.profile
    
    # Ensure the GeoDataFrame is in the same CRS as the raster mask
    if vect_obj.crs != mask_crs:
        print('Different CRS, reprojecting')
        vect_obj = vect_obj.to_crs(mask_crs)
    
    raster = rasterize(((geom, 1) for geom in vect_obj.geometry), out_shape=out_shape, transform=transform)
    # Save to a raster file
    with rio.open(out_file, 'w', driver='GTiff', 
                   height=out_shape[0], width=out_shape[1],
                   count=1, dtype=rio.uint8, 
                   crs=mask_crs, transform=transform) as dst:
        
        dst.write(raster, 1)

    src_data = rio.open(out_file).read(1).astype(int)
    write_geotiff_tags(src_data,out_profile,filename=out_file,nodata=0)           