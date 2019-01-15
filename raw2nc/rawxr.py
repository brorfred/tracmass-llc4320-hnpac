import os
import time

import numpy as np
import xarray as xr

nz = 40
nx = 1080
ny = 2700
nr = nx*ny

def read_horizontal_grid(fieldname="xc", datadir="./data/", dtype='>f8'):
    """Read in Grid variables from MITGCM data files"""
    fn = os.path.join(datadir, "%s.bin" % fieldname)
    vec = np.fromfile(fn, dtype=dtype, count=nr)
    return np.reshape(vec, (ny,nx))

def read_vertical_grid(datadir="data/"):
    fdrf="%s/DRF.data"%(droot)
    drf=np.fromfile(fdrf, dtype=dtyp,count=nz);
    zf=[0]
    zf=-np.concatenate((zf,np.cumsum(drf)))
    zc=0.5*( zf[0:-1]+zf[1:] )

def setup_time():
    """Create xarray time vec"""
    itvalLo = 233712 #144
    itvalHi = 1495008 #1259856
    itList = np.arange(itvalLo, itvalHi+144, 144)
    os.environ['TZ'] = 'UTC'
    tstrlist = []
    for i in itList:
        ts = time.gmtime(
            time.mktime(time.strptime('2011/09/11:UTC', '%Y/%m/%d:%Z'))+25.*i)
        tstrlist.append(time.strftime('%Y-%m-%dT%H:%M:%S', ts))
    return xr.DataArray(itList,coords={'Time':tstrlist},dims=('Time'))

def read_data(fieldname="uvel", time=862560,datadir="./data/"):
    """Read data from MITGCM binary file"""
    time = int(time)
    fldlist = ["U", "V", "W", "Salt", "Theta"]
    fldstmp = [fn for fn in fldlist if fn[0] in fieldname[0].upper()][0]
    filename = f"{time:010}_{fldstmp}_10800.8150.1_1080.3720.90"
    vec = np.fromfile(os.path.join(datadir,filename), dtype=">f4", count=nr*nz) 
    return np.reshape(vec, (nz, ny, nx))


def create_empty_xarray_object(tpos=10):
    """Create an xarray skeleton"""
    xc = read_horizontal_grid("xc")
    xg = read_horizontal_grid("xg")
    yc = read_horizontal_grid("yc")
    yg = read_horizontal_grid("yg")
    xu = 0.5*(xc[:,1:]+xc[:,:-1])[1 ,: ]
    yu = 0.5*(yc[1:,:]+yc[:-1,:])[1:,1 ]
    xv = 0.5*(xg[:,1:]+xg[:,:-1])[1 ,1:]
    yv = 0.5*(yg[1:,:]+yg[:-1,:])[: ,1 ]

    ds = xr.Dataset()

    def add_fld(fn, fld, desc, units):
        ds[fn] = fld
        ds.set_coords([fn])
        ds[fn].attrs.update({'Description':desc, 'units':units})

    add_fld("xu", xu, 'U velocity point longitudes', 'degrees_east')
    add_fld("yu", yu, 'U velocity point latitudes',  'degrees_north')
    add_fld("xv", xv, 'V velocity point longitudes', 'degrees_east')
    add_fld("yv", yv, 'V velocity point latitudes',  'degrees_north')
    add_fld("xc", xc[0,:], 'Scalar cell center point longs', 'degrees_east')
    add_fld("yc", yc[:,0], 'Scalar cell center point lats', 'degrees_north')
    add_fld("xg", xg[0,:], 'Scalar cell corner point longs', 'degrees_east')
    add_fld("yg", yg[:,0], 'Scalar cell corner point lats', 'degrees_north')
    #add_fld("zf", zf[:-1], 'W velocity point depths', 'm')
    #add_fld("zc", zc, 'Scalar cell center point depths', 'm')

    tvec = setup_time()
    ds['time']=[tvec[tpos].values*25.0];
    ds.set_coords(['time'])
    ds.time.attrs.update({'Description':'Time in seconds'})
    ds.time.attrs.update({'units':'seconds since 2011-09-11 00:00 UTC'})
    ds["time"].attrs.update({'time_origin':'11-SEP-2011 00:00:00 UTC'})
    ds["time"].attrs.update({'standard_name':'time'})

    return ds

def add_fields(ds):
    """Add data fields to an xarray object"""
    meta = {'uvel':['Zonal current speed',     'm/s'],
            'vvel':['Meridional current speed','m/s'],
            'wvel':['Vertical current speed',  'm/s'],
            #'salt':['Salinity',                'psu'],
            #'temp':['Potential temperature',   'deg C'],
            }
    
    for fn in meta.keys():
        mat = read_data(fieldname=fn, time=(ds.time.values/25)[0])
        ds[fn] = (['time', 'zc', 'yc', 'xc'], np.expand_dims(mat, axis=0))
        ds[fn].attrs.update({'Description':meta[fn][0], 'units':meta[fn][1]})



def all_data_to_netcdf():
     tvec = setup_time()
     for tpos in np.arange(len(tvec.values))[:100]:
         print(tpos)
         ds = rawxr.create_xarray_object(tpos=tpos)
         
         
     
