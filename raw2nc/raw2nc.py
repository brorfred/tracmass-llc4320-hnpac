import numpy as np
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import xarray as xr

#
# Plot various fields from LLC4320 Hawaii/North Pacific cutout region.
# Shows how to read files, what associated grid is, what times different 
# file name indices correspond to etc....
#

# Fields directories
droot='/nobackup1b/users/jahn/hinpac/grazsame3/run/run.0354';
sdflds=['offline-0604', 'offline']
sdfld=['THETA','SALT','UVEL','VVEL','WVEL']

##########################
# Read in Grid variables #
##########################
nz=40;nx=1080;ny=2700;
fxc="%s/XC.data"%(droot)
fxg="%s/XG.data"%(droot)
fyc="%s/YC.data"%(droot)
fyg="%s/YG.data"%(droot)

dtyp='>f4';nr=nx*ny;
RS = lambda phi: np.reshape(phi,(ny,nx))
xc=np.fromfile(fxc, dtype=dtyp,count=nr);
xg=np.fromfile(fxg, dtype=dtyp,count=nr);
yc=np.fromfile(fyc, dtype=dtyp,count=nr);
yg=np.fromfile(fyg, dtype=dtyp,count=nr);
print('West limit  %10.5fE'  % min(xc))
print('East limit  %10.5fE'  % max(xc))
print('North limit %10.5fN' % max(yc))
print('South limit %10.5fN' % min(yc))
xc=RS(xc);xg=RS(xg);yc=RS(yc);yg=RS(yg);

# Show horizontal Grid variables
fig=plt.figure(figsize=(20, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.subplot(1,4,1); plt.imshow(xc,origin='lower',cmap='gist_ncar');plt.title('XC - lon cell center');cbar=plt.colorbar()
ax=plt.subplot(1,4,2); plt.imshow(xg,origin='lower',cmap='gist_ncar');plt.title('XG - lon cell corner (SW)');cbar=plt.colorbar()
ax=plt.subplot(1,4,3); plt.imshow(yc,origin='lower',cmap='gist_ncar');plt.title('YC - lat cell center');cbar=plt.colorbar()
ax=plt.subplot(1,4,4); plt.imshow(yg,origin='lower',cmap='gist_ncar');plt.title('YG - lat cell corner (SW)');cbar=plt.colorbar()
plt.savefig('grid-plots.png', bbox_inches='tight')

# Show how latitudinal grid spacing decreases with latitude and longitudinal spacing is constant
fig=plt.figure(figsize=(20, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.subplot(1,2,1);
plt.plot(yc[1:-1,1]-yc[0:-2,1],yc[1:-1,1]);plt.plot(yc[1:-1,-1]-yc[0:-2,-1],yc[1:-1,-1]);
plt.title('Latitudinal spacing versus latitude')
plt.ylabel('Latitude');plt.xlabel('Latitudinal spacing');

plt.subplot(1,2,2);
plt.plot(xc[1,1:-1]-xc[1,0:-2],xc[1,0:-2]);plt.xlim((0.0195,0.022))
plt.title('Longitudinal spacing versus longitude')
plt.ylabel('Longitude');plt.xlabel('Longitudinal spacing');
plt.savefig('grid-line-plots.png', bbox_inches='tight')

# Show vertical levels information
fdrf="%s/DRF.data"%(droot)
drf=np.fromfile(fdrf, dtype=dtyp,count=nz);
zf=[0]
zf=-np.concatenate((zf,np.cumsum(drf)))
zc=0.5*( zf[0:-1]+zf[1:] )
fig=plt.figure(figsize=(20, 8), dpi= 80, facecolor='w', edgecolor='k')
plt.subplot(1,2,1);
plt.plot(zf,'x');
plt.title('Cell interface depths');plt.ylabel('Depth (m)')

plt.subplot(1,2,2);
plt.plot(zc,'.');
plt.title('Cell center depths');plt.ylabel('Depth (m)');
plt.savefig('vert-grid-line-plots.png', bbox_inches='tight')

###################################################################################################
# Read and plot physical fields (velocity components, temperature, salinity, sea-surface height). #
###################################################################################################

# Get times and create table of iteration numbers plus dates and times
import os
import time
itvalLo=144
itvalHi=1259856
itList=np.arange(itvalLo,itvalHi+144,144)
os.environ['TZ']='UTC'
tVals=[]
for i in itList:
 ts=time.gmtime(time.mktime(time.strptime('2011/09/11:UTC','%Y/%m/%d:%Z'))+25.*i)
 tstr=time.strftime('%Y-%m-%dT%H:%M:%S', ts)
 tVals.append(tstr)
 
tsLo=time.gmtime(time.mktime(time.strptime('2011/09/11:UTC','%Y/%m/%d:%Z'))+25.*itList[0])
tstr=time.strftime('%Y-%m-%dT%H:%M:%S', tsLo)
print("Initial time and time step number ", tstr, itList[0])
tsHi=time.gmtime(time.mktime(time.strptime('2011/09/11:UTC','%Y/%m/%d:%Z'))+25.*itList[-1])
tstr=time.strftime('%Y-%m-%dT%H:%M:%S', tsHi)
print("Final time and time step number ", tstr, itList[-1])

tsNumList=xr.DataArray(itList,coords={'Time':tVals},dims=('Time'))

# Get a particular timestep number
tN=10
print("Time step number: ",tsNumList[tN].values)
print("Corresponding time: ",tsNumList[tN].coords)

#
# Get and plot fields
#
itVal=tsNumList[tN].values
nrin=nz
lp=10
cm='gist_ncar'
def PLT():
  phi=np.fromfile(fn, dtype='>f4',count=nx*ny*nrin)
  phixyz=np.reshape(phi, (nrin, ny, nx)) 
  phixy=phixyz[lp,:,:]
  phixym=np.ma.masked_equal(phixy,0)
  fig=plt.figure(figsize=(10, 16), dpi= 600, facecolor='w', edgecolor='k')
  plt.title(pt)
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  plt.pcolormesh(X,Y,WIN(phixym),cmap=cm)
  plt.colorbar();
#
# Temperature
fn="%s/%s/%s/_.%10.10d.data"%(droot,sdflds[0],'THETA',itVal)
dep=zc[lp];td=tsNumList[tN].coords['Time'].values;
X=xg[:,:];Y=yg[:,:];pt='Potential temp (degress C) at z=%7.3fm T=%s'%(dep,td);
WIN = lambda phi: phi; PLT()
plt.savefig('potential-temperature.png', bbox_inches='tight')
#
# Vertical velocity
fn="%s/%s/%s/_.%10.10d.data"%(droot,sdflds[0],'WVEL',itVal)
dep=zf[lp];td=tsNumList[tN].coords['Time'].values;
X=xg[:,:];Y=yg[:,:];pt='W (m/s) at z=%7.3fm T=%s'%(dep,td);
WIN = lambda phi: phi; PLT()
plt.savefig('vertical-velocity.png', bbox_inches='tight')
#
# Zonal velocity
fn="%s/%s/%s/_.%10.10d.data"%(droot,sdflds[0],'UVEL',itVal)
dep=zc[lp];td=tsNumList[tN].coords['Time'].values;
X=0.5*(xg[1:,0:-1]+xg[1:,1:]); Y=0.5*(yc[0:-1,1:]+yc[1:,1:]); pt='U (m/s) at z=%7.3fm T=%s'%(dep,td);
WIN = lambda phi: phi[1:,1:]; PLT()
plt.savefig('zonal-velocity.png', bbox_inches='tight')
#
# Meridional velocity
fn="%s/%s/%s/_.%10.10d.data"%(droot,sdflds[0],'VVEL',itVal)
dep=zc[lp];td=tsNumList[tN].coords['Time'].values;
X=0.5*(xc[0:-1,1:]+xc[1:,1:]); Y=0.5*(yg[1:,0:-1]+yg[1:,1:]); pt='V (m/s) at z=%7.3fm T=%s'%(dep,td);
WIN = lambda phi: phi[1:,1:]; PLT()
plt.savefig('meridional-velocity.png', bbox_inches='tight')
#
# Salinity
fn="%s/%s/%s/_.%10.10d.data"%(droot,sdflds[0],'SALT',itVal)
dep=zc[lp];td=tsNumList[tN].coords['Time'].values;
X=xg[:,:];Y=yg[:,:];pt='Salinity (ppt) at z=%7.3fm T=%s'%(dep,td);
WIN = lambda phi: phi; PLT()
plt.savefig('salinity.png', bbox_inches='tight')

# Now write out some example netcdf using xarray
# First set up grid
xcbx=0.5*(xc[:,1:]+xc[:,:-1])
ycby=0.5*(yc[1:,:]+yc[:-1,:])
xgbx=0.5*(xg[:,1:]+xg[:,:-1])
ygby=0.5*(yg[1:,:]+yg[:-1,:])

xu=xcbx[1 ,: ];# print(xu.shape);print(xu[0:2],xu[-1]) # UVEL[1:-1,1:]
yu=ygby[1:,1 ];# print(yu.shape);print(yu[0:2],yu[-1])

xv=xgbx[1 ,1:];# print(xv.shape);print(xv[0:2],xv[-1]) # VVEL[1:,1:-1]
yv=ycby[: ,1 ];# print(yv.shape);print(yv[0:2],yv[-1])

ds=xr.Dataset()

ds['xu']=xu;ds.set_coords(['xu'])
ds.xu.attrs.update({'Description':'U velocity point longitudes'})
ds.xu.attrs.update({'units':'degrees_east'})

ds['yu']=yu;ds.set_coords(['yu'])
ds.yu.attrs.update({'Description':'U velocity point latitudes'})
ds.yu.attrs.update({'units':'degrees_north'})

ds['xv']=xv;ds.set_coords(['xv'])
ds.xv.attrs.update({'Description':'V velocity point longitudes'})
ds.xv.attrs.update({'units':'degrees_east'})

ds['yv']=yv;ds.set_coords(['yv']);
ds.yv.attrs.update({'Description':'V velocity point latitudes'});
ds.yv.attrs.update({'units':'degrees_north'});

ds['zf']=zf[:-1];ds.set_coords(['zf']);
ds.zf.attrs.update({'Description':'W velocity point depths'});
ds.zf.attrs.update({'units':'m'});

ds['zc']=zc;ds.set_coords(['zc']);
ds.zc.attrs.update({'Description':'Scalar quantity cell center point depths'});
ds.zc.attrs.update({'units':'m'});

ds['xc']=xc[0,:];ds.set_coords(['xc'])
ds.xc.attrs.update({'Description':'Scalar quantity cell center point longitudes'})
ds.xc.attrs.update({'units':'degrees_east'})

ds['yc']=yc[:,0];ds.set_coords(['yc'])
ds.yc.attrs.update({'Description':'Scalar quantity cell center point latitudes'})
ds.yc.attrs.update({'units':'degrees_north'})

ds['xg']=xg[0,:];ds.set_coords(['xg'])
ds.xg.attrs.update({'Description':'Scalar quantity cell corner point longitudes'})
ds.xg.attrs.update({'units':'degrees_east'})
ds.xg.attrs.update({'standard_name':'longitude'})

ds['yg']=yg[:,0];ds.set_coords(['yg'])
ds.yg.attrs.update({'Description':'Scalar quantity cell center point latitudes'})
ds.yg.attrs.update({'units':'degrees_north'})
ds.yg.attrs.update({'standard_name':'latitude'})

# ds['T']=tsNumList[tN].coords['Time'].values;ds.set_coords(['T'])
ds['time']=[tsNumList[tN].values*25.0];ds.set_coords(['time'])
ds.time.attrs.update({'Description':'Time in seconds'})
ds.time.attrs.update({'units':'seconds since 2011-09-11 00:00 UTC'})
ds.time.attrs.update({'time_origin':'11-SEP-2011 00:00:00 UTC'})
ds.time.attrs.update({'standard_name':'time'})


# Now add some fields
fn="%s/%s/%s/_.%10.10d.data"%(droot,sdflds[0],'THETA',itVal)
phi=np.fromfile(fn, dtype='>f4',count=nx*ny*nrin)
phixyz=np.reshape(phi, (1, nrin, ny, nx))
ds['theta']=(['time', 'zc', 'yc', 'xc'],  phixyz)
ds.theta.attrs.update({'Description':'Potential temperature'})
ds.theta.attrs.update({'units':'degrees_celcius'})

fn="%s/%s/%s/_.%10.10d.data"%(droot,sdflds[0],'SALT',itVal)
phi=np.fromfile(fn, dtype='>f4',count=nx*ny*nrin)
phixyz=np.reshape(phi, (1, nrin, ny, nx))
ds['salinity']=(['time', 'zc', 'yc', 'xc'],  phixyz)
ds.salinity.attrs.update({'Description':'Salinity'})
ds.salinity.attrs.update({'units':'psu'})

fn="%s/%s/%s/_.%10.10d.data"%(droot,sdflds[0],'UVEL',itVal)
phi=np.fromfile(fn, dtype='>f4',count=nx*ny*nrin)
phixyz=np.reshape(phi, (1, nrin, ny, nx))
ds['uvel']=(['time', 'zc', 'yu', 'xu'],  phixyz[:,:,1:-1,1:])
ds.uvel.attrs.update({'Description':'Zonal current speed'})
ds.uvel.attrs.update({'units':'m/s'})

fn="%s/%s/%s/_.%10.10d.data"%(droot,sdflds[0],'VVEL',itVal)
phi=np.fromfile(fn, dtype='>f4',count=nx*ny*nrin)
phixyz=np.reshape(phi, (1, nrin, ny, nx))
ds['vvel']=(['time', 'zc', 'yv', 'xv'],  phixyz[:,:,1:,1:-1])
ds.vvel.attrs.update({'Description':'Meridional current speed'})
ds.vvel.attrs.update({'units':'m/s'})

fn="%s/%s/%s/_.%10.10d.data"%(droot,sdflds[0],'WVEL',itVal)
phi=np.fromfile(fn, dtype='>f4',count=nx*ny*nrin)
phixyz=np.reshape(phi, (1, nrin, ny, nx))
ds['wvel']=(['time', 'zf', 'yc', 'xc'],  phixyz)
ds.wvel.attrs.update({'Description':'Vertical current speed'})
ds.wvel.attrs.update({'units':'m/s'})

fnnc='FIELDS_%10.10d.nc'%(tsNumList[tN].values)

ds.to_netcdf(path=fnnc)
