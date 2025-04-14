import xarray as xr
import numpy as np
import torch

def array_to_xarray(array, name_vars, dims, template):
        out=xr.Dataset(data_vars={name_vars[ind_nv]: (dims, array[:,ind_nv,:,:]) for ind_nv in range(len(name_vars))},
                     coords=template.coords)
        return out

def changeLongitudeProjection(grid):
    grid=grid.assign_coords({'longitude': list(np.arange(0,180+0.25,0.25))+list(np.arange(-179.75,0,0.25))})
    return grid

def get_date_ic_sfno(date_ic,path_data,vars):
      grid=[]
      for var in vars:
          print(var)
          grid_i=xr.open_dataset(path_data+var+'.nc').sel(valid_time=date_ic,method='nearest')
          grid.append(grid_i.drop_vars('pressure_level', errors='ignore'))
      grid=xr.merge(grid)
      grid=grid.expand_dims(dim ={'time': 1}, axis=0)
      return grid

def get_input_array(grid):
    inp = np.array(grid.load().to_array())
    # remove erronious singleton last dimension if present
    inp_shape = inp.shape
    if len(inp_shape)==5 and inp_shape[-1]==1:
        inp = inp.squeeze(-1)
    inp = np.transpose(inp, [1,0,2,3]) # re-arrange the 'var' dimension
    return inp

def scaleGrid_sfno(grid, path_data_mean, path_data_std, return_params=False):
    ## Load mean and std
    mean=np.load(path_data_mean)
    std=np.load(path_data_std)
    ## Put mean and std into xarray template
    mean_arr=np.ones((1,73,721,1440))
    std_arr=np.ones((1,73,721,1440))
    for v in range(73):
        mean_arr[:,v,:,:]=mean[:,v,:,:]
        std_arr[:,v,:,:]=std[:,v,:,:]
    mean_xrr=array_to_xarray(mean_arr,name_vars=list(grid.keys()),dims=grid['tcwv'].dims, template=grid).isel(time=0)
    std_xrr=array_to_xarray(std_arr,name_vars=list(grid.keys()),dims=grid['tcwv'].dims, template=grid).isel(time=0)
    ## Scale data
    if return_params is False:
        return (grid-mean_xrr)/std_xrr
    else:
        return (grid-mean_xrr)/std_xrr, mean_xrr, std_xrr

def load_ic(date_ic, sfno_vars, workdir, device):
    ##########################################
    # Initial condition
    data_ic=get_date_ic_sfno(date_ic=date_ic, path_data=workdir+'/data/era5/', vars=sfno_vars)
    # Scale data
    # Download the .npy files (means and standard deviations) from the ECMWF ai-models Github repository: 
    # https://github.com/ecmwf-lab/ai-models
    # Create a folder called "stats" in your working directory and store the .npy files here
    grid_delete, mean_xrr, sigma_xrr=scaleGrid_sfno(data_ic,
                                                    path_data_mean=workdir+'/sfno_data/global_means.npy',
                                                    path_data_std=workdir+'/sfno_data/global_stds.npy',
                                                    return_params=True)
    x=get_input_array(data_ic)
    mean_arr=np.array(mean_xrr.load().to_array())[None,:,:,:]
    sigma_arr=np.array(sigma_xrr.load().to_array())[None,:,:,:]
    x=torch.tensor(x).to(device, dtype=torch.float)
    x.requires_grad=True
    mean_arr=torch.tensor(mean_arr).to(device, dtype=torch.float)
    mean_arr.requires_grad=True
    sigma_arr=torch.tensor(sigma_arr).to(device, dtype=torch.float)
    sigma_arr.requires_grad=True
    return x, mean_arr, sigma_arr, data_ic

def xr4D(grid, levels, pressure_vars, surface_vars):
    grid_pressure=[]
    for pressure_var in pressure_vars:
        grid_var=[]
        for level in levels:
            var=pressure_var+str(level)
            grid_level=grid[var]
            grid_level=grid_level.assign_coords({'isobaric':level})
            grid_var.append(grid_level)
        grid_var=xr.concat(grid_var, dim='isobaric')
        grid_pressure.append(grid_var)
    grid_pressure=xr.merge(grid_pressure)
    grid_pressure=grid_pressure.rename({'ta50':'ta','z50':'z','ua50':'ua','va50':'va','hur50':'hur'})
    grid_surface=grid[surface_vars]
    grid=xr.merge([grid_surface,grid_pressure])
    return grid

