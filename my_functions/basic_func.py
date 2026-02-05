import xarray as xr
from typing import List, Union
import numpy as np

def get_area():
    import xarray as xr
    
    area = xr.open_dataset('/work/mh0066/m300909/mpiesm-landveg/mpiesm-landveg/experiments/fixed_SST_1co2/aquaplanet_sponge_layer_rotation_1x/outdata/grid_area.nc',decode_times=True)
    
    return area.cell_area[:]


def compute_area(lat_slice=None):
    """Compute the grid area and total area, optionally slicing by latitude."""
    grid_area = get_area()

    if lat_slice:
        grid_area = grid_area.sel(lat=lat_slice)

    total_area = grid_area.sum(dim=['lat', 'lon'])
    return grid_area, total_area


def weighted_mean(data, lat_slice=None, time_series=False):
    """Compute the weighted mean by weighing different latitudes."""
    grid_area, total_area = compute_area(lat_slice)
    data_area_weighted = (data * grid_area) / total_area
    data_area_weighted = data_area_weighted.sum(dim=['lat', 'lon'])

    if 'time' in data.dims:
        if time_series:
            return data_area_weighted
        return data_area_weighted.mean(dim='time')
    else:
        return data_area_weighted


def area_mean_weighted(data, lat_slice, time_series=False):
    """Compute the area-weighted mean for a given latitude slice."""
    return weighted_mean(data, lat_slice, time_series)


def global_mean_weighted(data, time_series=False):
    """Compute the global weighted mean."""
    return weighted_mean(data, time_series=time_series)


def global_std_error(data):

    import numpy as np
    
    data_area_weighted = global_mean_weighted(data, time_series = True)
    global_std = data_area_weighted.std(dim=['time'], ddof=1)
    
    sample_size = data_area_weighted.sizes['time']
    std_error = global_std / np.sqrt(sample_size)
        
    return std_error

def zonal_time_mean(data):
    """Compute the zonal time average"""
    return data.mean(dim=['lat','time'])

def potential_temperature(temperature, pressure):
    """
    Calculate the potential temperature.

    Parameters:
    - temperature (float or ndarray or xarray.DataArray): The temperature in Kelvin.
    - pressure (float or ndarray or xarray.DataArray): The pressure in Pascal.

    Returns:
    - float or ndarray or xarray.DataArray: The potential temperature in Kelvin.
    """
    # Exponent: R/cp
    exponent = 0.286
    p0 = 100000  # Reference pressure in Pascal

    theta = temperature * (p0 / pressure) ** exponent
    return theta


def static_stability(temp):

    import numpy as np
    import xarray as xr
    
    """
    Calculate the static stability.

    Parameters:
    - temp (xarray.DataArray): An xarray DataArray with a 'plev' coordinate.

    Returns:
    - xarray.DataArray: The static stability.
    """

    if 'plev' not in temp.coords:
        raise ValueError("The 'temp' DataArray must have a 'plev' coordinate.")

    pot_temp = potential_temperature(temp, temp.plev)
    stability = -(temp / pot_temp) * pot_temp.differentiate('plev')
    return stability


def dict_to_xarray(data, var, iterate_dim=('rotation', []), time_slice=None):

    import xarray as xr

    """
    Convert a dictionary of data into an xarray DataArray with a specified dimension.

    Parameters:
    - data: dict
        Dictionary containing data arrays.
    - iterate_dim: tuple
        A tuple where the first element is the dimension name (e.g., 'rotation')
        and the second element is a list of values for this dimension.
    - var: str
        Variable name to select from each data array.
    - time_slice: slice, optional
        Time slice to select from each data array.

    Returns:
    - xarray.DataArray
        Concatenated DataArray with the specified dimension.
    """
    dim_name, dim_values = iterate_dim
    data_ls = []
    
    for key, item in data.items():
        data_ls.append(item[var].sel(time=time_slice))

    # Convert the list of data arrays into an xarray DataArray
    data_da = xr.concat(data_ls, dim=dim_name)
    data_da[dim_name] = dim_values
    return data_da



def subsidence_mask(omega, plev=None, hadley=None):
    """
    Create a subsidence mask for the given omega dataset.

    Parameters:
    - omega: xarray DataArray containing the omega data.
    - plev: Pressure level to select from the omega data.
    - hadley: Latitude range for the Hadley cell to select.

    Returns:
    - A boolean mask where True indicates subsidence (omega > 0).
    """
    # Initialize an empty dictionary for selection criteria
    selection_criteria = {}

    # Add pressure level to selection criteria if specified
    if plev is not None:
        selection_criteria['plev'] = plev

    # Add latitude range to selection criteria if specified
    if hadley is not None:
        selection_criteria['lat'] = slice(hadley, -hadley)

    # Apply selection criteria to omega
    omega = omega.sel(**selection_criteria)
    
    # Return a boolean mask indicating subsidence
    return omega > 0


def feedback_parameter(N2: Union[np.ndarray, xr.DataArray], N1: Union[np.ndarray, xr.DataArray], 
                       T2: Union[np.ndarray, xr.DataArray], T1: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:
    """
    Computes the feedback parameter using radiative imbalance from different simulations.
    
    Parameters:
        N2 (np.ndarray or xr.DataArray): Variable from the warming experiment.
        N1 (np.ndarray or xr.DataArray): Variable from the control experiment.
        T2 (np.ndarray or xr.DataArray): Surface temperature of warming experiment.
        T1 (np.ndarray or xr.DataArray): Surface temperature of control experiment.
        
    Returns:
        np.ndarray or xr.DataArray: Feedback parameter (lambda).
    """
    var_change = N2 - N1
    temp_change = T2 - T1
    
    return var_change / temp_change


def cess_sensitivity(N2: Union[np.ndarray, xr.DataArray], N1: Union[np.ndarray, xr.DataArray], 
                       T2: Union[np.ndarray, xr.DataArray], T1: Union[np.ndarray, xr.DataArray]) -> Union[np.ndarray, xr.DataArray]:

    forcing = 3.6
    feedback = -1 * feedback_parameter(N2, N1, T2, T1)
    
    return forcing / feedback







