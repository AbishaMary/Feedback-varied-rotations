## Extent of hadley cell

import xarray as xr
from typing import List, Union
import numpy as np

import sys
# define the path where my modules are saved
module_path = 'home/m/m300909/clear sky feedback/'
# add this path tho the system path using the `sys` module
if not module_path in sys.path: sys.path.append(module_path)
# load them as they were regular python packages.
# (the file is called home/m/m300909/py_data_handling.py)
import basic_func as bf

def hadley_cell_extent(mastrfu: xr.DataArray, P_t: float, P_b: float, iterate_dim: str = 'rotation') -> List[float]:
    """
    Determine the extent of the Hadley cell.

    Parameters:
    mastrfu (xr.DataArray): The input data array containing the mass stream function.
    P_t (float): The top pressure level.
    P_b (float): The bottom pressure level.
    iterate_dim (str): Dimension to iterate over, either 'rotation' or 'time'.

    Returns:
    List[float]: Latitudinal extents of the Hadley cell.
    """
    # Reverse the latitude dimension
    reversed_array = mastrfu.isel(lat=slice(None, None, -1))
    extent = []

    # Iterate over each value in the specified dimension
    for dim_val in reversed_array[iterate_dim]:
        # Select the pressure levels and the current dimension value
        reversed_array_mean = reversed_array.sel(plev=slice(P_b, P_t), **{iterate_dim: dim_val}).mean(dim='plev')
        
        # Find the first occurrence where the mass stream function changes sign from positive to negative
        first_match = next((idx for idx, item in enumerate(reversed_array_mean) if item <= 0), None)
        
        if first_match is None:
            # If no sign change is found, use the last latitude value
            extent.append(reversed_array_mean.lat[-1].item())
        else:
            # Choose the previous latitude when the positive value is minimum
            extent.append(reversed_array_mean.lat[max(0, first_match - 1)].item())

    return extent




## Latitude weighted average

def global_mean_lat_weighted(data: xr.DataArray) -> xr.DataArray:
    """
    Computes the global mean using weightage for each latitude.
    
    Parameters:
    data (xr.DataArray): Input data in zonal mean format with a 'lat' dimension.
    
    Returns:
    xr.DataArray: Global mean by weighing each latitude.
    """
    
    # Calculate the weights for each latitude
    latitudes = data.lat
    weights = np.cos(np.deg2rad(latitudes))
    
    # Normalize the weights so that their mean is 1
    normalized_weights = weights / weights.mean()
    
    # Apply the weights to the data and calculate the weighted mean along the 'lat' dimension
    global_mean = (data * normalized_weights).mean(dim='lat')
    
    return global_mean




## Latitudinal mean profile of mastrfu

def mean_mastrfu_profile(mastrfu: xr.DataArray, iterate_dim: str = 'rotation') -> xr.DataArray:
    """
    Calculate the mean profile of the mastrfu DataArray.

    Parameters:
    mastrfu (xr.DataArray): Input DataArray with dimensions ['plev', 'time', 'lon', 'lat', 'rotation'].
    iterate_dim (str): Dimension to iterate over, either 'rotation' or 'time'.

    Returns:
    xr.DataArray: A DataArray of mean profiles for each value in the specified dimension.
    """
    
    # Initialize an empty list to store the mean profiles
    mean_profile = []

    # Calculate the mean over 'time' and 'lon' dimensions, and select the northern hemisphere
    mean_dims = ['lon'] if iterate_dim == 'time' else ['time', 'lon']
    mastrfu_north = mastrfu.mean(dim=mean_dims).sel(lat=slice(90, 5))

    # Determine the extent of the Hadley cell between 300hPa and 700hPa
    extent = hadley_cell_extent(mastrfu_north, 30000, 70000,iterate_dim=iterate_dim)

    # Iterate over each value in the specified dimension
    for i, dim_value in enumerate(mastrfu_north[iterate_dim]):
        try:
            # Select the data for the current dimension value and latitude slice
            hadley_cell = mastrfu_north.sel({iterate_dim: dim_value, 'lat': slice(extent[i], 0)})
            
            # Calculate the mean over the 'lat' dimension
            mean_mastrfu = global_mean_lat_weighted(hadley_cell)
            
            # Append the mean profile to the list
            mean_profile.append(mean_mastrfu)
            
        except KeyError as e:
            raise ValueError(f"Missing required data for {iterate_dim} {dim_value}: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing {iterate_dim} {dim_value}: {e}")

    # Convert the list of mean profiles into an xarray DataArray
    mean_profile_da = xr.concat(mean_profile, dim=iterate_dim)
    mean_profile_da[iterate_dim] = mastrfu[iterate_dim]

    # Return the DataArray of mean profiles
    return mean_profile_da



def max_mastrfu_profile(mastrfu: xr.DataArray, iterate_dim: str = 'rotation') -> xr.DataArray:
    """
    Calculate the max profile of the mastrfu DataArray.

    Parameters:
    mastrfu (xr.DataArray): Input DataArray with dimensions ['plev', 'time', 'lon', 'lat', 'rotation'].
    iterate_dim (str): Dimension to iterate over, either 'rotation' or 'time'.

    Returns:
    xr.DataArray: A DataArray of max profiles for each value in the specified dimension.
    """
    
    # Initialize an empty list to store the mean profiles
    max_profile = []

    # Calculate the mean over 'time' and 'lon' dimensions, and select the northern hemisphere
    mean_dims = ['lon'] if iterate_dim == 'time' else ['time', 'lon']
    mastrfu_north = mastrfu.mean(dim=mean_dims).sel(lat=slice(90, 5))

    # Determine the extent of the Hadley cell between 300hPa and 700hPa
    extent = hadley_cell_extent(mastrfu_north, 30000, 70000,iterate_dim=iterate_dim)

    # Iterate over each value in the specified dimension
    for i, dim_value in enumerate(mastrfu_north[iterate_dim]):
        try:
            # Select the data for the current dimension value and latitude slice
            hadley_cell = mastrfu_north.sel({iterate_dim: dim_value, 'lat': slice(extent[i], 0)})

            max_index = hadley_cell.argmax(dim='lat')

            lat = hadley_cell.lat[max_index]
            
            # Calculate the mean over the 'lat' dimension
            max_mastrfu = hadley_cell.sel(lat=lat)
            
            # Append the mean profile to the list
            max_profile.append(max_mastrfu)
            
        except KeyError as e:
            raise ValueError(f"Missing required data for {iterate_dim} {dim_value}: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while processing {iterate_dim} {dim_value}: {e}")

    # Convert the list of mean profiles into an xarray DataArray
    max_profile_da = xr.concat(max_profile, dim=iterate_dim)
    max_profile_da[iterate_dim] = mastrfu[iterate_dim]

    # Return the DataArray of mean profiles
    return max_profile_da
    


## Check the required dimensions

def check_required_dims(data_array, required_dims=None, optional_dims=None):
    """
    Check if required dimensions are present in the input data array.

    Parameters:
    data_array (xarray.DataArray): The input data array to check.
    required_dims (set, optional): The set of required dimensions. Defaults to a predefined set.
    optional_dims (set, optional): The set of optional dimensions. Defaults to a predefined set.

    Raises:
    ValueError: If the required dimensions are not present in the input data array.
    """
    if required_dims is None:
        required_dims = {'lat', 'lon'}
    
    if optional_dims is None:
        optional_dims = {'rotation','plev', 'time'}
    
    missing_required_dims = required_dims - set(data_array.dims)
    if missing_required_dims:
        raise ValueError(f"Input data array must contain dimensions: {required_dims}")
    
    missing_optional_dims = optional_dims - set(data_array.dims)
    if missing_optional_dims:
        print(f"Warning: Input data array is missing optional dimensions: {missing_optional_dims}")

# Example usage:
# check_required_dims(data_array)
# check_required_dims(data_array, required_dims={'lat', 'time', 'lon'}, optional_dims={'plev', 'rotation'})


## Find two maxima

from typing import Tuple

def find_two_maxima(data_array: xr.DataArray) -> Tuple[float, float]:
    """
    Finds the two maxima in a given DataArray. The first maximum is found in the first increasing part,
    and the second maximum is found in the second increasing part after the first decreasing part.

    Parameters:
    data_array (xr.DataArray): Input data array.

    Returns:
    Tuple[float, float]: Two maxima values.
    """
    
    # Find the index where the first decreasing part starts
    first_decreasing_index = np.argmax(np.diff(data_array) < 0)
    
    # Find the first maximum in the first increasing part
    first_max = np.max(data_array[:first_decreasing_index + 1])
    
    # Find the index where the second increasing part starts
    second_increasing_index = first_decreasing_index + np.argmax(np.diff(data_array[first_decreasing_index:]) > 0) + 1
    
    # Find the second maximum in the second increasing part
    second_max = np.max(data_array[second_increasing_index:])
    
    return first_max, second_max 


## Shallow and Deep mastrfu

def shallow_deep_mastrfu(mastrfu: xr.DataArray, iterate_dim: str = 'rotation') -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Process the mastrfu DataArray to find shallow and deep maxima.

    Parameters:
    mastrfu (xr.DataArray): Input DataArray containing mastrfu data.
    iterate_dim (str): Dimension to iterate over, either 'rotation' or 'time'.

    Returns:
    tuple: Two DataArrays containing shallow and deep maxima profiles respectively.
    """

    # Check required dimensions
    check_required_dims(mastrfu, required_dims={'plev', 'lat', 'lon'}, optional_dims={'rotation','time'})
    
    # Calculate the mean profile of mastrfu
    mastrfu_mean = mean_mastrfu_profile(mastrfu, iterate_dim=iterate_dim)

    shallow_maxima = []
    deep_maxima = []
    
    # Iterate over each value in the specified dimension to find shallow and deep maxima
    for dim_val in mastrfu_mean[iterate_dim]:
        # Select data for the current dimension value
        data_dim = mastrfu_mean.sel({iterate_dim: dim_val})

        # Find the two maxima for the current dimension value
        shallow_max, deep_max = find_two_maxima(data_dim)

        shallow_maxima.append(shallow_max)
        deep_maxima.append(deep_max)

    # Convert the list of maxima profiles into xarray DataArrays
    shallow_da = xr.concat(shallow_maxima, dim=iterate_dim)
    shallow_da[iterate_dim] = mastrfu[iterate_dim]

    deep_da = xr.concat(deep_maxima, dim=iterate_dim)
    deep_da[iterate_dim] = mastrfu[iterate_dim]
    
    return shallow_da, deep_da




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


def norm_warming(data_ctrl: Union[np.ndarray, xr.DataArray], data_warm: Union[np.ndarray, xr.DataArray], 
                 tsurf_ctrl: xr.DataArray, tsurf_warm: xr.DataArray) -> Union[np.ndarray, xr.DataArray]:
    """
    Normalizes the warming data using the feedback parameter.
    
    Parameters:
        data_ctrl (np.ndarray or xr.DataArray): Data from the control experiment.
        data_warm (np.ndarray or xr.DataArray): Data from the warming experiment.
        tsurf_ctrl (xr.DataArray): Surface temperature of the control experiment.
        tsurf_warm (xr.DataArray): Surface temperature of the warming experiment.
        
    Returns:
        np.ndarray or xr.DataArray: Normalized warming value.
    """
    check_required_dims(tsurf_ctrl)
    check_required_dims(tsurf_warm)

    #tsurf_ctrl_mean = bf.global_mean_weighted(tsurf_ctrl)
    #tsurf_warm_mean = bf.global_mean_weighted(tsurf_warm)

    norm_val = feedback_parameter(data_ctrl, data_warm, tsurf_ctrl, tsurf_warm)

    return norm_val



def find_nearest_plev(st, target_temp=273, iterate_dim: str = 'rotation'):
    """
    Find the pressure level (plev) where the temperature is closest to the target temperature
    for each rotation in the xarray.DataArray.

    Parameters:
    - st: xarray.DataArray containing temperature data with dimensions 'rotation' and 'plev'.
    - target_temp: The target temperature to find (default is 273).

    Returns:
    - xarray.DataArray with the nearest plev for each rotation.
    """
    # Initialize an empty list to store the results
    results = []

    # Iterate over each unique value in the 'rotation' coordinate
    for dim_val in st[iterate_dim]:
        # Select the temperature data corresponding to the current rotation
        temp_data = st.sel({iterate_dim: dim_val})
        
        # Calculate the index of the plev where the temperature is closest to the target_temp
        closest_index = np.abs(temp_data - target_temp).argmin(dim='plev')
        
        # Retrieve the plev value at the closest index
        closest_plev = temp_data['plev'][closest_index]
        
        # Append the rotation and corresponding closest plev to the results list
        results.append(closest_plev.item())

    # Convert the results list to an xarray.DataArray for structured output
    results_da = xr.DataArray(results, dims=iterate_dim, coords={iterate_dim: st.coords[iterate_dim]})
    
    # Return the DataArray containing the nearest plev for each rotation
    return results_da


def mean_mastrfu_max(mastrfu: xr.DataArray, iterate_dim: str = 'rotation') -> xr.DataArray:
    """
    Calculate the mean profile of the mastrfu DataArray.

    Parameters:
    mastrfu (xr.DataArray): Input DataArray with dimensions ['plev', 'rotation'].

    Returns:
    list: maximum of mastrfu values for each rotation.
    """
    
    # Initialize an empty list to store the mean profiles
    max_mastrfu = []

    for i, dim_val in enumerate(mastrfu[iterate_dim]):

        # Select the data for the current rotation
        max_val = mastrfu.sel({iterate_dim: dim_val}).max()
            
        max_mastrfu.append(max_val)
            
    # Convert the list of mean profiles into an xarray DataArray
    max_mastrfu_da = xr.concat(max_mastrfu, dim=iterate_dim)
    max_mastrfu_da[iterate_dim] = mastrfu[iterate_dim]

    # Return the DataArray of mean profiles
    return max_mastrfu_da



def within_hadley_cell(data, lat):

    data_hadley = data.sel(lat=slice(lat, -1*lat))

    return data_hadley

