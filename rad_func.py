import xarray as xr
from typing import List, Union
import numpy as np
from metpy.calc import lcl as calculate_lcl, dewpoint_from_relative_humidity
from metpy.units import units
import pandas as pd

import sys
# define the path where my modules are saved
module_path = 'home/m/m300909/clear sky feedback/'
# add this path tho the system path using the `sys` module
if not module_path in sys.path: sys.path.append(module_path)
# load them as they were regular python packages.
# (the file is called home/m/m300909/py_data_handling.py)
import basic_func as bf
import inversion_func as ivf
import cir_func as cf



def column_relative_humidity(q, T, plev_slice):
    """
    Calculate column relative humidity using specific humidity and saturated specific humidity.
    q: specific humidity
    qs: Saturated specific humidity
    Returns: Column relative humidity till 100hPa
    """

    integrated_water_vapour = q.sel(plev=plev_slice
                                   ).integrate(coord='plev')

    saturated_integrated_water_vapour = (ivf.saturated_specific_humidity(T,T.plev).
                                         sel(plev=plev_slice).
                                         integrate(coord='plev'))

    return integrated_water_vapour / saturated_integrated_water_vapour




def crh_binning(crh, hadley_extent):

    # Define bins for the vectorized lat-lon array
    bins = np.linspace(0, 1, 101)

    # Initialize a list to store the binned data for each rotation
    binned_data_list = []

    # Iterate over the rotation dimension
    for i, rotation_val in enumerate(crh['rotation']):
        # Select the data for the current rotation
        crh_data = crh.sel(rotation=rotation_val)
        hadley = hadley_extent[i]

        if hadley is not None:
            crh_data = cf.within_hadley_cell(crh_data, hadley)
            #print(rotation_data.lat[0])
        # Stack the latitude and longitude into a single dimension
        crh_data_stacked = crh_data.stack(lat_lon=('lat', 'lon', 'time'))

        # Convert the stacked data to a pandas DataFrame
        crh_df = pd.DataFrame({"crh": crh_data_stacked.values})

        # Bin the data using pandas and calculate the mean for each bin
        crh_df['bins'] = pd.cut(crh_df['crh'], bins=bins, labels=bins[:-1], right=False)
        binned_data = crh_df.groupby('bins')['crh'].mean()

        # Convert the binned data back to an xarray DataArray
        binned_data_xr = xr.DataArray(binned_data, dims=['bins'], coords={'bins': bins[:-1]})

        # Append the binned data to the list
        binned_data_list.append(binned_data_xr)

    # Combine the binned data back into a single xarray DataArray
    binned_data_combined = xr.concat(binned_data_list, dim='rotation')

    return binned_data_combined