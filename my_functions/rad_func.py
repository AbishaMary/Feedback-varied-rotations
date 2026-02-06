import xarray as xr
from typing import List, Union
import numpy as np
#from metpy.calc import lcl as calculate_lcl, dewpoint_from_relative_humidity
#from metpy.units import units
import pandas as pd




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



def zonal_time_mean(data):
    """Compute the zonal time average"""
    return data.mean(dim=['lon','time'])


def pdf(data, iterate_dim: str = 'rotation'):
    import numpy as np
    import xarray as xr
    from scipy.stats import norm

    pdf = []
    for dim_val in data[iterate_dim]: 

        dim_data = data.sel({iterate_dim: dim_val})
        mu, sigma = norm.fit(dim_data)
        crh = np.linspace(0.01, 1, 99)
        pdf.append(norm.pdf(crh, mu, sigma))

    pdf_da = xr.DataArray(pdf, dims=[iterate_dim, 'crh'], 
                          coords={iterate_dim: data.coords[iterate_dim], 'crh': crh})
        
    return pdf_da