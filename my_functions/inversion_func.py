import xarray as xr
from typing import List, Union
import numpy as np
from metpy.calc import lcl as calculate_lcl, dewpoint_from_relative_humidity
from metpy.units import units

from . import basic_func as bf


def lts(temp, tpot= None, plev1=80000, plev2=100000):
    """
    Calculate the Lower Tropospheric Stability (LTS) from temperature data.

    Parameters:
    - tpot: True if temp contains the potential temperature data.
    - temp: xarray DataArray containing the temperature data.
    - plev1: Pressure level for the upper layer (default is 80000 Pa).
    - plev2: Pressure level for the lower layer (default is 100000 Pa).

    Returns:
    - A DataArray representing the LTS.
    """
    # Calculate potential temperature at specified pressure levels
    if tpot is not None:
        tpot_plev1 = tpot.sel(plev=plev1, method='nearest')
        tpot_plev2 = tpot.sel(plev=plev2, method='nearest')
    else:
        tpot_plev1 = bf.potential_temperature(temp, temp.sel(plev=plev1, method='nearest'))
        tpot_plev2 = bf.potential_temperature(temp, temp.sel(plev=plev2, method='nearest'))

    # Calculate LTS
    lts = tpot_plev1 - tpot_plev2

    return lts



def moist_adiabat(T):
    """
    Calculate the moist adiabatic lapse rate for a given temperature.

    Parameters:
    T (float): Temperature in Kelvin.

    Returns:
    float: Moist adiabatic lapse rate in K/m.
    """

    # Constants
    g = 9.81  # m/s²
    cp = 1004  # J/(kg·K)
    Lv = 2.5e6  # J/kg
    Rd = 287  # J/(kg·K)
    Rv = 461.5  # J/(kg·K)

    # Calculate saturated specific humidity
    qs = saturated_specific_humidity(T, T.plev)

    # Calculate the moist adiabatic lapse rate
    lapse_rate = (g / cp) * (
        ((1 + (Lv * qs) / (Rd * T)) /
             (1 + (Lv**2 * qs) / (cp * Rv * T**2)))
    )

    return lapse_rate

def saturated_specific_humidity(T, P):
    """
    Calculate saturated specific humidity.
    T: Temperature in Kelvin
    P: Total atmospheric pressure in Pa
    Returns: Saturated specific humidity
    """
    es = saturation_vapor_pressure(T)
    qs = 0.622 * es / (P - es)
    return qs


def saturation_vapor_pressure(T):

    """
    Calculate saturation vapor pressure using the Classius-Clayperon formula.
    T: Temperature in Kelvin
    Returns: Saturation vapor pressure in hPa
    """

    Lv = 2.5e6  # Latent heat of vaporisation J/Kg
    eO = 611.2  # saturation vapour pressure at the triple point of water Pa
    Rv = 461.5  # gas constant of water vapour J/(Kg K)
    TO = 273.16  # Triple point of water K

    es = eO * np.exp((Lv / Rv) * ((1 / TO) - (1 / T)))
    return es



def pressure_to_height(T, q, ps, lcl_p, plev):
    """
    Compute LCL height using the hypsometric equation with
    vertical integration in log-pressure space.

    Parameters
    ----------
    T : xarray.DataArray
        Temperature (K) with dimension 'plev'
    q : xarray.DataArray
        Specific humidity (kg/kg)
    ps : xarray.DataArray
        Surface pressure (Pa)
    lcl_p : xarray.DataArray
        LCL pressure (Pa)
    plev : xarray.DataArray
        Pressure levels (Pa)

    Returns
    -------
    z : xarray.DataArray
        Height of LCL above surface (m)
    """

    Rd = 287.0
    g = 9.81

    # Virtual temperature
    Tv = T * (1 + 0.61 * q)

    # Broadcast pressure levels to match T dimensions
    p = plev.broadcast_like(T)

    # Create mask for integration bounds
    mask = (p <= ps) & (p >= lcl_p)

    # Compute d(ln p)
    dlnp = np.log(p.shift(plev=-1) / p)

    # Integrand
    integrand = Tv * dlnp

    # Apply mask
    integrand = integrand.where(mask)

    # Integrate vertically
    delta_z = -(Rd / g) * integrand.sum(dim="plev")

    return delta_z





def calculate_virtual_temperature(T, q):
    """
    Calculate the virtual temperature using temperature and specific humidity.

    Parameters:
    - temperature: Air temperature in Kelvin.
    - specific_humidity: Specific humidity (dimensionless, e.g., kg/kg).

    Returns:
    - Virtual temperature in Kelvin.
    """
    # Calculate the mixing ratio from specific humidity
    w = calculate_mixing_ratio(q)
    
    # Calculate virtual temperature
    virtual_temperature = T * (1 + 0.61 * w)
    
    return virtual_temperature


def calculate_mixing_ratio(q):
    """
    Calculate the mixing ratio from specific humidity.

    Parameters:
    - specific_humidity: Specific humidity (dimensionless, e.g., kg/kg).

    Returns:
    - Mixing ratio (dimensionless, e.g., kg/kg).
    """
    return q / (1 - q)




def dew_point(ts, rh):
    """
    Calculate the dew point temperature given temperature and relative humidity.

    Parameters:
    - ts: Temperature in degrees Celsius.
    - rh: Relative humidity in percent.

    Returns:
    - Dew point temperature in degrees Celsius.
    """

    return dewpoint_from_relative_humidity(ts * units.kelvin, rh * units.percent)



def compute_lcl(ts, ps, rh):
    """
    Calculate the Lifted Condensation Level (LCL) given temperature, pressure, and relative humidity.

    Parameters:
    - ts: Temperature in degrees Celsius.
    - ps: Pressure in hPa.
    - rh: Relative humidity in percent.

    Returns:
    - Tuple containing LCL pressure in hPa and LCL temperature in degrees Celsius.
    """
    rh = rh * 100
    dew_T = dew_point(ts, rh) 
    lcl_p, lcl_T = calculate_lcl(ps * units.hPa, ts * units.kelvin, dew_T)
    
    lcl_p = array_to_xarray(lcl_p, dims=("rotation","lat", "lon"), 
                            coords={"rotation":ts.rotation,
                                    "lat":ts.lat,
                                    "lon":ts.lon},
                            name="LCL pressure",
                            units="Pa",
                            long_name="LCL pressure"
                            )
    
    lcl_T = array_to_xarray(lcl_T, dims=("rotation","lat", "lon"), 
                            coords={"rotation":ts.rotation,
                                    "lat":ts.lat,
                                    "lon":ts.lon},
                            name="LCL temperature",
                            units="Kelvin",
                            long_name="LCL temperature"
                            )
    return lcl_p, lcl_T



def array_to_xarray(array, dims, coords=None, name="variable", units=None, long_name=None):
    """
    Convert a NumPy array to an xarray DataArray with specified dimensions and coordinates.

    This function is fully generic and works for arrays of any shape. It allows you to
    provide dimension names, coordinate arrays, and optional metadata such as variable
    name, units, and a descriptive long_name.

    Parameters
    ----------
    array : np.ndarray
        The input NumPy array to convert. Can be 1D, 2D, or 3D (or higher).
    dims : tuple of str
        Names of the dimensions corresponding to the axes of `array`.
        Length must match `array.ndim`.
    coords : dict of str: array_like, optional
        Dictionary mapping dimension names to coordinate arrays. Keys should match names in `dims`.
        Default is None.
    name : str, optional
        Name of the DataArray variable. Default is 'variable'.
    units : str, optional
        Units string for the variable. Default is None.
    long_name : str, optional
        Descriptive long name for the variable. Default is None.

    Returns
    -------
    xr.DataArray
        An xarray DataArray with the specified dimensions, coordinates, and metadata.

    Examples
    --------
    >>> import numpy as np
    >>> lat = np.linspace(-90, 90, 10)
    >>> lon = np.linspace(0, 360, 20, endpoint=False)
    >>> data = np.random.rand(10, 20)
    >>> da = array_to_xarray(data, dims=("lat", "lon"), coords={"lat": lat, "lon": lon},
    ...                       name="temperature", units="K", long_name="Surface Temperature")
    >>> print(da)
    """
    # Ensure dimension names match array shape
    if len(dims) != array.ndim:
        raise ValueError(f"Length of dims {dims} must match array.ndim {array.ndim}")

    # Create DataArray with optional coords and metadata
    da = xr.DataArray(
        array,
        dims=dims,
        coords=coords,
        name=name,
        attrs={k: v for k, v in {"units": units, "long_name": long_name}.items() if v is not None}
    )

    return da


def pressure_to_height_2(T, q, ps, plev):
    """
    Calculate the height (z) from pressure levels using the barometric formula.

    Parameters:
    - T: Temperature in Kelvin.
    - q: Specific humidity (dimensionless, e.g., kg/kg).
    - ps: Surface pressure in Pascals.
    - plev: Pressure level in Pascals.

    Returns:
    - Height (z) in meters.
    """
    # Constants
    Rd = 287.0  # Specific gas constant for dry air, J/(kg·K)
    g = 9.81    # Acceleration due to gravity, m/s²

    T_mean = calculate_virtual_temperature(T, q).mean(dim='plev')
    
    z = (Rd * T_mean / g) * np.log(ps / plev)
    return z


def inversion_plev_2_ht(T, q, theta, p, pmin=700.0, pmax=950.0):
    """
    theta: potential temperature (K)
    p: pressure levels (hPa), surface → upper troposphere
    """
        
    mask = (p <= pmax) & (p >= pmin)
    dtheta_dp = theta.where(mask, drop=True).differentiate('plev')
    p_inv = dtheta_dp.idxmin(dim="plev")   

    z_inv = pressure_to_height(T.where(mask, drop=False),
                                    q.where(mask, drop=False),
                                    pmax, p_inv)
    return z_inv


    
        