import xarray as xr
from typing import List, Union
import numpy as np
from metpy.calc import lcl as calculate_lcl, dewpoint_from_relative_humidity
from metpy.units import units



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
        tpot_plev1 = cf.potential_temperature(temp, temp.sel(plev=plev1, method='nearest'))
        tpot_plev2 = cf.potential_temperature(temp, temp.sel(plev=plev2, method='nearest'))

    # Calculate LTS
    lts = tpot_plev1 - tpot_plev2

    return lts

    

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



def lcl(ts, ps, rh):
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
    return lcl_p, lcl_T


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


def estimated_inversion_strength(
    theta_level,
    theta_surface,
    z_700,
    z_lcl,
    T_ref,
    p_ref=85000.0
):
    """
    Estimated Inversion Strength (EIS) with temperature-dependent
    moist adiabatic lapse rate.

    Parameters
    ----------
    theta_level : float or array-like
        Potential temperature at the specified pressure level (K)
    theta_surface : float or array-like
        Surface potential temperature (K)
    z_700 : float or array-like
        Geopotential height of the 700 hPa level (m)
    z_lcl : float or array-like
        Lifting Condensation Level height (m)
    T_ref : float or array-like
        Air temperature at the reference level used to compute
        the moist adiabatic lapse rate (K), typically at 850 hPa
    p_ref : float, optional
        Pressure of the reference level (Pa). Default is 85000 Pa

    Returns
    -------
    eis : float or array-like
        Estimated Inversion Strength (K)
    """

    # Lower-tropospheric stability
    lts = theta_level - theta_surface

    # Moist adiabatic lapse rate evaluated at reference level
    gamma_m = moist_adiabat(T_ref)

    # Estimated inversion strength
    eis = lts - gamma_m * (z_700 - z_lcl)

    return eis


def pressure_to_height(ts, ps, plev):
    """
    Calculate the height (z) from pressure levels using the barometric formula.

    Parameters:
    - ts: Surface temperature in Kelvin.
    - ps: Surface pressure in Pascals.
    - plev: Pressure level in Pascals.

    Returns:
    - Height (z) in meters.
    """
    # Constants
    Rd = 287.0  # Specific gas constant for dry air, J/(kg·K)
    g = 9.81    # Acceleration due to gravity, m/s²

    # Calculate height
    z = (Rd * ts / g) * np.log(ps / plev)
    return z


def calculate_mixing_ratio(q):
    """
    Calculate the mixing ratio from specific humidity.

    Parameters:
    - specific_humidity: Specific humidity (dimensionless, e.g., kg/kg).

    Returns:
    - Mixing ratio (dimensionless, e.g., kg/kg).
    """
    return q / (1 - q)



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





    
        