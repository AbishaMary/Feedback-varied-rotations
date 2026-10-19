Radiative feedbacks and circulation at different planetary rotation rates

Analysis code from my PhD at the Max Planck Institute for Meteorology (MPI-M). The project asks how the rotation rate of a planet changes its atmospheric circulation, clouds and radiative feedbacks under surface warming.

Experiments
Model: ECHAM (MPI-ESM atmosphere) in an aquaplanet configuration
Rotation rates: 1/32, 1/16, 1/8, 1/4, 1/2, 1 (Earth-like), 2, 4 and 8 times Earth's rotation
Warming: fixed sea surface temperature (SST) runs: a control run and uniform SST increases of +1 K, +2 K, +3 K and +4 K
Simulations were run on the DKRZ supercomputer (Linux, shell scripting, Fortran, Python)
What the analysis covers

Radiative response (radiative_response.ipynb)

Clear-sky and all-sky radiative feedback parameters (W m⁻² K⁻¹)
Shortwave and longwave cloud feedbacks
Changes in cloud fraction, cloud water and precipitable water
Links between feedbacks and changes in low cloud and cloud water

Circulation response (work_area/circulation_response.ipynb)

Meridional mass stream function and Hadley cell extent
Shallow (congestus) and deep circulation strength
Freezing level, for slow and fast rotation rates

Other notebooks (work_area/)

atmospheric_variables.ipynb: relative humidity, temperature, cloud fraction and mass stream function
congestus_circulation.ipynb: radiative cooling and the congestus (mid-level) circulation
Code structure
my_functions/
├── basic_func.py      # area-weighted means, standard errors, potential temperature,
│                      # static stability, subsidence mask, feedback parameter
├── cir_func.py        # Hadley cell extent, mass stream function profiles,
│                      # shallow vs deep circulation, freezing level
├── inversion_func.py  # lower-tropospheric stability (LTS), estimated inversion
│                      # strength (EIS, ECTEI), moist adiabat, saturation humidity
└── rad_func.py        # radiative helper functions

The feedback parameter is computed as the change in top-of-atmosphere net radiation divided by the change in global-mean surface temperature between two simulations. Area-weighted global means account for grid-cell area, and standard errors are used to check the robustness of the results.

Requirements

Python 3.10+ with numpy, xarray, pandas, matplotlib and cmocean.

Data

The model output (NetCDF) is stored on the DKRZ HPC system and is not included in this repository because of its size. File paths in the notebooks point to that system and need to be adjusted to run the code elsewhere.
