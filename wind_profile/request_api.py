#!/usr/bin/env python

import cdsapi

c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": [
            "100m_u_component_of_wind",
            "10m_u_component_of_wind",
            "2m_temperature",
            "charnock",
            "eastward_turbulent_surface_stress",
            "forecast_surface_roughness",
            "friction_velocity",
            "northward_turbulent_surface_stress",
            "surface_sensible_heat_flux",
        ],
        "year": "2021",
        "month": "09",
        "day": "01",
        "time": "06:00",
        "area": [
            80,
            -60,
            50,
            -40,
        ],
    },
    "download.nc",
)

c.retrieve(
    "reanalysis-era5-single-levels",
    {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": "air_density_over_the_oceans",
        "year": "2021",
        "month": "09",
        "day": "01",
        "time": "06:00",
        "area": [
            80,
            -60,
            50,
            -40,
        ],
    },
    "rho_air.nc",
)
