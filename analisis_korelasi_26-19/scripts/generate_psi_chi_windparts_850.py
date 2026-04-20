#!/usr/bin/env python3
"""
Generate streamfunction, velocity potential, and wind component fields
from ERA5 monthly mean zonal and meridional wind at 850 hPa.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr
from windspharm.xarray import VectorWind


# Edit these paths as needed.
INPUT_FILE = Path("/Users/rizzie/ClimateData/era5-monthly/u_v_global850.nc")
OUTPUT_FILE = Path("psi_chi_windparts_850.nc")

# Expected variable names in the input file.
U_NAME = "u"
V_NAME = "v"


def find_coord_name(ds, candidates, kind):
    """Return the first matching coordinate or dimension name."""
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(
        f"Could not find a {kind} coordinate/dimension. Tried: {', '.join(candidates)}"
    )


def select_pressure_level(da, level_name):
    """Select 850 hPa, accepting either 850 or 85000 if present."""
    level_values = np.asarray(da[level_name].values)

    if 850 in level_values:
        return da.sel({level_name: 850})
    if 850.0 in level_values:
        return da.sel({level_name: 850.0})
    if 85000 in level_values:
        return da.sel({level_name: 85000})
    if 85000.0 in level_values:
        return da.sel({level_name: 85000.0})

    raise ValueError(
        f"850 hPa level not found in '{level_name}'. Available values include: "
        f"{level_values[:10]}"
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    t0 = perf_counter()
    print(f"Opening {INPUT_FILE}...")
    with xr.open_dataset(INPUT_FILE) as ds:
        print(f"Opened dataset in {perf_counter() - t0:.1f} s")

        if U_NAME not in ds.variables:
            raise KeyError(f"Missing variable '{U_NAME}' in {INPUT_FILE}")
        if V_NAME not in ds.variables:
            raise KeyError(f"Missing variable '{V_NAME}' in {INPUT_FILE}")

        u = ds[U_NAME]
        v = ds[V_NAME]

        level_name = find_coord_name(ds, ("level", "lev", "pressure_level"), "pressure level")
        lat_name = find_coord_name(ds, ("lat", "latitude"), "latitude")
        lon_name = find_coord_name(ds, ("lon", "longitude"), "longitude")
        time_name = find_coord_name(ds, ("time", "valid_time"), "time")

        if time_name not in u.dims or time_name not in v.dims:
            raise ValueError(
                f"Expected a '{time_name}' dimension on both wind variables."
            )

        # Select 850 hPa and standardize coordinate names for windspharm.
        u = select_pressure_level(u, level_name)
        v = select_pressure_level(v, level_name)

        rename_map = {}
        if lat_name != "lat":
            rename_map[lat_name] = "lat"
        if lon_name != "lon":
            rename_map[lon_name] = "lon"
        if rename_map:
            u = u.rename(rename_map)
            v = v.rename(rename_map)

        if time_name != "time":
            u = u.rename({time_name: "time"})
            v = v.rename({time_name: "time"})

        # Ensure latitude is ordered north-to-south, which windspharm requires.
        lat_values = np.asarray(u["lat"].values)
        if lat_values[0] < lat_values[-1]:
            u = u.sortby("lat", ascending=False)
            v = v.sortby("lat", ascending=False)

        print(f"Loading wind fields into memory...")
        t_load = perf_counter()
        u = u.load()
        v = v.load()
        print(f"Loaded wind fields in {perf_counter() - t_load:.1f} s")

        # Compute the Helmholtz decomposition with windspharm.
        print("Computing streamfunction, velocity potential, and wind components...")
        t_wind = perf_counter()
        wind = VectorWind(u, v)
        psi = wind.streamfunction()
        chi = wind.velocitypotential()
        u_psi, v_psi = wind.nondivergentcomponent()
        u_chi, v_chi = wind.irrotationalcomponent()
        print(f"Computed decomposition in {perf_counter() - t_wind:.1f} s")

        out = xr.Dataset(
            data_vars={
                "psi": psi,
                "chi": chi,
                "u_psi": u_psi,
                "v_psi": v_psi,
                "u_chi": u_chi,
                "v_chi": v_chi,
            }
        ).assign_coords({level_name: u.coords[level_name]})

        # Basic metadata for the saved fields.
        out["psi"].attrs.update(
            {
                "long_name": "streamfunction of monthly mean wind",
                "units": "m2 s-1",
            }
        )
        out["chi"].attrs.update(
            {
                "long_name": "velocity potential of monthly mean wind",
                "units": "m2 s-1",
            }
        )
        out["u_psi"].attrs.update(
            {
                "long_name": "zonal non-divergent wind component",
                "units": "m s-1",
            }
        )
        out["v_psi"].attrs.update(
            {
                "long_name": "meridional non-divergent wind component",
                "units": "m s-1",
            }
        )
        out["u_chi"].attrs.update(
            {
                "long_name": "zonal irrotational wind component",
                "units": "m s-1",
            }
        )
        out["v_chi"].attrs.update(
            {
                "long_name": "meridional irrotational wind component",
                "units": "m s-1",
            }
        )

        out.attrs.update(
            {
                "title": "ERA5 850 hPa wind decomposition",
                "source_file": str(INPUT_FILE),
                "note": "Fields are computed from the raw monthly mean wind fields.",
            }
        )

        # Write the resulting dataset to disk.
        print(f"Writing {OUTPUT_FILE}...")
        t_write = perf_counter()
        out.to_netcdf(OUTPUT_FILE)
        print(f"Wrote NetCDF in {perf_counter() - t_write:.1f} s")

    print(f"Total elapsed time: {perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
