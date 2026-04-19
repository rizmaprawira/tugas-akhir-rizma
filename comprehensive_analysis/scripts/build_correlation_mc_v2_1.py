#!/usr/bin/env python3
"""
Build derived DJF correlation datasets on the current MC analysis domain.

The script computes the DJF seasonal means once on the same region used by
correlation_mc.ipynb, then computes correlations, p-values, and significance
masks against the Niño3.4 index. The resulting NetCDF files can be reused
when plotting extents change, so the expensive statistics do not need to be
recomputed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from scipy.stats import norm, t as student_t
from data_processing.config import RAINFALL_PATH
from data_processing.config import WIND_PATH
from data_processing.config import NINO34_PATH
from data_processing.config import MFC_PATH
from data_processing.config import SVP_PATH


FULL_YEARS = np.arange(1981, 2021)
PAST_YEARS = np.arange(1981, 2007)
RECENT_YEARS = np.arange(2007, 2021)
DJF_MONTHS = (12, 1, 2)
ANALYSIS_LAT_SLICE = slice(22, -32)
ANALYSIS_LON_SLICE = slice(88, 157)

RAINFALL_PATH = Path(
    rainfall_path = RAINFALL_PATH
)
WIND_PATH = Path(wind_path = WIND_PATH)
MFC_PATH = Path(mfc_path = MFC_PATH)
SVP_PATH = Path(
    svp_path = SVP_PATH
)
NINO34_PATH = Path(
    nino34_path = NINO34_PATH
)
NINO34_COLUMN = (
    "   Nino Anom 3.4 Index  using ersstv5 from CPC  missing value -99.99 "
    "https://psl.noaa.gov/data/timeseries/month/"
)

DEFAULT_MEANS_NAME = "correlation_mc_djf_means.nc"
DEFAULT_STATS_NAME = "correlation_mc_djf_stats.nc"


def find_coord_name(obj, candidates, kind):
    """Return the first matching coordinate or dimension name."""
    for name in candidates:
        if name in obj.coords or name in obj.dims:
            return name
    raise ValueError(
        f"Could not find a {kind} coordinate/dimension. Tried: {', '.join(candidates)}"
    )


def select_850_hpa_if_present(obj):
    """Select the 850 hPa level if the object carries a pressure dimension."""
    for level_name in ("pressure_level", "level", "lev"):
        if level_name not in obj.coords and level_name not in obj.dims:
            continue

        level_values = np.atleast_1d(np.asarray(obj[level_name].values))
        for target in (850, 850.0, 85000, 85000.0):
            if target in level_values:
                return obj.sel({level_name: target})

        raise ValueError(
            f"850 hPa level not found in '{level_name}'. Available values include: "
            f"{level_values[:10]}"
        )

    return obj


def standardize_obj(obj, chunk=False):
    """Standardize time/lat/lon names and longitude convention."""
    rename_map = {}
    if "valid_time" in obj.coords or "valid_time" in obj.dims:
        rename_map["valid_time"] = "time"
    if "latitude" in obj.coords or "latitude" in obj.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in obj.coords or "longitude" in obj.dims:
        rename_map["longitude"] = "lon"
    if rename_map:
        obj = obj.rename(rename_map)

    if "lon" in obj.coords:
        obj = obj.assign_coords(lon=(obj.lon % 360)).sortby("lon")

    if "lat" in obj.coords:
        lat_values = np.asarray(obj["lat"].values)
        if lat_values.size > 1 and lat_values[0] < lat_values[-1]:
            obj = obj.sortby("lat", ascending=False)

    if chunk:
        chunk_map = {}
        if "time" in obj.dims:
            chunk_map["time"] = min(12, int(obj.sizes["time"]))
        if "lat" in obj.dims:
            chunk_map["lat"] = min(45, int(obj.sizes["lat"]))
        if "lon" in obj.dims:
            chunk_map["lon"] = min(90, int(obj.sizes["lon"]))
        if chunk_map:
            obj = obj.chunk(chunk_map)

    return obj


def match_variable_name(ds, target):
    """Find a data variable name even if underscores differ slightly."""
    target_key = target.replace("_", "").lower()
    matches = [
        name
        for name in ds.data_vars
        if name.lower() == target.lower()
        or name.lower().replace("_", "") == target_key
    ]
    if not matches:
        matches = [
            name
            for name in ds.data_vars
            if target_key in name.lower().replace("_", "")
        ]
    if not matches:
        raise KeyError(f"Variable for {target} not found in the dataset")
    return matches[0]


def build_djf_seasonal_means(da, start, end):
    """Select DJF months, assign DJF year, and compute seasonal means."""
    da = da.sel(time=slice(start, end))
    da = crop_analysis_domain(da)
    month_mask = da.time.dt.month.isin(DJF_MONTHS)
    djf_year = xr.where(da.time.dt.month == 12, da.time.dt.year + 1, da.time.dt.year)
    da = da.sel(time=month_mask).assign_coords(
        djf_year=("time", djf_year.sel(time=month_mask).data)
    )
    da = da.sel(time=da.djf_year.isin(FULL_YEARS))

    clim = da
    past = clim.sel(time=clim.djf_year.isin(PAST_YEARS))
    recent = clim.sel(time=clim.djf_year.isin(RECENT_YEARS))

    return (
        clim.groupby("djf_year").mean("time"),
        past.groupby("djf_year").mean("time"),
        recent.groupby("djf_year").mean("time"),
    )


def build_nino34_djf_series(df):
    """Compute DJF Niño3.4 means as xarray DataArrays."""
    df = df.set_index("Date").loc["1980-12-01":"2020-02-29"]
    df = df[df.index.month.isin(DJF_MONTHS)].copy()
    df["djf_year"] = df.index.year + (df.index.month == 12).astype("int8")
    df = df[df["djf_year"].isin(FULL_YEARS)].copy()

    clim_series = df.groupby("djf_year")[NINO34_COLUMN].mean()
    past_series = df[df["djf_year"].isin(PAST_YEARS)].groupby("djf_year")[
        NINO34_COLUMN
    ].mean()
    recent_series = df[df["djf_year"].isin(RECENT_YEARS)].groupby("djf_year")[
        NINO34_COLUMN
    ].mean()

    clim = xr.DataArray(
        clim_series.to_numpy(),
        coords={"djf_year": clim_series.index.to_numpy()},
        dims="djf_year",
        name="nino34",
    )
    past = xr.DataArray(
        past_series.to_numpy(),
        coords={"djf_year": past_series.index.to_numpy()},
        dims="djf_year",
        name="nino34",
    )
    recent = xr.DataArray(
        recent_series.to_numpy(),
        coords={"djf_year": recent_series.index.to_numpy()},
        dims="djf_year",
        name="nino34",
    )

    return clim, past, recent


def safe_corr(corr):
    """Clip correlation values to keep Fisher/t-stat conversions finite."""
    return xr.where(np.abs(corr) >= 0.999999, np.sign(corr) * 0.999999, corr)


def gaussian_smooth_latlon(da, sigma=0.7):
    """Apply Gaussian spatial smoothing across lat/lon independently for each time slice."""
    if not {"time", "lat", "lon"}.issubset(da.dims):
        raise ValueError("Expected dimensions including time, lat, and lon")

    def _smooth_2d(arr):
        return gaussian_filter(arr, sigma=sigma)

    smoothed = xr.apply_ufunc(
        _smooth_2d,
        da,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        vectorize=True,
        dask="allowed",
        output_dtypes=[da.dtype],
    )
    smoothed = smoothed.transpose(*da.dims)
    smoothed = smoothed.assign_attrs(da.attrs)
    if da.name is not None:
        smoothed = smoothed.rename(da.name)
    return smoothed


def crop_analysis_domain(obj):
    """Crop a field or dataset to the buffered regional analysis domain."""
    if "lat" in obj.coords or "lat" in obj.dims:
        obj = obj.sel(lat=ANALYSIS_LAT_SLICE)
    if "lon" in obj.coords or "lon" in obj.dims:
        obj = obj.sel(lon=ANALYSIS_LON_SLICE)
    return obj


def corr_and_pvalues(field_clim, field_past, field_recent, nino_clim, nino_past, nino_recent):
    """Compute correlations, p-values, and the recent-minus-past difference test."""
    field_clim, nino_clim = xr.align(field_clim, nino_clim, join="inner")
    field_past, nino_past = xr.align(field_past, nino_past, join="inner")
    field_recent, nino_recent = xr.align(field_recent, nino_recent, join="inner")

    corr_clim = xr.corr(field_clim, nino_clim, dim="djf_year").compute()
    corr_past = xr.corr(field_past, nino_past, dim="djf_year").compute()
    corr_recent = xr.corr(field_recent, nino_recent, dim="djf_year").compute()
    corr_recent_minus_past = corr_recent - corr_past

    n_full = int(field_clim.sizes["djf_year"])
    n_past = int(field_past.sizes["djf_year"])
    n_recent = int(field_recent.sizes["djf_year"])

    corr_clim_safe = safe_corr(corr_clim)
    corr_past_safe = safe_corr(corr_past)
    corr_recent_safe = safe_corr(corr_recent)

    corr_clim_t = corr_clim_safe * np.sqrt((n_full - 2) / (1 - corr_clim_safe**2))
    corr_past_t = corr_past_safe * np.sqrt((n_past - 2) / (1 - corr_past_safe**2))
    corr_recent_t = corr_recent_safe * np.sqrt((n_recent - 2) / (1 - corr_recent_safe**2))

    corr_clim_p = xr.apply_ufunc(
        lambda x: 2 * student_t.sf(np.abs(x), df=n_full - 2),
        corr_clim_t,
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    ).compute()
    corr_past_p = xr.apply_ufunc(
        lambda x: 2 * student_t.sf(np.abs(x), df=n_past - 2),
        corr_past_t,
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    ).compute()
    corr_recent_p = xr.apply_ufunc(
        lambda x: 2 * student_t.sf(np.abs(x), df=n_recent - 2),
        corr_recent_t,
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    ).compute()

    z_stat = (np.arctanh(corr_recent_safe) - np.arctanh(corr_past_safe)) / np.sqrt(
        1 / (n_recent - 3) + 1 / (n_past - 3)
    )
    corr_recent_minus_past_p = xr.apply_ufunc(
        lambda x: 2 * norm.sf(np.abs(x)),
        z_stat,
        vectorize=True,
        dask="allowed",
        output_dtypes=[float],
    ).compute()

    return {
        "corr_clim": corr_clim,
        "corr_past": corr_past,
        "corr_recent": corr_recent,
        "corr_recent_minus_past": corr_recent_minus_past,
        "p_clim": corr_clim_p,
        "p_past": corr_past_p,
        "p_recent": corr_recent_p,
        "p_recent_minus_past": corr_recent_minus_past_p,
    }


def scalar_family_datasets(mean_prefix, stats_prefix, field_clim, field_past, field_recent, nino_clim, nino_past, nino_recent):
    """Build the mean and correlation datasets for a scalar field."""
    stats = corr_and_pvalues(field_clim, field_past, field_recent, nino_clim, nino_past, nino_recent)
    means = xr.Dataset(
        {
            f"{mean_prefix}_clim_djf_corr": field_clim,
            f"{mean_prefix}_past_djf_corr": field_past,
            f"{mean_prefix}_recent_djf_corr": field_recent,
        }
    )
    stats_ds = xr.Dataset(
        {
            f"{stats_prefix}_corr_clim": stats["corr_clim"],
            f"{stats_prefix}_corr_past": stats["corr_past"],
            f"{stats_prefix}_corr_recent": stats["corr_recent"],
            f"{stats_prefix}_corr_recent_minus_past": stats["corr_recent_minus_past"],
            f"{stats_prefix}_corr_clim_p": stats["p_clim"],
            f"{stats_prefix}_corr_past_p": stats["p_past"],
            f"{stats_prefix}_corr_recent_p": stats["p_recent"],
            f"{stats_prefix}_corr_recent_minus_past_p": stats["p_recent_minus_past"],
            f"{stats_prefix}_corr_clim_sig": (stats["p_clim"] < 0.05).astype("int8"),
            f"{stats_prefix}_corr_past_sig": (stats["p_past"] < 0.05).astype("int8"),
            f"{stats_prefix}_corr_recent_sig": (stats["p_recent"] < 0.05).astype("int8"),
            f"{stats_prefix}_corr_recent_minus_past_sig": (
                stats["p_recent_minus_past"] < 0.05
            ).astype("int8"),
        }
    )
    return means, stats_ds


def vector_family_datasets(
    family_prefix,
    mean_u_prefix,
    stats_u_prefix,
    u_clim,
    u_past,
    u_recent,
    mean_v_prefix,
    stats_v_prefix,
    v_clim,
    v_past,
    v_recent,
    nino_clim,
    nino_past,
    nino_recent,
):
    """Build the mean and correlation datasets for a vector field."""
    u_stats = corr_and_pvalues(u_clim, u_past, u_recent, nino_clim, nino_past, nino_recent)
    v_stats = corr_and_pvalues(v_clim, v_past, v_recent, nino_clim, nino_past, nino_recent)

    means = xr.Dataset(
        {
            f"{mean_u_prefix}_clim_djf_corr": u_clim,
            f"{mean_u_prefix}_past_djf_corr": u_past,
            f"{mean_u_prefix}_recent_djf_corr": u_recent,
            f"{mean_v_prefix}_clim_djf_corr": v_clim,
            f"{mean_v_prefix}_past_djf_corr": v_past,
            f"{mean_v_prefix}_recent_djf_corr": v_recent,
        }
    )

    stats_ds = xr.Dataset(
        {
            f"{stats_u_prefix}_corr_clim": u_stats["corr_clim"],
            f"{stats_u_prefix}_corr_past": u_stats["corr_past"],
            f"{stats_u_prefix}_corr_recent": u_stats["corr_recent"],
            f"{stats_u_prefix}_corr_recent_minus_past": u_stats["corr_recent_minus_past"],
            f"{stats_u_prefix}_corr_clim_p": u_stats["p_clim"],
            f"{stats_u_prefix}_corr_past_p": u_stats["p_past"],
            f"{stats_u_prefix}_corr_recent_p": u_stats["p_recent"],
            f"{stats_u_prefix}_corr_recent_minus_past_p": u_stats["p_recent_minus_past"],
            f"{stats_u_prefix}_corr_clim_sig": (u_stats["p_clim"] < 0.05).astype("int8"),
            f"{stats_u_prefix}_corr_past_sig": (u_stats["p_past"] < 0.05).astype("int8"),
            f"{stats_u_prefix}_corr_recent_sig": (u_stats["p_recent"] < 0.05).astype("int8"),
            f"{stats_u_prefix}_corr_recent_minus_past_sig": (
                u_stats["p_recent_minus_past"] < 0.05
            ).astype("int8"),
            f"{stats_v_prefix}_corr_clim": v_stats["corr_clim"],
            f"{stats_v_prefix}_corr_past": v_stats["corr_past"],
            f"{stats_v_prefix}_corr_recent": v_stats["corr_recent"],
            f"{stats_v_prefix}_corr_recent_minus_past": v_stats["corr_recent_minus_past"],
            f"{stats_v_prefix}_corr_clim_p": v_stats["p_clim"],
            f"{stats_v_prefix}_corr_past_p": v_stats["p_past"],
            f"{stats_v_prefix}_corr_recent_p": v_stats["p_recent"],
            f"{stats_v_prefix}_corr_recent_minus_past_p": v_stats["p_recent_minus_past"],
            f"{stats_v_prefix}_corr_clim_sig": (v_stats["p_clim"] < 0.05).astype("int8"),
            f"{stats_v_prefix}_corr_past_sig": (v_stats["p_past"] < 0.05).astype("int8"),
            f"{stats_v_prefix}_corr_recent_sig": (v_stats["p_recent"] < 0.05).astype("int8"),
            f"{stats_v_prefix}_corr_recent_minus_past_sig": (
                v_stats["p_recent_minus_past"] < 0.05
            ).astype("int8"),
            f"{family_prefix}_vector_sig_clim": (
                (u_stats["p_clim"] < 0.05) | (v_stats["p_clim"] < 0.05)
            ).astype("int8"),
            f"{family_prefix}_vector_sig_past": (
                (u_stats["p_past"] < 0.05) | (v_stats["p_past"] < 0.05)
            ).astype("int8"),
            f"{family_prefix}_vector_sig_recent": (
                (u_stats["p_recent"] < 0.05) | (v_stats["p_recent"] < 0.05)
            ).astype("int8"),
            f"{family_prefix}_vector_sig_recent_minus_past": (
                (u_stats["p_recent_minus_past"] < 0.05)
                | (v_stats["p_recent_minus_past"] < 0.05)
            ).astype("int8"),
        }
    )

    return means, stats_ds


def cast_float_and_mask_vars(ds):
    """Cast float vars to float32 and masks to int8 before writing."""
    ds = ds.copy()
    for name in list(ds.data_vars):
        dtype = ds[name].dtype
        if np.issubdtype(dtype, np.floating):
            ds[name] = ds[name].astype("float32")
        elif np.issubdtype(dtype, np.bool_):
            ds[name] = ds[name].astype("int8")
    return ds


def build_encoding(ds):
    """Compression settings for NetCDF output."""
    return {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}


def write_dataset(ds, path, label):
    """Write a dataset with light compression and report the target path."""
    ds = cast_float_and_mask_vars(ds)
    print(f"Writing {label}: {path}")
    ds.to_netcdf(path, encoding=build_encoding(ds))


def main():
    parser = argparse.ArgumentParser(
        description="Build derived DJF seasonal-mean and correlation datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for the derived NetCDF outputs.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    means_output = output_dir / DEFAULT_MEANS_NAME
    stats_output = output_dir / DEFAULT_STATS_NAME

    t0 = perf_counter()
    print("[1/5] Opening source datasets...")

    ds_rain = standardize_obj(xr.open_dataset(RAINFALL_PATH)[["precipitation"]])
    ds_rain = crop_analysis_domain(ds_rain).load()
    ds_wind = select_850_hpa_if_present(xr.open_dataset(WIND_PATH)[["u", "v"]])
    ds_wind = standardize_obj(ds_wind)
    ds_wind = crop_analysis_domain(ds_wind).load()
    ds_mfc = standardize_obj(xr.open_dataset(MFC_PATH)[["viwve", "viwvn"]])
    ds_mfc = crop_analysis_domain(ds_mfc).load()
    ds_svp = standardize_obj(xr.open_dataset(SVP_PATH))
    ds_svp = crop_analysis_domain(ds_svp).load()

    svp_var_map = {}
    for target in ["psi", "chi", "u_psi", "v_psi", "u_chi", "v_chi"]:
        svp_var_map[target] = match_variable_name(ds_svp, target)

    ds_nino34 = pd.read_csv(
        NINO34_PATH,
        usecols=["Date", NINO34_COLUMN],
        parse_dates=["Date"],
    )
    print(f"  opened in {perf_counter() - t0:.1f} s")

    print("[2/5] Building DJF seasonal means on the current MC domain...")

    rain_clim, rain_past, rain_recent = build_djf_seasonal_means(
        ds_rain["precipitation"],
        "1980-12-01",
        "2020-02-29",
    )

    u_clim, u_past, u_recent = build_djf_seasonal_means(
        ds_wind["u"],
        "1980-12-01",
        "2020-02-29",
    )
    v_clim, v_past, v_recent = build_djf_seasonal_means(
        ds_wind["v"],
        "1980-12-01",
        "2020-02-29",
    )

    psi_clim, psi_past, psi_recent = build_djf_seasonal_means(
        ds_svp[svp_var_map["psi"]],
        "1980-12-01",
        "2020-02-29",
    )
    chi_clim, chi_past, chi_recent = build_djf_seasonal_means(
        ds_svp[svp_var_map["chi"]],
        "1980-12-01",
        "2020-02-29",
    )
    u_psi_clim, u_psi_past, u_psi_recent = build_djf_seasonal_means(
        ds_svp[svp_var_map["u_psi"]],
        "1980-12-01",
        "2020-02-29",
    )
    v_psi_clim, v_psi_past, v_psi_recent = build_djf_seasonal_means(
        ds_svp[svp_var_map["v_psi"]],
        "1980-12-01",
        "2020-02-29",
    )
    u_chi_clim, u_chi_past, u_chi_recent = build_djf_seasonal_means(
        ds_svp[svp_var_map["u_chi"]],
        "1980-12-01",
        "2020-02-29",
    )
    v_chi_clim, v_chi_past, v_chi_recent = build_djf_seasonal_means(
        ds_svp[svp_var_map["v_chi"]],
        "1980-12-01",
        "2020-02-29",
    )

    mfc_qx_monthly = ds_mfc["viwve"].sel(time=slice("1980-12-01", "2020-02-29"))
    mfc_qy_monthly = ds_mfc["viwvn"].sel(time=slice("1980-12-01", "2020-02-29"))
    mfc_qx_monthly = crop_analysis_domain(mfc_qx_monthly)
    mfc_qy_monthly = crop_analysis_domain(mfc_qy_monthly)
    print("  computing MFC scalar field...")
    a = 6.371e6
    phi = np.deg2rad(mfc_qx_monthly["lat"])
    dqx_dlam = mfc_qx_monthly.differentiate("lon") * (180.0 / np.pi)
    dqycos_dphi = (mfc_qy_monthly * np.cos(phi)).differentiate("lat") * (180.0 / np.pi)
    div_q = (1.0 / (a * np.cos(phi))) * dqx_dlam + (1.0 / (a * np.cos(phi))) * dqycos_dphi
    mfc_monthly = (-div_q).rename("mfc")
    # Gaussian spatial smoothing of the MFC shading field, following Nuran's method.
    print("  applying Gaussian smoothing to MFC scalar field...")
    mfc_monthly = gaussian_smooth_latlon(mfc_monthly, sigma=0.7)
    mfc_monthly.attrs["units"] = "kg m^-2 s^-1"
    mfc_monthly.attrs["long_name"] = "moisture flux convergence"
    mfc_monthly.attrs["smoothing"] = "Gaussian spatial filter, sigma=0.7"

    qx_clim, qx_past, qx_recent = build_djf_seasonal_means(
        mfc_qx_monthly,
        "1980-12-01",
        "2020-02-29",
    )
    qy_clim, qy_past, qy_recent = build_djf_seasonal_means(
        mfc_qy_monthly,
        "1980-12-01",
        "2020-02-29",
    )
    mfc_clim, mfc_past, mfc_recent = build_djf_seasonal_means(
        mfc_monthly,
        "1980-12-01",
        "2020-02-29",
    )

    nino34_clim, nino34_past, nino34_recent = build_nino34_djf_series(ds_nino34)
    print(f"  seasonal means built in {perf_counter() - t0:.1f} s")

    print("[3/5] Computing correlations and significance...")

    rain_means, rain_stats = scalar_family_datasets(
        "rain",
        "rain",
        rain_clim,
        rain_past,
        rain_recent,
        nino34_clim,
        nino34_past,
        nino34_recent,
    )

    wind_means, wind_stats = vector_family_datasets(
        "wind",
        "u",
        "wind_u",
        u_clim,
        u_past,
        u_recent,
        "v",
        "wind_v",
        v_clim,
        v_past,
        v_recent,
        nino34_clim,
        nino34_past,
        nino34_recent,
    )

    mfc_scalar_means, mfc_scalar_stats = scalar_family_datasets(
        "mfc",
        "mfc",
        mfc_clim,
        mfc_past,
        mfc_recent,
        nino34_clim,
        nino34_past,
        nino34_recent,
    )
    mfc_vector_means, mfc_vector_stats = vector_family_datasets(
        "mfc",
        "qx",
        "mfc_qx",
        qx_clim,
        qx_past,
        qx_recent,
        "qy",
        "mfc_qy",
        qy_clim,
        qy_past,
        qy_recent,
        nino34_clim,
        nino34_past,
        nino34_recent,
    )

    psi_scalar_means, psi_scalar_stats = scalar_family_datasets(
        "psi",
        "psi",
        psi_clim,
        psi_past,
        psi_recent,
        nino34_clim,
        nino34_past,
        nino34_recent,
    )
    psi_vector_means, psi_vector_stats = vector_family_datasets(
        "psi",
        "u_psi",
        "u_psi",
        u_psi_clim,
        u_psi_past,
        u_psi_recent,
        "v_psi",
        "v_psi",
        v_psi_clim,
        v_psi_past,
        v_psi_recent,
        nino34_clim,
        nino34_past,
        nino34_recent,
    )

    chi_scalar_means, chi_scalar_stats = scalar_family_datasets(
        "chi",
        "chi",
        chi_clim,
        chi_past,
        chi_recent,
        nino34_clim,
        nino34_past,
        nino34_recent,
    )
    chi_vector_means, chi_vector_stats = vector_family_datasets(
        "chi",
        "u_chi",
        "u_chi",
        u_chi_clim,
        u_chi_past,
        u_chi_recent,
        "v_chi",
        "v_chi",
        v_chi_clim,
        v_chi_past,
        v_chi_recent,
        nino34_clim,
        nino34_past,
        nino34_recent,
    )
    print(f"  statistics computed in {perf_counter() - t0:.1f} s")

    print("[4/5] Merging derived datasets...")
    nino_means = xr.Dataset(
        {
            "nino34_clim_djf_corr": nino34_clim,
            "nino34_past_djf_corr": nino34_past,
            "nino34_recent_djf_corr": nino34_recent,
        }
    )

    means_ds = xr.merge(
        [
            rain_means,
            wind_means,
            mfc_scalar_means,
            mfc_vector_means,
            psi_scalar_means,
            psi_vector_means,
            chi_scalar_means,
            chi_vector_means,
            nino_means,
        ]
    )
    stats_ds = xr.merge(
        [
            rain_stats,
            wind_stats,
            mfc_scalar_stats,
            mfc_vector_stats,
            psi_scalar_stats,
            psi_vector_stats,
            chi_scalar_stats,
            chi_vector_stats,
        ]
    )
    print(f"  merged in {perf_counter() - t0:.1f} s")

    means_ds.attrs.update(
        {
            "title": "DJF seasonal means on the current MC analysis domain",
            "source_files": ", ".join(
                [
                    str(RAINFALL_PATH),
                    str(WIND_PATH),
                    str(MFC_PATH),
                    str(SVP_PATH),
                    str(NINO34_PATH),
                ]
            ),
            "note": "Derived once so correlation plots can be redrawn without recomputing the seasonal means.",
        }
    )
    stats_ds.attrs.update(
        {
            "title": "DJF correlations and significance on the current MC analysis domain",
            "source_files": ", ".join(
                [
                    str(RAINFALL_PATH),
                    str(WIND_PATH),
                    str(MFC_PATH),
                    str(SVP_PATH),
                    str(NINO34_PATH),
                ]
            ),
            "note": "Correlation, p-value, and significance outputs derived from the DJF seasonal means.",
        }
    )

    print("[5/5] Writing derived datasets...")
    write_dataset(means_ds, means_output, "DJF seasonal means")
    write_dataset(stats_ds, stats_output, "correlation/significance stats")

    print(f"Done in {perf_counter() - t0:.1f} s")
    print(f"Means output: {means_output}")
    print(f"Stats output: {stats_output}")


if __name__ == "__main__":
    main()
