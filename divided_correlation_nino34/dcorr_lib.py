"""
Divided Correlation V4 Library
==============================

Grid-level ENSO teleconnection analysis for DJF over the Maritime Continent.

Key design choices (V4):
- Domain: 90.0–152.5°E, 12.5°S–12.5°N (full Maritime Continent)
- Land mask: ALL land in domain (not just Indonesia)
- Season: DJF only (Dec of year Y-1 + Jan-Feb of year Y = DJF Y)
- Period: DJF 1981–2020 (40 years)
- Climatology: 1991–2020 for anomaly computation
- Detrending: Linear detrend applied to anomalies
- No standardization
- Changed thresholds: 0.4, 0.6, 0.8
- Running window: 15 years only
- Split layout: 2x2 (P1/P2 left, delta right)
- Colormaps: RdBu for correlation, BrBG for delta

This module is fully independent from v1 and v2.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cartopy.io import shapereader
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

# =============================================================================
# CONSTANTS
# =============================================================================

YEAR_START = 1981
YEAR_END = 2020
CLIM_START = 1991
CLIM_END = 2020

# Maritime Continent domain
ANALYSIS_DOMAIN = (90.0, 152.5, -12.5, 12.5)

# Split configuration
SPLIT_MIN_YEARS = 10
SPLIT_MAX_YEARS = 30

# Change detection thresholds
DEFAULT_CHANGED_THRESHOLDS = (0.40, 0.60, 0.80)
PRIMARY_CHANGED_THRESHOLD = 0.40
SIGN_FLIP_MIN_ABS_R = 0.20
STRENGTHEN_THRESHOLD = 0.20
WEAKEN_THRESHOLD = -0.20

# Running window
DEFAULT_RUNNING_WINDOWS = (15,)

# Hotspot selection
DEFAULT_N_HOTSPOTS = 6
DEFAULT_HOTSPOT_MIN_SEPARATION_DEG = 5.0

# Plotting defaults
DEFAULT_DPI = 180
DEFAULT_CORRELATION_CMAP = "RdBu"
DEFAULT_DELTA_CMAP = "BrBG"
DEFAULT_CORR_VMIN = -1.0
DEFAULT_CORR_VMAX = 1.0
DEFAULT_DELTA_VLIM = 1.0
MAP_LON_TICKS = np.arange(90.0, 151.0, 10.0)
MAP_LAT_TICKS = np.arange(-10.0, 11.0, 5.0)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class DividedCorrelationConfig:
    """Configuration for the Divided Correlation workflow."""

    root_dir: Path
    project_data_dir: Path
    results_dir: Path
    mswep_monthly_path: Path
    nino34_monthly_path: Path
    analysis_name: str = "divided_correlation"
    # Temporal parameters
    start_year: int = YEAR_START
    end_year: int = YEAR_END
    clim_start: int = CLIM_START
    clim_end: int = CLIM_END
    
    # Spatial domain
    lon_min: float = ANALYSIS_DOMAIN[0]
    lon_max: float = ANALYSIS_DOMAIN[1]
    lat_min: float = ANALYSIS_DOMAIN[2]
    lat_max: float = ANALYSIS_DOMAIN[3]
    
    # Split parameters
    split_min_years: int = SPLIT_MIN_YEARS
    split_max_years: int = SPLIT_MAX_YEARS
    
    # Change detection thresholds
    changed_thresholds: tuple[float, ...] = field(
        default_factory=lambda: DEFAULT_CHANGED_THRESHOLDS
    )
    primary_changed_threshold: float = PRIMARY_CHANGED_THRESHOLD
    sign_flip_min_abs_r: float = SIGN_FLIP_MIN_ABS_R
    strengthen_threshold: float = STRENGTHEN_THRESHOLD
    weaken_threshold: float = WEAKEN_THRESHOLD
    
    # Running diagnostics
    running_windows: tuple[int, ...] = field(
        default_factory=lambda: DEFAULT_RUNNING_WINDOWS
    )
    
    # Hotspot selection
    n_hotspots: int = DEFAULT_N_HOTSPOTS
    hotspot_min_separation_deg: float = DEFAULT_HOTSPOT_MIN_SEPARATION_DEG
    
    # Plotting
    dpi: int = DEFAULT_DPI
    correlation_cmap: str = DEFAULT_CORRELATION_CMAP
    delta_cmap: str = DEFAULT_DELTA_CMAP
    corr_vmin: float = DEFAULT_CORR_VMIN
    corr_vmax: float = DEFAULT_CORR_VMAX
    delta_vlim: float = DEFAULT_DELTA_VLIM
    
    @classmethod
    def from_repo_root(
        cls,
        root_dir: Path | str,
        mswep_monthly_path: Path | str | None = None,
        nino34_monthly_path: Path | str | None = None,
    ) -> "DividedCorrelationV4Config":
        """Create config by auto-detecting paths from repository root."""
        root = Path(root_dir).resolve()
        paths_cfg = _read_paths_config(root)
        
        climate_data_dir = None
        project_data_dir = root / "data"
        results_dir = root / "results"
        
        if paths_cfg:
            if climate_value := paths_cfg.get("climate_data_dir"):
                climate_data_dir = Path(climate_value)
            if project_value := paths_cfg.get("project_data_dir"):
                project_data_dir = Path(project_value)
            if results_value := paths_cfg.get("results_dir"):
                results_dir = Path(results_value)
        
        # Auto-detect MSWEP path
        mswep_path = Path(mswep_monthly_path) if mswep_monthly_path else _first_existing_path(
            [
                root / "data" / "mswep-monthly" / "mswep_monthly_combined.nc",
                project_data_dir / "mswep-monthly" / "mswep_monthly_combined.nc",
                *(
                    [climate_data_dir / "mswep-monthly" / "mswep_monthly_combined.nc"]
                    if climate_data_dir else []
                ),
            ],
            fallback=root / "data" / "mswep-monthly" / "mswep_monthly_combined.nc",
        )
        
        # Auto-detect Niño 3.4 path
        nino_path = Path(nino34_monthly_path) if nino34_monthly_path else _first_existing_path(
            [
                root / "data" / "index-monthly" / "nino34.anom.csv",
                project_data_dir / "index-monthly" / "nino34.anom.csv",
                *(
                    [climate_data_dir / "index-monthly" / "nino34.anom.csv"]
                    if climate_data_dir else []
                ),
            ],
            fallback=root / "data" / "index-monthly" / "nino34.anom.csv",
        )
        
        return cls(
            root_dir=root,
            project_data_dir=project_data_dir,
            results_dir=results_dir,
            mswep_monthly_path=mswep_path,
            nino34_monthly_path=nino_path,
        )
    
    # Directory properties
    @property
    def intermediate_dir(self) -> Path:
        return self.project_data_dir / "intermediate" / self.analysis_name
    
    @property
    def processed_dir(self) -> Path:
        return self.project_data_dir / "processed" / self.analysis_name
    
    @property
    def results_base_dir(self) -> Path:
        return self.results_dir / self.analysis_name
    
    @property
    def png_baseline_dir(self) -> Path:
        return self.results_base_dir / "png" / "baseline"
    
    @property
    def png_splits_dir(self) -> Path:
        return self.results_base_dir / "png" / "splits"
    
    @property
    def png_atlas_dir(self) -> Path:
        return self.results_base_dir / "png" / "atlas"
    
    @property
    def png_summaries_dir(self) -> Path:
        return self.results_base_dir / "png" / "summaries"
    
    @property
    def png_running_dir(self) -> Path:
        return self.results_base_dir / "png" / "running_stats"
    
    @property
    def tables_dir(self) -> Path:
        return self.results_base_dir / "tables"
    
    @property
    def logs_dir(self) -> Path:
        return self.results_base_dir / "logs"
    
    def ensure_output_dirs(self) -> None:
        """Create all output directories."""
        for directory in [
            self.intermediate_dir,
            self.processed_dir,
            self.png_baseline_dir,
            self.png_splits_dir,
            self.png_atlas_dir,
            self.png_summaries_dir,
            self.png_running_dir,
            self.tables_dir,
            self.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def to_manifest_dict(self) -> dict[str, object]:
        """Convert config to JSON-serializable dictionary."""
        payload = asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in payload.items()}
    
    @property
    def n_years(self) -> int:
        return self.end_year - self.start_year + 1
    
    @property
    def n_splits(self) -> int:
        return self.split_max_years - self.split_min_years + 1


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _read_paths_config(root_dir: Path) -> dict[str, object]:
    config_path = root_dir / "configs" / "paths.yml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _first_existing_path(candidates: Sequence[Path], fallback: Path) -> Path:
    for c in candidates:
        if c.exists():
            return c
    return fallback


def slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")


def save_png(fig: plt.Figure, out_path: Path, dpi: int = DEFAULT_DPI) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def export_summary_tables(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def export_gridded_outputs(ds: xr.Dataset, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)


def write_run_manifest(manifest: dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================


def resolve_name(available: Iterable[str], candidates: Sequence[str]) -> str | None:
    lookup = {n.lower(): n for n in available}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def select_precip_var(ds: xr.Dataset, configured_var: str | None = None) -> xr.DataArray:
    if configured_var and configured_var in ds.data_vars:
        return ds[configured_var]
    for var in ds.data_vars:
        dims_lower = [d.lower() for d in ds[var].dims]
        if any(d in ("time", "valid_time") for d in dims_lower):
            if any(d in ("lat", "latitude", "y") for d in dims_lower):
                if any(d in ("lon", "longitude", "x") for d in dims_lower):
                    return ds[var]
    raise ValueError(f"Could not infer precipitation variable from: {list(ds.data_vars)}")


def standardize_precip_coords(da: xr.DataArray) -> xr.DataArray:
    available = list(da.coords) + list(da.dims)
    rename_map: dict[str, str] = {}
    
    time_name = resolve_name(available, ("time", "valid_time"))
    lat_name = resolve_name(available, ("lat", "latitude", "y"))
    lon_name = resolve_name(available, ("lon", "longitude", "x"))
    
    if time_name is None or lat_name is None or lon_name is None:
        raise ValueError(f"Could not resolve time/lat/lon from {available}")
    
    if time_name != "time":
        rename_map[time_name] = "time"
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    
    if rename_map:
        da = da.rename(rename_map)
    
    extra_dims = [d for d in da.dims if d not in {"time", "lat", "lon"}]
    for d in extra_dims:
        da = da.isel({d: 0}, drop=True)
    
    da = da.assign_coords(time=pd.to_datetime(da["time"].values))
    da = da.sortby("time")
    da = da.assign_coords(lon=((da["lon"] + 360) % 360))
    da = da.sortby("lon")
    
    return da


def ensure_rainfall_mm_per_month(da: xr.DataArray) -> xr.DataArray:
    units = str(da.attrs.get("units", "")).lower()
    out = da.copy()
    
    if "mm/day" in units or "mm d-1" in units:
        out = da * da["time"].dt.days_in_month
        out.attrs = dict(da.attrs)
        out.attrs["units"] = "mm/month"
    elif units in {"m", "m/month"}:
        out = da * 1000.0
        out.attrs = dict(da.attrs)
        out.attrs["units"] = "mm/month"
    elif "m/day" in units or "m d-1" in units:
        out = da * 1000.0 * da["time"].dt.days_in_month
        out.attrs = dict(da.attrs)
        out.attrs["units"] = "mm/month"
    
    return out


def subset_domain(
    da: xr.DataArray,
    lon_min: float, lon_max: float,
    lat_min: float, lat_max: float,
) -> xr.DataArray:
    lat0, lat1 = float(da["lat"].values[0]), float(da["lat"].values[-1])
    lat_slice = slice(lat_max, lat_min) if lat0 > lat1 else slice(lat_min, lat_max)
    lon0, lon1 = float(da["lon"].values[0]), float(da["lon"].values[-1])
    lon_slice = slice(lon_min, lon_max) if lon0 <= lon1 else slice(lon_max, lon_min)
    return da.sel(lon=lon_slice, lat=lat_slice)


def load_mswep_monthly(path: Path | str, precip_var: str | None = None) -> xr.DataArray:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"MSWEP monthly file not found: {dataset_path}")
    
    ds = xr.open_dataset(dataset_path)
    da = select_precip_var(ds, configured_var=precip_var)
    da = standardize_precip_coords(da)
    da = ensure_rainfall_mm_per_month(da)
    da.name = da.name or "precipitation"
    return da


def load_nino34_monthly(path: Path | str) -> pd.Series:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Nino3.4 file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    if {"year", "month"}.issubset(df.columns):
        value_cols = [c for c in df.columns if c not in {"year", "month"}]
        if not value_cols:
            raise ValueError(f"No value column in {csv_path}")
        time = pd.to_datetime({
            "year": pd.to_numeric(df["year"], errors="coerce"),
            "month": pd.to_numeric(df["month"], errors="coerce"),
            "day": 1,
        }, errors="coerce")
        numeric = {c: pd.to_numeric(df[c], errors="coerce") for c in value_cols}
        value_col = max(numeric, key=lambda c: int(numeric[c].notna().sum()))
        values = numeric[value_col]
    elif "date" in df.columns:
        value_cols = [c for c in df.columns if c != "date"]
        if not value_cols:
            raise ValueError(f"No value column in {csv_path}")
        time = pd.to_datetime(df["date"], errors="coerce")
        numeric = {c: pd.to_numeric(df[c], errors="coerce") for c in value_cols}
        value_col = max(numeric, key=lambda c: int(numeric[c].notna().sum()))
        values = numeric[value_col]
    else:
        raise ValueError(f"CSV must have 'date' or 'year'/'month' columns: {csv_path}")
    
    series = pd.Series(
        np.asarray(values, dtype=float),
        index=pd.to_datetime(np.asarray(time), errors="coerce"),
        name="nino34",
    )
    series = series.replace([-9999, -9999.0, -99.99], np.nan).dropna()
    
    if series.empty:
        raise ValueError(f"No valid Nino3.4 values: {csv_path}")
    
    series.index = series.index.to_period("M").to_timestamp(how="start")
    series = series.groupby(series.index).mean().sort_index()
    return series


# =============================================================================
# DJF CONSTRUCTION
# =============================================================================


def build_complete_djf_field(
    monthly_da: xr.DataArray,
    start_year: int,
    end_year: int,
) -> xr.DataArray:
    """
    Build DJF seasonal mean field.
    Dec(Y-1) + Jan(Y) + Feb(Y) = DJF Y
    """
    month = monthly_da["time"].dt.month
    djf_monthly = monthly_da.sel(time=month.isin([12, 1, 2]))
    
    djf_year = xr.where(
        djf_monthly["time"].dt.month == 12,
        djf_monthly["time"].dt.year + 1,
        djf_monthly["time"].dt.year,
    )
    djf_monthly = djf_monthly.assign_coords(djf_year=("time", djf_year.data))
    
    month_count = djf_monthly["time"].groupby("djf_year").count()
    full_years = month_count["djf_year"].where(month_count == 3, drop=True).values.astype(int)
    keep_years = [y for y in full_years if start_year <= y <= end_year]
    
    if not keep_years:
        raise ValueError(f"No complete DJF years in range {start_year}-{end_year}")
    
    djf = djf_monthly.groupby("djf_year").mean("time")
    djf = djf.sel(djf_year=keep_years).rename({"djf_year": "year"})
    djf = djf.transpose("year", "lat", "lon")
    djf = djf.assign_coords(year=np.array(keep_years, dtype=int))
    return djf


def build_complete_djf_series(
    monthly_series: pd.Series,
    start_year: int,
    end_year: int,
) -> pd.Series:
    """Build DJF seasonal mean series."""
    if monthly_series.empty:
        raise ValueError("Monthly series is empty")
    
    frame = monthly_series.copy()
    frame.index = pd.to_datetime(frame.index)
    data = frame.to_frame("value").reset_index().rename(columns={"index": "time"})
    data["year"] = data["time"].dt.year
    data["month"] = data["time"].dt.month
    data = data[data["month"].isin([12, 1, 2])].copy()
    data["djf_year"] = np.where(data["month"] == 12, data["year"] + 1, data["year"])
    
    grouped = data.groupby("djf_year").agg(
        value=("value", "mean"),
        month_count=("month", "nunique"),
    )
    grouped = grouped[grouped["month_count"] == 3]
    grouped = grouped.loc[(grouped.index >= start_year) & (grouped.index <= end_year), "value"]
    grouped.index.name = "year"
    grouped.name = "nino34"
    
    if grouped.empty:
        raise ValueError(f"No complete DJF values in range {start_year}-{end_year}")
    
    return grouped.astype(float)


def validate_djf_year_axis(years: Sequence[int], start_year: int, end_year: int) -> None:
    expected = np.arange(start_year, end_year + 1)
    actual = np.asarray(years, dtype=int)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValueError(f"Expected years {expected[0]}-{expected[-1]}; got {actual.tolist()}")


# =============================================================================
# ANOMALY AND DETRENDING
# =============================================================================


def compute_rain_climatology(djf_da: xr.DataArray, clim_start: int, clim_end: int) -> xr.DataArray:
    return djf_da.sel(year=slice(clim_start, clim_end)).mean("year")


def compute_series_climatology(djf_series: pd.Series, clim_start: int, clim_end: int) -> float:
    return float(djf_series.loc[clim_start:clim_end].mean())


def compute_djf_anomalies(djf_da: xr.DataArray, climatology: xr.DataArray) -> xr.DataArray:
    out = djf_da - climatology
    out.name = f"{djf_da.name or 'rain'}_anom"
    return out


def compute_series_anomalies(djf_series: pd.Series, climatology: float) -> pd.Series:
    out = djf_series - climatology
    out.name = f"{djf_series.name or 'series'}_anom"
    return out


def detrend_series(series: pd.Series) -> pd.Series:
    values = series.to_numpy(dtype=float)
    x = np.arange(values.size, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        raise ValueError("Need at least two finite values to detrend")
    slope, intercept = np.polyfit(x[mask], values[mask], 1)
    trend = slope * x + intercept
    detrended = values.copy()
    detrended[mask] = values[mask] - trend[mask] + np.nanmean(values[mask])
    detrended[~mask] = np.nan
    return pd.Series(detrended, index=series.index, name=f"{series.name}_detrended")


def linear_detrend_field_along_year(da: xr.DataArray) -> xr.DataArray:
    stacked = da.stack(point=("lat", "lon")).transpose("year", "point")
    arr = np.asarray(stacked.values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    x = np.arange(arr.shape[0], dtype=float)
    
    for idx in range(arr.shape[1]):
        col = arr[:, idx]
        mask = np.isfinite(col)
        if mask.sum() < 2:
            continue
        slope, intercept = np.polyfit(x[mask], col[mask], 1)
        trend = slope * x[mask] + intercept
        adjusted = col.copy()
        adjusted[mask] = col[mask] - trend + np.nanmean(col[mask])
        out[:, idx] = adjusted
    
    detrended = xr.DataArray(out, coords=stacked.coords, dims=stacked.dims)
    detrended = detrended.unstack("point").transpose("year", "lat", "lon")
    detrended.name = f"{da.name or 'field'}_detrended"
    return detrended


# =============================================================================
# LAND MASK (ALL LAND IN DOMAIN, NOT JUST INDONESIA)
# =============================================================================


def build_valid_all_years_mask(djf_da: xr.DataArray) -> xr.DataArray:
    mask = np.all(np.isfinite(djf_da.values), axis=0)
    return xr.DataArray(mask, coords={"lat": djf_da.lat, "lon": djf_da.lon}, dims=["lat", "lon"])


def _load_maritime_continent_land_geometry():
    """Load ALL land geometry in the domain using Natural Earth physical land."""
    shp_path = shapereader.natural_earth(
        resolution="10m",
        category="physical",
        name="land",
    )
    reader = shapereader.Reader(shp_path)
    land_geoms = [record.geometry for record in reader.records()]
    return unary_union(land_geoms)


def build_land_mask(lats: np.ndarray, lons: np.ndarray) -> xr.DataArray:
    """Build land mask for ALL land in the Maritime Continent domain."""
    geom = _load_maritime_continent_land_geometry()
    prepared_geom = prep(geom)
    
    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            point = Point(lon, lat)
            mask[i, j] = prepared_geom.contains(point)
    
    return xr.DataArray(
        mask,
        coords={"lat": lats, "lon": lons},
        dims=["lat", "lon"],
        name="land_mask",
    )


def build_analysis_valid_land_mask(
    land_mask: xr.DataArray,
    valid_mask: xr.DataArray,
) -> xr.DataArray:
    combined = land_mask.astype(bool) & valid_mask.astype(bool)
    combined.name = "analysis_valid_land_mask"
    return combined


def build_analysis_valid_ocean_mask(
    land_mask: xr.DataArray,
    valid_mask: xr.DataArray,
) -> xr.DataArray:
    combined = (~land_mask.astype(bool)) & valid_mask.astype(bool)
    combined.name = "analysis_valid_ocean_mask"
    return combined


def build_analysis_valid_all_mask(valid_mask: xr.DataArray) -> xr.DataArray:
    combined = valid_mask.astype(bool)
    combined.name = "analysis_valid_all_mask"
    return combined


# =============================================================================
# PERIOD SPLITTING
# =============================================================================


def generate_split_periods(
    start_year: int,
    end_year: int,
    min_p1_years: int,
    max_p1_years: int,
) -> pd.DataFrame:
    """Generate all period splits from min_p1_years to max_p1_years."""
    total_years = end_year - start_year + 1
    
    records = []
    for p1_size in range(min_p1_years, max_p1_years + 1):
        p2_size = total_years - p1_size
        p1_start = start_year
        p1_end = start_year + p1_size - 1
        p2_start = p1_end + 1
        p2_end = end_year
        
        records.append({
            "split_id": f"{p1_size}v{p2_size}",
            "p1_start": p1_start,
            "p1_end": p1_end,
            "p1_years": p1_size,
            "p2_start": p2_start,
            "p2_end": p2_end,
            "p2_years": p2_size,
        })
    
    split_df = pd.DataFrame(records)
    split_df["split_id"] = split_df["split_id"].map(str).astype(object)
    return split_df


def _complete_year_subset(djf_da: xr.DataArray, start_year: int, end_year: int) -> xr.DataArray:
    years = djf_da["year"].values
    keep = [y for y in years if start_year <= y <= end_year]
    return djf_da.sel(year=keep)


# =============================================================================
# CORRELATION AND REGRESSION
# =============================================================================


def compute_grid_correlation(rain_da: xr.DataArray, nino_series: pd.Series) -> xr.DataArray:
    years = rain_da["year"].values
    nino_aligned = nino_series.reindex(years).values
    rain_arr = rain_da.values
    
    rain_mean = np.nanmean(rain_arr, axis=0)
    nino_mean = np.nanmean(nino_aligned)
    
    rain_centered = rain_arr - rain_mean[np.newaxis, :, :]
    nino_centered = nino_aligned - nino_mean
    
    cov = np.nanmean(rain_centered * nino_centered[:, np.newaxis, np.newaxis], axis=0)
    rain_std = np.nanstd(rain_arr, axis=0, ddof=0)
    nino_std = np.nanstd(nino_aligned, ddof=0)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / (rain_std * nino_std)
    
    corr = np.where(np.isfinite(corr), corr, np.nan)
    
    return xr.DataArray(
        corr,
        coords={"lat": rain_da.lat, "lon": rain_da.lon},
        dims=["lat", "lon"],
        name="correlation",
    )


def compute_grid_regression_slope(rain_da: xr.DataArray, nino_series: pd.Series) -> xr.DataArray:
    years = rain_da["year"].values
    nino_aligned = nino_series.reindex(years).values
    rain_arr = rain_da.values
    
    rain_mean = np.nanmean(rain_arr, axis=0)
    nino_mean = np.nanmean(nino_aligned)
    
    rain_centered = rain_arr - rain_mean[np.newaxis, :, :]
    nino_centered = nino_aligned - nino_mean
    
    cov = np.nanmean(rain_centered * nino_centered[:, np.newaxis, np.newaxis], axis=0)
    nino_var = np.nanvar(nino_aligned, ddof=0)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = cov / nino_var
    
    slope = np.where(np.isfinite(slope), slope, np.nan)
    
    return xr.DataArray(
        slope,
        coords={"lat": rain_da.lat, "lon": rain_da.lon},
        dims=["lat", "lon"],
        name="regression_slope",
    )


def compute_fullperiod_maps(rain_da: xr.DataArray, nino_series: pd.Series) -> xr.Dataset:
    corr = compute_grid_correlation(rain_da, nino_series)
    slope = compute_grid_regression_slope(rain_da, nino_series)
    return xr.Dataset({"correlation": corr, "regression_slope": slope})


def compute_split_delta_maps(
    rain_da: xr.DataArray,
    nino_series: pd.Series,
    split_df: pd.DataFrame,
) -> xr.Dataset:
    n_splits = len(split_df)
    lats = rain_da.lat.values
    lons = rain_da.lon.values
    
    corr_p1 = np.full((n_splits, len(lats), len(lons)), np.nan)
    corr_p2 = np.full((n_splits, len(lats), len(lons)), np.nan)
    slope_p1 = np.full((n_splits, len(lats), len(lons)), np.nan)
    slope_p2 = np.full((n_splits, len(lats), len(lons)), np.nan)
    
    for idx, row in split_df.iterrows():
        rain_p1 = _complete_year_subset(rain_da, row["p1_start"], row["p1_end"])
        nino_p1 = nino_series.loc[row["p1_start"]:row["p1_end"]]
        corr_p1[idx] = compute_grid_correlation(rain_p1, nino_p1).values
        slope_p1[idx] = compute_grid_regression_slope(rain_p1, nino_p1).values
        
        rain_p2 = _complete_year_subset(rain_da, row["p2_start"], row["p2_end"])
        nino_p2 = nino_series.loc[row["p2_start"]:row["p2_end"]]
        corr_p2[idx] = compute_grid_correlation(rain_p2, nino_p2).values
        slope_p2[idx] = compute_grid_regression_slope(rain_p2, nino_p2).values
    
    delta_corr = corr_p2 - corr_p1
    delta_slope = slope_p2 - slope_p1
    split_ids = np.asarray(split_df["split_id"].map(str).tolist(), dtype=object)
    
    return xr.Dataset(
        {
            "corr_p1": (["split", "lat", "lon"], corr_p1),
            "corr_p2": (["split", "lat", "lon"], corr_p2),
            "delta_corr": (["split", "lat", "lon"], delta_corr),
            "slope_p1": (["split", "lat", "lon"], slope_p1),
            "slope_p2": (["split", "lat", "lon"], slope_p2),
            "delta_slope": (["split", "lat", "lon"], delta_slope),
        },
        coords={"split": split_ids, "lat": lats, "lon": lons},
    )


# =============================================================================
# SENSITIVITY METRICS
# =============================================================================


def _flatten_mask_values(data_array: xr.DataArray, mask_da: xr.DataArray) -> np.ndarray:
    mask = mask_da.values.astype(bool)
    return data_array.values[mask]


def summarize_split_metrics(
    split_ds: xr.Dataset,
    split_df: pd.DataFrame,
    land_mask: xr.DataArray,
    ocean_mask: xr.DataArray,
    all_mask: xr.DataArray,
    changed_thresholds: tuple[float, ...],
    sign_flip_min_abs_r: float,
    strengthen_threshold: float,
    weaken_threshold: float,
) -> pd.DataFrame:
    records = []
    n_land = int(land_mask.values.sum())
    n_ocean = int(ocean_mask.values.sum())
    n_all = int(all_mask.values.sum())
    
    for _, row in split_df.iterrows():
        split_id = row["split_id"]
        
        delta_r = split_ds["delta_corr"].sel(split=split_id)
        delta_slope = split_ds["delta_slope"].sel(split=split_id)
        corr_p1 = split_ds["corr_p1"].sel(split=split_id)
        corr_p2 = split_ds["corr_p2"].sel(split=split_id)
        
        delta_r_land = _flatten_mask_values(delta_r, land_mask)
        delta_slope_land = _flatten_mask_values(delta_slope, land_mask)
        corr_p1_land = _flatten_mask_values(corr_p1, land_mask)
        corr_p2_land = _flatten_mask_values(corr_p2, land_mask)
        delta_r_ocean = _flatten_mask_values(delta_r, ocean_mask)
        delta_r_all = _flatten_mask_values(delta_r, all_mask)
        
        abs_delta = np.abs(delta_r_land)
        abs_delta_ocean = np.abs(delta_r_ocean)
        abs_delta_all = np.abs(delta_r_all)
        record = {
            "split_id": split_id,
            "p1_start": row["p1_start"],
            "p1_end": row["p1_end"],
            "p2_start": row["p2_start"],
            "p2_end": row["p2_end"],
            "n_land_cells": n_land,
            "n_ocean_cells": n_ocean,
            "n_all_cells": n_all,
            "mean_abs_delta_r": np.nanmean(abs_delta),
            "median_abs_delta_r": np.nanmedian(abs_delta),
            "std_delta_r": np.nanstd(delta_r_land),
            "mean_delta_r": np.nanmean(delta_r_land),
            "median_delta_r": np.nanmedian(delta_r_land),
            "mean_delta_slope": np.nanmean(delta_slope_land),
            "median_delta_slope": np.nanmedian(delta_slope_land),
            "mean_abs_delta_r_ocean": np.nanmean(abs_delta_ocean),
            "mean_abs_delta_r_all": np.nanmean(abs_delta_all),
        }
        
        # Changed cells at thresholds (absolute delta) for land, ocean, and all valid grid cells.
        for thresh in changed_thresholds:
            thresh_key = f"{int(thresh * 100):03d}"

            n_changed = np.sum(abs_delta > thresh)
            record[f"n_changed_{thresh_key}"] = n_changed
            record[f"frac_changed_{thresh_key}"] = n_changed / n_land if n_land > 0 else 0.0
            n_positive = np.sum(delta_r_land > thresh)
            record[f"n_delta_r_gt_{thresh_key}"] = n_positive
            record[f"frac_delta_r_gt_{thresh_key}"] = n_positive / n_land if n_land > 0 else 0.0

            n_changed_ocean = np.sum(abs_delta_ocean > thresh)
            record[f"n_changed_{thresh_key}_ocean"] = n_changed_ocean
            record[f"frac_changed_{thresh_key}_ocean"] = n_changed_ocean / n_ocean if n_ocean > 0 else 0.0

            n_changed_all = np.sum(abs_delta_all > thresh)
            record[f"n_changed_{thresh_key}_all"] = n_changed_all
            record[f"frac_changed_{thresh_key}_all"] = n_changed_all / n_all if n_all > 0 else 0.0
        
        # Sign flips
        sign_flip_mask = (
            (np.sign(corr_p1_land) != np.sign(corr_p2_land)) &
            (np.abs(corr_p1_land) >= sign_flip_min_abs_r) &
            (np.abs(corr_p2_land) >= sign_flip_min_abs_r)
        )
        record["n_sign_flip"] = np.sum(sign_flip_mask)
        record["frac_sign_flip"] = np.sum(sign_flip_mask) / n_land if n_land > 0 else 0.0
        
        # Strengthen vs weaken
        record["n_strengthen"] = np.sum(delta_r_land > strengthen_threshold)
        record["n_weaken"] = np.sum(delta_r_land < weaken_threshold)
        record["frac_strengthen"] = record["n_strengthen"] / n_land if n_land > 0 else 0.0
        record["frac_weaken"] = record["n_weaken"] / n_land if n_land > 0 else 0.0
        
        records.append(record)
    
    return pd.DataFrame(records)


def rank_splits(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    df["rank_mean_abs_delta"] = df["mean_abs_delta_r"].rank(ascending=False)
    if "frac_changed_040" in df.columns:
        df["rank_frac_changed"] = df["frac_changed_040"].rank(ascending=False)
    rank_cols = [c for c in df.columns if c.startswith("rank_")]
    if rank_cols:
        df["composite_rank"] = df[rank_cols].mean(axis=1)
        df = df.sort_values("composite_rank")
    return df


def sort_splits_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """Return splits ordered from 10v30 through 30v10 for plotting."""
    ordered = df.copy()
    if {"p1_start", "p1_end"}.issubset(ordered.columns):
        return ordered.sort_values(["p1_start", "p1_end"]).reset_index(drop=True)
    if "split_id" in ordered.columns:
        p1_years = ordered["split_id"].str.extract(r"^(\d+)v", expand=False).astype(int)
        ordered = ordered.assign(_p1_years=p1_years)
        ordered = ordered.sort_values("_p1_years").drop(columns="_p1_years")
    return ordered.reset_index(drop=True)


# =============================================================================
# HOTSPOT SELECTION
# =============================================================================


def _candidate_points_for_hotspots(
    delta_ds: xr.Dataset,
    land_mask: xr.DataArray,
    split_id: str,
    top_n: int = 50,
) -> pd.DataFrame:
    delta_r = delta_ds["delta_corr"].sel(split=split_id)
    abs_delta = np.abs(delta_r.values)
    mask = land_mask.values.astype(bool)
    abs_delta_masked = np.where(mask, abs_delta, np.nan)
    
    lats = delta_r.lat.values
    lons = delta_r.lon.values
    
    records = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if mask[i, j] and np.isfinite(abs_delta_masked[i, j]):
                records.append({
                    "lat": lat,
                    "lon": lon,
                    "abs_delta_r": abs_delta_masked[i, j],
                    "delta_r": delta_r.values[i, j],
                })
    
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.nlargest(top_n, "abs_delta_r")


def _is_far_enough(lat: float, lon: float, chosen: list[tuple[float, float]], min_sep: float) -> bool:
    for clat, clon in chosen:
        dist = np.sqrt((lat - clat) ** 2 + (lon - clon) ** 2)
        if dist < min_sep:
            return False
    return True


def select_hotspot_cells(
    delta_ds: xr.Dataset,
    land_mask: xr.DataArray,
    split_id: str,
    n_hotspots: int,
    min_separation_deg: float,
) -> pd.DataFrame:
    candidates = _candidate_points_for_hotspots(delta_ds, land_mask, split_id, top_n=100)
    
    if candidates.empty:
        return pd.DataFrame(columns=["hotspot_id", "lat", "lon", "abs_delta_r", "delta_r"])
    
    chosen: list[tuple[float, float]] = []
    used: set[tuple[float, float]] = set()
    hotspots: list[dict[str, float]] = []

    def add_hotspot(row: pd.Series) -> None:
        point = (row["lat"], row["lon"])
        chosen.append(point)
        used.add(point)
        hotspots.append({
            "hotspot_id": len(hotspots) + 1,
            "lat": row["lat"],
            "lon": row["lon"],
            "abs_delta_r": row["abs_delta_r"],
            "delta_r": row["delta_r"],
        })
    
    for _, row in candidates.iterrows():
        if len(hotspots) >= n_hotspots:
            break
        if _is_far_enough(row["lat"], row["lon"], chosen, min_separation_deg):
            add_hotspot(row)

    # Backfill from the remaining strongest candidates so v4 always returns
    # at least the requested number of hotspots when enough land cells exist.
    if len(hotspots) < n_hotspots:
        for _, row in candidates.iterrows():
            if len(hotspots) >= n_hotspots:
                break
            point = (row["lat"], row["lon"])
            if point in used:
                continue
            add_hotspot(row)
    
    return pd.DataFrame(hotspots)


def extract_hotspot_series(rain_da: xr.DataArray, hotspot_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in hotspot_df.iterrows():
        lat, lon = row["lat"], row["lon"]
        values = rain_da.sel(lat=lat, lon=lon, method="nearest").values
        years = rain_da.year.values
        for year, value in zip(years, values):
            records.append({
                "hotspot_id": row["hotspot_id"],
                "lat": lat,
                "lon": lon,
                "year": year,
                "rain_anom": value,
            })
    return pd.DataFrame(records)


# =============================================================================
# RUNNING DIAGNOSTICS
# =============================================================================


def compute_centered_running_correlation(series_a: pd.Series, series_b: pd.Series, window: int) -> pd.DataFrame:
    half_win = window // 2
    years = series_a.index.values
    records = []
    
    for i, year in enumerate(years):
        start_idx = max(0, i - half_win)
        end_idx = min(len(years), i + half_win + 1)
        
        if end_idx - start_idx < window:
            records.append({"year": year, "correlation": np.nan})
            continue
        
        a_window = series_a.iloc[start_idx:end_idx].values
        b_window = series_b.iloc[start_idx:end_idx].values
        mask = np.isfinite(a_window) & np.isfinite(b_window)
        
        if mask.sum() < 3:
            records.append({"year": year, "correlation": np.nan})
            continue
        
        corr = np.corrcoef(a_window[mask], b_window[mask])[0, 1]
        records.append({"year": year, "correlation": corr})
    
    return pd.DataFrame(records)


def compute_centered_running_variance(series: pd.Series, window: int, value_name: str) -> pd.DataFrame:
    half_win = window // 2
    years = series.index.values
    records = []
    
    for i, year in enumerate(years):
        start_idx = max(0, i - half_win)
        end_idx = min(len(years), i + half_win + 1)
        
        if end_idx - start_idx < window:
            records.append({"year": year, value_name: np.nan})
            continue
        
        window_vals = series.iloc[start_idx:end_idx].values
        mask = np.isfinite(window_vals)
        
        if mask.sum() < 2:
            records.append({"year": year, value_name: np.nan})
            continue
        
        var = np.var(window_vals[mask], ddof=1)
        records.append({"year": year, value_name: var})
    
    return pd.DataFrame(records)


def compute_centered_running_covariance(rain_series: pd.Series, nino_series: pd.Series, window: int) -> pd.DataFrame:
    half_win = window // 2
    years = rain_series.index.values
    records = []
    
    for i, year in enumerate(years):
        start_idx = max(0, i - half_win)
        end_idx = min(len(years), i + half_win + 1)
        
        if end_idx - start_idx < window:
            records.append({"year": year, "covariance": np.nan})
            continue
        
        rain_window = rain_series.iloc[start_idx:end_idx].values
        nino_window = nino_series.iloc[start_idx:end_idx].values
        mask = np.isfinite(rain_window) & np.isfinite(nino_window)
        
        if mask.sum() < 2:
            records.append({"year": year, "covariance": np.nan})
            continue
        
        cov = np.cov(rain_window[mask], nino_window[mask])[0, 1]
        records.append({"year": year, "covariance": cov})
    
    return pd.DataFrame(records)


def compute_running_stats(rain_series: pd.Series, nino_series: pd.Series, window: int) -> pd.DataFrame:
    corr_df = compute_centered_running_correlation(rain_series, nino_series, window)
    rain_var_df = compute_centered_running_variance(rain_series, window, "rain_variance")
    nino_var_df = compute_centered_running_variance(nino_series, window, "nino_variance")
    cov_df = compute_centered_running_covariance(rain_series, nino_series, window)
    
    result = corr_df.copy()
    result["rain_variance"] = rain_var_df["rain_variance"]
    result["nino_variance"] = nino_var_df["nino_variance"]
    result["covariance"] = cov_df["covariance"]
    return result


def compute_running_stats_for_hotspots(
    rain_da: xr.DataArray,
    nino_series: pd.Series,
    hotspot_df: pd.DataFrame,
    windows: tuple[int, ...],
) -> dict[int, dict[int, pd.DataFrame]]:
    results: dict[int, dict[int, pd.DataFrame]] = {}
    
    for _, row in hotspot_df.iterrows():
        hotspot_id = row["hotspot_id"]
        lat, lon = row["lat"], row["lon"]
        
        rain_at_point = rain_da.sel(lat=lat, lon=lon, method="nearest")
        rain_series = pd.Series(rain_at_point.values, index=rain_at_point.year.values, name="rain")
        
        results[hotspot_id] = {}
        for window in windows:
            results[hotspot_id][window] = compute_running_stats(rain_series, nino_series, window)
    
    return results


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def _add_map_context(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    *,
    label_bottom: bool = True,
    label_left: bool = True,
    tick_labelsize: int = 8,
) -> None:
    """Add coastlines and lat/lon ticks on all sides, labels on bottom/left only."""
    ax.coastlines(resolution="50m", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_xticks(MAP_LON_TICKS, crs=ccrs.PlateCarree())
    ax.set_yticks(MAP_LAT_TICKS, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.gridlines(
        draw_labels=False,
        xlocs=MAP_LON_TICKS,
        ylocs=MAP_LAT_TICKS,
        linewidth=0.3,
        alpha=0.5,
        linestyle="--",
    )
    ax.tick_params(
        top=True,
        bottom=True,
        left=True,
        right=True,
        labeltop=False,
        labelbottom=label_bottom,
        labelleft=label_left,
        labelright=False,
        labelsize=tick_labelsize,
    )


def plot_land_mask(
    land_mask: xr.DataArray,
    config: DividedCorrelationV4Config,
    out_path: Path,
) -> None:
    extent = (config.lon_min, config.lon_max, config.lat_min, config.lat_max)
    
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    
    land_mask.astype(int).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Greens",
        vmin=0, vmax=1,
        add_colorbar=True,
    )
    _add_map_context(ax, extent)
    ax.set_title(f"Land Mask ({int(land_mask.sum().values)} land cells)")
    
    save_png(fig, out_path, config.dpi)


def plot_split_timeline(split_df: pd.DataFrame, config: DividedCorrelationV4Config, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    n_splits = len(split_df)
    colors_p1 = plt.cm.Blues(0.6)
    colors_p2 = plt.cm.Oranges(0.6)
    
    for idx, row in split_df.iterrows():
        y = n_splits - idx - 1
        ax.barh(y, row["p1_years"], left=row["p1_start"], height=0.7, color=colors_p1, edgecolor="black", linewidth=0.5)
        ax.barh(y, row["p2_years"], left=row["p2_start"], height=0.7, color=colors_p2, edgecolor="black", linewidth=0.5)
    
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels(split_df["split_id"].astype(str).tolist()[::-1])
    ax.set_xlabel("DJF Year")
    ax.set_ylabel("Split Configuration")
    ax.set_title("Period Split Experiment Timeline")
    ax.axvline(2000, color="gray", linestyle="--", alpha=0.5)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=colors_p1, label="Period 1"),
            plt.Rectangle((0, 0), 1, 1, color=colors_p2, label="Period 2"),
        ],
        loc="lower right",
    )
    ax.tick_params(left=True, right=True, top=True, bottom=True,
                   labelleft=True, labelright=False, labeltop=False, labelbottom=True)
    
    plt.tight_layout()
    save_png(fig, out_path, config.dpi)


def plot_correlation_map(
    corr_da: xr.DataArray,
    config: DividedCorrelationV4Config,
    out_path: Path,
    title: str = "Correlation Map",
) -> None:
    extent = (config.lon_min, config.lon_max, config.lat_min, config.lat_max)
    
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    
    corr_da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=config.correlation_cmap,
        vmin=config.corr_vmin,
        vmax=config.corr_vmax,
        add_colorbar=True,
        cbar_kwargs={"label": "Pearson Correlation"},
    )
    _add_map_context(ax, extent)
    ax.set_title(title)
    
    save_png(fig, out_path, config.dpi)


def plot_delta_map(
    delta_da: xr.DataArray,
    config: DividedCorrelationV4Config,
    out_path: Path,
    title: str = "Delta Correlation Map",
) -> None:
    extent = (config.lon_min, config.lon_max, config.lat_min, config.lat_max)
    
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    
    delta_da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=config.delta_cmap,
        vmin=-config.delta_vlim,
        vmax=config.delta_vlim,
        add_colorbar=True,
        cbar_kwargs={"label": "Δr (P2 - P1)"},
    )
    _add_map_context(ax, extent)
    ax.set_title(title)
    
    save_png(fig, out_path, config.dpi)


def plot_split_2x2(
    split_ds: xr.Dataset,
    split_id: str,
    config: DividedCorrelationV4Config,
    out_path: Path,
    p1_label: str = "Period 1",
    p2_label: str = "Period 2",
) -> None:
    """
    Plot 2x2 layout: left column P1 (top) and P2 (bottom), right column delta (top).
    P1 and P2 share RdBu colorbar, delta uses BrBG.
    """
    extent = (config.lon_min, config.lon_max, config.lat_min, config.lat_max)
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)
    
    ax_p1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_p2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax_delta = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    
    corr_p1 = split_ds["corr_p1"].sel(split=split_id)
    corr_p2 = split_ds["corr_p2"].sel(split=split_id)
    delta = split_ds["delta_corr"].sel(split=split_id)
    
    # P1 correlation
    im1 = corr_p1.plot(
        ax=ax_p1,
        transform=ccrs.PlateCarree(),
        cmap=config.correlation_cmap,
        vmin=config.corr_vmin,
        vmax=config.corr_vmax,
        add_colorbar=False,
    )
    _add_map_context(ax_p1, extent)
    ax_p1.set_title(f"{p1_label}\nCorrelation", fontsize=10)
    
    # P2 correlation
    im2 = corr_p2.plot(
        ax=ax_p2,
        transform=ccrs.PlateCarree(),
        cmap=config.correlation_cmap,
        vmin=config.corr_vmin,
        vmax=config.corr_vmax,
        add_colorbar=False,
    )
    _add_map_context(ax_p2, extent)
    ax_p2.set_title(f"{p2_label}\nCorrelation", fontsize=10)
    
    # Shared colorbar for P1 and P2 (bottom left)
    cbar_ax1 = fig.add_axes([0.08, 0.06, 0.38, 0.02])
    cbar1 = plt.colorbar(im2, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('Pearson Correlation', fontsize=9)
    
    # Delta correlation
    im3 = delta.plot(
        ax=ax_delta,
        transform=ccrs.PlateCarree(),
        cmap=config.delta_cmap,
        vmin=-config.delta_vlim,
        vmax=config.delta_vlim,
        add_colorbar=False,
    )
    _add_map_context(ax_delta, extent)
    ax_delta.set_title("Δr (P2 - P1)", fontsize=10)
    
    # Colorbar for delta (top right area)
    cbar_ax2 = fig.add_axes([0.55, 0.55, 0.38, 0.02])
    cbar2 = plt.colorbar(im3, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Δr (P2 - P1)', fontsize=9)
    
    fig.suptitle(f"Split {split_id}: DJF Rainfall vs Niño 3.4", fontsize=12, y=0.98)
    save_png(fig, out_path, config.dpi)


def plot_split_atlas(
    split_ds: xr.Dataset,
    split_df: pd.DataFrame,
    config: DividedCorrelationV4Config,
    out_path: Path,
    metric: str = "delta_corr",
    n_cols: int = 3,
) -> None:
    n_splits = len(split_df)
    n_rows = (n_splits + n_cols - 1) // n_cols
    extent = (config.lon_min, config.lon_max, config.lat_min, config.lat_max)
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, (_, row) in enumerate(split_df.iterrows()):
        split_id = row["split_id"]
        ax = axes[idx]
        data = split_ds[metric].sel(split=split_id)
        
        if metric == "delta_corr":
            data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=config.delta_cmap,
                vmin=-config.delta_vlim,
                vmax=config.delta_vlim,
                add_colorbar=False,
            )
        else:
            data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=config.correlation_cmap,
                vmin=config.corr_vmin,
                vmax=config.corr_vmax,
                add_colorbar=False,
            )
        
        _add_map_context(ax, extent, tick_labelsize=7)
        ax.set_title(split_id, fontsize=10)
    
    for idx in range(n_splits, len(axes)):
        axes[idx].set_visible(False)
    
    metric_label = "Δr (P2-P1)" if metric == "delta_corr" else metric
    fig.suptitle(f"Split Atlas: {metric_label}", fontsize=14, y=1.02)
    plt.tight_layout()
    save_png(fig, out_path, config.dpi)


def plot_split_metric_lines(
    metrics_df: pd.DataFrame,
    config: DividedCorrelationV4Config,
    out_path: Path,
    y_col: str = "mean_abs_delta_r",
    y_label: str = "Mean |Δr|",
) -> None:
    """Summary plot with x-axis sorted 10v30 to 30v10, ticks on all sides, labels bottom/left."""
    metrics_df = sort_splits_for_plotting(metrics_df)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_df))
    ax.plot(x, metrics_df[y_col].values, marker="o", linewidth=2, markersize=6)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["split_id"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_xlabel("Split Configuration")
    ax.set_ylabel(y_label)
    ax.set_title(f"Split Sensitivity: {y_label}")
    ax.grid(True, alpha=0.3)
    
    # Ticks on all sides, labels only bottom/left
    ax.tick_params(left=True, right=True, top=True, bottom=True,
                   labelleft=True, labelright=False, labeltop=False, labelbottom=True)
    
    plt.tight_layout()
    save_png(fig, out_path, config.dpi)


def plot_split_changed_cells(
    metrics_df: pd.DataFrame,
    config: DividedCorrelationV4Config,
    out_path: Path,
    column_suffix: str = "",
    cell_label: str = "Land",
) -> None:
    """Plot changed-cell counts/fractions for thresholds 0.4, 0.6, 0.8."""
    metrics_df = sort_splits_for_plotting(metrics_df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(metrics_df))
    
    # Left: Absolute count
    for thresh in config.changed_thresholds:
        thresh_key = f"{int(thresh * 100):03d}"
        col = f"n_changed_{thresh_key}{column_suffix}"
        if col in metrics_df.columns:
            axes[0].plot(x, metrics_df[col].values, marker="o", label=f"|Δr| > {thresh}")
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_df["split_id"].astype(str).tolist(), rotation=45, ha="right")
    axes[0].set_xlabel("Split Configuration")
    axes[0].set_ylabel(f"Number of Changed {cell_label} Cells")
    axes[0].set_title(f"Changed {cell_label} Cells (Count)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(left=True, right=True, top=True, bottom=True,
                        labelleft=True, labelright=False, labeltop=False, labelbottom=True)
    
    # Right: Fraction
    for thresh in config.changed_thresholds:
        thresh_key = f"{int(thresh * 100):03d}"
        col = f"frac_changed_{thresh_key}{column_suffix}"
        if col in metrics_df.columns:
            axes[1].plot(x, metrics_df[col].values * 100, marker="o", label=f"|Δr| > {thresh}")
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_df["split_id"].astype(str).tolist(), rotation=45, ha="right")
    axes[1].set_xlabel("Split Configuration")
    axes[1].set_ylabel(f"Fraction of {cell_label} Cells (%)")
    axes[1].set_title(f"Changed {cell_label} Cells (Percentage)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(left=True, right=True, top=True, bottom=True,
                        labelleft=True, labelright=False, labeltop=False, labelbottom=True)
    
    plt.tight_layout()
    save_png(fig, out_path, config.dpi)


def plot_positive_delta_cells(
    metrics_df: pd.DataFrame,
    config: DividedCorrelationV4Config,
    out_path: Path,
) -> None:
    """Plot land-cell counts with positive delta correlation above each threshold."""
    metrics_df = sort_splits_for_plotting(metrics_df)
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics_df))
    for thresh in config.changed_thresholds:
        thresh_key = f"{int(thresh * 100):03d}"
        col = f"n_delta_r_gt_{thresh_key}"
        if col in metrics_df.columns:
            ax.plot(x, metrics_df[col].values, marker="o", linewidth=2, label=f"Δr > {thresh}")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["split_id"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_xlabel("Split Configuration")
    ax.set_ylabel("Number of Land Cells")
    ax.set_title("Positive Delta Correlation Cells")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(
        left=True,
        right=True,
        top=True,
        bottom=True,
        labelleft=True,
        labelright=False,
        labeltop=False,
        labelbottom=True,
    )

    plt.tight_layout()
    save_png(fig, out_path, config.dpi)


def plot_strengthen_weaken_balance(
    metrics_df: pd.DataFrame,
    config: DividedCorrelationV4Config,
    out_path: Path,
) -> None:
    metrics_df = sort_splits_for_plotting(metrics_df)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_df))
    width = 0.35
    
    ax.bar(x - width / 2, metrics_df["n_strengthen"].values, width, label="Strengthening", color="forestgreen")
    ax.bar(x + width / 2, metrics_df["n_weaken"].values, width, label="Weakening", color="firebrick")
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["split_id"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_xlabel("Split Configuration")
    ax.set_ylabel("Number of Land Cells")
    ax.set_title(f"Strengthening vs Weakening (threshold: {config.strengthen_threshold})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(left=True, right=True, top=True, bottom=True,
                   labelleft=True, labelright=False, labeltop=False, labelbottom=True)
    
    plt.tight_layout()
    save_png(fig, out_path, config.dpi)


def plot_sign_flip_count(
    metrics_df: pd.DataFrame,
    config: DividedCorrelationV4Config,
    out_path: Path,
) -> None:
    metrics_df = sort_splits_for_plotting(metrics_df)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_df))
    ax.bar(x, metrics_df["n_sign_flip"].values, color="purple", alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["split_id"].astype(str).tolist(), rotation=45, ha="right")
    ax.set_xlabel("Split Configuration")
    ax.set_ylabel("Number of Sign-Flip Cells")
    ax.set_title(f"Sign Flips (|r| ≥ {config.sign_flip_min_abs_r} in both periods)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(left=True, right=True, top=True, bottom=True,
                   labelleft=True, labelright=False, labeltop=False, labelbottom=True)
    
    plt.tight_layout()
    save_png(fig, out_path, config.dpi)


def plot_hotspot_location_map(
    hotspot_df: pd.DataFrame,
    land_mask: xr.DataArray,
    config: DividedCorrelationV4Config,
    out_path: Path,
    title: str = "Selected Hotspots",
) -> None:
    extent = (config.lon_min, config.lon_max, config.lat_min, config.lat_max)
    
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    
    land_mask.astype(int).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Greens",
        alpha=0.3,
        add_colorbar=False,
    )
    
    for _, row in hotspot_df.iterrows():
        color = "red" if row["delta_r"] > 0 else "blue"
        ax.scatter(row["lon"], row["lat"], s=100, c=color, edgecolor="black",
                   transform=ccrs.PlateCarree(), zorder=10)
        ax.text(row["lon"] + 0.5, row["lat"] + 0.5, str(row["hotspot_id"]),
                fontsize=9, fontweight="bold", transform=ccrs.PlateCarree())
    
    _add_map_context(ax, extent)
    ax.set_title(title)
    
    save_png(fig, out_path, config.dpi)


def plot_hotspot_running_by_metric(
    running_dict: dict[int, dict[int, pd.DataFrame]],
    hotspot_df: pd.DataFrame,
    window: int,
    config: DividedCorrelationV4Config,
    out_dir: Path,
) -> None:
    """
    Plot running statistics organized by metric.
    One plot per metric, all 6+ hotspots on same plot.
    """
    metrics = ["correlation", "rain_variance", "nino_variance", "covariance"]
    metric_labels = {
        "correlation": "Running Correlation",
        "rain_variance": "Running Rainfall Variance",
        "nino_variance": "Running Niño 3.4 Variance",
        "covariance": "Running Covariance",
    }
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for _, row in hotspot_df.iterrows():
            hotspot_id = row["hotspot_id"]
            df = running_dict[hotspot_id][window]
            label = f"HS{hotspot_id} ({row['lat']:.1f}°, {row['lon']:.1f}°)"
            ax.plot(df["year"], df[metric], marker=".", linewidth=1.5, markersize=4, label=label, alpha=0.8)
        
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("DJF Year (center of window)")
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(f"{metric_labels[metric]} (window={window})")
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.tick_params(left=True, right=True, top=True, bottom=True,
                       labelleft=True, labelright=False, labeltop=False, labelbottom=True)
        
        plt.tight_layout()
        out_path = out_dir / f"dcorr_v4_djf_running_{metric}_w{window}.png"
        save_png(fig, out_path, config.dpi)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================


def run_workflow(config: DividedCorrelationV4Config) -> dict[str, object]:
    """Execute the complete divided correlation V4 workflow."""
    import datetime
    
    print("=" * 60)
    print("DIVIDED CORRELATION V4 WORKFLOW")
    print("=" * 60)
    
    run_start = datetime.datetime.now()
    
    # Step 1: Setup
    print("\n[1/13] Creating output directories...")
    config.ensure_output_dirs()
    
    # Step 2: Load data
    print("\n[2/13] Loading data...")
    print(f"  MSWEP: {config.mswep_monthly_path}")
    print(f"  Niño 3.4: {config.nino34_monthly_path}")
    
    rain_monthly = load_mswep_monthly(config.mswep_monthly_path)
    rain_monthly = subset_domain(
        rain_monthly,
        config.lon_min, config.lon_max,
        config.lat_min, config.lat_max,
    )
    nino_monthly = load_nino34_monthly(config.nino34_monthly_path)
    
    print(f"  Rain monthly shape: {rain_monthly.shape}")
    print(f"  Niño monthly range: {nino_monthly.index[0]} to {nino_monthly.index[-1]}")
    
    # Step 3: Build DJF fields
    print("\n[3/13] Building DJF fields...")
    djf_rain_raw = build_complete_djf_field(rain_monthly, config.start_year, config.end_year)
    djf_nino_raw = build_complete_djf_series(nino_monthly, config.start_year, config.end_year)
    
    validate_djf_year_axis(djf_rain_raw.year.values, config.start_year, config.end_year)
    validate_djf_year_axis(djf_nino_raw.index.values, config.start_year, config.end_year)
    
    print(f"  DJF rain shape: {djf_rain_raw.shape}")
    print(f"  DJF years: {config.start_year}-{config.end_year} ({config.n_years} years)")
    
    export_gridded_outputs(
        djf_rain_raw.to_dataset(name="djf_rain"),
        config.intermediate_dir / f"djf_rain_raw_{config.start_year}_{config.end_year}.nc",
    )
    
    # Step 4: Anomalies and detrending
    print("\n[4/13] Computing anomalies and detrending...")
    rain_clim = compute_rain_climatology(djf_rain_raw, config.clim_start, config.clim_end)
    nino_clim = compute_series_climatology(djf_nino_raw, config.clim_start, config.clim_end)
    
    djf_rain_anom = compute_djf_anomalies(djf_rain_raw, rain_clim)
    djf_nino_anom = compute_series_anomalies(djf_nino_raw, nino_clim)
    
    djf_rain = linear_detrend_field_along_year(djf_rain_anom)
    djf_nino = detrend_series(djf_nino_anom)
    
    print(f"  Climatology period: {config.clim_start}-{config.clim_end}")
    
    export_gridded_outputs(
        djf_rain.to_dataset(name="djf_rain_anom_detrended"),
        config.intermediate_dir / f"djf_rain_anom_detrended_{config.start_year}_{config.end_year}.nc",
    )
    
    # Step 5: Land mask (ALL land in domain)
    print("\n[5/13] Building land mask (all Maritime Continent land)...")
    valid_mask = build_valid_all_years_mask(djf_rain)
    land_mask = build_land_mask(djf_rain.lat.values, djf_rain.lon.values)
    analysis_mask = build_analysis_valid_land_mask(land_mask, valid_mask)
    ocean_mask = build_analysis_valid_ocean_mask(land_mask, valid_mask)
    all_mask = build_analysis_valid_all_mask(valid_mask)
    
    n_land = int(analysis_mask.sum().values)
    n_ocean = int(ocean_mask.sum().values)
    n_all = int(all_mask.sum().values)
    print(f"  Land cells: {n_land}")
    print(f"  Ocean cells: {n_ocean}")
    print(f"  All valid cells: {n_all}")
    
    export_gridded_outputs(land_mask.to_dataset(name="land_mask"), config.intermediate_dir / "mc_land_mask.nc")
    export_gridded_outputs(analysis_mask.to_dataset(name="analysis_valid_land_mask"), config.intermediate_dir / "analysis_valid_land_mask.nc")
    export_gridded_outputs(ocean_mask.to_dataset(name="analysis_valid_ocean_mask"), config.intermediate_dir / "analysis_valid_ocean_mask.nc")
    export_gridded_outputs(all_mask.to_dataset(name="analysis_valid_all_mask"), config.intermediate_dir / "analysis_valid_all_mask.nc")
    
    # Step 6: Split periods
    print("\n[6/13] Generating split periods...")
    split_max_years = min(config.split_max_years, config.n_years - config.split_min_years)
    if split_max_years < config.split_min_years:
        raise ValueError(
            f"Invalid split bounds: min_p1_years={config.split_min_years}, "
            f"max_p1_years={split_max_years}, n_years={config.n_years}"
        )
    split_df = generate_split_periods(config.start_year, config.end_year, config.split_min_years, split_max_years)
    print(f"  Number of splits: {len(split_df)}")
    
    # Step 7: Full-period maps
    print("\n[7/13] Computing full-period correlation maps...")
    fullperiod_ds = compute_fullperiod_maps(djf_rain, djf_nino)
    export_gridded_outputs(fullperiod_ds, config.processed_dir / "dcorr_v4_djf_fullperiod_maps.nc")
    
    # Step 8: Split maps
    print("\n[8/13] Computing split correlation maps...")
    split_ds = compute_split_delta_maps(djf_rain, djf_nino, split_df)
    split_ds = split_ds.assign_coords(split=np.asarray(split_df["split_id"].map(str).tolist(), dtype=object))
    export_gridded_outputs(split_ds, config.processed_dir / "dcorr_v4_djf_split_maps.nc")
    
    # Step 9: Sensitivity metrics
    print("\n[9/13] Computing sensitivity metrics...")
    metrics_df = summarize_split_metrics(
        split_ds, split_df, analysis_mask, ocean_mask, all_mask,
        config.changed_thresholds,
        config.sign_flip_min_abs_r,
        config.strengthen_threshold,
        config.weaken_threshold,
    )
    metrics_df = rank_splits(metrics_df)
    
    export_summary_tables(metrics_df, config.tables_dir / f"dcorr_v4_djf_split_metrics_{config.start_year}_{config.end_year}.csv")
    export_summary_tables(split_df, config.tables_dir / f"dcorr_v4_djf_split_periods_{config.start_year}_{config.end_year}.csv")
    
    # Step 10: Hotspots
    print("\n[10/13] Selecting hotspot cells...")
    best_split = metrics_df.loc[metrics_df["mean_abs_delta_r"].idxmax(), "split_id"]
    hotspot_df = select_hotspot_cells(split_ds, analysis_mask, best_split, config.n_hotspots, config.hotspot_min_separation_deg)
    print(f"  Selected {len(hotspot_df)} hotspots from split {best_split}")
    
    export_summary_tables(hotspot_df, config.tables_dir / f"dcorr_v4_djf_hotspots_{config.start_year}_{config.end_year}.csv")
    
    # Step 11: Running diagnostics
    print("\n[11/13] Computing running diagnostics...")
    running_stats = compute_running_stats_for_hotspots(djf_rain, djf_nino, hotspot_df, config.running_windows)
    
    # Step 12: Generate plots
    print("\n[12/13] Generating plots...")
    
    # Baseline
    plot_correlation_map(fullperiod_ds["correlation"], config,
                         config.png_baseline_dir / f"dcorr_v4_djf_{config.start_year}_{config.end_year}_fullperiod_correlation.png",
                         f"DJF {config.start_year}-{config.end_year} Full Period Correlation")
    plot_correlation_map(fullperiod_ds["regression_slope"], config,
                         config.png_baseline_dir / f"dcorr_v4_djf_{config.start_year}_{config.end_year}_fullperiod_slope.png",
                         f"DJF {config.start_year}-{config.end_year} Full Period Regression Slope")
    plot_land_mask(analysis_mask, config, config.png_baseline_dir / f"dcorr_v4_djf_{config.start_year}_{config.end_year}_land_mask.png")
    plot_split_timeline(split_df, config, config.png_baseline_dir / f"dcorr_v4_djf_{config.start_year}_{config.end_year}_split_timeline.png")
    
    # Per-split 2x2
    print("  Generating per-split maps...")
    for _, row in split_df.iterrows():
        split_id = row["split_id"]
        p1_label = f"P1: {row['p1_start']}-{row['p1_end']}"
        p2_label = f"P2: {row['p2_start']}-{row['p2_end']}"
        out_path = config.png_splits_dir / f"dcorr_v4_djf_split_{split_id}_2x2.png"
        plot_split_2x2(split_ds, split_id, config, out_path, p1_label, p2_label)
    
    # Atlas
    print("  Generating atlas plots...")
    plot_split_atlas(split_ds, split_df, config, config.png_atlas_dir / "dcorr_v4_djf_atlas_delta_corr.png", metric="delta_corr")
    plot_split_atlas(split_ds, split_df, config, config.png_atlas_dir / "dcorr_v4_djf_atlas_corr_p1.png", metric="corr_p1")
    plot_split_atlas(split_ds, split_df, config, config.png_atlas_dir / "dcorr_v4_djf_atlas_corr_p2.png", metric="corr_p2")
    
    # Summary
    print("  Generating summary plots...")
    plot_split_metric_lines(metrics_df, config, config.png_summaries_dir / "dcorr_v4_djf_sensitivity_mean_abs_delta_r.png",
                            y_col="mean_abs_delta_r", y_label="Mean |Δr| over land")
    plot_split_changed_cells(
        metrics_df,
        config,
        config.png_summaries_dir / "dcorr_v4_djf_sensitivity_changed_cells.png",
        cell_label="Land",
    )
    plot_split_changed_cells(
        metrics_df,
        config,
        config.png_summaries_dir / "dcorr_v4_djf_sensitivity_changed_cells_ocean.png",
        column_suffix="_ocean",
        cell_label="Ocean",
    )
    plot_split_changed_cells(
        metrics_df,
        config,
        config.png_summaries_dir / "dcorr_v4_djf_sensitivity_changed_cells_all_grid.png",
        column_suffix="_all",
        cell_label="All Grid",
    )
    plot_positive_delta_cells(metrics_df, config, config.png_summaries_dir / "dcorr_v4_djf_sensitivity_positive_delta_cells.png")
    plot_strengthen_weaken_balance(metrics_df, config, config.png_summaries_dir / "dcorr_v4_djf_sensitivity_strengthen_weaken.png")
    plot_sign_flip_count(metrics_df, config, config.png_summaries_dir / "dcorr_v4_djf_sensitivity_sign_flips.png")
    
    # Hotspots and running
    if len(hotspot_df) > 0:
        plot_hotspot_location_map(hotspot_df, analysis_mask, config,
                                  config.png_summaries_dir / "dcorr_v4_djf_hotspots_map.png",
                                  title=f"Hotspots (from split {best_split})")
        for window in config.running_windows:
            plot_hotspot_running_by_metric(running_stats, hotspot_df, window, config, config.png_running_dir)
    
    # Step 13: Write manifest
    print("\n[13/13] Writing run manifest...")
    run_end = datetime.datetime.now()
    
    manifest = {
        "config": config.to_manifest_dict(),
        "run_start": run_start.isoformat(),
        "run_end": run_end.isoformat(),
        "run_duration_seconds": (run_end - run_start).total_seconds(),
        "n_splits": len(split_df),
        "n_land_cells": n_land,
        "n_ocean_cells": n_ocean,
        "n_all_cells": n_all,
        "n_hotspots": len(hotspot_df),
        "best_split_for_hotspots": best_split,
    }
    
    write_run_manifest(manifest, config.logs_dir / f"dcorr_v4_run_{run_start.strftime('%Y%m%d_%H%M%S')}.json")
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"Duration: {(run_end - run_start).total_seconds():.1f} seconds")
    print(f"Results: {config.results_base_dir}")
    
    return manifest
