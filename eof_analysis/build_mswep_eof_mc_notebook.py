from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path("/Users/rizzie/TugasAkhir")
PROJECT_DIR = ROOT / "notebook" / "eof-mc_v1"
NOTEBOOK_PATH = PROJECT_DIR / "mswep_eof_djf_wmc_emc_comprehensive.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # MSWEP DJF EOF for the Maritime Continent

        This notebook rebuilds the logic of the source EOF notebook into a more complete DJF workflow for two Maritime Continent subregions:

        - **West Maritime Continent (WMC)**: lon `92.5-120.0`, lat `-12.5 to 12.5`
        - **East Maritime Continent (EMC)**: lon `120.0-152.5`, lat `-12.5 to 12.5`

        The workflow keeps the main source-notebook ideas:

        - build complete DJF seasonal means
        - subset the analysis by region
        - use PCA on the DJF anomaly matrix as the EOF solver
        - apply a simple sign convention so EOF signs are stable

        The requested changes are applied here:

        - only **Nino3.4** is used
        - **DMI is removed**
        - rainfall is **linearly detrended before EOF**
        - only **EOF1** and **EOF2** are retained
        - every analysis block has its own markdown explanation
        - every figure is saved to the local output folder as **PNG**
        """
    ),
    md(
        """
        ## Environment Setup

        This first code block imports the packages, sets plotting defaults, and defines a writable Matplotlib cache inside the project folder. The font sizes follow the requested rule:

        - figure titles: `12`
        - colorbar labels: `14`
        - map longitude and latitude labels: `14`
        - x and y labels: `14`
        """
    ),
    code(
        """
        import os
        from pathlib import Path

        PROJECT_DIR = Path("/Users/rizzie/TugasAkhir/notebook/eof-mc_v1")
        MPLCONFIGDIR = PROJECT_DIR / ".mplconfig"
        MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
        os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_DIR / ".cache"))
        Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

        import numpy as np
        import pandas as pd
        import xarray as xr
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import pycwt as wavelet

        from scipy import signal
        from sklearn.decomposition import PCA

        if not hasattr(np, "int"):
            np.int = int

        TITLE_FONTSIZE = 12
        LABEL_FONTSIZE = 14
        TICK_FONTSIZE = 12
        LEGEND_FONTSIZE = 11
        DPI = 180

        plt.rcParams.update(
            {
                "axes.titlesize": TITLE_FONTSIZE,
                "axes.labelsize": LABEL_FONTSIZE,
                "xtick.labelsize": TICK_FONTSIZE,
                "ytick.labelsize": TICK_FONTSIZE,
                "legend.fontsize": LEGEND_FONTSIZE,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            }
        )
        """
    ),
    md(
        """
        ## Configuration

        This section defines the file paths, the study boxes, the larger map extent for the overview figures, the running window, and the output folders. The notebook uses the local monthly MSWEP archive rather than the incomplete `mswep.nc` shortcut file.
        """
    ),
    code(
        """
        ROOT = Path("/Users/rizzie/TugasAkhir")
        DATA_MSWEP_COMBINED = ROOT / "data" / "mswep-monthly" / "mswep_monthly_combined.nc"
        NINO34_PATH = ROOT / "data" / "index" / "nino34.anom.csv"

        OUT_ROOT = PROJECT_DIR / "output"
        OUT_PNG = OUT_ROOT / "png"
        OUT_TABLES = OUT_ROOT / "tables"
        for directory in (OUT_ROOT, OUT_PNG, OUT_TABLES):
            directory.mkdir(parents=True, exist_ok=True)

        CLIM_BASELINE_START = 1991
        CLIM_BASELINE_END = 2020
        RUNNING_WINDOW = 15
        EOF_NMODES = 2
        MAP_EXTENT = [85.0, 160.0, -20.0, 20.0]
        DATA_EXTENT = [80.0, 160.0, -25.0, 25.0]

        REGIONS = {
            "wmc": {
                "label": "West Maritime Continent (WMC)",
                "short": "WMC",
                "lon_min": 92.5,
                "lon_max": 120.0,
                "lat_min": -12.5,
                "lat_max": 12.5,
                "color": "tab:blue",
            },
            "emc": {
                "label": "East Maritime Continent (EMC)",
                "short": "EMC",
                "lon_min": 120.0,
                "lon_max": 152.5,
                "lat_min": -12.5,
                "lat_max": 12.5,
                "color": "tab:green",
            },
        }

        NINO_COLOR = "tab:red"
        PC_COLORS = {"PC1": "tab:purple", "PC2": "tab:orange"}

        print("Rainfall combined file:", DATA_MSWEP_COMBINED)
        print("Nino3.4 file:", NINO34_PATH)
        print("Output PNG folder:", OUT_PNG)
        print("Output table folder:", OUT_TABLES)
        """
    ),
    md(
        """
        ## Helper Functions

        These helper functions are the reusable core of the notebook:

        - harmonize the rainfall grid and coordinates
        - build complete DJF seasonal means
        - read the monthly Nino3.4 file
        - subset WMC and EMC consistently
        - linearly detrend rainfall while preserving the mean level
        - compute the two-mode EOF solution
        - compute running statistics and running correlations
        - compute wavelet coherence with `pycwt`
        - save figures and exported tables

        The EOF implementation intentionally stays close to the source notebook logic by using PCA on the reshaped DJF anomaly matrix.
        """
    ),
    code(
        """
        def slugify(text: str) -> str:
            return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


        def save_figure(fig: plt.Figure, filename: str) -> Path:
            out_path = OUT_PNG / filename
            fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
            plt.show()
            plt.close(fig)
            print("Saved:", out_path)
            return out_path


        def add_map_features(ax, extent, draw_labels=True):
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.coastlines(resolution="10m", linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4)
            gl = ax.gridlines(
                draw_labels=draw_labels,
                linewidth=0.4,
                color="gray",
                alpha=0.5,
                linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False
            if draw_labels:
                gl.xlabel_style = {"size": LABEL_FONTSIZE}
                gl.ylabel_style = {"size": LABEL_FONTSIZE}
            return gl


        def draw_region_boxes(ax):
            for region in REGIONS.values():
                width = region["lon_max"] - region["lon_min"]
                height = region["lat_max"] - region["lat_min"]
                rect = mpatches.Rectangle(
                    (region["lon_min"], region["lat_min"]),
                    width,
                    height,
                    facecolor=region["color"],
                    edgecolor=region["color"],
                    linewidth=2.0,
                    alpha=0.18,
                    transform=ccrs.PlateCarree(),
                )
                ax.add_patch(rect)
                ax.text(
                    region["lon_min"] + 0.6,
                    region["lat_max"] + 0.6,
                    region["short"],
                    color=region["color"],
                    fontsize=LABEL_FONTSIZE,
                    fontweight="bold",
                    transform=ccrs.PlateCarree(),
                )


        def build_complete_djf(da: xr.DataArray) -> xr.DataArray:
            month = da["time"].dt.month
            is_djf = month.isin([12, 1, 2])
            da_djf = da.sel(time=is_djf)

            djf_year = xr.where(
                da_djf["time"].dt.month == 12,
                da_djf["time"].dt.year + 1,
                da_djf["time"].dt.year,
            )
            da_djf = da_djf.assign_coords(djf_year=("time", djf_year.data))

            month_count = da_djf["time"].groupby("djf_year").count()
            full_years = month_count["djf_year"].where(month_count == 3, drop=True).values.astype(int)

            djf = da_djf.groupby("djf_year").mean("time").sel(djf_year=full_years)
            djf = djf.rename({"djf_year": "year"}).sortby("year")
            return djf


        def read_monthly_index(csv_path: Path, value_name: str) -> xr.DataArray:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]

            date_col = df.columns[0]
            value_col = df.columns[1]

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
            df.loc[df[value_col] <= -999, value_col] = np.nan
            df = df.dropna(subset=[date_col, value_col]).copy()

            return xr.DataArray(
                df[value_col].values,
                dims=("time",),
                coords={"time": pd.DatetimeIndex(df[date_col].values)},
                name=value_name,
            )


        def subset_region(da: xr.DataArray, region: dict) -> xr.DataArray:
            lat_slice = (
                slice(region["lat_max"], region["lat_min"])
                if float(da["lat"].values[0]) > float(da["lat"].values[-1])
                else slice(region["lat_min"], region["lat_max"])
            )
            return da.sel(
                lon=slice(region["lon_min"], region["lon_max"]),
                lat=lat_slice,
            )


        def linear_detrend_series(series: pd.Series):
            values = series.to_numpy(dtype=float)
            x = np.arange(len(values), dtype=float)
            mask = np.isfinite(values)
            if mask.sum() < 2:
                raise ValueError("Need at least two finite values to detrend a series")

            slope, intercept = np.polyfit(x[mask], values[mask], 1)
            trend = slope * x + intercept
            detrended = values.copy()
            detrended[mask] = values[mask] - trend[mask] + np.nanmean(values[mask])
            detrended[~mask] = np.nan

            return (
                pd.Series(detrended, index=series.index, name=f"{series.name}_detrended"),
                pd.Series(trend, index=series.index, name=f"{series.name}_trend"),
                slope,
                intercept,
            )


        def detrend_field_preserve_mean(da: xr.DataArray) -> xr.DataArray:
            stacked = da.stack(point=("lat", "lon")).transpose("year", "point")
            arr = np.asarray(stacked.values, dtype=float)
            out = np.full_like(arr, np.nan, dtype=float)

            for idx in range(arr.shape[1]):
                col = arr[:, idx]
                mask = np.isfinite(col)
                if mask.sum() < 2:
                    continue
                x = np.arange(mask.sum(), dtype=float)
                slope, intercept = np.polyfit(x, col[mask], 1)
                trend = slope * x + intercept
                adjusted = col.copy()
                adjusted[mask] = col[mask] - trend + np.nanmean(col[mask])
                out[:, idx] = adjusted

            detrended = xr.DataArray(out, coords=stacked.coords, dims=stacked.dims)
            detrended = detrended.unstack("point").transpose("year", "lat", "lon")
            detrended.name = f"{da.name}_detrended" if da.name else "detrended"
            return detrended


        def apply_sign_convention(components: np.ndarray, scores: np.ndarray):
            comp = components.copy()
            pcs = scores.copy()
            for mode in range(comp.shape[0]):
                anchor = int(np.nanargmax(np.abs(comp[mode, :])))
                if comp[mode, anchor] < 0:
                    comp[mode, :] *= -1.0
                    pcs[:, mode] *= -1.0
            return comp, pcs


        def compute_eof(anom: xr.DataArray, n_modes: int = 2):
            years = pd.Index(anom["year"].values.astype(int), name="year")
            arr = np.asarray(anom.values, dtype=float).reshape(len(years), -1)
            valid_mask = np.all(np.isfinite(arr), axis=0)
            x_valid = arr[:, valid_mask]
            x_centered = x_valid - x_valid.mean(axis=0, keepdims=True)

            pca = PCA(n_components=n_modes, svd_solver="full")
            pcs = pca.fit_transform(x_centered)
            components, pcs = apply_sign_convention(pca.components_, pcs)

            eof_flat = np.full((n_modes, arr.shape[1]), np.nan, dtype=float)
            eof_flat[:, valid_mask] = components
            eof_maps = eof_flat.reshape(n_modes, len(anom["lat"]), len(anom["lon"]))

            eof_da = xr.DataArray(
                eof_maps,
                dims=("mode", "lat", "lon"),
                coords={
                    "mode": np.arange(1, n_modes + 1),
                    "lat": anom["lat"],
                    "lon": anom["lon"],
                },
                name="eof_loading",
            )
            pc_df = pd.DataFrame(
                pcs,
                index=years,
                columns=[f"PC{i}" for i in range(1, n_modes + 1)],
            )
            variance = pd.DataFrame(
                {
                    "mode": np.arange(1, n_modes + 1),
                    "explained_variance_ratio": pca.explained_variance_ratio_,
                    "explained_variance_percent": pca.explained_variance_ratio_ * 100.0,
                    "cumulative_percent": np.cumsum(pca.explained_variance_ratio_) * 100.0,
                }
            )
            return {
                "eof_da": eof_da,
                "pc_df": pc_df,
                "variance": variance,
                "valid_mask": valid_mask,
            }


        def zscore(series: pd.Series) -> pd.Series:
            std = series.std(ddof=0)
            if std == 0 or np.isnan(std):
                return pd.Series(np.nan, index=series.index, name=series.name)
            return (series - series.mean()) / std


        def compute_running_stats(rain_series: pd.Series, nino_series: pd.Series, window: int) -> pd.DataFrame:
            pair = pd.concat(
                [rain_series.rename("rain"), nino_series.rename("nino34")],
                axis=1,
            ).dropna()
            pair["rain_z"] = zscore(pair["rain"])
            pair["nino34_z"] = zscore(pair["nino34"])

            if len(pair) < window:
                raise ValueError(f"Need at least {window} paired years for running statistics")

            rows = []
            for start in range(len(pair) - window + 1):
                chunk = pair.iloc[start : start + window]
                midpoint = int(chunk.index[window // 2])
                rows.append(
                    {
                        "mid_year": midpoint,
                        "rain_var": chunk["rain_z"].var(ddof=0),
                        "nino34_var": chunk["nino34_z"].var(ddof=0),
                        "rain_nino34_cov": chunk["rain_z"].cov(chunk["nino34_z"], ddof=0),
                    }
                )
            return pd.DataFrame(rows)


        def compute_running_correlation(series_a: pd.Series, series_b: pd.Series, window: int) -> pd.DataFrame:
            pair = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1).dropna()
            if len(pair) < window:
                raise ValueError(f"Need at least {window} paired years for running correlation")

            rows = []
            for start in range(len(pair) - window + 1):
                chunk = pair.iloc[start : start + window]
                rows.append(
                    {
                        "mid_year": int(chunk.index[window // 2]),
                        "correlation": chunk["a"].corr(chunk["b"]),
                    }
                )
            return pd.DataFrame(rows)


        def compute_wavelet(series_a: pd.Series, series_b: pd.Series) -> dict:
            pair = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1).dropna()
            years = pair.index.to_numpy(dtype=float)
            a = zscore(pair["a"]).to_numpy(dtype=float)
            b = zscore(pair["b"]).to_numpy(dtype=float)
            dt = 1.0

            try:
                wct, phase, coi, freq, signif = wavelet.wct(
                    a,
                    b,
                    dt,
                    dj=1 / 12,
                    s0=2 * dt,
                    J=-1,
                    significance_level=0.95,
                    wavelet="morlet",
                    normalize=True,
                )
            except TypeError:
                wct, phase, coi, freq, signif = wavelet.wct(a, b, dt)

            return {
                "years": years,
                "wct": np.abs(wct),
                "phase": phase,
                "coi": coi,
                "freq": freq,
                "period": 1.0 / freq,
                "signif": signif,
            }


        def get_climatology_years(years: pd.Index) -> pd.Index:
            years = pd.Index(years.astype(int), name="year")
            clim_years = years[(years >= CLIM_BASELINE_START) & (years <= CLIM_BASELINE_END)]
            if len(clim_years) == 0:
                clim_years = years
            return clim_years


        def save_region_tables(result: dict):
            prefix = slugify(result["region"]["short"])
            result["series_table"].to_csv(OUT_TABLES / f"{prefix}_djf_series.csv", index=False)
            result["pc_nino_table"].to_csv(OUT_TABLES / f"{prefix}_pc_nino34.csv", index=False)
            result["variance_table"].to_csv(OUT_TABLES / f"{prefix}_eof_variance.csv", index=False)
            result["running_stats"].to_csv(OUT_TABLES / f"{prefix}_running_stats_window{RUNNING_WINDOW}.csv", index=False)
            result["running_corr_pc1"].to_csv(OUT_TABLES / f"{prefix}_running_corr_pc1_window{RUNNING_WINDOW}.csv", index=False)
            result["running_corr_pc2"].to_csv(OUT_TABLES / f"{prefix}_running_corr_pc2_window{RUNNING_WINDOW}.csv", index=False)


        def analyze_region(region_key: str, region: dict, pr_djf: xr.DataArray, nino_djf: pd.Series, clim_years: pd.Index) -> dict:
            pr_region = subset_region(pr_djf, region)
            rain_raw = pr_region.mean(dim=("lat", "lon"), skipna=True).to_pandas()
            rain_raw.name = f"{region['short']}_rain_raw_mm"
            rain_raw.index = pd.Index(rain_raw.index.astype(int), name="year")

            rain_detrended, rain_trend, slope, intercept = linear_detrend_series(rain_raw)
            pr_detrended = detrend_field_preserve_mean(pr_region)

            clim = pr_detrended.sel(year=clim_years).mean("year")
            anom = pr_detrended - clim
            eof_result = compute_eof(anom, n_modes=EOF_NMODES)

            pc_df = eof_result["pc_df"].copy()
            variance_table = eof_result["variance"].copy()
            pc_nino_table = pc_df.join(nino_djf.rename("nino34_djf"), how="inner")

            running_stats = compute_running_stats(rain_detrended, nino_djf, RUNNING_WINDOW)
            running_corr_pc1 = compute_running_correlation(pc_nino_table["PC1"], pc_nino_table["nino34_djf"], RUNNING_WINDOW)
            running_corr_pc2 = compute_running_correlation(pc_nino_table["PC2"], pc_nino_table["nino34_djf"], RUNNING_WINDOW)

            series_table = pd.DataFrame(
                {
                    "year": rain_raw.index.astype(int),
                    "rain_raw_mm": rain_raw.to_numpy(dtype=float),
                    "rain_trend_mm": rain_trend.to_numpy(dtype=float),
                    "rain_detrended_mm": rain_detrended.to_numpy(dtype=float),
                    "nino34_djf": nino_djf.reindex(rain_raw.index).to_numpy(dtype=float),
                }
            )

            return {
                "region_key": region_key,
                "region": region,
                "pr_region_djf": pr_region,
                "pr_region_detrended": pr_detrended,
                "rain_raw": rain_raw,
                "rain_trend": rain_trend,
                "rain_detrended": rain_detrended,
                "nino_djf": nino_djf,
                "anom": anom,
                "eof_da": eof_result["eof_da"],
                "pc_df": pc_df,
                "variance_table": variance_table,
                "pc_nino_table": pc_nino_table.reset_index(),
                "series_table": series_table,
                "running_stats": running_stats,
                "running_corr_pc1": running_corr_pc1,
                "running_corr_pc2": running_corr_pc2,
                "slope_per_index_step": slope,
                "intercept": intercept,
                "mean_spatial_field": pr_region.mean("year"),
                "overall_rain_nino_corr": rain_raw.corr(nino_djf.reindex(rain_raw.index)),
                "summary": {
                    "region": region["label"],
                    "n_years": int(len(rain_raw)),
                    "slope_per_step": float(slope),
                    "overall_rain_nino_corr": float(rain_raw.corr(nino_djf.reindex(rain_raw.index))),
                },
            }


        def plot_region_definition_map():
            fig = plt.figure(figsize=(10, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            add_map_features(ax, MAP_EXTENT, draw_labels=True)
            draw_region_boxes(ax)
            ax.set_title("Study regions: WMC and EMC", fontsize=TITLE_FONTSIZE)
            save_figure(fig, "01_region_definition_map.png")


        def plot_rainfall_timeseries(regional_series: dict):
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
            for ax, key in zip(axes, ["wmc", "emc"]):
                series = regional_series[key]
                region = REGIONS[key]
                ax.plot(series.index, series.values, color=region["color"], linewidth=2.0)
                ax.set_ylabel("Rainfall (mm)", fontsize=LABEL_FONTSIZE)
                ax.set_title(f"{region['label']} DJF rainfall time series", fontsize=TITLE_FONTSIZE)
                ax.grid(True, linestyle="--", alpha=0.4)
            axes[-1].set_xlabel("DJF year", fontsize=LABEL_FONTSIZE)
            save_figure(fig, "02_djf_rainfall_timeseries.png")


        def plot_nino_timeseries(nino_djf: pd.Series):
            fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
            ax.plot(nino_djf.index, nino_djf.values, color=NINO_COLOR, linewidth=2.0)
            ax.axhline(0.0, color="0.3", linewidth=0.9, linestyle="--")
            ax.set_title(
                "Nino3.4 DJF time series (common ENSO index used for both WMC and EMC)",
                fontsize=TITLE_FONTSIZE,
            )
            ax.set_xlabel("DJF year", fontsize=LABEL_FONTSIZE)
            ax.set_ylabel("Nino3.4 anomaly", fontsize=LABEL_FONTSIZE)
            ax.grid(True, linestyle="--", alpha=0.4)
            save_figure(fig, "03_nino34_djf_timeseries.png")


        def plot_rain_vs_nino(regional_series: dict, nino_djf: pd.Series):
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
            for ax, key in zip(axes, ["wmc", "emc"]):
                region = REGIONS[key]
                rain = regional_series[key]
                pair = pd.concat([rain.rename("rain"), nino_djf.rename("nino34")], axis=1).dropna()
                corr = pair["rain"].corr(pair["nino34"])
                ax2 = ax.twinx()
                ax.plot(pair.index, pair["rain"], color=region["color"], linewidth=2.0, label=f"{region['short']} rain")
                ax2.plot(pair.index, pair["nino34"], color=NINO_COLOR, linewidth=1.8, linestyle="--", label="Nino3.4")
                ax.set_ylabel("Rainfall (mm)", fontsize=LABEL_FONTSIZE, color=region["color"])
                ax2.set_ylabel("Nino3.4 anomaly", fontsize=LABEL_FONTSIZE, color=NINO_COLOR)
                ax.set_title(
                    f"{region['label']}: raw DJF rainfall vs raw Nino3.4 | r = {corr:.3f}",
                    fontsize=TITLE_FONTSIZE,
                )
                ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
                ax2.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
                ax.grid(True, linestyle="--", alpha=0.35)
            axes[-1].set_xlabel("DJF year", fontsize=LABEL_FONTSIZE)
            save_figure(fig, "04_djf_rainfall_vs_nino34_raw.png")


        def plot_spatial_mean_djf_rainfall(pr_djf: xr.DataArray):
            mean_field = pr_djf.mean("year")
            fig = plt.figure(figsize=(10, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            add_map_features(ax, MAP_EXTENT, draw_labels=True)
            mesh = ax.pcolormesh(
                mean_field["lon"],
                mean_field["lat"],
                mean_field,
                cmap="Blues",
                shading="auto",
                transform=ccrs.PlateCarree(),
            )
            draw_region_boxes(ax)
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.88)
            cbar.set_label("DJF rainfall (mm)", fontsize=LABEL_FONTSIZE)
            ax.set_title("Mean DJF rainfall over all available years", fontsize=TITLE_FONTSIZE)
            save_figure(fig, "05_spatial_mean_djf_rainfall.png")


        def plot_detrend_comparison(result: dict):
            region = result["region"]
            fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
            ax.plot(result["rain_raw"].index, result["rain_raw"].values, color=region["color"], linewidth=2.0, label="Raw DJF rainfall")
            ax.plot(result["rain_detrended"].index, result["rain_detrended"].values, color="black", linewidth=1.8, linestyle="--", label="Linearly detrended rainfall")
            ax.plot(result["rain_trend"].index, result["rain_trend"].values, color="0.5", linewidth=1.2, linestyle=":", label="Fitted trend")
            ax.set_title(f"{region['label']}: raw vs detrended DJF rainfall", fontsize=TITLE_FONTSIZE)
            ax.set_xlabel("DJF year", fontsize=LABEL_FONTSIZE)
            ax.set_ylabel("Rainfall (mm)", fontsize=LABEL_FONTSIZE)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best")
            save_figure(fig, f"{slugify(region['short'])}_01_raw_vs_detrended_rainfall.png")


        def plot_running_stats(result: dict):
            region = result["region"]
            stats = result["running_stats"]
            fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
            ax.plot(stats["mid_year"], stats["rain_var"], color=region["color"], linewidth=2.0, label="Running variance of detrended rain")
            ax.plot(stats["mid_year"], stats["nino34_var"], color=NINO_COLOR, linewidth=2.0, linestyle="--", label="Running variance of Nino3.4")
            ax.plot(stats["mid_year"], stats["rain_nino34_cov"], color="black", linewidth=2.0, linestyle=":", label="Running covariance of rain and Nino3.4")
            ax.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
            ax.set_title(
                f"{region['label']}: {RUNNING_WINDOW}-year running variance and covariance using standardized inputs",
                fontsize=TITLE_FONTSIZE,
            )
            ax.set_xlabel("Midpoint DJF year", fontsize=LABEL_FONTSIZE)
            ax.set_ylabel("Normalized statistic", fontsize=LABEL_FONTSIZE)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best")
            save_figure(fig, f"{slugify(region['short'])}_02_running_variance_covariance.png")


        def plot_scree(result: dict):
            region = result["region"]
            variance = result["variance_table"]
            fig, ax1 = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
            ax1.bar(variance["mode"], variance["explained_variance_percent"], color=region["color"], alpha=0.8)
            ax1.set_xlabel("EOF mode", fontsize=LABEL_FONTSIZE)
            ax1.set_ylabel("Explained variance (%)", fontsize=LABEL_FONTSIZE)
            ax1.set_xticks(variance["mode"])

            ax2 = ax1.twinx()
            ax2.plot(variance["mode"], variance["cumulative_percent"], color="black", marker="o", linewidth=1.8)
            ax2.set_ylabel("Cumulative variance (%)", fontsize=LABEL_FONTSIZE)
            ax2.set_ylim(0, 100)
            ax1.set_title(f"{region['label']}: EOF scree plot (EOF1-EOF2)", fontsize=TITLE_FONTSIZE)
            save_figure(fig, f"{slugify(region['short'])}_03_eof_scree.png")


        def plot_eof_maps(result: dict):
            region = result["region"]
            eof_da = result["eof_da"]
            variance = result["variance_table"]
            proj = ccrs.PlateCarree()
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), subplot_kw={"projection": proj}, constrained_layout=True)

            for idx, ax in enumerate(axes, start=1):
                eof_map = eof_da.sel(mode=idx)
                vmax = float(np.nanmax(np.abs(eof_map.values)))
                if not np.isfinite(vmax) or vmax == 0:
                    vmax = 1e-12
                mesh = ax.pcolormesh(
                    eof_map["lon"],
                    eof_map["lat"],
                    eof_map,
                    cmap="RdBu_r",
                    shading="auto",
                    vmin=-vmax,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                )
                add_map_features(
                    ax,
                    [region["lon_min"], region["lon_max"], region["lat_min"], region["lat_max"]],
                    draw_labels=True,
                )
                ax.set_title(
                    f"EOF{idx} loading ({variance.loc[idx - 1, 'explained_variance_percent']:.1f}%)",
                    fontsize=TITLE_FONTSIZE,
                )
                cbar = fig.colorbar(mesh, ax=ax, pad=0.02, shrink=0.9)
                cbar.set_label("EOF loading", fontsize=LABEL_FONTSIZE)

            save_figure(fig, f"{slugify(region['short'])}_04_eof12_loading_maps.png")


        def plot_pc_vs_nino(result: dict):
            region = result["region"]
            pair = result["pc_nino_table"].set_index("year")
            variance = result["variance_table"]
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

            for ax, pc_name in zip(axes, ["PC1", "PC2"]):
                corr = pair[pc_name].corr(pair["nino34_djf"])
                ev = float(variance.loc[variance["mode"] == int(pc_name[-1]), "explained_variance_percent"].iloc[0])
                ax2 = ax.twinx()
                ax.plot(pair.index, pair[pc_name], color=PC_COLORS[pc_name], linewidth=2.0)
                ax2.plot(pair.index, pair["nino34_djf"], color=NINO_COLOR, linewidth=1.8, linestyle="--")
                ax.set_ylabel(f"{pc_name} (raw)", fontsize=LABEL_FONTSIZE, color=PC_COLORS[pc_name])
                ax2.set_ylabel("Nino3.4 (raw)", fontsize=LABEL_FONTSIZE, color=NINO_COLOR)
                ax.set_title(
                    f"{region['label']}: {pc_name} vs Nino3.4 | r = {corr:.3f} | EV = {ev:.1f}%",
                    fontsize=TITLE_FONTSIZE,
                )
                ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
                ax2.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
                ax.grid(True, linestyle="--", alpha=0.35)

            axes[-1].set_xlabel("DJF year", fontsize=LABEL_FONTSIZE)
            save_figure(fig, f"{slugify(region['short'])}_05_pc_vs_nino34_raw.png")


        def plot_running_pc_corr(result: dict):
            region = result["region"]
            rc1 = result["running_corr_pc1"]
            rc2 = result["running_corr_pc2"]
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)

            axes[0].plot(rc1["mid_year"], rc1["correlation"], color=PC_COLORS["PC1"], linewidth=2.0)
            axes[0].axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
            axes[0].set_title(f"{region['label']}: {RUNNING_WINDOW}-year running correlation of PC1 and Nino3.4", fontsize=TITLE_FONTSIZE)
            axes[0].set_ylabel("Correlation", fontsize=LABEL_FONTSIZE)
            axes[0].grid(True, linestyle="--", alpha=0.4)

            axes[1].plot(rc2["mid_year"], rc2["correlation"], color=PC_COLORS["PC2"], linewidth=2.0)
            axes[1].axhline(0.0, color="0.4", linewidth=0.8, linestyle="--")
            axes[1].set_title(f"{region['label']}: {RUNNING_WINDOW}-year running correlation of PC2 and Nino3.4", fontsize=TITLE_FONTSIZE)
            axes[1].set_ylabel("Correlation", fontsize=LABEL_FONTSIZE)
            axes[1].set_xlabel("Midpoint DJF year", fontsize=LABEL_FONTSIZE)
            axes[1].grid(True, linestyle="--", alpha=0.4)

            save_figure(fig, f"{slugify(region['short'])}_06_running_correlation_pc_vs_nino34.png")


        def plot_wavelet_panel(result: dict):
            region = result["region"]
            pair = result["pc_nino_table"].set_index("year")
            wave_pc1 = compute_wavelet(pair["PC1"], pair["nino34_djf"])
            wave_pc2 = compute_wavelet(pair["PC2"], pair["nino34_djf"])

            fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

            for ax, wave_result, pc_name in zip(axes, [wave_pc1, wave_pc2], ["PC1", "PC2"]):
                years = wave_result["years"]
                period = wave_result["period"]
                power = wave_result["wct"]
                signif = wave_result["signif"]
                cf = ax.contourf(
                    years,
                    period,
                    power,
                    levels=np.linspace(0, 1, 11),
                    cmap="viridis",
                    extend="both",
                )

                try:
                    sig_arr = np.asarray(signif)
                    if sig_arr.ndim == 1 and sig_arr.shape[0] == power.shape[0]:
                        sig2 = np.repeat(sig_arr[:, None], power.shape[1], axis=1)
                        ax.contour(years, period, power - sig2, levels=[0], colors="white", linewidths=1.0)
                except Exception:
                    pass

                if wave_result["coi"] is not None and len(wave_result["coi"]) == len(years):
                    ax.plot(years, wave_result["coi"], color="white", linewidth=1.2, linestyle="--")
                    ax.fill_between(years, wave_result["coi"], np.nanmax(period), color="white", alpha=0.3)

                ax.set_yscale("log", base=2)
                ax.set_ylim(max(np.nanmin(period), 1), min(np.nanmax(period), 16))
                ax.set_title(
                    f"{region['label']}: wavelet coherence of {pc_name} and Nino3.4 (normalized inputs)",
                    fontsize=TITLE_FONTSIZE,
                )
                ax.set_ylabel("Period (years)", fontsize=LABEL_FONTSIZE)
                ax.grid(False)
                cbar = fig.colorbar(cf, ax=ax, pad=0.01)
                cbar.set_label("Wavelet coherence", fontsize=LABEL_FONTSIZE)

            axes[-1].set_xlabel("DJF year", fontsize=LABEL_FONTSIZE)
            save_figure(fig, f"{slugify(region['short'])}_07_wavelet_coherence_pc_vs_nino34.png")
        """
    ),
    md(
        """
        ## Load and Harmonize the Input Data

        This block opens the pre-merged MSWEP monthly file directly, slices the dataset on `lon` and `lat`, and only then pulls the `precipitation` variable. The combined file already uses `lon` from `-180 to 180` and `lat` from `90 to -90`, so the load step stays minimal.
        """
    ),
    code(
        """
        if not DATA_MSWEP_COMBINED.exists():
            raise FileNotFoundError(f"Missing combined MSWEP file: {DATA_MSWEP_COMBINED}")

        ds = xr.open_dataset(DATA_MSWEP_COMBINED)
        ds = ds.sel(
            lon=slice(DATA_EXTENT[0], DATA_EXTENT[1]),
            lat=slice(DATA_EXTENT[3], DATA_EXTENT[2]),
        )

        if "precipitation" not in ds.data_vars:
            raise KeyError(f"Expected 'precipitation' in combined MSWEP file, found {list(ds.data_vars)}")

        PR = ds["precipitation"]
        NINO34_MONTHLY = read_monthly_index(NINO34_PATH, "nino34")

        print("Opened MSWEP combined file:", DATA_MSWEP_COMBINED.name)
        print("Rainfall monthly range:", str(PR["time"].values[0]), "to", str(PR["time"].values[-1]))
        print("Nino3.4 monthly range:", str(NINO34_MONTHLY["time"].values[0]), "to", str(NINO34_MONTHLY["time"].values[-1]))
        print("Rainfall grid:", PR.sizes)
        """
    ),
    md(
        """
        ## Build the Complete DJF Time Series

        The source notebook groups December-January-February into a DJF year labeled by the January year. This section applies the same logic, keeps only complete 3-month DJF seasons, and then finds the common years shared by rainfall and Nino3.4.

        The climatology years are chosen from the requested `1991-2020` baseline if that baseline overlaps the local file. If not, the notebook falls back to the full common DJF range.
        """
    ),
    code(
        """
        PR_DJF_FULL = build_complete_djf(PR)
        NINO_DJF_FULL = build_complete_djf(NINO34_MONTHLY).to_pandas()
        NINO_DJF_FULL.name = "nino34_djf"
        NINO_DJF_FULL.index = pd.Index(NINO_DJF_FULL.index.astype(int), name="year")

        COMMON_YEARS = pd.Index(
            sorted(set(PR_DJF_FULL["year"].values.astype(int)) & set(NINO_DJF_FULL.index.astype(int))),
            name="year",
        )
        PR_DJF = PR_DJF_FULL.sel(year=COMMON_YEARS)
        NINO_DJF = NINO_DJF_FULL.loc[COMMON_YEARS]
        CLIM_YEARS = get_climatology_years(COMMON_YEARS)

        REGIONAL_RAW_SERIES = {
            key: subset_region(PR_DJF, region).mean(dim=("lat", "lon"), skipna=True).to_pandas().rename(f"{region['short']}_rain_raw_mm")
            for key, region in REGIONS.items()
        }
        for key in REGIONAL_RAW_SERIES:
            REGIONAL_RAW_SERIES[key].index = pd.Index(REGIONAL_RAW_SERIES[key].index.astype(int), name="year")

        print("Common complete DJF years:", int(COMMON_YEARS.min()), "to", int(COMMON_YEARS.max()))
        print("Number of common DJF years:", len(COMMON_YEARS))
        print("Climatology years used:", int(CLIM_YEARS.min()), "to", int(CLIM_YEARS.max()))
        REGION_RESULTS = {}
        """
    ),
    md(
        """
        ## Preliminary Plot 1: Study-Area Definition Map

        This overview map shows the two analysis boxes on a larger Maritime Continent extent so the study boundaries are easy to see. The notebook uses **blue** for WMC and **green** for EMC, and this color mapping is kept consistent through the rest of the figures.
        """
    ),
    code("plot_region_definition_map()"),
    md(
        """
        ## Preliminary Plot 2: DJF Rainfall Time Series

        This figure summarizes the raw area-mean DJF rainfall series for both regions before any detrending or EOF analysis. It is a quick first check on how WMC and EMC vary through the common DJF period.
        """
    ),
    code("plot_rainfall_timeseries(REGIONAL_RAW_SERIES)"),
    md(
        """
        ## Preliminary Plot 3: Nino3.4 DJF Time Series

        Nino3.4 is a single basin-scale ENSO index, so it is not spatially averaged over WMC or EMC. Instead, the same DJF Nino3.4 series is used as the common climate driver for both regional analyses.
        """
    ),
    code("plot_nino_timeseries(NINO_DJF)"),
    md(
        """
        ## Preliminary Plot 4: Raw DJF Rainfall Versus Raw Nino3.4

        These panels compare the raw area-mean DJF rainfall series with the raw DJF Nino3.4 index using twin axes. The overall Pearson correlation for each region is printed in the subplot title.
        """
    ),
    code("plot_rain_vs_nino(REGIONAL_RAW_SERIES, NINO_DJF)"),
    md(
        """
        ## Preliminary Plot 5: Mean Spatial DJF Rainfall

        This map shows the average DJF rainfall field over all available common years on the larger Maritime Continent domain. The WMC and EMC boxes are overlaid so the study regions can be interpreted against the broader rainfall climatology.
        """
    ),
    code("plot_spatial_mean_djf_rainfall(PR_DJF)"),
]


for region_key, region_name in [("wmc", "West Maritime Continent (WMC)"), ("emc", "East Maritime Continent (EMC)")]:
    cells.extend(
        [
            md(
                f"""
                ## {region_name}: Regional Preprocessing

                This block prepares the full regional analysis package for `{region_key.upper()}`:

                - raw area-mean DJF rainfall
                - linear detrending of the regional rainfall series
                - gridpoint-wise linear detrending of the DJF rainfall field
                - EOF analysis on the detrended DJF anomalies
                - running statistics, running correlations, and wavelet inputs

                The resulting tables are also exported to the local `output/tables` folder.
                """
            ),
            code(
                f"""
                region_key = "{region_key}"
                REGION_RESULTS[region_key] = analyze_region(region_key, REGIONS[region_key], PR_DJF, NINO_DJF, CLIM_YEARS)
                save_region_tables(REGION_RESULTS[region_key])
                REGION_RESULTS[region_key]["summary"]
                """
            ),
            md(
                f"""
                ## {region_name}: Detrending Check

                The requested EOF analysis uses linearly detrended DJF rainfall. Before moving to EOFs, this figure compares the raw regional DJF rainfall series with the mean-preserving linearly detrended version so the effect of detrending is visible.
                """
            ),
            code(f'plot_detrend_comparison(REGION_RESULTS["{region_key}"])'),
            md(
                f"""
                ## {region_name}: Running Variance and Running Covariance

                This diagnostic uses the detrended regional rainfall series together with the DJF Nino3.4 index. Both inputs are standardized first, then a `{15}`-year running variance of rainfall, running variance of Nino3.4, and running covariance between rainfall and Nino3.4 are computed and plotted.
                """
            ),
            code(f'plot_running_stats(REGION_RESULTS["{region_key}"])'),
            md(
                f"""
                ## {region_name}: EOF Scree Plot

                The EOF solver is limited to **EOF1** and **EOF2** only, as requested. The scree plot shows the explained variance of these two retained modes and their cumulative contribution.
                """
            ),
            code(f'plot_scree(REGION_RESULTS["{region_key}"])'),
            md(
                f"""
                ## {region_name}: EOF1 and EOF2 Loading Maps

                This cartopy figure shows the spatial loading patterns for EOF1 and EOF2 computed from the **linearly detrended DJF rainfall anomalies** in the selected region.
                """
            ),
            code(f'plot_eof_maps(REGION_RESULTS["{region_key}"])'),
            md(
                f"""
                ## {region_name}: Raw PC Time Series Versus Raw Nino3.4

                These panels compare the raw principal-component scores for PC1 and PC2 against the raw DJF Nino3.4 index. Each title reports the overall correlation and the explained variance of that EOF mode.
                """
            ),
            code(f'plot_pc_vs_nino(REGION_RESULTS["{region_key}"])'),
            md(
                f"""
                ## {region_name}: Running Correlation of PCs and Nino3.4

                This figure tracks the temporal stability of the EOF-Nino3.4 relationship with a `{15}`-year running correlation. The raw PC scores and the raw DJF Nino3.4 index are used directly, without standardization.
                """
            ),
            code(f'plot_running_pc_corr(REGION_RESULTS["{region_key}"])'),
            md(
                f"""
                ## {region_name}: Wavelet Coherence of PCs and Nino3.4

                The final diagnostic looks at time-frequency dependence between each PC and Nino3.4 using wavelet coherence. As requested, the wavelet calculation uses **normalized** input series, while the cone of influence and significance contour are added when available from `pycwt`.
                """
            ),
            code(f'plot_wavelet_panel(REGION_RESULTS["{region_key}"])'),
        ]
    )


cells.extend(
    [
        md(
            """
            ## Output Inventory

            This final cell lists the saved PNG and table outputs produced by the notebook so the generated files can be checked quickly from one place.
            """
        ),
        code(
            """
            print("PNG outputs:")
            for path in sorted(OUT_PNG.glob("*.png")):
                print(" -", path.name)

            print("\\nTable outputs:")
            for path in sorted(OUT_TABLES.glob("*.csv")):
                print(" -", path.name)
            """
        ),
    ]
)


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}

PROJECT_DIR.mkdir(parents=True, exist_ok=True)
with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Wrote notebook: {NOTEBOOK_PATH}")
