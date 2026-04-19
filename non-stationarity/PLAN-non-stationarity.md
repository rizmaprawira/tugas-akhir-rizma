# MSWEP-nino34 DJF Non-Stationarity Notebook Plan

## Summary
1. Build one notebook at [mswep_nino34_djf_nonstationarity.ipynb](/Users/rizzie/Academic/9_TugasAkhir/notebook/non-stationarity/mswep_nino34_djf_nonstationarity.ipynb) that computes strict DJF rainfall anomalies and DJF NINO3.4, then generates 8 non-stationarity diagnostics.
2. Use MSWEP only from [data/mswep](/Users/rizzie/Academic/9_TugasAkhir/data/mswep), NINO34 from [nino34.anom.csv](/Users/rizzie/Academic/9_TugasAkhir/data/index/nino34.anom.csv), and regions from [domain.json](/Users/rizzie/Academic/9_TugasAkhir/data/all_data/domain.json).
3. Save every figure as PNG under [output](/Users/rizzie/Academic/9_TugasAkhir/notebook/non-stationarity/output) and display each figure inline in notebook cells.

## Locked Decisions
1. DJF analysis period is DJF 1980 to DJF 2024 (strict complete DJF seasons only).
2. Regions use all 9 domain boxes from `domain.json` with the same inferred names/order as prior running-correlation workflow.
3. Lag convention is fixed: positive lag means nino34 leads rainfall by that many DJF seasons.
4. Lag scope is fixed: plots 1-2 use lags `-2..+2`; plots 3, 5, 6, 7 use lag `0` and lag `+1`; plot 8 uses lag `0`.
5. Regime scatter will include both fixed regimes and change-point-derived regimes.
6. Notebook installs `ruptures` and `pycwt` for plots 7 and 8.

## Files and Outputs
1. Notebook file: [mswep_nino34_djf_nonstationarity.ipynb](/Users/rizzie/Academic/9_TugasAkhir/notebook/non-stationarity/mswep_nino34_djf_nonstationarity.ipynb).
2. Output root: [output](/Users/rizzie/Academic/9_TugasAkhir/notebook/non-stationarity/output) with `png`, `tables`, and `logs` subfolders.
3. Table output file 1: `djf_region_timeseries_1980_2024.csv`.
4. Table output file 2: `running_corr_lags_m2_p2_window15.csv`.
5. Table output file 3: `running_regression_lag0_lag1_window15.csv`.
6. Table output file 4: `changepoints_running_slope.csv`.
7. Figure file set includes:
8. `01_running_corr_multilag_all_regions.png`.
9. `02_lag_heatmap_<region>.png` (one per region).
10. `03_running_slope_r2_lag0_all_regions.png`.
11. `03_running_slope_r2_lag1_all_regions.png`.
12. `04_regime_scatter_fixed_lag0_all_regions.png`.
13. `04_regime_scatter_fixed_lag1_all_regions.png`.
14. `04_regime_scatter_cp_lag0_all_regions.png`.
15. `04_regime_scatter_cp_lag1_all_regions.png`.
16. `05_surrogate_envelope_lag0_all_regions.png`.
17. `05_surrogate_envelope_lag1_all_regions.png`.
18. `06_expanding_corr_slope_lag0_all_regions.png`.
19. `06_expanding_corr_slope_lag1_all_regions.png`.
20. `07_changepoint_running_slope_lag0_all_regions.png`.
21. `07_changepoint_running_slope_lag1_all_regions.png`.
22. `08_wavelet_coherence_lag0_<region>.png` (one per region).

## Notebook Structure (Decision Complete)
1. Intro markdown cell with goal, data sources, strict DJF definition, and lag sign convention.
2. Setup code cell with imports, random seed, plotting style, output directory creation, and `save_and_show(fig, filename)` helper.
3. Dependency cell with `%pip install ruptures pycwt` and import verification.
4. Region cell that loads `domain.json`, infers names with existing centroid logic, and fixes region order.
5. Data load cell that opens MSWEP monthly data and nino34 monthly data; nino34 `-9999` values converted to missing.
6. DJF builder cell implementing strict `Dec(y-1)+Jan(y)+Feb(y)` with complete-season requirement.
7. Rain preprocessing cell computing area-mean monthly rainfall per region, DJF means, 1991-2020 fixed climatology anomalies, and z-scores.
8. nino34 preprocessing cell computing strict DJF nino34 for the same year labels.
9. Merge/lag cell producing master table and lagged nino34 columns.
10. QC cell with assertions on DJF membership for 1980 and 2024, climatology sample size 30 years, and expected window-center counts.

## Plot Method Specs
1. Plot 1 running correlation: centered 15-year Pearson `r` for each region and lags `-2..+2`; show sign flips, amplitude changes, and lag-of-max shifts.
2. Plot 2 lag heatmap: per region heatmap of running correlation (year vs lag), plus ridge of max absolute correlation and ridge sign.
3. Plot 3 running regression: per region and lag (0,+1), fit `Rain_anom = a*nino34_lag + b` per moving window; plot `a(t)` and `R²(t)` with moving-block bootstrap CIs.
4. Plot 4 regime scatter: nino34 vs rainfall anomaly by regime, with regression line and slope CI for fixed regimes and change-point regimes.
5. Plot 5 surrogate envelope: phase-randomized nino34 surrogates (`n=1000`) under stationarity, running-correlation envelope (5th-95th), and observed curve overlay.
6. Plot 6 expanding window: cumulative correlation and slope from 1980 to year `y` (minimum span 15 years).
7. Plot 7 change points: detect 1-3 breakpoints on running slope via `ruptures` dynamic programming, select breakpoint count by BIC, plot breaks and segment means.
8. Plot 8 wavelet coherence: region-wise lag-0 coherence via `pycwt`, including 95% significance mask, red-noise background test, ENSO 2-7 year band, and cone of influence.

## Public Interfaces and Types
1. Helper function interface: `load_domains(path) -> OrderedDict[str, dict]`.
2. Helper function interface: `build_djf(monthly_series, start_year, end_year, require_complete=True) -> pd.Series`.
3. Helper function interface: `compute_region_monthly_mswep(da, box) -> pd.Series`.
4. Helper function interface: `make_lagged_nino34(nino34_series, lag) -> pd.Series`.
5. Helper function interface: `running_corr(x, y, window=15) -> pd.Series`.
6. Helper function interface: `running_regression_with_bootstrap(x, y, window=15, block_len=3, n_boot=1000) -> pd.DataFrame`.
7. Helper function interface: `phase_randomized_surrogates(x, n=1000, seed=42) -> np.ndarray`.
8. Master table schema: `region:str, year:int, rain_djf_mm:float, rain_anom_mm:float, rain_z:float, nino34_djf:float`.

## Test Cases and Acceptance Criteria
1. DJF construction test passes for explicit dates: DJF 1980 uses Dec 1979 + Jan 1980 + Feb 1980; DJF 2024 uses Dec 2023 + Jan 2024 + Feb 2024.
2. Climatology test confirms exactly 30 DJF seasons in 1991-2020 per region.
3. Lag alignment test confirms `lag=+1` uses nino34 from `year-1`.
4. Running-window test confirms 31 center years for 15-year windows over 1980-2024.
5. Bootstrap output test confirms finite CI bounds for slope and `R²`.
6. Surrogate output test confirms envelope arrays have expected dimensions and finite percentiles.
7. Change-point test confirms at most 3 breakpoints and minimum segment length of 4 centers.
8. Output test confirms all PNGs and CSV tables are created and each plot is shown inline in notebook execution.

## Assumptions and Defaults
1. DJF 1979 is excluded because December 1978 is unavailable in MSWEP; strict DJF is preserved.
2. Requested period is implemented as DJF years 1980-2024 for complete-season consistency.
3. Base climatology and z-score parameters are fixed to 1991-2020 and not recomputed in moving windows.
4. Positive lag always means nino34 leads rainfall; labels explicitly state this in every lagged plot.
5. If package installation fails during implementation, notebook will log explicit warnings and continue with plots 1-6; plots 7-8 become conditional.
6. No AGENTS skill is applied here because available skills are for skill creation/installation, not this climate-analysis notebook workflow.
