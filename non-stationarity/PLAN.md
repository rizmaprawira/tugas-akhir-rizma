# Switch DJF Running-Correlation Index From ONI to Nino3.4

## Summary
1. Update [djf_runningcorr_domainjson_layoutAC.py](/Users/rizzie/Academic/9_TugasAkhir/notebook/runningcorrelation/djf_runningcorr_domainjson_layoutAC.py) to use `/Users/rizzie/Academic/9_TugasAkhir/data/index/nina34.anom.csv` as the only ENSO index source.
2. Replace ONI-specific variable names, labels, and output filenames with Nino3.4 naming.
3. Keep all existing DJF logic, windows, domains, and rainfall processing unchanged.

## Important API/Interface Changes
1. Config constant:
2. `ONI_CSV` becomes `NINO34_CSV = Path("/Users/rizzie/Academic/9_TugasAkhir/data/index/nina34.anom.csv")`.
3. Loader function:
4. `load_oni_monthly(path)` becomes `load_nino34_monthly(path)`.
5. Function output series name becomes `"nino34"` instead of `"oni"`.
6. Internal data interface:
7. `oni_monthly`, `oni_djf` variables become `nino34_monthly`, `nino34_djf`.
8. Output table name:
9. `oni_djf_{START_YEAR}_{END_YEAR}.csv` becomes `nino34_djf_{START_YEAR}_{END_YEAR}.csv`.
10. Plot and text interface:
11. All titles/messages currently containing `"ONI"` become `"Nino3.4"`.

## Implementation Steps
1. Update constants and top-level docstring text from ONI wording to Nino3.4 wording.
2. Replace ONI loader with a robust Nino3.4 parser:
3. Read CSV, lowercase/strip column names.
4. Require `date` column.
5. Use the first non-`date` column as index values (your file has one long anomaly column name).
6. Parse values as numeric.
7. Convert sentinel missing values `-9999`, `-9999.0`, `-99.99` to `NaN`.
8. Convert date to monthly timestamps and group by month as in current workflow.
9. In `main()`, compute `nino34_djf = monthly_to_djf_series_relaxed(nino34_monthly, START_YEAR, END_YEAR)`.
10. Update running-correlation call sites to pass `nino34_djf`.
11. Rename local merged columns in `centered_running_corr` from `"oni"` to `"nino34"` for clarity.
12. Rename ONI CSV output and update print statements accordingly.
13. Replace all plot titles/subtitles containing ONI with Nino3.4.

## Test Cases and Scenarios
1. Parsing test:
2. Confirm loader reads [nina34.anom.csv](/Users/rizzie/Academic/9_TugasAkhir/data/index/nina34.anom.csv) without depending on exact anomaly column header text.
3. Confirm sentinel values become missing and are excluded.
4. DJF index test:
5. Confirm DJF year mapping still follows `Dec(y-1), Jan(y), Feb(y)` for index series.
6. Confirm generated seasonal CSV exists as `nino34_djf_1979_2020.csv` in `csv_domainjson`.
7. Pipeline test:
8. Run script end-to-end and verify running-correlation CSV/figures are produced successfully.
9. Verify figure titles and console logs mention Nino3.4, not ONI.
10. Regression safety test:
11. Confirm rainfall anomaly output schema and values are unchanged except for index-driven correlation values.

## Assumptions and Defaults
1. Full replacement is intended: ONI is no longer used in this script.
2. The second CSV column in `nina34.anom.csv` is treated as the Nino3.4 anomaly series regardless of verbose header text.
3. Existing analysis period (`START_YEAR=1979`, `END_YEAR=2020`) remains unchanged unless edited separately.
4. No change is made to domain definitions, window lengths, or precipitation preprocessing.
