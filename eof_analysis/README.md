# MSWEP DJF EOF Workflow for WMC and EMC

This folder contains a new notebook workflow derived from the logic in [mswep_eof_djf_v2.ipynb](/Users/rizzie/TugasAkhir/notebook/non-stationarity/mswep_eof_djf_v2.ipynb), but rebuilt for the Maritime Continent split requested here:

- `WMC`: lon `92.5-120.0`, lat `-12.5 to 12.5`
- `EMC`: lon `120.0-152.5`, lat `-12.5 to 12.5`

Key changes from the source notebook:

- uses only `Nino3.4`
- removes all `DMI` logic
- limits EOF analysis to `EOF1` and `EOF2`
- linearly detrends DJF rainfall before EOF analysis
- adds step-by-step markdown explanations throughout the notebook
- saves every figure as PNG into the local output folder

Files:

- [build_mswep_eof_mc_notebook.py](/Users/rizzie/TugasAkhir/notebook/eof-mc_v1/build_mswep_eof_mc_notebook.py): generates the notebook
- [mswep_eof_djf_wmc_emc_comprehensive.ipynb](/Users/rizzie/TugasAkhir/notebook/eof-mc_v1/mswep_eof_djf_wmc_emc_comprehensive.ipynb): generated notebook
- `output/png`: figure outputs
- `output/tables`: optional exported tables from the notebook

Data sources used by the notebook:

- `/Users/rizzie/TugasAkhir/data/mswep-monthly/mswep_monthly_combined.nc`
- `/Users/rizzie/TugasAkhir/data/index/nino34.anom.csv`

The notebook opens the pre-merged MSWEP file directly with `xarray.open_dataset`, subsets it to the Maritime Continent extent, and then auto-detects the common complete DJF overlap with Nino3.4.
