#!/usr/bin/env python3
"""
Script to generate the combined El Niño & La Niña clustering notebook.
This creates a comprehensive notebook with helper functions and timing.
"""

import json
import os

# Define notebook structure
notebook_cells = [
    # ===== CELL 1: Title =====
    {
        "cell_type": "markdown",
        "id": "#VSC-cell-001",
        "metadata": {},
        "source": [
            "# COMBINED EL NIÑO & LA NIÑA OLR CLUSTERING ANALYSIS (DJF)\n",
            "\n",
            "This notebook combines K-Means clustering analysis for both El Niño and La Niña events using DJF (December-January-February) OLR and precipitation data.\n",
            "Helper functions provide flexible visualization with adjustable parameters (vmin/vmax, lat/lon ticks, suptitle, figsize, cbar, etc.)"
        ]
    },
    # ===== CELL 2: Libraries & Timing =====
    {
        "cell_type": "markdown",
        "id": "#VSC-cell-002",
        "metadata": {},
        "source": ["# 1. IMPORT LIBRARIES & START TIMING"]
    },
    {
        "cell_type": "code",
        "id": "#VSC-cell-003",
        "metadata": {},
        "source": [
            "import time\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import xarray as xr\n",
            "import scipy as stats\n",
            "import cartopy.feature as cfeature\n",
            "import cartopy.crs as ccrs\n",
            "from matplotlib.colors import BoundaryNorm\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore', category=RuntimeWarning)\n",
            "from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter\n",
            "from sklearn.cluster import KMeans\n",
            "from sklearn.metrics import silhouette_score\n",
            "from sklearn.preprocessing import StandardScaler\n",
            "from scipy.stats import ttest_1samp, ttest_ind, pearsonr\n",
            "\n",
            "# Global notebook timing\n",
            "start_time_notebook = time.time()\n",
            "print(f'Notebook execution started at: {time.strftime(\"%Y-%m-%d %H:%M:%S\")}')"
        ]
    },
    # ===== CELL 3: Helper Functions =====
    {
        "cell_type": "markdown",
        "id": "#VSC-cell-004",
        "metadata": {},
        "source": ["# 2. HELPER FUNCTIONS FOR PLOTTING"]
    },
    {
        "cell_type": "code",
        "id": "#VSC-cell-005",
        "metadata": {},
        "source": [
            "def plot_composite_maps(field_dict, n_clusters, vmin=-30, vmax=30, step=3,\n",
            "                        figsize=(16, 5), lat_ticks=None, lon_ticks=None,\n",
            "                        cbar_label='OLR anomaly (W/m²)', cbar_size=[0.15, 0.04, 0.70, 0.05],\n",
            "                        suptitle=None, cmap='seismic', show_significance=False, sig_dict=None):\n",
            "    \"\"\"\n",
            "    Helper function to plot composite anomaly maps.\n",
            "    \n",
            "    Parameters:\n",
            "    -----------\n",
            "    field_dict : dict\n",
            "        Dictionary with cluster indices as keys and field data (xarray) as values\n",
            "    n_clusters : int\n",
            "        Number of clusters\n",
            "    vmin, vmax, step : float\n",
            "        Min, max, and step for anomaly levels\n",
            "    figsize : tuple\n",
            "        Figure size (width, height)\n",
            "    lat_ticks, lon_ticks : list\n",
            "        Custom latitude and longitude ticks\n",
            "    cbar_label : str\n",
            "        Colorbar label\n",
            "    cbar_size : list\n",
            "        Colorbar position [left, bottom, width, height]\n",
            "    suptitle : str\n",
            "        Figure suptitle\n",
            "    cmap : str\n",
            "        Colormap name\n",
            "    show_significance : bool\n",
            "        Whether to show significance masks\n",
            "    sig_dict : dict\n",
            "        Dictionary with significance masks\n",
            "    \"\"\"\n",
            "    if lat_ticks is None:\n",
            "        lat_ticks = [-15, -10, -5, 0, 5, 10, 15]\n",
            "    if lon_ticks is None:\n",
            "        lon_ticks = [90, 100, 110, 120, 130, 140, 150]\n",
            "    \n",
            "    anom_levels = np.arange(vmin, vmax + step, step)\n",
            "    proj = ccrs.PlateCarree()\n",
            "    fig, axes = plt.subplots(1, n_clusters, figsize=figsize, subplot_kw={'projection': proj})\n",
            "    axes = np.atleast_1d(axes)\n",
            "    norm = BoundaryNorm(anom_levels, ncolors=256)\n",
            "    cf = None\n",
            "    \n",
            "    for c in range(n_clusters):\n",
            "        ax = axes[c]\n",
            "        if c not in field_dict:\n",
            "            ax.set_axis_off()\n",
            "            continue\n",
            "        \n",
            "        field = field_dict[c]\n",
            "        lon = field['longitude'].values\n",
            "        lat = field['latitude'].values\n",
            "        \n",
            "        cf = ax.contourf(lon, lat, field, levels=anom_levels, norm=norm,\n",
            "                        transform=proj, cmap=cmap, extend='both')\n",
            "        \n",
            "        if show_significance and sig_dict is not None and c in sig_dict:\n",
            "            mask = sig_dict[c]\n",
            "            ax.contourf(lon, lat, mask, levels=[0.5, 1.5], hatches=['..'],\n",
            "                       colors='none', transform=proj)\n",
            "        \n",
            "        ax.coastlines()\n",
            "        ax.add_feature(cfeature.BORDERS, linewidth=0.3)\n",
            "        \n",
            "        ax.set_yticks(lat_ticks, crs=proj)\n",
            "        if c == 0:\n",
            "            ax.tick_params(labelleft=True, labelsize=13)\n",
            "            ax.yaxis.set_major_formatter(LatitudeFormatter())\n",
            "        else:\n",
            "            ax.tick_params(labelleft=False)\n",
            "        \n",
            "        ax.set_xticks(lon_ticks, crs=proj)\n",
            "        ax.xaxis.set_major_formatter(LongitudeFormatter())\n",
            "        ax.tick_params(labelbottom=True, labelsize=13)\n",
            "        \n",
            "        ax.set_extent([90, 155, -15, 15], crs=proj)\n",
            "        ax.set_title(f'C{c+1}', fontsize=16, fontweight='bold')\n",
            "    \n",
            "    cax = fig.add_axes(cbar_size)\n",
            "    cbar = fig.colorbar(cf, cax=cax, orientation='horizontal')\n",
            "    cbar.ax.tick_params(labelsize=14)\n",
            "    cbar.set_label(cbar_label, fontsize=14)\n",
            "    \n",
            "    plt.subplots_adjust(top=0.85, bottom=0.22, wspace=0.1)\n",
            "    if suptitle:\n",
            "        plt.suptitle(suptitle, fontsize=14, fontweight='bold')\n",
            "    plt.show()\n",
            "\n",
            "\n",
            "print('Helper functions defined successfully')"
        ]
    },
    # ===== CELL 4: Data Loading =====
    {
        "cell_type": "markdown",
        "id": "#VSC-cell-006",
        "metadata": {},
        "source": ["# 3. LOAD & PREPARE DATA"]
    },
    {
        "cell_type": "code",
        "id": "#VSC-cell-007",
        "metadata": {},
        "source": [
            "start_time_data = time.time()\n",
            "\n",
            "# Load OLR data\n",
            "olr = xr.open_dataset('/Users/rizzie/Academic/9_TugasAkhir/notebook/cluster-lanina/datastamul/olr-era5.nc')\n",
            "olr = olr['avg_tnlwrf'].sel(latitude=slice(15, -15), longitude=slice(90, 155))\n",
            "olr = olr * -1  # Flip sign for positive anomaly = enhanced convection\n",
            "\n",
            "# Compute DJF seasonal means (aligned by January year)\n",
            "# DJF Y = Dec(Y-1) + Jan(Y) + Feb(Y)\n",
            "start_year, end_year = 1979, 2020\n",
            "valid_time = olr.valid_time\n",
            "seasonal_means = []\n",
            "\n",
            "for year in range(start_year, end_year + 1):\n",
            "    dec_prev = valid_time[(valid_time.dt.year == year - 1) & (valid_time.dt.month == 12)]\n",
            "    jan_feb = valid_time[(valid_time.dt.year == year) & (valid_time.dt.month.isin([1, 2]))]\n",
            "    selected_times = xr.concat([dec_prev, jan_feb], dim='valid_time')\n",
            "    if selected_times.size != 3:\n",
            "        continue\n",
            "    seasonal_means.append(olr.sel(valid_time=selected_times).mean(dim='valid_time').assign_coords(year=year))\n",
            "\n",
            "olr_djf = xr.concat(seasonal_means, dim='year')\n",
            "print(f'OLR data loaded. Shape: {olr_djf.shape}')\n",
            "\n",
            "# Load ONI index\n",
            "oni_csv_path = '/Users/rizzie/Academic/9_TugasAkhir/notebook/clusterolr/oni_year_month.csv'\n",
            "oni_monthly = pd.read_csv(oni_csv_path)\n",
            "oni_monthly.columns = [c.strip().lower() for c in oni_monthly.columns]\n",
            "oni_monthly = oni_monthly[['year', 'month', 'oni']].copy()\n",
            "oni_monthly['year'] = oni_monthly['year'].astype(int)\n",
            "oni_monthly['month'] = oni_monthly['month'].astype(int)\n",
            "oni_monthly['oni'] = oni_monthly['oni'].astype(float)\n",
            "oni_monthly = oni_monthly[oni_monthly['oni'] > -9000].copy()\n",
            "\n",
            "oni_monthly = oni_monthly[oni_monthly['month'].isin([12, 1, 2])].copy()\n",
            "oni_monthly['djf_year'] = np.where(oni_monthly['month'] == 12, oni_monthly['year'] + 1, oni_monthly['year'])\n",
            "\n",
            "oni_djf = oni_monthly.groupby('djf_year').agg(oni_djf=('oni', 'mean'), n_months=('month', 'nunique')).reset_index()\n",
            "oni_djf = oni_djf[oni_djf['n_months'] == 3].drop(columns='n_months')\n",
            "oni_djf = oni_djf[(oni_djf['djf_year'] >= 1979) & (oni_djf['djf_year'] <= 2020)].copy()\n",
            "\n",
            "# Identify El Niño and La Niña years\n",
            "el_nino_years = oni_djf.loc[oni_djf['oni_djf'] >= 0.5, 'djf_year'].astype(int).tolist()\n",
            "la_nina_years = oni_djf.loc[oni_djf['oni_djf'] <= -0.5, 'djf_year'].astype(int).tolist()\n",
            "\n",
            "print(f'El Niño years (ONI >= 0.5): {el_nino_years}')\n",
            "print(f'La Niña years (ONI <= -0.5): {la_nina_years}')\n",
            "print(f'Data loading time: {time.time() - start_time_data:.2f} seconds')"
        ]
    },
    # ===== CELL 5: El Niño Section Header =====
    {
        "cell_type": "markdown",
        "id": "#VSC-cell-008",
        "metadata": {},
        "source": ["# ========== EL NIÑO ANALYSIS (ONI ≥ 0.5) =========="]
    },
    {
        "cell_type": "code",
        "id": "#VSC-cell-009",
        "metadata": {},
        "source": [
            "# Calculate El Niño OLR anomalies\n",
            "base_period_start, base_period_end = 1991, 2020\n",
            "base_period_data = olr_djf.sel(year=slice(base_period_start, base_period_end))\n",
            "base_period_mean = base_period_data.mean(dim='year')\n",
            "\n",
            "olr_djf_nino_anomalies_list = []\n",
            "for year in el_nino_years:\n",
            "    current_year_data = olr_djf.sel(year=year)\n",
            "    olr_djf_nino_anomalies_list.append(current_year_data - base_period_mean)\n",
            "\n",
            "olr_djf_nino_anomalies = xr.concat(olr_djf_nino_anomalies_list, dim='year')\n",
            "print(f'\\n=== EL NIÑO ===\\nOLR anomalies calculated. Shape: {olr_djf_nino_anomalies.shape}')"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "#VSC-cell-010",
        "metadata": {},
        "source": ["## El Niño - K-Means Clustering"]
    },
    {
        "cell_type": "code",
        "id": "#VSC-cell-011",
        "metadata": {},
        "source": [
            "start_time_clustering = time.time()\n",
            "\n",
            "# Prepare data for clustering\n",
            "X_nino = olr_djf_nino_anomalies.stack(space=('latitude', 'longitude')).values\n",
            "scaler_nino = StandardScaler()\n",
            "X_nino_scaled = scaler_nino.fit_transform(X_nino)\n",
            "\n",
            "# Elbow and Silhouette analysis\n",
            "max_K_nino = min(10, X_nino_scaled.shape[0] - 1)\n",
            "K_range_nino = range(2, max_K_nino + 1)\n",
            "sse_nino, sil_nino = [], []\n",
            "\n",
            "for k in K_range_nino:\n",
            "    km = KMeans(n_clusters=k, n_init=500, max_iter=800, random_state=42)\n",
            "    km.fit(X_nino_scaled)\n",
            "    sse_nino.append(km.inertia_)\n",
            "    sil_nino.append(silhouette_score(X_nino_scaled, km.labels_))\n",
            "\n",
            "# Plot elbow and silhouette\n",
            "fig, ax = plt.subplots(1, 2, figsize=(12, 5))\n",
            "ax[0].plot(list(K_range_nino), sse_nino, marker='o', linewidth=2)\n",
            "ax[0].set_xlabel('Number of clusters K', fontsize=12)\n",
            "ax[0].set_ylabel('Within-cluster SSE', fontsize=12)\n",
            "ax[0].set_title('El Niño - Elbow Plot', fontsize=13, fontweight='bold')\n",
            "ax[0].grid(True, alpha=0.3)\n",
            "\n",
            "ax[1].plot(list(K_range_nino), sil_nino, marker='s', color='orange', linewidth=2)\n",
            "ax[1].set_xlabel('Number of clusters K', fontsize=12)\n",
            "ax[1].set_ylabel('Mean Silhouette Score', fontsize=12)\n",
            "ax[1].set_title('El Niño - Silhouette Analysis', fontsize=13, fontweight='bold')\n",
            "ax[1].grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f'Elbow & Silhouette analysis time: {time.time() - start_time_clustering:.2f} seconds')"
        ]
    },
    {
        "cell_type": "code",
        "id": "#VSC-cell-012",
        "metadata": {},
        "source": [
            "# Perform K-Means with K=2\n",
            "n_clusters = 2\n",
            "kmeans_nino = KMeans(n_clusters=n_clusters, n_init=500, max_iter=800, random_state=42)\n",
            "kmeans_nino.fit(X_nino_scaled)\n",
            "\n",
            "olr_djf_nino_anomalies.coords['cluster'] = ('year', kmeans_nino.labels_)\n",
            "\n",
            "# Build cluster table\n",
            "cluster_table_nino = pd.DataFrame({\n",
            "    'year': olr_djf_nino_anomalies['year'].values,\n",
            "    'cluster': kmeans_nino.labels_\n",
            "}).sort_values(['cluster', 'year']).reset_index(drop=True)\n",
            "\n",
            "print(f'\\nEl Niño Cluster Assignment:')\n",
            "print(cluster_table_nino)\n",
            "print(f'Total El Niño events: {len(el_nino_years)}')\n",
            "print(f'Cluster 1: {len(cluster_table_nino[cluster_table_nino[\"cluster\"]==0])} events')\n",
            "print(f'Cluster 2: {len(cluster_table_nino[cluster_table_nino[\"cluster\"]==1])} events')"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "#VSC-cell-013",
        "metadata": {},
        "source": ["## El Niño - Composite Maps with Significance"]
    },
    {
        "cell_type": "code",
        "id": "#VSC-cell-014",
        "metadata": {},
        "source": [
            "# Calculate composites with equal sampling and significance\n",
            "n_samples_nino = cluster_table_nino['cluster'].value_counts().min()\n",
            "print(f'Equalizing to {n_samples_nino} samples per cluster')\n",
            "\n",
            "olr_composite_nino = {}\n",
            "olr_sig_nino = {}\n",
            "\n",
            "for c in range(n_clusters):\n",
            "    years_c = cluster_table_nino.loc[cluster_table_nino['cluster'] == c, 'year'].values\n",
            "    if len(years_c) > n_samples_nino:\n",
            "        np.random.seed(42)\n",
            "        sampled_years = np.random.choice(years_c, n_samples_nino, replace=False)\n",
            "    else:\n",
            "        sampled_years = years_c\n",
            "    \n",
            "    data_sample = olr_djf_nino_anomalies.sel(year=sampled_years)\n",
            "    olr_composite_nino[c] = data_sample.mean('year')\n",
            "    \n",
            "    # Significance test\n",
            "    t_stat, p_val = ttest_1samp(data_sample.values, 0, axis=0, nan_policy='omit')\n",
            "    olr_sig_nino[c] = p_val < 0.05\n",
            "    print(f'C{c+1} members: {sampled_years}')\n",
            "\n",
            "# Plot with helper function\n",
            "plot_composite_maps(\n",
            "    olr_composite_nino, n_clusters,\n",
            "    vmin=-30, vmax=30, step=3,\n",
            "    figsize=(16, 5),\n",
            "    cbar_label='OLR anomaly (W/m²)',\n",
            "    suptitle=f'El Niño OLR Composites – K={n_clusters} (Stippled p < 0.05)',\n",
            "    cmap='seismic',\n",
            "    show_significance=True,\n",
            "    sig_dict=olr_sig_nino\n",
            ")"
        ]
    },
    # ===== CELL 6: La Niña Section Header =====
    {
        "cell_type": "markdown",
        "id": "#VSC-cell-015",
        "metadata": {},
        "source": ["# ========== LA NIÑA ANALYSIS (ONI ≤ -0.5) =========="]
    },
    {
        "cell_type": "code",
        "id": "#VSC-cell-016",
        "metadata": {},
        "source": [
            "# Calculate La Niña OLR anomalies\n",
            "olr_djf_nina_anomalies_list = []\n",
            "for year in la_nina_years:\n",
            "    current_year_data = olr_djf.sel(year=year)\n",
            "    olr_djf_nina_anomalies_list.append(current_year_data - base_period_mean)\n",
            "\n",
            "olr_djf_nina_anomalies = xr.concat(olr_djf_nina_anomalies_list, dim='year')\n",
            "print(f'\\n=== LA NIÑA ===\\nOLR anomalies calculated. Shape: {olr_djf_nina_anomalies.shape}')"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "#VSC-cell-017",
        "metadata": {},
        "source": ["## La Niña - K-Means Clustering"]
    },
    {
        "cell_type": "code",
        "id": "#VSC-cell-018",
        "metadata": {},
        "source": [
            "# Prepare and cluster La Niña data\n",
            "X_nina = olr_djf_nina_anomalies.stack(space=('latitude', 'longitude')).values\n",
            "scaler_nina = StandardScaler()\n",
            "X_nina_scaled = scaler_nina.fit_transform(X_nina)\n",
            "\n",
            "# Elbow and Silhouette\n",
            "max_K_nina = min(10, X_nina_scaled.shape[0] - 1)\n",
            "K_range_nina = range(2, max_K_nina + 1)\n",
            "sse_nina, sil_nina = [], []\n",
            "\n",
            "for k in K_range_nina:\n",
            "    km = KMeans(n_clusters=k, n_init=500, max_iter=800, random_state=42)\n",
            "    km.fit(X_nina_scaled)\n",
            "    sse_nina.append(km.inertia_)\n",
            "    sil_nina.append(silhouette_score(X_nina_scaled, km.labels_))\n",
            "\n",
            "fig, ax = plt.subplots(1, 2, figsize=(12, 5))\n",
            "ax[0].plot(list(K_range_nina), sse_nina, marker='o', linewidth=2)\n",
            "ax[0].set_xlabel('Number of clusters K', fontsize=12)\n",
            "ax[0].set_ylabel('Within-cluster SSE', fontsize=12)\n",
            "ax[0].set_title('La Niña - Elbow Plot', fontsize=13, fontweight='bold')\n",
            "ax[0].grid(True, alpha=0.3)\n",
            "\n",
            "ax[1].plot(list(K_range_nina), sil_nina, marker='s', color='green', linewidth=2)\n",
            "ax[1].set_xlabel('Number of clusters K', fontsize=12)\n",
            "ax[1].set_ylabel('Mean Silhouette Score', fontsize=12)\n",
            "ax[1].set_title('La Niña - Silhouette Analysis', fontsize=13, fontweight='bold')\n",
            "ax[1].grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "id": "#VSC-cell-019",
        "metadata": {},
        "source": [
            "# K-Means with K=2 for La Niña\n",
            "kmeans_nina = KMeans(n_clusters=n_clusters, n_init=500, max_iter=800, random_state=42)\n",
            "kmeans_nina.fit(X_nina_scaled)\n",
            "\n",
            "olr_djf_nina_anomalies.coords['cluster'] = ('year', kmeans_nina.labels_)\n",
            "\n",
            "# Build cluster table\n",
            "cluster_table_nina = pd.DataFrame({\n",
            "    'year': olr_djf_nina_anomalies['year'].values,\n",
            "    'cluster': kmeans_nina.labels_\n",
            "}).sort_values(['cluster', 'year']).reset_index(drop=True)\n",
            "\n",
            "print(f'\\nLa Niña Cluster Assignment:')\n",
            "print(cluster_table_nina)\n",
            "print(f'Total La Niña events: {len(la_nina_years)}')\n",
            "print(f'Cluster 1: {len(cluster_table_nina[cluster_table_nina[\"cluster\"]==0])} events')\n",
            "print(f'Cluster 2: {len(cluster_table_nina[cluster_table_nina[\"cluster\"]==1])} events')"
        ]
    },
    {
        "cell_type": "markdown",
        "id": "#VSC-cell-020",
        "metadata": {},
        "source": ["## La Niña - Composite Maps with Significance"]
    },
    {
        "cell_type": "code",
        "id": "#VSC-cell-021",
        "metadata": {},
        "source": [
            "# Calculate composites with significance\n",
            "n_samples_nina = cluster_table_nina['cluster'].value_counts().min()\n",
            "print(f'Equalizing to {n_samples_nina} samples per cluster')\n",
            "\n",
            "olr_composite_nina = {}\n",
            "olr_sig_nina = {}\n",
            "\n",
            "for c in range(n_clusters):\n",
            "    years_c = cluster_table_nina.loc[cluster_table_nina['cluster'] == c, 'year'].values\n",
            "    if len(years_c) > n_samples_nina:\n",
            "        np.random.seed(42)\n",
            "        sampled_years = np.random.choice(years_c, n_samples_nina, replace=False)\n",
            "    else:\n",
            "        sampled_years = years_c\n",
            "    \n",
            "    data_sample = olr_djf_nina_anomalies.sel(year=sampled_years)\n",
            "    olr_composite_nina[c] = data_sample.mean('year')\n",
            "    \n",
            "    t_stat, p_val = ttest_1samp(data_sample.values, 0, axis=0, nan_policy='omit')\n",
            "    olr_sig_nina[c] = p_val < 0.05\n",
            "    print(f'C{c+1} members: {sampled_years}')\n",
            "\n",
            "# Plot with helper function\n",
            "plot_composite_maps(\n",
            "    olr_composite_nina, n_clusters,\n",
            "    vmin=-30, vmax=30, step=3,\n",
            "    figsize=(16, 5),\n",
            "    cbar_label='OLR anomaly (W/m²)',\n",
            "    suptitle=f'La Niña OLR Composites – K={n_clusters} (Stippled p < 0.05)',\n",
            "    cmap='seismic',\n",
            "    show_significance=True,\n",
            "    sig_dict=olr_sig_nina\n",
            ")"
        ]
    },
    # ===== CELL 7: Notebook Timing Summary =====
    {
        "cell_type": "markdown",
        "id": "#VSC-cell-022",
        "metadata": {},
        "source": ["# EXECUTION TIME SUMMARY"]
    },
    {
        "cell_type": "code",
        "id": "#VSC-cell-023",
        "metadata": {},
        "source": [
            "end_time_notebook = time.time()\n",
            "total_time = end_time_notebook - start_time_notebook\n",
            "\n",
            "print('='*60)\n",
            "print('NOTEBOOK EXECUTION TIME SUMMARY')\n",
            "print('='*60)\n",
            "print(f'Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)')\n",
            "print(f'Notebook completed at: {time.strftime(\"%Y-%m-%d %H:%M:%S\")}')\n",
            "print('='*60)"
        ]
    }
]

# Create notebook structure
notebook = {
    "cells": notebook_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Write notebook to file
output_path = '/Users/rizzie/Academic/9_TugasAkhir/notebook/clusterolr/cluster_enso_djf_combined.ipynb'

with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✓ Combined notebook created successfully!")
print(f"  Location: {output_path}")
print(f"\nNotebook Structure:")
print(f"  - Libraries & Timing")
print(f"  - Helper Functions (plot_composite_maps)")
print(f"  - Data Loading (OLR + ONI)")
print(f"  - EL NIÑO ANALYSIS (with clustering & composites)")
print(f"  - LA NIÑA ANALYSIS (with clustering & composites)")
print(f"  - Execution Time Summary")
print(f"\nFeatures:")
print(f"  ✓ Helper functions with adjustable parameters")
print(f"  ✓ Timing for each major section")
print(f"  ✓ Both El Niño and La Niña analysis")
print(f"  ✓ K-Means clustering (K=2)")
print(f"  ✓ Significance testing (p < 0.05)")
print(f"  ✓ Equal sampling for balanced composites")
print(f"  ✓ Customizable visualization parameters")
