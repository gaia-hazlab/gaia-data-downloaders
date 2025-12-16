# gaia-data-downloaders
scripts to download weather data


### Installation Instructions

**Run these commands in your terminal BEFORE running this notebook:**

**Option 1: Using conda (recommended for reliability)**
```bash
conda create -n watershed python=3.11
conda activate watershed
conda install -c conda-forge geopandas obspy matplotlib pandas
pip install pygeohydro
```

**Option 2: Using pip (requires system libraries)**
```bash
# macOS: Install GDAL first
brew install gdal

# Then install Python packages
pip install geopandas obspy pygeohydro matplotlib pandas xarray netcdf4 rasterio
```

**After installation:**
- If using conda, select the `watershed` kernel in this notebook
- If using pip, restart your kernel before running the imports below