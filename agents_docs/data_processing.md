# Data Processing Patterns and Techniques

## Table of Contents

1. [Common Processing Patterns](#common-processing-patterns)
2. [Spatial Operations](#spatial-operations)
3. [Temporal Operations](#temporal-operations)
4. [Coordinate Transformations](#coordinate-transformations)
5. [Variable Derivation](#variable-derivation)
6. [Optimization Techniques](#optimization-techniques)
7. [Quality Control](#quality-control)

## Common Processing Patterns

### Pattern 1: Load-Subset-Process-Save

The most common pattern across notebooks:

```python
# 1. Load data (lazy)
ds = xr.open_dataset('input.nc')  # or open_zarr, open_mfdataset

# 2. Spatial subset
boundary = gpd.read_file('boundary.json')
ds_clipped = ds.rio.clip(boundary.geometry)

# 3. Temporal subset
ds_clipped = ds_clipped.sel(time=slice('2020-01-01', '2020-12-31'))

# 4. Select variables
ds_clipped = ds_clipped[['temp', 'precip', 'wind']]

# 5. Derive new variables
ds_clipped['windspeed'] = np.sqrt(ds_clipped['u']**2 + ds_clipped['v']**2)

# 6. Compute (if dask)
ds_clipped = ds_clipped.compute()  # or let to_zarr() compute

# 7. Save optimized
ds_clipped.to_zarr('output.zarr', mode='w', consolidated=True)
```

**Key Points**:
- Subset early (reduce data volume)
- Lazy evaluation until compute/save
- Zarr for efficient storage

---

### Pattern 2: Multi-File Concatenation

For time series from multiple files:

```python
# Open all files with pattern matching
ds = xr.open_mfdataset(
    '../data/weather_data/*.nc',
    combine='nested',          # Or 'by_coords'
    concat_dim='time',         # Dimension to concatenate along
    parallel=True,             # Use dask for parallel reads
    engine='netcdf4'           # Or 'h5netcdf' for better performance
)

# Alternative: Manual concatenation for more control
files = glob.glob('../data/weather_data/*.nc')
datasets = [xr.open_dataset(f) for f in sorted(files)]
ds = xr.concat(datasets, dim='time')

# Clean up coordinates if needed
ds = ds.sortby('time')
```

**Considerations**:
- `combine='nested'` when files have consistent structure
- `combine='by_coords'` when coordinates vary
- Use `parallel=True` for large datasets
- Sort by time after concatenation

---

### Pattern 3: Parallel Downloads

Using ThreadPoolExecutor for I/O-bound operations:

```python
from concurrent.futures import ThreadPoolExecutor
import requests

def download_file(url, output_path):
    """Download single file."""
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return output_path

# Generate URL list
urls = [f'https://example.com/data/{date:%Y%m%d}.nc'
        for date in pd.date_range('2020-01-01', '2020-01-31', freq='1D')]

# Download in parallel
with ThreadPoolExecutor(max_workers=8) as executor:
    files = list(executor.map(
        lambda url: download_file(url, f'data/{os.path.basename(url)}'),
        urls
    ))

print(f'Downloaded {len(files)} files')
```

**Guidelines**:
- Use 4-24 workers depending on network/server limits
- Monitor rate limits (add delays if needed)
- Handle exceptions per-file
- Log failures for retry

---

## Spatial Operations

### Clipping to Polygon Boundary

**Method 1: Using rioxarray (for regular grids)**

```python
import rioxarray as rxr
import geopandas as gpd

# Load boundary
boundary = gpd.read_file('boundary.geojson')

# Load raster
ds = rxr.open_rasterio('input.tif', masked=True)

# Clip (handles CRS transformation automatically)
clipped = ds.rio.clip(boundary.geometry, boundary.crs, drop=True)
```

**Method 2: Using regionmask (for lat/lon grids)**

```python
import regionmask

# Create regions from GeoDataFrame
regions = regionmask.Regions.from_geopandas(
    boundary,
    names='name',      # Column with region names
    abbrevs='abbrev'   # Column with abbreviations (optional)
)

# Create mask
mask = regions.mask(ds.lon, ds.lat)

# Apply mask
ds_masked = ds.where(mask.notnull())
```

**Method 3: Using shapely (for irregular grids)**

```python
from shapely.vectorized import contains

# Boolean mask
in_boundary = contains(
    boundary.geometry[0],  # Shapely polygon
    ds.lon.values,         # Longitude array
    ds.lat.values          # Latitude array
)

# Apply mask
ds_masked = ds.where(in_boundary)
```

**Choosing a Method**:
- **rioxarray**: Regular grids with known CRS
- **regionmask**: Climate model grids (lat/lon arrays)
- **shapely**: Irregular grids, maximum control

---

### Bounding Box Extraction

```python
# Define bounding box
lon_min, lon_max = -123.0, -120.0
lat_min, lat_max = 47.0, 49.0

# Method 1: xarray .sel() with slice
ds_bbox = ds.sel(
    lon=slice(lon_min, lon_max),
    lat=slice(lat_min, lat_max)
)

# Method 2: Boolean indexing
mask_lon = (ds.lon >= lon_min) & (ds.lon <= lon_max)
mask_lat = (ds.lat >= lat_min) & (ds.lat <= lat_max)
ds_bbox = ds.where(mask_lon & mask_lat, drop=True)

# Method 3: Using GeoDataFrame (create bbox first)
from shapely.geometry import box
bbox = gpd.GeoDataFrame(
    {'geometry': [box(lon_min, lat_min, lon_max, lat_max)]},
    crs='EPSG:4326'
)
ds_bbox = ds.rio.clip(bbox.geometry)
```

---

### Spatial Resampling/Regridding

```python
import xesmf as xe  # xESMF for regridding

# Define target grid
target_grid = xr.Dataset({
    'lat': (['lat'], np.arange(47, 49, 0.1)),
    'lon': (['lon'], np.arange(-123, -120, 0.1))
})

# Create regridder
regridder = xe.Regridder(
    ds,
    target_grid,
    method='bilinear',  # or 'conservative', 'nearest_s2d'
    periodic=False
)

# Regrid data
ds_regridded = regridder(ds)
```

**Methods**:
- `bilinear`: Smooth interpolation
- `conservative`: Preserves totals (good for fluxes)
- `nearest_s2d`: Nearest neighbor
- `patch`: Higher-order patch recovery

---

## Temporal Operations

### Time Selection

```python
# Select specific dates
ds_day = ds.sel(time='2020-01-15')

# Select date range
ds_range = ds.sel(time=slice('2020-01-01', '2020-01-31'))

# Select by year
ds_2020 = ds.sel(time=ds.time.dt.year == 2020)

# Select by month
ds_jan = ds.sel(time=ds.time.dt.month == 1)

# Select by season
ds_winter = ds.sel(time=ds.time.dt.season == 'DJF')

# Select specific hours
ds_noon = ds.sel(time=ds.time.dt.hour == 12)
```

---

### Temporal Aggregation

```python
# Daily mean from hourly
ds_daily = ds.resample(time='1D').mean()

# Monthly sum (e.g., precipitation)
ds_monthly = ds.resample(time='1M').sum()

# Seasonal climatology
ds_seasonal = ds.groupby('time.season').mean('time')

# Annual cycle (monthly climatology)
ds_monthly_clim = ds.groupby('time.month').mean('time')

# Custom aggregation
ds_weekly = ds.resample(time='1W').apply(
    lambda x: x.quantile(0.95, dim='time')
)
```

---

### Rolling Windows

```python
# 7-day moving average
ds_smoothed = ds.rolling(time=7, center=True).mean()

# 30-day rolling sum (precipitation)
ds_30day = ds.rolling(time=30, center=False).sum()

# Cumulative sum
ds_cumsum = ds.cumsum(dim='time')
```

---

### Time Parsing and Formatting

**Common Issue**: WRF time format

```python
# WRF times: 'YYYY-MM-DD_HH:MM:SS' as bytes or strings
time_strs = ds['Times'].astype(str)
time_strs = [t.replace('_', ' ') for t in time_strs.values]
dts = pd.to_datetime(time_strs).floor('h')

ds = ds.rename({'Time': 'time'})
ds = ds.assign_coords(time=dts)
ds = ds.drop_vars('Times')
```

**USGS time zones**:

```python
# Parse with timezone offset
if 'tz_cd' in data.columns:
    tz_offset = data['tz_cd'].map(_TZ_shift).fillna('+00:00')
    dt_str = data['datetime'].astype(str).str.strip() + tz_offset
    dt_utc = pd.to_datetime(dt_str, errors='coerce', utc=True)
else:
    dt_utc = pd.to_datetime(data['datetime'], errors='coerce', utc=True)

data['datetime'] = dt_utc
```

---

## Coordinate Transformations

### Adding Lat/Lon to Model Grids

**WRF Example**:

```python
# Load metadata file with coordinates
metadata = xr.open_dataset('wrfinput_d02.nc')

# Extract lat/lon (2D arrays)
lat_2d = metadata['XLAT'].isel(Time=0).values
lon_2d = metadata['XLONG'].isel(Time=0).values

# Create DataArrays with proper dimensions
lat = xr.DataArray(lat_2d, dims=['y', 'x'])
lon = xr.DataArray(lon_2d, dims=['y', 'x'])

# Assign to dataset
ds = ds.assign_coords(lat=lat, lon=lon)
```

---

### CRS Transformations

```python
import rioxarray as rxr

# Set CRS if not present
ds = ds.rio.write_crs('EPSG:4326')

# Transform to different CRS
ds_transformed = ds.rio.reproject('EPSG:3857')  # Web Mercator

# Get CRS info
crs = ds.rio.crs
print(crs.to_proj4())
```

---

### Longitude Wrapping (0-360 to -180-180)

```python
# Method 1: Simple arithmetic
ds['lon'] = (ds['lon'] + 180) % 360 - 180

# Method 2: xarray where
ds['lon'] = xr.where(ds['lon'] > 180, ds['lon'] - 360, ds['lon'])

# Sort if needed
ds = ds.sortby('lon')
```

---

## Variable Derivation

### Wind Speed from Components

```python
# 2D wind speed
ds['windspeed'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
ds['windspeed'].attrs = {
    'long_name': '10-meter wind speed',
    'units': 'm s-1'
}

# Wind direction (meteorological convention)
ds['wind_dir'] = np.arctan2(-ds['u10'], -ds['v10']) * 180 / np.pi
ds['wind_dir'] = (ds['wind_dir'] + 360) % 360
```

---

### Relative Humidity from Specific Humidity

**Using MetPy**:

```python
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units

ds['RH'] = relative_humidity_from_specific_humidity(
    ds['pressure'] * units.Pa,
    ds['temperature'] * units.K,
    ds['specific_humidity']
)
```

**Using Magnus Formula** (approximation):

```python
# Temperature in Celsius
T_c = ds['T2'] - 273.15

# Pressure in kPa
p_kpa = ds['PSFC'] / 1000.0

# Specific humidity (dimensionless)
q = ds['Q2']

# Saturation vapor pressure (kPa)
es_kpa = 0.6108 * np.exp((17.27 * T_c) / (T_c + 237.3))

# Actual vapor pressure (kPa)
e_kpa = (q * p_kpa) / (0.622 + 0.378 * q)

# Relative humidity (%)
ds['RH'] = (e_kpa / es_kpa).clip(0, 1) * 100.0

# Vapor Pressure Deficit (kPa)
ds['VPD'] = (es_kpa - e_kpa).clip(min=0)
```

---

### Total Precipitation

**From WRF/CONUS404**:

```python
# Sum convective, non-convective, and snow
ds['PRECIP_TOT'] = ds['RAINC'] + ds['RAINNC']

if 'SNOWNC' in ds:
    ds['PRECIP_TOT'] = ds['PRECIP_TOT'] + ds['SNOWNC']

ds['PRECIP_TOT'].attrs = {
    'long_name': 'Total precipitation (rain + snow)',
    'units': 'mm'
}
```

**Convert Units**:

```python
# Inches to mm
ds['precip_mm'] = ds['precip_inches'] * 25.4

# mm to inches
ds['precip_inches'] = ds['precip_mm'] / 25.4
```

---

### Extreme Rainfall Multiplier (ERM)

```python
# Load climatological reference
with xr.open_dataset('max_daily_precip_climatology.nc') as ref:
    precip_clim = ref['precip'].values
    lon = ref['lon'].values
    lat = ref['lat'].values

# Load observed precipitation
with xr.open_dataset('observed_precip.nc') as obs:
    precip_obs = obs['precip'].values

# Calculate ERM
erm = precip_obs / precip_clim

# For composite ERM (max of multiple periods)
erm_composite = np.maximum.reduce([erm_1day, erm_2day, erm_3day])
```

---

## Optimization Techniques

### Chunking Strategy

**Principles**:
- Chunk size: 10-100 MB per chunk
- Align with access patterns
- Balance parallelism and overhead

**Time Series Access** (read along time):

```python
ds = ds.chunk({'time': 30, 'lat': 100, 'lon': 100})
```

**Spatial Access** (read spatial slices):

```python
ds = ds.chunk({'time': 1, 'lat': 500, 'lon': 500})
```

**Zarr Storage**:

```python
# Encoding with chunk specification
encoding = {
    'temperature': {'chunks': (30, 100, 100), 'compressor': 'default'},
    'precipitation': {'chunks': (30, 100, 100), 'compressor': 'default'}
}

ds.to_zarr('output.zarr', encoding=encoding, mode='w')
```

---

### Compression

**Zarr Compression**:

```python
from numcodecs import Blosc

# Use Blosc with zstd
compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

encoding = {
    var: {'compressor': compressor}
    for var in ds.data_vars
}

ds.to_zarr('output.zarr', encoding=encoding)
```

**NetCDF Compression**:

```python
encoding = {
    var: {'zlib': True, 'complevel': 4}
    for var in ds.data_vars
}

ds.to_netcdf('output.nc', encoding=encoding)
```

---

### Lazy Evaluation

**Dask Arrays**:

```python
# Operations are lazy (not computed immediately)
ds_subset = ds.sel(time=slice('2020-01-01', '2020-12-31'))
ds_daily = ds_subset.resample(time='1D').mean()

# No computation yet! Just task graph built

# Trigger computation
ds_daily_computed = ds_daily.compute()  # Load into memory

# Or compute while saving
ds_daily.to_zarr('output.zarr', mode='w')  # Computes and streams to disk
```

**Progress Bars**:

```python
from dask.diagnostics import ProgressBar

with ProgressBar():
    ds.to_zarr('output.zarr', mode='w')
```

---

### Memory Management

**Explicit Cleanup**:

```python
import gc

# Close datasets
ds.close()

# Delete variables
del ds, ds_subset

# Force garbage collection
gc.collect()
```

**Context Managers**:

```python
# Automatically closes
with xr.open_dataset('input.nc') as ds:
    result = ds['temp'].mean().compute()
# ds is closed here
```

---

## Quality Control

### Data Validation

**Check for NaN values**:

```python
# Count NaNs per variable
nan_counts = {var: ds[var].isnull().sum().item() for var in ds.data_vars}
print(nan_counts)

# Fraction of NaNs
nan_fractions = {var: ds[var].isnull().mean().item() for var in ds.data_vars}
```

**Check value ranges**:

```python
# Statistics per variable
for var in ds.data_vars:
    print(f"{var}:")
    print(f"  Min: {ds[var].min().item():.2f}")
    print(f"  Max: {ds[var].max().item():.2f}")
    print(f"  Mean: {ds[var].mean().item():.2f}")
    print(f"  Std: {ds[var].std().item():.2f}")
```

---

### Visualization for QC

**Spatial Plot**:

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig, ax = plt.subplots(
    subplot_kw={'projection': ccrs.PlateCarree()},
    figsize=(12, 8)
)

ds['temperature'].isel(time=0).plot(ax=ax, transform=ccrs.PlateCarree())

ax.coastlines()
ax.add_feature(ccrs.cartopy.feature.BORDERS)
ax.gridlines(draw_labels=True)

plt.title('Temperature - First Time Step')
plt.show()
```

**Time Series Plot**:

```python
# Spatial mean over time
ts = ds['temperature'].mean(dim=['lat', 'lon'])
ts.plot(figsize=(12, 4))
plt.title('Basin-Mean Temperature')
plt.xlabel('Time')
plt.ylabel('Temperature (K)')
plt.grid(True, alpha=0.3)
plt.show()
```

**Distribution Plot**:

```python
# Histogram
ds['precipitation'].plot.hist(bins=50)
plt.xlabel('Precipitation (mm)')
plt.ylabel('Frequency')
plt.title('Precipitation Distribution')
plt.show()
```

---

### Outlier Detection

```python
# Z-score method
from scipy import stats

# Compute z-scores
z_scores = np.abs(stats.zscore(ds['temperature'].values, axis=None, nan_policy='omit'))

# Flag outliers (|z| > 3)
outliers = z_scores > 3
print(f'Outlier fraction: {outliers.sum() / outliers.size:.2%}')

# Remove outliers
ds['temperature_clean'] = ds['temperature'].where(~outliers)
```

---

### Metadata Preservation

```python
# Add metadata to derived variables
ds['windspeed'].attrs = {
    'long_name': '10-meter wind speed',
    'standard_name': 'wind_speed',
    'units': 'm s-1',
    'description': 'Calculated from U10 and V10 components'
}

# Global attributes
ds.attrs = {
    'title': 'Processed CONUS404 Data for Skagit Watershed',
    'source': 'CONUS404 Reanalysis',
    'processing_date': pd.Timestamp.now().isoformat(),
    'processing_notes': 'Clipped to Skagit boundary, derived variables added',
    'contact': 'user@institution.edu'
}
```

---

## Best Practices Summary

1. **Subset Early**: Reduce data volume before processing
2. **Use Lazy Evaluation**: Build task graphs, compute once
3. **Chunk Appropriately**: Match access patterns
4. **Compress**: Save disk space and improve I/O
5. **Validate**: Check ranges, NaNs, and distributions
6. **Visualize**: Always QC with plots
7. **Document**: Preserve metadata and processing history
8. **Handle Failures**: Graceful error handling, retries
9. **Optimize Storage**: Zarr for arrays, Parquet for tables
10. **Clean Up**: Close datasets, delete large objects

---

**Last Updated**: 2026-02-10
