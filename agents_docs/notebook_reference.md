# Notebook Reference Guide

## Table of Contents

1. [Overview](#overview)
2. [Weather Model Downloaders](#weather-model-downloaders)
3. [Precipitation Downloaders](#precipitation-downloaders)
4. [Hydrological Data Downloaders](#hydrological-data-downloaders)
5. [Inventory and Analysis Notebooks](#inventory-and-analysis-notebooks)
6. [Infrastructure Notebooks](#infrastructure-notebooks)
7. [Quick Reference Table](#quick-reference-table)

## Overview

This document provides detailed reference information for all 18 Jupyter notebooks in the repository. Each entry includes:
- Purpose and scope
- Key dependencies
- Input/output specifications
- Execution notes
- Related notebooks

### Notebook Categories

```
Weather Models (4):
├── CONUS404_Downloader.ipynb
├── HRRR_Downloader.ipynb
├── WRF_Downloader.ipynb
└── PRISM_Downloader.ipynb

Precipitation (2):
├── StageIV_Downloader.ipynb
└── ORNL_Downloader.ipynb

Hydrological (4):
├── USGS_Stream_Flow_Bulk_Downloader.ipynb
├── USGS_Stream_Gage_Site_Metadata_Downloader.ipynb
├── USGS_IV_Downloader_Demo_ipynb.ipynb
└── USGS_IRIS_crossref.ipynb

Inventory & Analysis (5):
├── merged-inventory.ipynb
├── select_stations_by_watershed_nc.ipynb
├── hydroclimatology_huc_local.ipynb
├── NC_Plot_snotel.ipynb
└── weather.ipynb

Infrastructure (3):
├── S3-bucket.ipynb
├── download_DEM.ipynb
└── converting_data_workflow.ipynb
```

---

## Weather Model Downloaders

### CONUS404_Downloader.ipynb

**Purpose**: Download and process CONUS404 reanalysis data for Skagit watershed

**Size**: 14 KB (small - mainly code)

**Data Source**: CONUS404 on OSN Pod (AWS S3-compatible)

**Key Operations**:
1. Open remote Zarr store (daily or hourly)
2. Subset by time range (2014-2023)
3. Select meteorological variables
4. Apply spatial mask (Skagit boundary)
5. Derive variables (wind speed, total precip, RH, VPD)
6. Optimize chunking for local storage
7. Save to local Zarr store
8. Generate QC plots

**Dependencies**:
- xarray, numpy, pandas
- geopandas, regionmask, shapely
- rioxarray, pyproj
- matplotlib
- dask (for progress bar)

**Configuration**:
```python
DATASET_KIND = "daily"  # or "hourly"
START = "2014-10-01"
END = "2023-12-31"
VARS = ["T2", "Q2", "U10", "V10", "PSFC", "SWDNB", "LWDNB",
        "RAINC", "RAINNC", "SNOWNC", "SMOIS", "TSLB", "HFX", "LH", "HGT"]
BOUNDARY_JSON = "../data/GIS/SkagitBoundary.json"
OUT_ZARR = "conus404_skagit_daily.zarr"
```

**Derived Variables**:
- `WS10`: Wind speed from U10, V10
- `PRECIP_TOT`: Sum of RAINC, RAINNC, SNOWNC
- `RH`: Relative humidity (Magnus formula)
- `VPD`: Vapor pressure deficit

**Outputs**:
- Zarr store at specified path
- QC plots: spatial map and time series

**Execution Time**: ~10-30 minutes (depends on time range and network speed)

**Special Notes**:
- Uses OSN Pod endpoint (not standard AWS)
- Requires spatial boundary file
- Memory efficient (dask lazy evaluation)
- Can handle large date ranges

**Related Notebooks**: WRF_Downloader.ipynb (similar workflow)

---

### HRRR_Downloader.ipynb

**Purpose**: Download HRRR weather model data using Herbie library

**Size**: 8.6 KB

**Data Source**: NOAA HRRR model via Herbie/AWS

**Key Operations**:
1. Define date range and parameters
2. Download GRIB2 files with FastHerbie
3. Create inventory files for wgrib2
4. Geographic limiting with wgrib2
5. Open with cfgrib, merge datasets
6. Apply polygon mask
7. Save to Zarr

**Dependencies**:
- herbie-data (FastHerbie, wgrib2 wrapper)
- xarray, cfgrib
- geopandas, shapely
- rioxarray
- **System**: wgrib2 executable in PATH

**Configuration**:
```python
model = 'hrrr'
product = 'sfc'  # 2D surface fields
date_range = pd.date_range('2020-03-01', '2020-03-02', freq='1h')
parameters = {
    'TMP': 'surface',
    'RH': '2 m above ground',
    'WIND': '10 m above ground',
    'APCP': 'surface:0-1 hour acc fcst',
    'DSWRF': 'surface',
    'DLWRF': 'surface'
}
aoi_path = '../data/GIS/SkagitBoundary.json'
```

**Helper Functions**:
- `download_parameters()`: Extract specific variables
- `parseGeoJson()`: Get bounding box from GeoJSON
- `limitGeographicRange()`: Use wgrib2 to crop

**Processing Steps**:
```python
# 1. Download
fh = FastHerbie(date_range, model='hrrr', product='sfc', fxx=range(0,2))
fh_files = download_parameters(parameters, fh)

# 2. Geographic limit
bounds = parseGeoJson(aoi_path)
geo_limited_files = limitGeographicRange(bounds, fh_files)

# 3. Open and merge
datasets = []
for f in geo_limited_files:
    unmerged = cfgrib.open_datasets(f, indexpath='', decode_timedelta=False)
    merged = xr.merge([...])  # Complex logic to handle forecast steps
    datasets.append(merged)

# 4. Combine and mask
combined = xr.concat(datasets, dim='time')
masked = combined.where(shapely.contains_xy(...))

# 5. Save
masked.to_zarr('hrrr.zarr', mode='w')
```

**Outputs**:
- Zarr store: `../data/weather_data/hrrr.zarr`

**Execution Time**: ~5-20 minutes (depends on date range)

**Special Notes**:
- **Critical**: wgrib2 must be in PATH
- GRIB2 format requires special handling
- Forecast step logic is complex (see notebook)
- Longitude wrapping needed (0-360 → -180-180)

**Troubleshooting**:
- If wgrib2 not found: Set PATH or install via conda
- MergeError on max_10si: Use `compat='override'`

**Related Notebooks**: CONUS404_Downloader.ipynb (similar structure)

---

### WRF_Downloader.ipynb

**Purpose**: Download WRF-CMIP6 downscaled climate projections from AWS S3

**Size**: 2.0 MB (large - includes outputs)

**Data Source**: WRF-CMIP6 on AWS S3 (anonymous access)

**Key Operations**:
1. Generate file list for date range and model
2. Parallel download from S3 (24 workers)
3. Open and combine NetCDF files
4. Add lat/lon coordinates from metadata file
5. Parse WRF time format
6. Apply spatial mask
7. Calculate derived variables
8. Save to Zarr
9. Generate visualizations

**Dependencies**:
- boto3 (AWS SDK)
- xarray, xcdat
- geopandas, shapely
- metpy (for relative humidity)
- cartopy (for plotting)
- ThreadPoolExecutor

**Configuration**:
```python
BUCKET_NAME = 'wrf-cmip6-noversioning'
OUTPUT_DIR = '../data/weather_data/'

# File generation
files = generateFileNames(
    start_date='2020-01-01',
    end_date='2020-01-03',
    model='cesm2_r11i1p1f1_ssp245',
    data_tier=2,  # 1=wrfout, 2=auxhist
    domain=2,     # d02 = 9km
    historical=False,
    bias_correction=False
)
```

**Helper Functions**:
```python
def generateFileNames(start, end, model, data_tier, domain, historical, bc):
    """Generate S3 paths for WRF output files"""
    # Complex logic to handle yearly directories starting Sept 1

def downloadS3File(bucket, file, output_dir):
    """Download single file from S3"""

def downloadMetadataFile(domain, output_dir, coord=False):
    """Download wrfinput file for coordinates"""

def getLatLonHgtFromMetadata(metadata_file):
    """Extract lat/lon/elevation from metadata"""
```

**Processing Pipeline**:
```python
# 1. Download files in parallel
with ThreadPoolExecutor(24) as executor:
    files = executor.map(lambda f: downloadS3File(...), file_list)

# 2. Open and combine
x = xr.open_mfdataset("../data/weather_data/*.nc", combine='nested', concat_dim='Time')

# 3. Add coordinates
lat, lon, hgt = getLatLonHgtFromMetadata('wrfinput_d02.nc')
x = x.assign_coords(lat=lat, lon=lon).assign(hgt=hgt)

# 4. Rename dimensions
x = x.rename({'south_north': 'y', 'west_east': 'x', 'Time': 'time'})

# 5. Parse times
time_strs = x['Times'].astype(str)
dts = pd.to_datetime(time_strs).floor('h')
x = x.assign(time=dts).drop_vars('Times')

# 6. Subset to AOI
mask = vectorized.contains(boundary.geometry[0], x.lon, x.lat)
masked = x.where(mask)

# 7. Save
masked.to_zarr('wrf.zarr')
```

**Derived Variables** (example):
```python
# Calculate RH from Q2, T2, PSFC using MetPy
x = x.assign(RH=relative_humidity_from_specific_humidity(
    x.PSFC * units.Pa,
    x.T2 * units.degK,
    x.Q2
))

# Calculate wind speed
x = x.assign(WS10=np.sqrt(x.U10**2 + x.V10**2))

# Calculate total precip
x = x.assign(TP=x.RAINC + x.RAINNC)
```

**Outputs**:
- Zarr store: `wrf.zarr`
- Plots: Temperature maps, animations (optional)

**Execution Time**: ~5-15 minutes (depends on number of files)

**Special Notes**:
- S3 unsigned access (no credentials)
- Lambert Conformal projection (need metadata for coords)
- Time format: `YYYY-MM-DD_HH:MM:SS` (needs parsing)
- Files organized by water year (Oct 1 - Sep 30)

**Troubleshooting**:
- Missing lat/lon: Download metadata file first
- Time parsing errors: Check Times variable format

**Related Notebooks**: CONUS404_Downloader.ipynb

---

### PRISM_Downloader.ipynb

**Purpose**: Download and process PRISM gridded climate observations

**Size**: 258 KB (medium - includes plots)

**Data Source**: PRISM Climate Group (Oregon State University)

**Key Operations**:
1. Download multiple variables in parallel
2. Iterate through downloaded BIL files
3. Clip to watershed boundary
4. Create xarray Dataset with time dimension
5. Save to Zarr
6. Generate cartographic visualizations

**Dependencies**:
- pyPRISMClimate (PRISM wrapper)
- rioxarray, xarray
- geopandas
- matplotlib, cartopy
- ThreadPoolExecutor

**Configuration**:
```python
dates = pd.date_range('2017-02-01', '2017-02-08', freq='1d', inclusive='right')
variables = ['tmean', 'tmax', 'tmin', 'ppt', 'vpdmax', 'vpdmin']
dest_path = '../data/weather_data/'
mask = gpd.read_file('../data/GIS/SkagitBoundary.json')
```

**Download Pattern**:
```python
with ThreadPoolExecutor(4) as executor:
    executor.map(
        lambda var: pyPRISMClimate.get_prism_dailys(
            var,
            min_date=dates[0].strftime("%Y-%m-%d"),
            max_date=dates[-1].strftime("%Y-%m-%d"),
            dest_path=dest_path,
            keep_zip=False
        ),
        variables
    )
```

**Processing Function**:
```python
def create_dataset(min_date, max_date, dest_path, boundaries_gdf):
    rasters = []
    for f in pyPRISMClimate.utils.prism_iterator(dest_path):
        # Open raster
        raster = rxr.open_rasterio(f['full_path'], masked=True)
        # Clip to boundary
        raster = raster.rio.clip(boundaries_gdf.to_crs(raster.rio.crs).geometry)
        # Extract date and add time coordinate
        date = dt.strptime(f['date'], "%Y-%m-%d")
        raster = raster.expand_dims(dim='time')
        raster.coords['time'] = ('time', [date])
        # Clean up and rename
        raster = raster.drop_vars(['spatial_ref']).sel(band=1).drop_vars(['band'])
        raster = raster.rename(f['variable']).rename({'x': 'lon', 'y': 'lat'})
        rasters.append(raster)

    # Merge all variables
    weather_dataset = xr.merge(rasters)
    weather_dataset.to_zarr(output_file, mode='a')
    return weather_dataset
```

**Cleanup**:
```python
# Remove raw BIL files and metadata after processing
weather_files = [f['full_path'] for f in pyPRISMClimate.utils.prism_iterator(dest_path)]
meta_data_files = glob.glob(os.path.join(dest_path, 'PRISM_*_bil.*'))
clean_up_files(weather_files + meta_data_files)
```

**Outputs**:
- Zarr store: `{min_date}_{max_date}_PRISM_data.zarr`
- Plots: Lambert Conformal maps with cartopy

**Execution Time**: ~2-5 minutes per week of data

**Special Notes**:
- Downloads create many temporary files (.bil, .hdr, .aux.xml)
- Use `keep_zip=False` to auto-delete zip files
- Clean up temp files after Zarr creation
- Iterator handles file discovery automatically

**Related Notebooks**: CONUS404_Downloader.ipynb (similar spatial processing)

---

## Precipitation Downloaders

### StageIV_Downloader.ipynb

**Purpose**: Download NOAA Stage IV precipitation and calculate Extreme Rainfall Multiplier (ERM)

**Size**: 953 KB (large - includes plots)

**Data Source**: NOAA Stage IV QPE (Quantitative Precipitation Estimate)

**Key Operations**:
1. Download 3 days of Stage IV NetCDF files
2. Download ERM climatology reference data
3. Calculate 1-day, 2-day, 3-day precipitation
4. Compute ERM for each period
5. Create composite ERM (max of all periods)
6. Generate publication-quality plots

**Dependencies**:
- xarray, numpy
- matplotlib, cartopy
- datetime, os

**Configuration**:
```python
analysis_date = dt.datetime(2025, 12, 11)
plot_area = [-126, -120, 45, 50]  # [lon_min, lon_max, lat_min, lat_max]
mt_baker_lat_lon = (48.7758, -121.8199)
httpdir = "https://water.noaa.gov/resources/downloads/precip/stageIV"
outdir = "../data/stageIV/"
plotdir = "../plots/stageIV/"
```

**Download URLs**:
```python
# Stage IV data
url = f"{httpdir}/{date:%Y/%m/%d}/nws_precip_1day_{date:%Y%m%d}_conus.nc"

# ERM climatology (from UW)
erm_1day = "https://orca.atmos.washington.edu/~bkerns/data/erm/erm_thresholds/max_daily_precip.MEDIAN.s4.2006_2020.nc"
erm_2day = "https://orca.atmos.washington.edu/~bkerns/data/erm/erm_thresholds/max_2day_precip.MEDIAN.s4.2006_2020.nc"
erm_3day = "https://orca.atmos.washington.edu/~bkerns/data/erm/erm_thresholds/max_3day_precip.MEDIAN.s4.2006_2020.nc"
```

**ERM Calculation**:
```python
# Read reference data
with xr.open_dataset(f"{outdir}/max_daily_precip.MEDIAN.s4.2006_2020.nc") as ds:
    erm_ref = ds['precip'].values
    lon = ds['lon'].values
    lat = ds['lat'].values

# Read Stage IV precipitation
with xr.open_dataset(f"{outdir}/nws_precip_1day_{date:%Y%m%d}_conus.nc") as ds:
    precip = ds['observation'].values[::-1, :]  # Flip latitudes

# Calculate ERM
precip_mm = precip * 25.4  # Convert inches to mm
erm = precip_mm / erm_ref  # ERM value

# For composite: max of 1-day, 2-day, 3-day
erm_composite = np.maximum.reduce([erm_1day, erm_2day, erm_3day])
```

**Plotting Functions**:
```python
def get_colormap():
    """Custom colormap for precipitation (12 colors)"""
    # Modifies jet colormap with custom colors
    # Lightgrey for trace, increasing intensity for heavier rain

def add_rainfall(ax, X, Y, Z, min_value=5.0, vmin=1, vmax=22):
    """Add rainfall to cartopy axis with custom colormap"""
    # Converts mm to dBR (logarithmic scale)
    Z_dbr = 10.0 * np.log10(Z / 25.4)
```

**Outputs**:
- Downloaded files: `../data/stageIV/nws_precip_1day_{date}_conus.nc`
- ERM reference: `../data/stageIV/max_{period}_precip.MEDIAN.s4.2006_2020.nc`
- Plots:
  - `StageIV_precip_and_ERM.{date}.daily.png`
  - `StageIV_precip_and_ERM.{date}.2day.png`
  - `StageIV_precip_and_ERM.{date}.3day.png`
  - `StageIV_composite_ERM.{date}.png`

**Plot Features**:
- Dual-panel: Precipitation and ERM
- Custom precipitation colormap (logarithmic scale)
- ERM contours (levels: 0.5-3.0, interval 0.25)
- Bold contours for ERM > 1.0
- Mt. Baker reference marker
- Coastlines, borders, state boundaries

**Execution Time**: ~2-5 minutes

**Special Notes**:
- Precipitation in inches (need conversion)
- Latitude array needs flipping (stored top-to-bottom)
- ERM interpretation:
  - < 1.0: Below normal
  - 1.0-2.0: Elevated risk
  - 2.0-3.0: High risk
  - > 3.0: Extreme risk
- 24-hour accumulation: 12Z-12Z UTC

**Related Notebooks**: PRISM_Downloader.ipynb (precipitation data)

---

### ORNL_Downloader.ipynb

**Purpose**: Download ORNL DAYMET data (not fully implemented in current version)

**Size**: 366 KB

**Data Source**: Oak Ridge National Laboratory Distributed Active Archive Center (DAAC)

**Status**: Notebook template exists but limited implementation shown

**Expected Features** (based on typical DAYMET usage):
- Download daily meteorological data
- North American coverage at 1 km resolution
- Variables: tmin, tmax, prcp, vp, srad, swe, dayl

**Note**: This notebook may require further development for production use.

---

## Hydrological Data Downloaders

### USGS_Stream_Flow_Bulk_Downloader.ipynb

**Purpose**: Download time series data from USGS stream gauges

**Size**: 53 KB

**Data Source**: USGS NWIS (National Water Information System)

**Key Operations**:
1. Load site metadata CSV
2. Iterate through stations
3. Query USGS API for each station
4. Parse RDB format response
5. Save per-station CSV and header
6. Handle time zones and data quality codes

**Dependencies**:
- pandas, requests
- pathlib, collections.defaultdict

**Configuration**:
```python
PWD = Path().cwd()
DATADIR = PWD / 'USGS_Stream_Gage'
SITE_CSV = DATADIR / 'usgs_gage_site_metadata.csv'
batchsize = 10  # Not currently used (sites processed serially)

# Time range
t0 = pd.Timestamp('2025-11-01 00:00:00', tz='US/Pacific')
t1 = pd.Timestamp('2025-12-31 23:59:59', tz='US/Pacific')

# Parameters to download
params = ['stage', 'discharge']  # Maps to USGS codes 00065, 00060
```

**API Configuration**:
```python
_BASE_URL = 'https://nwis.waterservices.usgs.gov/nwis/iv/?'
_PARCD_MAP = {
    'stage': '00065',
    'discharge': '00060',
    'temperature': '00010'
}
_UNITS = {'stage': 'ft', 'discharge': 'cfs', 'temperature': 'C'}
```

**Time Zone Handling**:
```python
_TZ_shift = {
    'UTC': '+00:00', 'PST': '-08:00', 'PDT': '-07:00',
    'MST': '-07:00', 'MDT': '-06:00', 'CST': '-06:00',
    'CDT': '-05:00', 'EST': '-05:00', 'EDT': '-04:00',
    'AKST': '-09:00', 'AKDT': '-08:00', 'HST': '-10:00'
}
```

**Processing Loop**:
```python
for idx, row in df_site.iterrows():
    # Build query URL
    site_str = str(idx)
    param_str = ','.join([_PARCD_MAP[p] for p in params])
    url = (f'{_BASE_URL}sites={site_str}&agencyCd=USGS&'
           f'parameterCd={param_str}&'
           f'startDT={t0.isoformat()}&endDT={t1.isoformat()}&'
           f'format=rdb')

    # Fetch and parse
    request = requests.get(url)
    lines = request.text.split('\n')

    # Parse header, column names, format string, data
    # Convert to DataFrame with proper data types
    # Handle time zones (convert to UTC)

    # Save per-station
    savedir = DATADIR / str(idx)
    data.to_csv(savedir / f'{idx}_data.csv')
    with open(savedir / f'{idx}_header.txt', 'w') as f:
        for hdr_line in hdr:
            f.write(f'{hdr_line}\n')
```

**Output Structure**:
```
USGS_Stream_Gage/
└── {station_id}/
    ├── {station_id}_data.csv
    │   Columns:
    │     - datetime (UTC, index)
    │     - discharge_cfs (float)
    │     - discharge_qual (str, quality code)
    │     - gage_height_ft (float)
    │     - gage_height_qual (str)
    │
    └── {station_id}_header.txt
        (Metadata lines starting with #)
```

**Execution Time**: ~5-15 minutes for 743 stations (serial processing)

**Special Notes**:
- Processes sites serially (no parallelism)
- Handles missing time zones gracefully
- Drops unparseable datetime rows
- Status code 400 = bad query (skips station)
- USGS API limit: 100 sites per query (not utilized)

**Error Handling**:
```python
if request.status_code == 400:
    print('status_code: 400 - bad query - skipping')
    continue

bad_dt = dt_utc.isna()
if bad_dt.any():
    print(f'Warning: dropping {int(bad_dt.sum())} rows with unparseable datetime')
```

**Related Notebooks**:
- `USGS_Stream_Gage_Site_Metadata_Downloader.ipynb` (prerequisite)
- `USGS_IV_Downloader_Demo_ipynb.ipynb` (simpler demo)

---

### USGS_Stream_Gage_Site_Metadata_Downloader.ipynb

**Purpose**: Download USGS stream gauge site metadata and create spatial inventory

**Size**: 24 KB

**Data Source**: USGS WaterWatch / NWIS

**Key Operations**:
1. Query USGS WaterWatch for active gauges
2. Parse metadata (location, status, discharge)
3. Create GeoDataFrame
4. Save as CSV and GeoJSON

**Note**: Specific implementation details would require reading the full notebook.

**Outputs**:
- `USGS_Stream_Gage/usgs_gage_site_metadata.csv`
- `stations_by_basin.csv`
- `stations_by_basin_with_gages.csv`

**Related Notebooks**:
- `USGS_Stream_Flow_Bulk_Downloader.ipynb` (uses output CSV)

---

### USGS_IV_Downloader_Demo_ipynb.ipynb

**Purpose**: Demonstration of USGS instantaneous values API

**Size**: 22 KB

**Scope**: Simplified example for learning the USGS API

**Related Notebooks**:
- `USGS_Stream_Flow_Bulk_Downloader.ipynb` (production version)

---

### USGS_IRIS_crossref.ipynb

**Purpose**: Cross-reference USGS stream gauges with IRIS seismic stations

**Size**: 174 KB

**Operations**:
- Find USGS gauges near seismic stations
- Identify co-located or nearby monitoring sites
- Support multi-hazard analysis

**Use Case**: Enable integrated hydrological-seismic analysis

---

## Inventory and Analysis Notebooks

### merged-inventory.ipynb

**Purpose**: Create unified station inventory from multiple sources (precipitation, streamflow, seismic)

**Size**: 15 KB

**Data Sources**:
- Synoptic Data (precipitation stations)
- USGS (streamflow stations)
- IRIS (seismic stations)

**Key Operations**:
1. Query Synoptic API for precipitation stations in WA
2. Filter active, unrestricted stations
3. Query USGS streamflow stations (via Synoptic)
4. Query IRIS seismic stations (via ObsPy)
5. Add SimpleStyle properties for web visualization
6. Save individual and combined GeoJSON files
7. Create interactive Folium map

**Dependencies**:
- requests, os (for API access)
- pandas, geopandas
- obspy (for seismic data)

**Configuration**:
```python
TOKEN = os.environ.get('SYNOPTIC_TOKEN')
baseAPI = "https://api.synopticdata.com/v2"
```

**Precipitation Stations Query**:
```python
params = dict(
    state="wa",
    token=TOKEN,
    sensorvars=True,
    vars='precip_accum',  # Filter to precip-only
    output='geojson'
)
response = requests.get(baseAPI + "/stations/metadata", params=params)
gfp = gpd.GeoDataFrame.from_features(response.json()['features'])

# Filter
gfp = gfp[(gfp.status == 'ACTIVE') & (gfp.restricted_data == False)]
```

**Streamflow Stations Query**:
```python
params = dict(
    state="wa",
    token=TOKEN,
    vars='stream_flow',
    output='geojson'
)
# Similar processing...
```

**Seismic Stations Query**:
```python
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

client = Client('IRIS')
aoi = gpd.read_file("WA_state_boundary.geojson")
minlon, minlat, maxlon, maxlat = aoi.total_bounds

inv = client.get_stations(
    channel='EHZ,HHZ,ENZ,HNZ,BNZ',  # Vertical components
    minlatitude=minlat, maxlatitude=maxlat,
    minlongitude=minlon, maxlongitude=maxlon,
    starttime=UTCDateTime('2025-01-01'),
    endtime=UTCDateTime('2025-12-31'),
    level='station'
)

# Convert to DataFrame, then GeoDataFrame
```

**Styling**:
```python
# Precipitation stations
gfp['marker-color'] = '#0000FF'  # Blue
gfp['marker-size'] = 'small'
gfp['marker-symbol'] = 'water'

# Streamflow stations
gfs['marker-color'] = '#00FFFF'  # Cyan
gfs['marker-symbol'] = 'waterfall'

# Seismic stations
gf_seis['marker-color'] = '#FF00FF'  # Magenta
gf_seis['marker-symbol'] = 'defibrillator'
```

**Outputs**:
- `precip-stations-wa-styled.geojson`
- `streamflow-stations-wa-styled.geojson`
- `seismic-stations-wa-styled.geojson`
- `combined-stations-wa-styled.geojson`

**GeoJSON Combination**:
```python
import json, glob

geojson_files = glob.glob("*.geojson")
all_features = []

for fname in geojson_files:
    with open(fname) as f:
        data = json.load(f)
        all_features.extend(data['features'])

# Write combined FeatureCollection
with open("combined-stations-wa-styled.geojson", "w") as f:
    f.write('{"type": "FeatureCollection", "features": [...]}')
```

**Execution Time**: ~1-2 minutes

**Special Notes**:
- Requires Synoptic API token in environment
- SimpleStyle spec for GeoJSON styling (supported by geojson.io, GitHub)
- Maki icons for marker symbols
- Combined file enables multi-layer web maps

**Related Notebooks**:
- `select_stations_by_watershed_nc.ipynb` (spatial selection)
- `S3-bucket.ipynb` (publishing to web)

---

### select_stations_by_watershed_nc.ipynb

**Purpose**: Select stations within specific watersheds/basins

**Size**: 67 MB (very large - likely has extensive outputs)

**Operations**:
- Load watershed boundaries (HUC polygons)
- Spatial join with station inventories
- Filter stations by basin membership

**Outputs**:
- `stations_by_basin.csv` (without USGS gages)
- `stations_by_basin_with_gages.csv` (with USGS)

**Use Case**: Create basin-specific station lists for hydrological modeling

---

### hydroclimatology_huc_local.ipynb

**Purpose**: Hydroclimatological analysis at HUC (Hydrologic Unit Code) scale

**Size**: 12 MB (large - analysis results)

**Scope**: Analyze climate and hydrological patterns within specific HUCs

**Note**: Specific details would require full notebook inspection.

---

### NC_Plot_snotel.ipynb

**Purpose**: Plot and analyze SNOTEL (Snow Telemetry) data from NetCDF files

**Size**: 110 KB

**Operations**:
- Read SNOTEL data (snow depth, SWE)
- Generate time series plots
- Spatial analysis across SNOTEL network

**Use Case**: Snow hydrology analysis for mountainous watersheds

---

### weather.ipynb

**Purpose**: General weather data loading and analysis

**Size**: 3.4 MB (large - includes plots)

**Scope**: Exploratory analysis and visualization of weather data

**Note**: This appears to be a working/scratch notebook for various analyses.

---

## Infrastructure Notebooks

### S3-bucket.ipynb

**Purpose**: Demonstrate interaction with CRESST AWS S3 bucket

**Size**: 2.0 MB (includes outputs and examples)

**Key Operations**:
1. Upload derived data products to S3
2. Read remote data from S3
3. Demonstrate cloud-optimized workflows
4. Create and share data products

**Dependencies**:
- obstore (preferred S3 library)
- boto3 (alternative)
- xarray, rasterio, rioxarray

**S3 Configuration**:
```python
from obstore.auth.boto3 import Boto3CredentialProvider
from obstore.store import S3Store

store = S3Store(
    "cresst",
    credential_provider=Boto3CredentialProvider(),
    region="us-west-2"
)
```

**Example Workflow: NISAR Data**:
```python
# 1. Download NISAR soil moisture product
url = 'https://nisar.asf.earthdatacloud.nasa.gov/...'
!wget -nc {url}

# 2. Extract single algorithm result
algorithm = 'DSG'
ds = xr.open_dataset(f'NETCDF:{filename}', group=f'/science/LSAR/SME2/grids/algorithmCandidates/{algorithm}', engine='rasterio')

# 3. Convert to Cloud-Optimized GeoTIFF
output_filename = filename.replace('.h5', f'_{algorithm}.tif')
ds['soilMoisture'].rio.to_raster(output_filename, driver='COG', compress='DEFLATE')

# 4. Upload to S3
with open(output_filename, 'rb') as content:
    obstore.put(store, f'scottsfiles/{output_filename}', content)

# 5. Read from S3 (demonstrate access)
da = xr.open_dataarray(f's3://cresst/scottsfiles/{output_filename}', engine='rasterio')
```

**Bucket Organization**:
```
s3://cresst/
├── README.md
├── {username}/
├── stehekin/planet/  # Satellite imagery
└── scottsfiles/      # Shared products
```

**Key Patterns**:
- Use `obstore` for modern Python S3 access
- Cloud-Optimized GeoTIFF (COG) for raster products
- Include metadata in file names
- Organize by user/project

**Outputs**:
- Uploaded files in S3 bucket
- Example: `scottsfiles/NISAR_..._DSG.tif`

**Execution Time**: Variable (depends on upload size and network)

**Special Notes**:
- Bucket allows anonymous read
- Write requires AWS credentials
- Use `AWS_PROFILE` environment variable
- Credentials in `~/.aws/credentials`

**Related Notebooks**: All downloaders (can publish outputs)

---

### download_DEM.ipynb

**Purpose**: Download and process digital elevation models

**Size**: 195 KB

**Data Source**: SRTM via `elevation` library

**Key Operations**:
1. Download SRTM DEM for bounding box
2. Load and clip to polygon boundary
3. Visualize terrain

**Dependencies**:
- elevation (handles GDAL/SRTM)
- rioxarray, rasterio
- geopandas

**Configuration**:
```python
# Skagit bounds
bounding_box = (-122.6, 47.5, -120, 49.5)  # (W, S, E, N)
file_path = '../data/GIS/SkagitRiver_90mDEM.tif'
boundary_path = '../data/GIS/SkagitBoundary.json'
```

**Workflow**:
```python
# 1. Download DEM
elevation.clip(bounds=bounding_box, output=file_path)

# 2. Load boundary
sf = gpd.read_file(boundary_path)

# 3. Load DEM and clip
dem = rioxarray.open_rasterio(file_path, masked=True)
skagit_dem = dem.rio.clip(sf.geometry)

# 4. Plot
skagit_dem.plot()
```

**Outputs**:
- GeoTIFF: `../data/GIS/SkagitRiver_90mDEM.tif`
- Plot of clipped DEM

**Execution Time**: ~1-2 minutes

**Special Notes**:
- `elevation` library handles tile downloads automatically
- Data cached in `~/.cache/elevation/`
- SRTM3 (90m) is default, SRTM1 (30m) available for US

**Related Notebooks**: Any notebook requiring terrain data

---

### converting_data_workflow.ipynb

**Purpose**: Template/example for data format conversions

**Size**: 2.9 KB (small - template)

**Scope**: Demonstrate conversions between formats (NetCDF, Zarr, GeoTIFF, CSV)

**Use Case**: Standardize data formats for interoperability

---

## Quick Reference Table

| Notebook | Size | Primary Purpose | Execution Time | Dependencies |
|----------|------|-----------------|----------------|--------------|
| CONUS404_Downloader.ipynb | 14KB | Download CONUS404 reanalysis | 10-30 min | xarray, geopandas, regionmask, s3fs |
| HRRR_Downloader.ipynb | 8.6KB | Download HRRR forecasts | 5-20 min | herbie, cfgrib, wgrib2* |
| WRF_Downloader.ipynb | 2.0MB | Download WRF-CMIP6 projections | 5-15 min | boto3, metpy, cartopy |
| PRISM_Downloader.ipynb | 258KB | Download PRISM daily data | 2-5 min/week | pyPRISMClimate, cartopy |
| StageIV_Downloader.ipynb | 953KB | Download Stage IV + ERM | 2-5 min | cartopy, xarray |
| ORNL_Downloader.ipynb | 366KB | Download DAYMET (template) | N/A | TBD |
| USGS_Stream_Flow_Bulk_Downloader.ipynb | 53KB | Download USGS time series | 5-15 min | requests, pandas |
| USGS_Stream_Gage_Site_Metadata_Downloader.ipynb | 24KB | Download USGS site metadata | 1-2 min | requests, geopandas |
| USGS_IV_Downloader_Demo_ipynb.ipynb | 22KB | USGS API demo | <1 min | requests |
| USGS_IRIS_crossref.ipynb | 174KB | Cross-reference USGS/IRIS | 2-5 min | geopandas, obspy |
| merged-inventory.ipynb | 15KB | Create unified station inventory | 1-2 min | requests, obspy, Synoptic token* |
| select_stations_by_watershed_nc.ipynb | 67MB | Select stations by basin | Variable | geopandas |
| hydroclimatology_huc_local.ipynb | 12MB | HUC-scale analysis | Variable | xarray, geopandas |
| NC_Plot_snotel.ipynb | 110KB | Plot SNOTEL data | Variable | xarray, matplotlib |
| weather.ipynb | 3.4MB | General weather analysis | Variable | Various |
| S3-bucket.ipynb | 2.0MB | S3 integration demo | Variable | obstore/boto3, AWS creds* |
| download_DEM.ipynb | 195KB | Download elevation data | 1-2 min | elevation, rioxarray |
| converting_data_workflow.ipynb | 2.9KB | Format conversion template | N/A | xarray, rioxarray |

*System requirements or credentials needed

---

**Last Updated**: 2026-02-10
