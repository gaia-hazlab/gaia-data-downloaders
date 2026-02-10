# Data Sources Reference

## Table of Contents

1. [Overview](#overview)
2. [Weather and Climate Models](#weather-and-climate-models)
3. [Precipitation Products](#precipitation-products)
4. [Hydrological Data](#hydrological-data)
5. [Terrain Data](#terrain-data)
6. [Station Networks](#station-networks)
7. [API Authentication](#api-authentication)
8. [Data Access Patterns](#data-access-patterns)

## Overview

This document provides detailed information about all data sources integrated in the gaia-data-downloaders system, including access methods, data formats, temporal/spatial coverage, and usage guidelines.

### Data Source Summary

| Source | Type | Coverage | Resolution | Update Frequency | Access Method |
|--------|------|----------|------------|------------------|---------------|
| CONUS404 | Reanalysis | CONUS | 4 km, hourly/daily | Historical (1979-2023) | AWS S3 (anonymous) |
| HRRR | Weather Model | CONUS | 3 km, hourly | Real-time + 2-day archive | AWS S3 (Herbie) |
| WRF-CMIP6 | Climate Model | Western US | 9 km, hourly | Historical + Projections | AWS S3 (anonymous) |
| PRISM | Gridded Obs | CONUS | 4 km, daily | Daily updates | HTTP download |
| Stage IV | Radar+Gauge | CONUS | 4 km, daily | Daily (12Z-12Z) | NOAA HTTP |
| USGS Gages | Stream Obs | National | Point, 15-min | Real-time | REST API |
| ORNL DAYMET | Gridded Obs | North America | 1 km, daily | Annual updates | ORNL DAAC |
| SRTM DEM | Elevation | Global | 90 m | Static | HTTP via elevation lib |
| Synoptic | Weather Stations | US | Point, variable | Real-time | REST API (token) |
| IRIS | Seismic | Global | Point, continuous | Real-time | FDSN web services |

## Weather and Climate Models

### CONUS404

**Description**: High-resolution atmospheric reanalysis for the Contiguous United States, 1979-2023, produced by NCAR using WRF model.

**Notebook**: `CONUS404_Downloader.ipynb`

**Access Details**:
- **Endpoint**: `s3://hytest/conus404/conus404_daily.zarr` (daily) or `conus404_hourly.zarr` (hourly)
- **Protocol**: S3 via OSN Pod (Open Storage Network)
- **Authentication**: Anonymous access
- **Storage Options**: `{'anon': True, 'client_kwargs': {'endpoint_url': 'https://usgs.osn.mghpcc.org'}}`

**Coverage**:
- **Spatial**: CONUS domain (approximately 25°N-53°N, 125°W-67°W)
- **Temporal**: October 1979 - December 2023
- **Resolution**: 4 km horizontal, hourly or daily aggregated

**Key Variables**:
- Meteorology: `T2` (temperature), `Q2` (humidity), `U10`/`V10` (wind), `PSFC` (pressure)
- Radiation: `SWDNB` (shortwave down), `LWDNB` (longwave down)
- Precipitation: `RAINC` (convective), `RAINNC` (non-convective), `SNOWNC` (snow)
- Soil: `SMOIS` (moisture), `TSLB` (temperature)
- Surface: `HFX` (sensible heat), `LH` (latent heat), `HGT` (terrain height)

**Derived Variables** (computed in notebook):
- `WS10`: Wind speed = sqrt(U10² + V10²)
- `PRECIP_TOT`: Total precip = RAINC + RAINNC + SNOWNC
- `RH`: Relative humidity from T2, Q2, PSFC
- `VPD`: Vapor pressure deficit

**Usage Patterns**:
```python
import xarray as xr

# Open remote Zarr store
ds = xr.open_zarr(
    store='s3://hytest/conus404/conus404_daily.zarr',
    storage_options={
        'anon': True,
        'client_kwargs': {'endpoint_url': 'https://usgs.osn.mghpcc.org'}
    },
    consolidated=True
)

# Subset time and variables
ds = ds.sel(time=slice('2014-10-01', '2023-12-31'))
ds = ds[['T2', 'PRECIP_TOT', 'WS10']]
```

**Limitations**:
- Large file sizes (daily ~1 TB, hourly ~10 TB)
- Subset spatially before downloading
- Use dask for lazy evaluation

**Citation**: Liu et al. (2017), *Bulletin of the American Meteorological Society*

---

### HRRR (High-Resolution Rapid Refresh)

**Description**: NOAA operational weather model providing hourly forecasts for CONUS at 3 km resolution.

**Notebook**: `HRRR_Downloader.ipynb`

**Access Details**:
- **Library**: `herbie-data` (Python wrapper)
- **Backend**: AWS S3 and NOAA archives
- **Authentication**: Anonymous
- **Data Format**: GRIB2 (requires wgrib2 executable)

**Coverage**:
- **Spatial**: CONUS
- **Temporal**: 2014-present (varies by archive)
- **Resolution**: 3 km horizontal, hourly forecasts
- **Forecast Horizon**: 0-48 hours

**Key Variables**:
- `TMP:surface` - Temperature
- `RH:2 m above ground` - Relative Humidity
- `WIND:10 m above ground` - Wind Speed
- `APCP:surface:0-1 hour acc fcst` - Accumulated Precipitation (hourly)
- `DSWRF:surface` - Downward Shortwave Radiation
- `DLWRF:surface` - Downward Longwave Radiation

**Usage Patterns**:
```python
from herbie import FastHerbie
import pandas as pd

# Define date range and parameters
dates = pd.date_range('2020-03-01', '2020-03-02', freq='1h')
parameters = {
    'TMP': 'surface',
    'RH': '2 m above ground',
    'WIND': '10 m above ground',
    'APCP': 'surface:0-1 hour acc fcst'
}

# Download with FastHerbie
fh = FastHerbie(dates, model='hrrr', product='sfc', fxx=range(0,2))
fields = [f":{param}:{level}" for param, level in parameters.items()]
param_regex = fr"^(?:{'|'.join(fields)})"
files = fh.download(param_regex)
```

**Special Requirements**:
- **wgrib2**: Must be in system PATH for GRIB2 processing
- **Environment**: Set `PATH` to include wgrib2 binary

**Limitations**:
- GRIB2 format requires specialized tools
- Limited historical archive depth
- Individual files per hour (many small files)

**Citation**: NOAA/NCEP/EMC

---

### WRF-CMIP6

**Description**: Weather Research and Forecasting (WRF) model dynamically downscaling CMIP6 global climate models for Western US.

**Notebook**: `WRF_Downloader.ipynb`

**Access Details**:
- **Endpoint**: `s3://wrf-cmip6-noversioning/downscaled_products/gcm/`
- **Protocol**: S3 via boto3 (anonymous)
- **Authentication**: `Config(signature_version=UNSIGNED)`
- **Browser**: https://wrf-cmip6-noversioning.s3.amazonaws.com/index.html

**Coverage**:
- **Spatial**: Western US (domains d01: 27km, d02: 9km)
- **Temporal**: Historical (1950-2014) and SSP scenarios (2015-2100)
- **Resolution**: 9 km horizontal (domain 2), hourly
- **Models**: Multiple CMIP6 GCMs (e.g., CESM2, MPI-ESM)

**File Structure**:
- Path: `downscaled_products/gcm/{model}_{scenario}/hourly/{year}/d02/`
- Files: `wrfout_d01_YYYY-MM-DD_HH:MM:SS` (tier 1) or `auxhist_d01_...` (tier 2)

**Key Variables**:
- Basic: `T2`, `Q2`, `PSFC`, `U10`, `V10`
- Precipitation: `RAINC`, `RAINNC`, `SNOW`
- Radiation: `SWDNB`, `LWDNB`, `SWUPB`, `LWUPB`
- Soil/Surface: `HFX`, `LH`, `RUNSF`, `RUNSB`

**Derived Variables**:
- `RH`: Relative humidity using MetPy
- `WS10`: Wind speed from U10, V10
- `TP`: Total precipitation (RAINC + RAINNC)

**Usage Patterns**:
```python
import boto3
from botocore import UNSIGNED
from botocore.client import Config

# Setup S3 client
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Generate file list
def generateFileNames(start, end, model, domain, historical=True, bc=False):
    dates = pd.date_range(start, end, freq='1h')
    path = f'downscaled_products/gcm/{model}{"_historical" if historical else ""}{"_bc" if bc else ""}/hourly'
    return [f'{path}/{d.year}/d0{domain}/wrfout_d01_{d:%Y-%m-%d_%H:%M:%S}' for d in dates]

# Parallel download
with ThreadPoolExecutor(24) as executor:
    files = executor.map(lambda f: s3.download_file('wrf-cmip6-noversioning', f, f'local/{f}'), file_list)
```

**Special Notes**:
- Bias-corrected versions available (suffix `_bc`)
- Requires coordinate metadata file: `wrfinput_d02.nc` or `wrfinput_d02_coord.nc`
- WRF uses Lambert Conformal projection (need to assign lat/lon coords)

**Limitations**:
- Large data volume (hourly files add up quickly)
- Non-standard time coordinate (needs parsing)
- Projection requires metadata file for lat/lon

**Citation**: University of Washington / Pacific Northwest National Laboratory

---

## Precipitation Products

### PRISM (Parameter-elevation Regressions on Independent Slopes Model)

**Description**: Gridded climate observations for CONUS, combining weather station data with terrain analysis.

**Notebook**: `PRISM_Downloader.ipynb`

**Access Details**:
- **Library**: `pyPRISMClimate` (Python wrapper)
- **Backend**: Oregon State University PRISM Climate Group
- **Protocol**: HTTP download
- **Authentication**: None (public access)

**Coverage**:
- **Spatial**: CONUS
- **Temporal**: 1895-present (daily data)
- **Resolution**: 4 km (~800 m available for smaller regions)

**Available Variables**:
- `tmean`: Mean temperature (°C)
- `tmax`: Maximum temperature (°C)
- `tmin`: Minimum temperature (°C)
- `ppt`: Precipitation (mm)
- `vpdmax`: Maximum vapor pressure deficit (hPa)
- `vpdmin`: Minimum vapor pressure deficit (hPa)
- `tdmean`: Mean dew point temperature (°C)

**Usage Patterns**:
```python
import pyPRISMClimate
from concurrent.futures import ThreadPoolExecutor

# Download multiple variables in parallel
variables = ['tmean', 'tmax', 'tmin', 'ppt', 'vpdmax', 'vpdmin']
with ThreadPoolExecutor(4) as executor:
    executor.map(
        lambda var: pyPRISMClimate.get_prism_dailys(
            var,
            min_date='2017-02-01',
            max_date='2017-02-08',
            dest_path='../data/weather_data/',
            keep_zip=False
        ),
        variables
    )
```

**Data Format**:
- Downloaded as BIL (Band Interleaved by Line) raster
- Metadata in separate files (.bil.aux.xml, .bil.hdr)
- Can be opened with rioxarray

**Processing Pattern**:
```python
import rioxarray as rxr

# Iterate through downloaded files
for f in pyPRISMClimate.utils.prism_iterator('../data/weather_data/'):
    raster = rxr.open_rasterio(f['full_path'], masked=True)
    raster = raster.rio.clip(boundary.geometry)
    # Extract date from filename
    date = dt.strptime(f['date'], '%Y-%m-%d')
    # Add to time series
```

**Limitations**:
- Daily data only (no sub-daily)
- CONUS only
- Individual files per variable/day (cleanup recommended)

**Citation**: PRISM Climate Group, Oregon State University

---

### NOAA Stage IV Precipitation Analysis

**Description**: Multi-sensor precipitation analysis combining NEXRAD radar and rain gauge observations, quality-controlled by NWS River Forecast Centers.

**Notebook**: `StageIV_Downloader.ipynb`

**Access Details**:
- **Endpoint**: `https://water.noaa.gov/resources/downloads/precip/stageIV/`
- **Protocol**: HTTP download (wget)
- **Format**: NetCDF
- **Authentication**: None

**Coverage**:
- **Spatial**: CONUS
- **Temporal**: 2002-present (near real-time)
- **Resolution**: 4 km horizontal
- **Aggregation**: Daily (12Z-12Z)

**File Naming**:
- Pattern: `nws_precip_1day_YYYYMMDD_conus.nc`
- URL: `{endpoint}/{YYYY}/{MM}/{DD}/{filename}`

**Variables**:
- `observation`: 24-hour accumulated precipitation (inches)
- Grid: Polar stereographic projection

**Usage Patterns**:
```python
import xarray as xr
import datetime as dt

# Download 3 days of data
analysis_date = dt.datetime(2025, 12, 11)
for i in range(3):
    date = analysis_date - dt.timedelta(days=i)
    url = f"https://water.noaa.gov/resources/downloads/precip/stageIV/{date:%Y/%m/%d}/nws_precip_1day_{date:%Y%m%d}_conus.nc"
    os.system(f"wget -nc -O ../data/stageIV/{os.path.basename(url)} {url}")

# Open and process
ds = xr.open_dataset('../data/stageIV/nws_precip_1day_20251211_conus.nc')
precip = ds['observation'].values[::-1, :]  # Flip latitudes
```

**Extreme Rainfall Multiplier (ERM)**:
- Notebook calculates ERM by comparing to climatological thresholds
- Reference data: `max_daily_precip.MEDIAN.s4.2006_2020.nc` (from UW)
- ERM = observed_precip / climatological_threshold
- Computed for 1-day, 2-day, 3-day accumulations
- Composite ERM = max(ERM_1day, ERM_2day, ERM_3day)

**Special Notes**:
- Times in UTC (12Z-12Z accumulation)
- Precipitation in inches (convert to mm: × 25.4)
- Latitudes need flipping (stored top-to-bottom)

**Citation**: NOAA National Weather Service River Forecast Centers

---

## Hydrological Data

### USGS Water Data Services

**Description**: United States Geological Survey stream gauge network providing real-time and historical water data.

**Notebooks**:
- `USGS_Stream_Flow_Bulk_Downloader.ipynb` - Download time series data
- `USGS_Stream_Gage_Site_Metadata_Downloader.ipynb` - Download site metadata
- `USGS_IV_Downloader_Demo_ipynb.ipynb` - Instantaneous values demo

**Access Details**:
- **Endpoint**: `https://waterdata.usgs.gov/nwis/` (web interface)
- **API**: `https://nwis.waterservices.usgs.gov/nwis/iv/` (REST API for instantaneous values)
- **Protocol**: HTTP REST API (no authentication)
- **Format**: RDB (tab-delimited text with metadata)

**Coverage**:
- **Spatial**: United States + territories
- **Temporal**: Varies by site (some 100+ years)
- **Resolution**: Typically 15-minute (instantaneous values)
- **Stations**: 743 active stations in Washington State (as of query)

**Parameter Codes**:
- `00060`: Discharge (cubic feet per second)
- `00065`: Gage height (feet)
- `00010`: Temperature (°C)

**API Query Structure**:
```
Base URL: https://nwis.waterservices.usgs.gov/nwis/iv/?
Parameters:
  - sites={station_id}          # USGS site number
  - agencyCd=USGS              # Agency code
  - parameterCd={param_codes}  # Comma-separated parameter codes
  - startDT={iso_timestamp}    # Start datetime (ISO 8601)
  - endDT={iso_timestamp}      # End datetime (ISO 8601)
  - format=rdb                 # Output format
```

**Usage Patterns**:
```python
import requests
import pandas as pd

# Build query URL
site = '12200500'  # Skagit River at Mt. Vernon
params = '00060,00065'  # Discharge and stage
url = (f'https://nwis.waterservices.usgs.gov/nwis/iv/?'
       f'sites={site}&agencyCd=USGS&'
       f'parameterCd={params}&'
       f'startDT=2025-11-01T00:00:00-08:00&'
       f'endDT=2025-12-31T23:59:59-08:00&'
       f'format=rdb')

# Fetch data
response = requests.get(url)

# Parse RDB format (skip header lines starting with #)
lines = [l for l in response.text.split('\n') if not l.startswith('#')]
# Parse columns and data...
```

**RDB Format Details**:
- Lines starting with `#` are comments/metadata
- First non-comment line: column headers
- Second line: data types (e.g., `5s`, `10n`)
- Remaining lines: data (tab-separated)

**Output Structure** (per station):
```
USGS_Stream_Gage/
└── {station_id}/
    ├── {station_id}_data.csv       # Parsed time series
    │   Columns: datetime, discharge_cfs, discharge_qual,
    │            gage_height_ft, gage_height_qual
    └── {station_id}_header.txt     # Metadata from API response
```

**Time Zones**:
- Data returned in local time zone (PST/PDT for WA)
- Notebook converts to UTC for consistency
- Timezone mapping: `_TZ_shift` dictionary

**Rate Limiting**:
- 100 sites per query maximum
- No explicit rate limit, but be respectful
- Notebook processes sites serially to avoid overload

**Metadata Fields**:
- `id`: USGS station number
- `name`: Station name
- `lat`, `lng`: Coordinates (WGS84)
- `huc_cd`: Hydrologic Unit Code
- `class`: Stream classification
- `url`: Web link to station page

**Limitations**:
- Limited to 120 days per query (use multiple queries for longer periods)
- Time zone handling requires care
- Data quality codes need interpretation

**Citation**: U.S. Geological Survey National Water Information System

---

## Terrain Data

### SRTM Digital Elevation Model

**Description**: Shuttle Radar Topography Mission global elevation data at 90m resolution.

**Notebook**: `download_DEM.ipynb`

**Access Details**:
- **Library**: `elevation` (Python wrapper for GDAL/SRTM data)
- **Backend**: NASA SRTM archive
- **Protocol**: Automatic download and mosaicking via GDAL
- **Authentication**: None

**Coverage**:
- **Spatial**: Near-global (60°N - 56°S)
- **Resolution**: 90 m (SRTM3), 30 m (SRTM1 for US)
- **Vertical Accuracy**: ~16 m absolute, ~6 m relative

**Usage Patterns**:
```python
import elevation
import rioxarray

# Download and clip to bounding box
bounds = (-122.6, 47.5, -120, 49.5)  # (W, S, E, N)
output_path = '../data/GIS/SkagitRiver_90mDEM.tif'
elevation.clip(bounds=bounds, output=output_path)

# Open and clip to polygon
dem = rioxarray.open_rasterio(output_path, masked=True)
boundary = gpd.read_file('../data/GIS/SkagitBoundary.json')
dem_clipped = dem.rio.clip(boundary.geometry)
```

**Data Format**:
- GeoTIFF
- Vertical datum: EGM96 (Earth Gravitational Model 1996)
- Horizontal datum: WGS84

**Special Notes**:
- The `elevation` library handles tile mosaicking automatically
- Data is cached locally (check ~/.cache/elevation/)
- For large areas, consider pre-downloading tiles

**Alternative Sources**:
- USGS 3DEP: Higher resolution for US (10m, 30m)
- Copernicus DEM: Global 30m
- ASTER GDEM: Global 30m (alternative)

**Citation**: NASA/JPL SRTM Mission

---

## Station Networks

### Synoptic Weather Stations

**Description**: MesoWest/Synoptic Data API providing access to weather station observations from multiple networks.

**Notebook**: `merged-inventory.ipynb`

**Access Details**:
- **Endpoint**: `https://api.synopticdata.com/v2/stations/metadata`
- **Protocol**: REST API (HTTPS)
- **Authentication**: Token required (academic accounts available)
- **Token Source**: Environment variable `SYNOPTIC_TOKEN`

**Coverage**:
- **Spatial**: Global (primary US coverage)
- **Networks**: ASOS, RAWS, SNOTEL, CWOP, personal weather stations, etc.
- **Variables**: Temperature, precipitation, wind, snow, etc.

**Query Parameters**:
- `state`: Two-letter state code (e.g., `wa`)
- `token`: API authentication token
- `vars`: Filter by variable (e.g., `precip_accum`)
- `sensorvars`: Include sensor variable info
- `output`: Format (`json` or `geojson`)

**Usage Patterns**:
```python
import requests
import os

TOKEN = os.environ.get('SYNOPTIC_TOKEN')
url = 'https://api.synopticdata.com/v2/stations/metadata'

# Query precipitation stations in Washington
params = {
    'state': 'wa',
    'token': TOKEN,
    'vars': 'precip_accum',
    'output': 'geojson'
}
response = requests.get(url, params=params)
data = response.json()

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame.from_features(data['features'], crs='EPSG:4326')
```

**Output Styling**:
- Notebook adds SimpleStyle properties for web visualization:
  - `marker-color`: `#0000FF` (blue) for precip stations
  - `marker-size`: `small`
  - `marker-symbol`: `water` (Maki icon)

**Station Filtering**:
- `status == 'ACTIVE'`: Only currently operational
- `restricted_data == False`: Publicly accessible data

**Key Fields**:
- `id`, `stid`: Station identifiers
- `name`: Station name
- `elevation`: Elevation in meters
- `period_of_record`: Start and end datetimes
- `station_info`: URL to station details

**Limitations**:
- API rate limits (varies by account type)
- Academic token required for full access
- Data quality varies by network

**Citation**: Synoptic Data PBC (formerly MesoWest)

---

### IRIS Seismic Network

**Description**: Incorporated Research Institutions for Seismology (IRIS) network providing seismic data via FDSN web services.

**Notebook**: `merged-inventory.ipynb`

**Access Details**:
- **Library**: `obspy` (Python framework for seismology)
- **Client**: `obspy.clients.fdsn.Client('IRIS')`
- **Protocol**: FDSN web services
- **Authentication**: None (public data)

**Coverage**:
- **Spatial**: Global
- **Networks**: Multiple (e.g., UW, CC, UO for Pacific Northwest)
- **Channels**: Broadband, short-period, strong-motion seismometers

**Query Parameters**:
```python
staqkwargs = {
    'channel': 'EHZ,HHZ,ENZ,HNZ,BNZ',  # Vertical component channels
    'minlatitude': 45.0,
    'maxlatitude': 50.0,
    'minlongitude': -125.0,
    'maxlongitude': -120.0,
    'starttime': UTCDateTime('2025-01-01'),
    'endtime': UTCDateTime('2025-12-31'),
    'level': 'station'  # Metadata level
}
```

**Usage Patterns**:
```python
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

client = Client('IRIS')

# Get station inventory
inv = client.get_stations(
    channel='EHZ,HHZ',
    minlatitude=47.0,
    maxlatitude=49.0,
    minlongitude=-123.0,
    maxlongitude=-120.0,
    starttime=UTCDateTime('2025-01-01'),
    level='station'
)

# Extract to DataFrame
stations = []
for net in inv.networks:
    for sta in net.stations:
        stations.append({
            'network': net.code,
            'station': sta.code,
            'latitude': sta.latitude,
            'longitude': sta.longitude,
            'elevation': sta.elevation,
            'start_datetime': sta.start_date.datetime,
            'end_datetime': sta.end_date.datetime if sta.end_date else None
        })
```

**Output Styling**:
- `marker-color`: `#FF00FF` (magenta)
- `marker-size`: `small`
- `marker-symbol`: `defibrillator` (represents monitoring/sensing)

**Key Fields**:
- `network`: Network code (e.g., `UW`, `CC`)
- `station`: Station code (e.g., `MORA`, `STW`)
- `latitude`, `longitude`: WGS84 coordinates
- `elevation`: Elevation in meters
- `station_info`: URL to IRIS metadata aggregator

**Station URL Pattern**:
```
https://ds.iris.edu/mda/{network}/{station}
```

**Limitations**:
- FDSN services have rate limits
- Some networks restrict access
- Metadata may be incomplete for older stations

**Citation**: IRIS Data Management Center

---

## API Authentication

### Synoptic API Token

**Required For**: Synoptic weather station data

**Setup**:
1. Register for academic account at https://synopticdata.com
2. Obtain API token from account dashboard
3. Set environment variable:
   ```bash
   export SYNOPTIC_TOKEN='your_token_here'
   ```
4. Access in Python:
   ```python
   import os
   TOKEN = os.environ.get('SYNOPTIC_TOKEN')
   ```

**Usage Limits**:
- Academic: 5000 API calls/day
- Check terms of service for latest limits

---

### AWS S3 Credentials

**Required For**: Writing to CRESST S3 bucket

**Setup**:
1. Obtain AWS credentials from project administrator
2. Create `~/.aws/credentials` file:
   ```ini
   [cresst-user]
   aws_access_key_id = AKIA...
   aws_secret_access_key = ...
   ```
3. Set environment variable:
   ```bash
   export AWS_PROFILE=cresst-user
   ```

**Read-Only Access**:
- Most data sources support anonymous read
- No credentials needed for downloading

**Write Access** (notebook example):
```python
from obstore.auth.boto3 import Boto3CredentialProvider
from obstore.store import S3Store

store = S3Store(
    "cresst",
    credential_provider=Boto3CredentialProvider(),
    region="us-west-2"
)
```

---

### NASA Earthdata

**Required For**: Some ORNL DAAC data, NISAR data

**Setup**:
1. Register at https://urs.earthdata.nasa.gov/
2. Create `~/.netrc` file:
   ```
   machine urs.earthdata.nasa.gov
   login your_username
   password your_password
   ```
3. Set permissions: `chmod 600 ~/.netrc`

**Used in**: `S3-bucket.ipynb` (NISAR data example)

---

## Data Access Patterns

### Pattern 1: Direct HTTP Download

**Used by**: PRISM, Stage IV, DEM

**Characteristics**:
- Simple URL-based access
- wget or requests library
- No authentication (usually)
- Individual files per time step

**Example**:
```python
import requests

url = 'https://water.noaa.gov/resources/downloads/precip/stageIV/2025/12/11/nws_precip_1day_20251211_conus.nc'
response = requests.get(url)
with open('output.nc', 'wb') as f:
    f.write(response.content)
```

---

### Pattern 2: REST API Query

**Used by**: USGS, Synoptic

**Characteristics**:
- Parameterized queries
- JSON or XML responses
- May require authentication
- Rate limiting

**Example**:
```python
import requests

params = {
    'sites': '12200500',
    'parameterCd': '00060',
    'startDT': '2025-01-01',
    'format': 'json'
}
response = requests.get('https://nwis.waterservices.usgs.gov/nwis/iv/', params=params)
data = response.json()
```

---

### Pattern 3: Cloud Object Storage (S3)

**Used by**: CONUS404, HRRR (via Herbie), WRF-CMIP6

**Characteristics**:
- Direct access to object store
- Supports partial reads (byte-range requests)
- Anonymous or authenticated
- Optimized for cloud compute

**Example**:
```python
import xarray as xr

ds = xr.open_zarr(
    's3://hytest/conus404/conus404_daily.zarr',
    storage_options={
        'anon': True,
        'client_kwargs': {'endpoint_url': 'https://usgs.osn.mghpcc.org'}
    }
)
```

---

### Pattern 4: Specialized Libraries

**Used by**: HRRR (Herbie), PRISM (pyPRISMClimate), Seismic (obspy)

**Characteristics**:
- Domain-specific API wrappers
- Hide complexity of data access
- Often include caching
- May have additional dependencies

**Example**:
```python
from herbie import Herbie

H = Herbie('2020-01-01', model='hrrr', product='sfc')
H.download(':TMP:2 m above ground')
```

---

### Comparison Table

| Pattern | Ease of Use | Flexibility | Performance | Authentication | Caching |
|---------|-------------|-------------|-------------|----------------|---------|
| Direct HTTP | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Usually not needed | Manual |
| REST API | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Sometimes | Manual |
| Cloud Storage | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Optional | Built-in |
| Specialized Lib | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Handled | Built-in |

---

**Last Updated**: 2026-02-10
