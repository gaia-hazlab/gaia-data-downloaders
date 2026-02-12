# AWS S3 Integration Guide

## Table of Contents

1. [Overview](#overview)
2. [Bucket Configuration](#bucket-configuration)
3. [Authentication](#authentication)
4. [Reading from S3](#reading-from-s3)
5. [Writing to S3](#writing-to-s3)
6. [Cloud-Optimized Workflows](#cloud-optimized-workflows)
7. [Best Practices](#best-practices)
8. [Cost Considerations](#cost-considerations)

## Overview

The gaia-data-downloaders project integrates with AWS S3 for both data access and storage:

**Use Cases**:
1. **Data Access**: Read large datasets from public S3 buckets (CONUS404, HRRR, WRF-CMIP6)
2. **Data Sharing**: Upload derived products to CRESST bucket for collaboration
3. **Cloud Computing**: Enable analysis directly on cloud-stored data
4. **Backup**: Archive processed datasets

**Key Libraries**:
- `obstore`: Modern, performant object store library (recommended)
- `boto3`: Official AWS SDK (fallback)
- `s3fs`: S3 filesystem interface for xarray/pandas
- `zarr`: Cloud-optimized array storage format

---

## Bucket Configuration

### CRESST Bucket

**Bucket Name**: `cresst`
**Region**: `us-west-2` (Oregon)
**Endpoint**: Standard AWS S3

**Access Policies**:
- **Read**: Public (anonymous) for most data
- **Write**: Authenticated (requires credentials)

**Organization**:
```
s3://cresst/
├── README.md                  # Bucket documentation
├── {username}/                # Per-user directories
│   └── derived_products/
├── stehekin/                  # Project-specific data
│   └── planet/
│       ├── catalog.json
│       └── PSScene/
├── scottsfiles/              # Shared derived products
└── {project}/                # Project directories
```

---

### Public Data Buckets

**CONUS404** (OSN Pod):
- Bucket: `s3://hytest/conus404/`
- Endpoint: `https://usgs.osn.mghpcc.org`
- Access: Anonymous
- Format: Zarr

**HRRR** (AWS Registry):
- Bucket: `s3://noaa-hrrr-bdp-pds/`
- Endpoint: Standard AWS
- Access: Anonymous
- Format: GRIB2

**WRF-CMIP6**:
- Bucket: `s3://wrf-cmip6-noversioning/`
- Endpoint: Standard AWS
- Access: Anonymous
- Format: NetCDF

---

## Authentication

### AWS Credentials Setup

**Option 1: AWS CLI** (recommended)

```bash
# Install AWS CLI
# macOS
brew install awscli

# Linux
sudo apt-get install awscli

# Configure credentials
aws configure --profile cresst-user
# Enter when prompted:
#   AWS Access Key ID: AKIA...
#   AWS Secret Access Key: ...
#   Default region: us-west-2
#   Default output format: json
```

**Option 2: Manual Configuration**

Create `~/.aws/credentials`:
```ini
[cresst-user]
aws_access_key_id = AKIAXXXXXXXXXXXXX
aws_secret_access_key = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Create `~/.aws/config`:
```ini
[profile cresst-user]
region = us-west-2
output = json
```

**Option 3: Environment Variables**

```bash
export AWS_ACCESS_KEY_ID=AKIAXXXXXXXXXXXXX
export AWS_SECRET_ACCESS_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export AWS_DEFAULT_REGION=us-west-2
```

---

### Using Credentials in Code

**With obstore (recommended)**:

```python
from obstore.auth.boto3 import Boto3CredentialProvider
from obstore.store import S3Store
import os

# Set profile
os.environ['AWS_PROFILE'] = 'cresst-user'

# Create store
store = S3Store(
    "cresst",
    credential_provider=Boto3CredentialProvider(),
    region="us-west-2"
)
```

**With boto3**:

```python
import boto3
from botocore.client import Config

# With profile
session = boto3.Session(profile_name='cresst-user')
s3 = session.client('s3')

# Anonymous access (public buckets)
s3 = boto3.client(
    's3',
    config=Config(signature_version=UNSIGNED)
)
```

**With s3fs**:

```python
import s3fs

# Authenticated
fs = s3fs.S3FileSystem(profile='cresst-user')

# Anonymous
fs = s3fs.S3FileSystem(anon=True)
```

---

## Reading from S3

### Reading Zarr Stores

**xarray + s3fs**:

```python
import xarray as xr
import s3fs

# Anonymous access (public data)
ds = xr.open_zarr(
    's3://hytest/conus404/conus404_daily.zarr',
    storage_options={
        'anon': True,
        'client_kwargs': {'endpoint_url': 'https://usgs.osn.mghpcc.org'}
    },
    consolidated=True
)

# Authenticated access
ds = xr.open_zarr(
    's3://cresst/mydata/processed.zarr',
    storage_options={
        'profile': 'cresst-user',
        'client_kwargs': {'region_name': 'us-west-2'}
    }
)

# Subset remotely (no download)
subset = ds.sel(
    time=slice('2020-01-01', '2020-12-31'),
    lat=slice(47, 49),
    lon=slice(-123, -120)
)

# Compute only what's needed
result = subset['temperature'].mean('time').compute()
```

---

### Reading Individual Files

**With obstore**:

```python
from obstore.store import S3Store
from obstore.auth.boto3 import Boto3CredentialProvider
import io

store = S3Store(
    "cresst",
    credential_provider=Boto3CredentialProvider(),
    region="us-west-2"
)

# Get file metadata
metadata = store.head('path/to/file.tif')
print(f"Size: {metadata['size']} bytes")

# Read file
data = store.get('path/to/file.tif')

# Read to BytesIO (for rasterio, etc.)
with io.BytesIO(data) as buffer:
    import rasterio
    with rasterio.open(buffer) as src:
        array = src.read()
```

---

### Reading NetCDF Files

**xarray + s3fs**:

```python
import xarray as xr

# Single file
ds = xr.open_dataset(
    's3://wrf-cmip6-noversioning/downscaled_products/gcm/model/file.nc',
    engine='h5netcdf',
    storage_options={'anon': True}
)

# Multiple files (pattern matching)
ds = xr.open_mfdataset(
    's3://bucket/path/*.nc',
    engine='h5netcdf',
    combine='by_coords',
    storage_options={'anon': True}
)
```

---

### Reading GeoTIFF

**rioxarray**:

```python
import rioxarray as rxr

# Read COG (Cloud-Optimized GeoTIFF)
da = rxr.open_rasterio(
    's3://cresst/scottsfiles/derived_product.tif',
    storage_options={'profile': 'cresst-user'}
)

# Efficient windowed read
da_subset = da.rio.clip_box(
    minx=-123, miny=47,
    maxx=-120, maxy=49
)
```

---

### Listing Files

**With obstore**:

```python
from obstore.store import S3Store

store = S3Store("cresst", ...)

# List all objects in prefix
objects = store.list("stehekin/planet/").collect()

for obj in objects:
    print(f"{obj['path']}: {obj['size']} bytes, {obj['last_modified']}")

# Recursive listing
all_objects = store.list_with_delimiter("").collect()
```

**With boto3**:

```python
import boto3

s3 = boto3.client('s3', ...)

# List objects
response = s3.list_objects_v2(
    Bucket='cresst',
    Prefix='stehekin/',
    MaxKeys=1000
)

for obj in response.get('Contents', []):
    print(f"{obj['Key']}: {obj['Size']} bytes")
```

---

## Writing to S3

### Uploading Files

**With obstore (recommended)**:

```python
from obstore.store import S3Store
from obstore.auth.boto3 import Boto3CredentialProvider

store = S3Store(
    "cresst",
    credential_provider=Boto3CredentialProvider(),
    region="us-west-2"
)

# Upload from file
with open('local_file.tif', 'rb') as f:
    store.put('myusername/derived_product.tif', f)

# Upload from bytes
data = b'...'  # Your data
store.put('myusername/data.bin', data)

# Upload with metadata
store.put(
    'myusername/file.nc',
    file_content,
    attributes={'content-type': 'application/netcdf'}
)
```

---

### Writing Zarr Stores

**xarray → Zarr → S3**:

```python
import xarray as xr
import s3fs

# Method 1: Direct write (single-threaded)
store = s3fs.S3FileSystem(profile='cresst-user')
store_path = 'cresst/myusername/processed_data.zarr'

ds.to_zarr(
    store.get_mapper(store_path),
    mode='w',
    consolidated=True
)

# Method 2: Write locally, then upload (recommended for large datasets)
# Write to local Zarr
ds.to_zarr('local_data.zarr', mode='w', consolidated=True)

# Upload to S3
import subprocess
subprocess.run([
    'aws', 's3', 'sync',
    'local_data.zarr',
    's3://cresst/myusername/data.zarr',
    '--profile', 'cresst-user'
])
```

---

### Writing GeoTIFFs

**Cloud-Optimized GeoTIFF (COG)**:

```python
import rioxarray as rxr

# Create derived product
da = ds['soilMoisture'].isel(time=0)

# Write to local COG
output_file = 'derived_product.tif'
da.rio.to_raster(
    output_file,
    driver='COG',
    compress='DEFLATE',
    predictor=2,
    blocksize=512,
    tiled=True
)

# Upload to S3
from obstore.store import S3Store
from obstore.auth.boto3 import Boto3CredentialProvider

store = S3Store("cresst", credential_provider=Boto3CredentialProvider(), region="us-west-2")

with open(output_file, 'rb') as f:
    store.put(f'myusername/{output_file}', f)

print(f"Uploaded to s3://cresst/myusername/{output_file}")
```

---

### Batch Uploads with AWS CLI

**Upload directory**:

```bash
aws s3 sync \
    ./local_directory/ \
    s3://cresst/myusername/data/ \
    --profile cresst-user \
    --exclude "*.tmp" \
    --exclude ".*"
```

**Upload with metadata**:

```bash
aws s3 cp \
    local_file.nc \
    s3://cresst/myusername/data.nc \
    --profile cresst-user \
    --metadata project=gaia,date=2026-02-10 \
    --content-type application/netcdf
```

---

## Cloud-Optimized Workflows

### Pattern 1: Subset on Cloud, Compute Locally

**Advantages**:
- Minimize data transfer
- Leverage cloud storage performance
- No need for local storage of full dataset

**Example**:

```python
import xarray as xr

# Open large remote dataset (lazy)
ds = xr.open_zarr(
    's3://hytest/conus404/conus404_daily.zarr',
    storage_options={
        'anon': True,
        'client_kwargs': {'endpoint_url': 'https://usgs.osn.mghpcc.org'}
    }
)

# Subset (still lazy)
subset = ds.sel(
    time=slice('2020-01-01', '2020-01-31'),
    lat=slice(47, 49),
    lon=slice(-123, -120)
)[['T2', 'PRECIP_TOT']]

# Compute only subset (download ~100 MB instead of ~1 TB)
subset_computed = subset.compute()

# Process locally
result = subset_computed.mean('time')

# Save locally or upload to S3
result.to_netcdf('result.nc')
```

---

### Pattern 2: Process on Cloud, Download Results

**Use Case**: When computation is large but results are small

**Example**:

```python
# Open remote dataset
ds = xr.open_zarr('s3://bucket/large_dataset.zarr', ...)

# Compute aggregation (processed on cloud/dask cluster)
monthly_mean = ds['temperature'].resample(time='1M').mean()

# Download only results
result = monthly_mean.compute()  # Small dataset

# Save locally
result.to_netcdf('monthly_means.nc')
```

---

### Pattern 3: Cloud-to-Cloud Transfer

**Use Case**: Derive products from public data, save to CRESST bucket

**Example**:

```python
import xarray as xr
import s3fs

# Read from public bucket (OSN)
ds = xr.open_zarr(
    's3://hytest/conus404/conus404_daily.zarr',
    storage_options={
        'anon': True,
        'client_kwargs': {'endpoint_url': 'https://usgs.osn.mghpcc.org'}
    }
)

# Process
subset = ds.sel(time=slice('2020-01-01', '2020-12-31'))
monthly = subset.resample(time='1M').mean()

# Write to CRESST bucket
store = s3fs.S3FileSystem(profile='cresst-user')
monthly.to_zarr(
    store.get_mapper('cresst/myusername/monthly_2020.zarr'),
    mode='w'
)
```

---

### Pattern 4: Incremental Updates

**Use Case**: Append new time steps to existing Zarr store

**Example**:

```python
import xarray as xr
import s3fs

store = s3fs.S3FileSystem(profile='cresst-user')
zarr_path = 'cresst/myusername/timeseries.zarr'

# Download new data
new_data = download_latest_data()

# Append to existing Zarr (mode='a')
new_data.to_zarr(
    store.get_mapper(zarr_path),
    mode='a',
    append_dim='time'
)
```

---

## Best Practices

### Data Organization

1. **Use User Directories**: `s3://cresst/{username}/`
2. **Descriptive Names**: `processed_prism_2020_skagit.zarr`
3. **Include Metadata**: Date, version, processing info
4. **Document**: Add README files in S3

**Good Structure**:
```
s3://cresst/myusername/
├── README.md
├── raw/
│   └── downloaded_2026-02-10/
├── processed/
│   ├── daily/
│   │   └── skagit_prism_2020.zarr
│   └── monthly/
│       └── skagit_prism_2020_monthly.zarr
└── derived/
    └── extreme_precip_analysis.nc
```

---

### Performance Optimization

**1. Use Cloud-Optimized Formats**:
- Zarr for arrays (chunked, compressed)
- COG for rasters (tiled, overviews)
- Parquet for tables (columnar)

**2. Optimize Chunking**:
```python
# Match expected access pattern
ds.to_zarr(
    zarr_path,
    encoding={
        'temperature': {
            'chunks': (30, 100, 100),  # (time, lat, lon)
            'compressor': zarr.Blosc(cname='zstd', clevel=3)
        }
    }
)
```

**3. Use Consolidated Metadata**:
```python
ds.to_zarr(zarr_path, consolidated=True)
```

**4. Parallelize When Possible**:
```python
# Use dask for parallel writes
ds.to_zarr(zarr_path, compute=True, mode='w')
```

---

### Security

**1. Never Commit Credentials**:
```bash
# Add to .gitignore
.aws/credentials
*.pem
```

**2. Use IAM Roles** (when on AWS EC2/Lambda):
```python
# No credentials needed - uses instance role
s3 = boto3.client('s3')
```

**3. Least Privilege**:
- Request minimal necessary permissions
- Separate read-only and write access

**4. Rotate Keys**:
- Change access keys periodically
- Revoke unused keys

---

### Error Handling

```python
from botocore.exceptions import ClientError
import time

def upload_with_retry(store, key, data, max_retries=3):
    """Upload with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            store.put(key, data)
            print(f"Upload successful: {key}")
            return
        except ClientError as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"Upload failed, retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"Upload failed after {max_retries} attempts")
                raise
```

---

### Monitoring

**Check Storage Usage**:

```bash
# Total bucket size
aws s3 ls s3://cresst/ --recursive --human-readable --summarize --profile cresst-user

# Size of specific prefix
aws s3 ls s3://cresst/myusername/ --recursive --human-readable --summarize --profile cresst-user
```

**List Recent Uploads**:

```python
import boto3
from datetime import datetime, timedelta

s3 = boto3.client('s3', ...)

# List objects modified in last 7 days
cutoff = datetime.now() - timedelta(days=7)

response = s3.list_objects_v2(Bucket='cresst', Prefix='myusername/')

for obj in response.get('Contents', []):
    if obj['LastModified'].replace(tzinfo=None) > cutoff:
        print(f"{obj['Key']}: {obj['Size']} bytes")
```

---

## Cost Considerations

### AWS S3 Pricing (us-west-2)

**Storage** (as of 2024):
- First 50 TB: $0.023 per GB/month
- Next 450 TB: $0.022 per GB/month

**Data Transfer**:
- IN to S3: Free
- OUT to internet: $0.09 per GB (after 100 GB free tier)
- S3 to CloudFront: $0.00 per GB

**Requests**:
- PUT/POST: $0.005 per 1,000 requests
- GET: $0.0004 per 1,000 requests
- LIST: $0.005 per 1,000 requests

### Cost Optimization

**1. Use Efficient Formats**:
- Zarr with compression (3-10x reduction)
- COG vs uncompressed GeoTIFF (5-10x reduction)

**2. Minimize Requests**:
- Use consolidated metadata for Zarr
- Batch small files
- Use S3 Select for filtering

**3. Lifecycle Policies**:
```python
# Archive old data to Glacier
import boto3

s3 = boto3.client('s3')

lifecycle_config = {
    'Rules': [{
        'Id': 'Archive old data',
        'Prefix': 'myusername/raw/',
        'Status': 'Enabled',
        'Transitions': [{
            'Days': 90,
            'StorageClass': 'GLACIER'
        }]
    }]
}

s3.put_bucket_lifecycle_configuration(
    Bucket='cresst',
    LifecycleConfiguration=lifecycle_config
)
```

**4. Delete Unnecessary Data**:
```bash
# Delete old test files
aws s3 rm s3://cresst/myusername/test/ --recursive --profile cresst-user
```

---

### Estimating Costs

**Example: 100 GB dataset**

```
Storage (1 month): 100 GB × $0.023 = $2.30
Uploads (initial): 1 upload × $0.005/1000 = negligible
Downloads (10×): 1 TB × $0.09 = $90.00

Total: ~$92.30/month (mostly egress)
```

**Tip**: Keep data in cloud for cloud compute, download only results.

---

## Example: Complete Workflow

**Scenario**: Download PRISM data, process, upload to S3

```python
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
from obstore.store import S3Store
from obstore.auth.boto3 import Boto3CredentialProvider
import pyPRISMClimate

# 1. Download PRISM data (local)
pyPRISMClimate.get_prism_dailys(
    'tmean',
    min_date='2020-01-01',
    max_date='2020-01-31',
    dest_path='./data/',
    keep_zip=False
)

# 2. Load and process
boundary = gpd.read_file('../data/GIS/SkagitBoundary.json')
rasters = []

for f in pyPRISMClimate.utils.prism_iterator('./data/'):
    raster = rxr.open_rasterio(f['full_path'], masked=True)
    raster = raster.rio.clip(boundary.geometry)
    rasters.append(raster)

ds = xr.concat(rasters, dim='time')

# 3. Optimize and save to Zarr (local)
ds.to_zarr('processed_prism.zarr', mode='w', consolidated=True)

# 4. Upload to S3
store = S3Store(
    "cresst",
    credential_provider=Boto3CredentialProvider(),
    region="us-west-2"
)

# Option A: Upload entire Zarr store
import subprocess
subprocess.run([
    'aws', 's3', 'sync',
    'processed_prism.zarr',
    's3://cresst/myusername/prism_jan2020.zarr',
    '--profile', 'cresst-user'
])

# Option B: Export to COG and upload
monthly_mean = ds.mean('time')
monthly_mean.rio.to_raster('jan2020_mean.tif', driver='COG', compress='DEFLATE')

with open('jan2020_mean.tif', 'rb') as f:
    store.put('myusername/prism_jan2020_mean.tif', f)

print("Upload complete!")
print("Access at: s3://cresst/myusername/prism_jan2020_mean.tif")
```

---

**Last Updated**: 2026-02-10
