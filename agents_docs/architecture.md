# Architecture Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Design Philosophy](#design-philosophy)
3. [Architectural Patterns](#architectural-patterns)
4. [Component Architecture](#component-architecture)
5. [Data Flow](#data-flow)
6. [Storage Architecture](#storage-architecture)
7. [Performance Considerations](#performance-considerations)
8. [Evolution and Technical Debt](#evolution-and-technical-debt)

## System Overview

The gaia-data-downloaders system is a **notebook-based data acquisition and processing pipeline** designed for hydroclimatological research. Unlike traditional software architectures with separate layers for data access, business logic, and presentation, this system uses Jupyter notebooks as the primary architectural unit.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Jupyter Notebook Layer                    │
│  (18 notebooks - each is a complete data pipeline)          │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Weather    │  Hydrological│   Terrain    │   Stations     │
│   Models     │     Data     │     Data     │   Inventory    │
├──────────────┴──────────────┴──────────────┴────────────────┤
│              Python Scientific Stack                         │
│  (xarray, pandas, geopandas, rioxarray, obspy)              │
├─────────────────────────────────────────────────────────────┤
│                  Data Source APIs                            │
│  (AWS S3, NOAA, USGS, IRIS, Synoptic, OSN Pod)             │
├─────────────────────────────────────────────────────────────┤
│                  Storage Layer                               │
│  (Local FS, AWS S3, Zarr, NetCDF, CSV, GeoJSON)            │
└─────────────────────────────────────────────────────────────┘
```

### Architectural Characteristics

**Strengths**:
- **Transparency**: Complete workflow visible in single notebook
- **Reproducibility**: Self-contained execution environments
- **Flexibility**: Easy to modify parameters and experiment
- **Documentation**: Code and documentation colocated
- **Prototyping**: Rapid development and iteration

**Tradeoffs**:
- **Code Reuse**: Limited - similar patterns repeated across notebooks
- **Testing**: Manual - no automated test suite
- **Modularity**: Coarse-grained - notebook is smallest unit
- **Maintenance**: Scattered - changes may need to be replicated
- **Scalability**: Limited - designed for researcher workstation scale

## Design Philosophy

### Core Principles

1. **Self-Contained Workflows**
   - Each notebook can be run independently
   - Dependencies are explicit in import cells
   - Data sources are documented inline
   - Outputs are self-describing (metadata preserved)

2. **Researcher-Centric Design**
   - Prioritize readability over abstraction
   - Include inline documentation and examples
   - Generate visualizations for QC
   - Support exploratory analysis

3. **Geographic Consistency**
   - Most pipelines target Skagit River watershed
   - Common boundary file: `../data/GIS/SkagitBoundary.json`
   - Consistent spatial reference (WGS84/EPSG:4326)

4. **Format Optimization**
   - Prefer Zarr for large multidimensional data
   - Use GeoJSON for vector data interchange
   - CSV for tabular USGS data
   - Cloud-optimized formats (COG) for rasters

5. **Parallel Processing**
   - ThreadPoolExecutor for I/O-bound downloads
   - Dask for lazy evaluation of large arrays
   - Batch processing for multiple stations/time periods

### Design Decisions

#### Why Jupyter Notebooks?

**Rationale**: Research workflows benefit from narrative + code integration
- Exploratory data analysis requires iteration
- Scientific workflows need documentation at code level
- Reproducibility enhanced by literate programming
- Easier onboarding for research scientists

**Alternatives Considered**:
- Python modules with CLI: More maintainable but less exploratory
- Workflow engines (Airflow, Prefect): Too heavyweight for research
- Scripts + config files: Less discoverable, harder to document

#### Why No Shared Library?

**Current State**: Common patterns duplicated across notebooks
- Each notebook imports standard libraries independently
- Download patterns similar but not abstracted
- Spatial clipping code repeated

**Rationale**:
- Notebooks remain self-contained and portable
- Easier to customize for specific needs
- Lower cognitive overhead (no abstraction layers)

**Future**: Consider extracting to `gaia_downloader_utils.py`:
- Common functions: `clip_to_aoi()`, `download_parallel()`, `to_optimized_format()`
- Would improve maintainability
- May reduce flexibility

## Architectural Patterns

### Pattern 1: Download-Process-Save Pipeline

**Used in**: CONUS404, HRRR, WRF, PRISM, Stage IV downloaders

**Structure**:
```python
# 1. Configuration
dates = pd.date_range(start, end, freq='1h')
variables = ['temp', 'precip', 'wind']
aoi_boundary = gpd.read_file('boundary.json')

# 2. Download (parallel)
with ThreadPoolExecutor(max_workers=4) as executor:
    files = executor.map(download_func, dates)

# 3. Load and combine
datasets = [xr.open_dataset(f) for f in files]
combined = xr.concat(datasets, dim='time')

# 4. Spatial subset
clipped = combined.rio.clip(aoi_boundary.geometry)

# 5. Process/derive variables
clipped['windspeed'] = np.sqrt(clipped['u']**2 + clipped['v']**2)

# 6. Optimize and save
clipped.to_zarr('output.zarr', mode='w')

# 7. QC visualization
clipped['temp'].plot()
```

**Key Benefits**:
- Clear separation of concerns
- Easy to debug at each stage
- Can restart from any checkpoint
- Memory-efficient with lazy evaluation

### Pattern 2: Station Inventory Pattern

**Used in**: USGS downloaders, merged-inventory notebook

**Structure**:
```python
# 1. Query metadata
stations_df = query_station_metadata(region='WA')

# 2. Filter and validate
active_stations = stations_df[stations_df.status == 'ACTIVE']

# 3. Iterate over stations
for station_id, metadata in active_stations.iterrows():
    # Download time series data
    data = download_station_data(station_id, date_range)

    # Save per-station
    save_dir = f'USGS_Stream_Gage/{station_id}'
    data.to_csv(f'{save_dir}/{station_id}_data.csv')

    # Track failures
    if download_failed:
        with open('cache/failed_ids.txt', 'a') as f:
            f.write(f'{station_id}\n')

# 4. Create spatial inventory
gdf = gpd.GeoDataFrame(active_stations,
                        geometry=gpd.points_from_xy(
                            active_stations.lon,
                            active_stations.lat
                        ))
gdf.to_file('stations_inventory.geojson')
```

**Key Benefits**:
- Robust to partial failures
- Can resume interrupted downloads
- Maintains spatial reference
- Supports web visualization

### Pattern 3: Cloud-Optimized Data Pattern

**Used in**: S3-bucket notebook, CONUS404 downloader

**Structure**:
```python
# 1. Access cloud data (anonymous or authenticated)
import s3fs
fs = s3fs.S3FileSystem(anon=True)

# 2. Open remote dataset
ds = xr.open_zarr(
    's3://bucket/dataset.zarr',
    storage_options={'anon': True}
)

# 3. Subset remotely (no full download)
subset = ds.sel(time=slice('2020-01-01', '2020-12-31'))
subset = subset.sel(lat=slice(47, 49), lon=slice(-123, -120))

# 4. Compute only what's needed
result = subset['variable'].mean('time').compute()

# 5. Upload derived products
with open('derived_product.tif', 'rb') as f:
    obstore.put(store, 'path/in/bucket', f)
```

**Key Benefits**:
- Minimize data transfer
- Leverage cloud storage performance
- Enable collaborative data access
- Support large datasets (TB+)

### Pattern 4: Spatial Masking Pattern

**Used in**: Almost all downloaders

**Structure**:
```python
# 1. Load boundary
mask = gpd.read_file('boundary.json')

# 2. Method A: Vector clipping (for rasters)
ds = xr.open_rasterio('input.tif')
clipped = ds.rio.clip(mask.geometry, mask.crs)

# 3. Method B: Boolean masking (for irregular grids)
from shapely.vectorized import contains
in_bounds = contains(mask.geometry[0],
                     ds.lon.values,
                     ds.lat.values)
masked = ds.where(in_bounds)

# 4. Method C: Regionmask (for labeled regions)
import regionmask
regions = regionmask.Regions.from_geopandas(mask)
region_mask = regions.mask(ds.lon, ds.lat)
masked = ds.where(region_mask.notnull())
```

**Key Benefits**:
- Reduces data volume early in pipeline
- Ensures spatial consistency
- Handles multiple CRS
- Supports complex geometries

## Component Architecture

### Core Components

#### 1. Data Acquisition Layer

**Responsibilities**:
- Query external APIs
- Download raw data files
- Handle authentication
- Implement retry logic
- Cache responses

**Implementation**:
- HTTP clients: `requests`, `aiohttp` (with caching)
- S3 clients: `boto3`, `s3fs`, `obstore`
- Specialized clients: `herbie` (HRRR), `pyPRISMClimate`, `pygeohydro`
- FTP access: Built-in Python `urllib`

**Key Patterns**:
- ThreadPoolExecutor for parallel downloads
- Exponential backoff for rate limiting
- SQLite cache for HTTP responses (`cache/aiohttp_cache.sqlite`)
- Failed ID tracking (`cache/failed_ids_*.txt`)

#### 2. Data Processing Layer

**Responsibilities**:
- Open and parse data formats
- Coordinate transformations
- Spatial and temporal subsetting
- Derive variables
- Quality control

**Implementation**:
- Multidimensional: `xarray`, `dask`
- Raster: `rioxarray`, `rasterio`, `elevation`
- Vector: `geopandas`, `shapely`
- Tabular: `pandas`
- Meteorology: `metpy`
- Seismic: `obspy`

**Key Patterns**:
- Lazy evaluation with dask arrays
- Chunked processing for memory efficiency
- CRS-aware clipping with rioxarray
- Chained transformations with xarray

#### 3. Storage Layer

**Responsibilities**:
- Write processed data
- Optimize storage format
- Maintain metadata
- Enable efficient reads

**Implementation**:
- Array storage: Zarr (preferred), NetCDF
- Raster: GeoTIFF (COG format)
- Vector: GeoJSON, Shapefile
- Tabular: CSV, Parquet
- Cloud: S3 via obstore/boto3

**Key Patterns**:
- Zarr for chunked array storage
- Appropriate chunk sizes for access patterns
- Compression (DEFLATE, ZSTD)
- Metadata preservation

#### 4. Visualization Layer

**Responsibilities**:
- Generate QC plots
- Create static maps
- Build interactive visualizations
- Export for web

**Implementation**:
- Plotting: `matplotlib`
- Cartographic: `cartopy`, `contextily`
- Interactive: `folium` (web maps)
- Styling: SimplStyle spec for GeoJSON

**Key Patterns**:
- Cartopy for projected maps
- Contextily for basemaps
- Folium for shareable HTML maps
- Matplotlib for time series and arrays

## Data Flow

### Typical Data Flow (Weather Model Example)

```
┌──────────────────┐
│  External API    │ (e.g., AWS S3 bucket with CONUS404)
│  or Data Archive │
└────────┬─────────┘
         │ HTTP/S3 download (parallel)
         ▼
┌──────────────────┐
│  Raw Data Files  │ (.nc, .grib2, .zarr)
│  (local cache)   │
└────────┬─────────┘
         │ xr.open_dataset()
         ▼
┌──────────────────┐
│  Xarray Dataset  │ (lazy loaded, dask arrays)
│  (full extent)   │
└────────┬─────────┘
         │ .sel() or .rio.clip()
         ▼
┌──────────────────┐
│  Spatial Subset  │ (Skagit watershed)
│  (Xarray)        │
└────────┬─────────┘
         │ Derive variables, compute
         ▼
┌──────────────────┐
│  Processed Data  │ (computed dask arrays)
│  (Xarray)        │
└────────┬─────────┘
         │ .to_zarr() or .to_netcdf()
         ▼
┌──────────────────┐
│  Optimized Store │ (.zarr with compression)
│  (local or S3)   │
└────────┬─────────┘
         │ xr.open_zarr() for analysis
         ▼
┌──────────────────┐
│  Visualization   │ (plots, maps, HTML)
└──────────────────┘
```

### USGS Stream Gauge Data Flow

```
┌──────────────────┐
│  USGS WaterWatch │ (REST API)
│  API             │
└────────┬─────────┘
         │ Query site metadata
         ▼
┌──────────────────┐
│  Site Metadata   │ (pandas DataFrame)
│  (CSV)           │
└────────┬─────────┘
         │ Filter active stations
         ▼
┌──────────────────┐
│  Station List    │ (743 stations in WA)
└────────┬─────────┘
         │ For each station (serial)
         ▼
┌──────────────────┐
│  Time Series     │ (USGS RDB format)
│  Query           │
└────────┬─────────┘
         │ Parse and convert
         ▼
┌──────────────────┐
│  Per-Station CSV │ (USGS_Stream_Gage/{id}/{id}_data.csv)
└────────┬─────────┘
         │ Aggregate metadata
         ▼
┌──────────────────┐
│  Spatial Index   │ (stations_by_basin.csv, GeoJSON)
└──────────────────┘
```

## Storage Architecture

### Local Storage Organization

```
gaia-data-downloaders/
├── USGS_Stream_Gage/          # Downloaded data (in repo)
│   └── {station_id}/
│       ├── {id}_data.csv      # Time series data
│       └── {id}_header.txt    # Metadata
├── cache/                      # HTTP cache (in repo)
│   ├── aiohttp_cache.sqlite   # Response cache
│   └── failed_ids_*.txt       # Download failure tracking
├── stations_by_basin*.csv     # Inventories (in repo)
│
└── ../data/                    # External data (not in repo)
    ├── weather_data/           # Weather/climate model outputs
    │   ├── *.zarr/            # Optimized array storage
    │   ├── *.nc               # NetCDF files
    │   └── *.grib2            # GRIB2 files
    ├── GIS/                    # Boundary files
    │   ├── SkagitBoundary.json
    │   └── *.tif              # DEM files
    ├── stageIV/               # NOAA precipitation
    └── plots/                  # Generated visualizations
```

### Cloud Storage Architecture (S3)

**Bucket**: `s3://cresst` (AWS us-west-2)

**Organization**:
```
cresst/
├── README.md
├── {researcher}/               # Per-user directories
│   └── *.tif                  # Derived products
├── stehekin/                  # Project-specific data
│   └── planet/                # Satellite imagery
│       ├── catalog.json
│       └── PSScene/           # Planet data products
└── scottsfiles/               # Shared derived products
```

**Access Patterns**:
- Anonymous read: Public data
- Authenticated write: AWS credentials required
- Integration: `obstore` library for Python access

### Storage Format Selection

| Format | Use Case | Pros | Cons |
|--------|----------|------|------|
| **Zarr** | Large multidimensional arrays | Chunked, cloud-native, fast random access | Less universal than NetCDF |
| **NetCDF** | Climate model outputs | Standard format, CF compliant | Slower for cloud access |
| **GeoTIFF (COG)** | Raster DEMs, derived products | Universal, cloud-optimized | Single time slice only |
| **GeoJSON** | Station inventories, boundaries | Web-compatible, human-readable | Large file size for many features |
| **CSV** | USGS time series | Simple, universal | No spatial reference, not optimized |
| **Parquet** | Large tabular data | Columnar, compressed | Not used yet (opportunity) |

## Performance Considerations

### Bottlenecks

1. **Network I/O**: Downloading large datasets (10+ GB)
   - Mitigation: Parallel downloads, spatial subsetting before download

2. **Disk I/O**: Writing large arrays
   - Mitigation: Zarr chunking, compression

3. **Memory**: Loading full datasets into RAM
   - Mitigation: Dask lazy evaluation, chunked processing

4. **CPU**: Coordinate transformations, interpolation
   - Mitigation: Use NumPy/Dask parallelism

### Optimization Strategies

**Download Optimization**:
- ThreadPoolExecutor with 4-24 workers
- Resume interrupted downloads (check file existence)
- Cache HTTP responses (aiohttp_cache)
- Spatial subsetting via API when possible

**Processing Optimization**:
- Lazy evaluation with dask arrays
- Appropriate chunk sizes (balance memory vs. parallelism)
- Early spatial filtering (clip before compute)
- Vectorized operations over loops

**Storage Optimization**:
- Zarr chunking aligned with access patterns
- Compression (DEFLATE for compatibility, ZSTD for performance)
- Remove encoding conflicts before write
- Consolidate metadata (Zarr consolidated=True)

### Scalability Limits

**Current Scale**:
- Datasets: 1 GB - 100 GB per source
- Time ranges: Days to years
- Spatial extent: Watershed to regional
- Stations: 100s to 1000s

**Scaling Challenges**:
- Continental-scale analysis would require cluster computing
- Real-time processing not supported (batch only)
- No orchestration for multi-step pipelines
- Limited error recovery (manual restart)

## Evolution and Technical Debt

### Historical Context

The repository appears to have evolved organically through research needs:
1. Initial USGS stream gauge downloaders
2. Added weather model integrations (HRRR, PRISM)
3. Expanded to climate models (CONUS404, WRF)
4. Station inventory management
5. Cloud storage integration (S3)

### Technical Debt

**Code Duplication**:
- Spatial clipping logic repeated across notebooks
- Download patterns similar but not abstracted
- Visualization code duplicated

**Testing Gap**:
- No automated tests
- Manual QC via visualization
- Limited error handling in some notebooks

**Documentation Inconsistency**:
- Some notebooks well-documented, others sparse
- API endpoints not consistently cited
- Data licenses not always specified

**Configuration Management**:
- Hard-coded paths (e.g., `../data/`)
- Mixed use of environment variables
- No centralized configuration file

### Modernization Opportunities

**Extract Common Library**:
```python
# Proposed: gaia_utils.py
def clip_to_aoi(dataset, boundary_path):
    """Standard spatial clipping with CRS handling."""

def download_parallel(urls, max_workers=4):
    """Robust parallel downloader with retry."""

def to_optimized_zarr(dataset, path, chunks=None):
    """Write zarr with best practices."""
```

**Add Testing Framework**:
- pytest for utility functions
- Integration tests for download pipelines
- Validation tests for data QC

**Implement Configuration**:
- YAML or TOML config file
- Environment variable support
- Path resolution relative to project root

**Workflow Orchestration**:
- Consider Prefect or Dagster for complex pipelines
- Enable scheduled updates
- Support incremental processing

### Migration Path

**Phase 1: Documentation** (Current)
- Create comprehensive docs (this file)
- Document all data sources
- Standardize notebook structure

**Phase 2: Consolidation**
- Extract common functions to utilities module
- Standardize error handling
- Implement configuration management

**Phase 3: Testing**
- Add unit tests for utilities
- Integration tests for key pipelines
- Automated QC validation

**Phase 4: Orchestration** (Future)
- Workflow engine for multi-step pipelines
- Scheduled data updates
- Monitoring and alerting

---

**Last Updated**: 2026-02-10
