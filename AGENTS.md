# GAIA Data Downloaders - Agent Documentation

**Repository**: gaia-data-downloaders
**Organization**: SSEC (Space Science and Engineering Center)
**Project**: GAIA / CRESST Hydroclimatological Data Collection System
**Primary Language**: Python (Jupyter Notebooks)
**License**: Not specified

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Quick Start](#quick-start)
3. [Repository Architecture](#repository-architecture)
4. [Key Concepts](#key-concepts)
5. [Working with the Codebase](#working-with-the-codebase)
6. [Data Sources and Pipelines](#data-sources-and-pipelines)
7. [Development Workflow](#development-workflow)
8. [Detailed Documentation](#detailed-documentation)

## Executive Summary

This repository is a collection of Jupyter notebooks designed to download, process, and analyze hydroclimatological data for the GAIA (Geophysical Applications for Interdisciplinary Advancement) and CRESST projects. The primary focus is on the Pacific Northwest region, particularly the Skagit River watershed in Washington State.

**Core Purpose**: Provide reproducible workflows for acquiring multi-source environmental data including:
- Weather and climate model outputs (CONUS404, HRRR, WRF, PRISM)
- Precipitation analysis (NOAA Stage IV)
- Hydrological data (USGS stream gauges)
- Topographic data (DEMs)
- Seismic station data (IRIS)
- Station inventory management

**Target Users**: Researchers, data scientists, and engineers working with environmental data for hydrological modeling, hazard assessment, and climate analysis.

## Quick Start

### Prerequisites

- Python 3.11+
- Pixi package manager (recommended) or Conda
- 10+ GB disk space for data downloads
- AWS credentials for S3 bucket access (optional, for data sharing)

### Installation

**Option 1: Using Pixi (Recommended)**

```bash
cd /Users/lsetiawan/Repos/SSEC/gaia-data-downloaders
pixi install
```

Then in VSCode, select the Python kernel at `.pixi/envs/default/bin/python`.

**Option 2: Using Conda**

```bash
conda env create -f environment.yml
conda activate gaia-hazlab
```

### Quick Test

Open any downloader notebook (e.g., `PRISM_Downloader.ipynb`) and run the first few cells to verify your environment is configured correctly.

## Repository Architecture

### High-Level Structure

```
gaia-data-downloaders/
├── AGENTS.md                    # This file - main documentation entry point
├── agents_docs/                 # Detailed reference documentation
│   ├── architecture.md          # System architecture and design decisions
│   ├── data_sources.md          # Data source details and APIs
│   ├── notebook_reference.md    # Individual notebook documentation
│   ├── data_processing.md       # Data processing patterns and workflows
│   ├── environment_setup.md     # Environment and dependency management
│   └── aws_s3_integration.md    # AWS S3 bucket integration guide
├── *.ipynb                      # Data downloader notebooks (18 total)
├── pixi.toml                    # Pixi package configuration
├── environment.yml              # Conda environment specification
├── USGS_Stream_Gage/           # Downloaded USGS stream gauge data
│   └── {station_id}/           # Per-station directories
│       ├── {station_id}_data.csv
│       └── {station_id}_header.txt
├── cache/                       # HTTP cache and failed download tracking
│   ├── aiohttp_cache.sqlite
│   └── failed_ids_*.txt
├── stations_by_basin*.csv       # Station inventory CSVs
├── qc_basins_stations.png       # QC visualization
└── remote-data.html             # Folium map visualization

Data directories (not in repo):
├── ../data/weather_data/        # Weather/climate model outputs
├── ../data/GIS/                 # GIS boundary files
├── ../data/stageIV/            # NOAA Stage IV precipitation
└── ../plots/                    # Generated plots
```

### Architecture Philosophy

The repository follows a **notebook-first, workflow-oriented architecture**:

1. **Modularity**: Each notebook is self-contained for a specific data source
2. **Reproducibility**: Complete workflows from download to visualization
3. **Geographic Focus**: Most workflows clip data to Skagit River watershed
4. **Cloud-Ready**: Supports both local and cloud (S3) data storage
5. **Parallelization**: Uses ThreadPoolExecutor for concurrent downloads

See [agents_docs/architecture.md](agents_docs/architecture.md) for detailed architectural documentation.

## Key Concepts

### 1. Data Pipeline Pattern

Most notebooks follow this pattern:

```python
# 1. Define AOI (Area of Interest)
mask = gpd.read_file('../data/GIS/SkagitBoundary.json')

# 2. Download data (often with date range)
dates = pd.date_range('2017-02-01', '2017-02-08', freq='1d')
download_data(dates, dest_path='../data/weather_data/')

# 3. Open and spatial subset
ds = xr.open_dataset(file_path)
ds_clipped = ds.rio.clip(mask.geometry)

# 4. Process/derive variables
ds['windspeed'] = np.sqrt(ds['U10']**2 + ds['V10']**2)

# 5. Save to optimized format
ds_clipped.to_zarr('output.zarr', mode='w')

# 6. Visualize
ds_clipped['variable'].plot()
```

### 2. Geographic Scope

**Primary AOI**: Skagit River watershed, Washington State
- Boundary file: `../data/GIS/SkagitBoundary.json` (GeoJSON)
- Approximate bounds: (-122.6, 47.5, -120, 49.5) in (lon_min, lat_min, lon_max, lat_max)
- Reference point: Mt. Baker (48.7758, -121.8199)

**Secondary Coverage**: Pacific Northwest / Washington State
- Used for station inventories and broader analysis

### 3. Data Formats

**Input Formats**:
- NetCDF (.nc) - climate model outputs
- GRIB2 (.grib2) - weather model data
- GeoTIFF (.tif) - elevation data
- CSV - tabular data (USGS stream gauges)
- GeoJSON - vector boundaries

**Output Formats**:
- Zarr - chunked array storage (preferred for large datasets)
- NetCDF - compatible format
- GeoJSON - station inventories
- PNG - plots and visualizations

### 4. Coordinate Systems

Most data uses WGS84 (EPSG:4326) lat/lon coordinates, but model grids may use:
- Lambert Conformal Conic (WRF, HRRR, CONUS404)
- EASE-Grid 2.0 (some satellite products)

The notebooks handle CRS transformations using `rioxarray` and `geopandas`.

## Working with the Codebase

### For AI Agents

**Navigation Strategy**:
1. Start with `AGENTS.md` (this file) for overview
2. Consult `agents_docs/notebook_reference.md` for specific data sources
3. Reference `agents_docs/data_processing.md` for common patterns
4. Check `agents_docs/architecture.md` for design decisions

**Common Operations**:

1. **Adding a new data source**: Create a new notebook following the naming pattern `{SOURCE}_Downloader.ipynb`
2. **Modifying AOI**: Update the boundary file reference in `../data/GIS/`
3. **Changing output format**: Modify the `to_zarr()` or `to_netcdf()` calls
4. **Adding variables**: Extend the `variables` or `params` dictionaries in download functions

**Code Quality Guidelines**:
- Use descriptive variable names (e.g., `analysis_date`, not `d`)
- Document API endpoints and data sources in markdown cells
- Include data citations and licenses
- Handle failed downloads gracefully (see cache directory pattern)
- Use ThreadPoolExecutor for I/O-bound operations

### For Human Developers

**Getting Started**:
1. Clone the repository
2. Install dependencies with `pixi install`
3. Open any notebook to explore example workflows
4. Modify date ranges and parameters as needed
5. Run notebooks sequentially (some depend on downloaded data)

**Best Practices**:
- Always verify the AOI boundary file exists before clipping
- Check disk space before large downloads
- Use `mode='a'` for incremental zarr writes
- Cache HTTP requests when possible (aiohttp_cache.sqlite)
- Document data provenance in notebook markdown cells

**Common Issues**:
- API rate limits: Use backoff/retry logic (see USGS notebooks)
- Memory errors: Process data in chunks, use dask
- CRS mismatches: Always check and transform coordinates
- Missing dependencies: Ensure wgrib2 is in PATH for HRRR/GRIB2 data

See [agents_docs/environment_setup.md](agents_docs/environment_setup.md) for detailed troubleshooting.

## Data Sources and Pipelines

The repository integrates data from 10+ sources:

| Data Source | Notebook | Frequency | Coverage |
|-------------|----------|-----------|----------|
| CONUS404 | `CONUS404_Downloader.ipynb` | Hourly/Daily | CONUS, 4km |
| HRRR | `HRRR_Downloader.ipynb` | Hourly | CONUS, 3km |
| WRF-CMIP6 | `WRF_Downloader.ipynb` | Hourly | Western US, 9km |
| PRISM | `PRISM_Downloader.ipynb` | Daily | CONUS, 4km |
| Stage IV | `StageIV_Downloader.ipynb` | Daily | CONUS, 4km |
| USGS Gages | `USGS_Stream_Flow_Bulk_Downloader.ipynb` | 15-min | National |
| ORNL | `ORNL_Downloader.ipynb` | Various | Regional |
| DEM | `download_DEM.ipynb` | Static | Global, 90m |
| Seismic | `merged-inventory.ipynb` | Real-time | IRIS network |
| Weather Stations | `merged-inventory.ipynb` | Variable | Synoptic API |

See [agents_docs/data_sources.md](agents_docs/data_sources.md) for detailed information on each source including:
- API endpoints and authentication
- Data variables available
- Temporal and spatial coverage
- Usage limitations and best practices

## Development Workflow

### Standard Workflow for Adding New Data

1. **Create notebook**: `{DataSource}_Downloader.ipynb`
2. **Import libraries**: Follow common imports pattern
3. **Define parameters**: Date range, variables, AOI
4. **Implement download**: Use ThreadPoolExecutor for parallelism
5. **Spatial subset**: Clip to Skagit boundary or custom AOI
6. **Process/derive**: Calculate additional variables
7. **Save optimized**: Write to Zarr with appropriate chunking
8. **Visualize**: Create QC plots
9. **Document**: Add markdown cells explaining data source and workflow

### Git Workflow

```bash
# The repository tracks changes to notebooks but ignores data outputs
git add *.ipynb
git commit -m "Add {DataSource} downloader with {feature}"
git push origin main
```

**Ignored patterns** (`.gitignore`):
- `*.h5` - HDF5 files
- `*.tif` - GeoTIFF files
- `USGS_Stream_Gage/*` - Downloaded stream gauge data
- `.pixi/*` - Pixi environment (except config)

### Testing and QA

**Manual Testing**:
1. Run notebook with small date range (1-3 days)
2. Verify output file creation and format
3. Check spatial extent matches AOI
4. Validate data values are reasonable
5. Generate visualization for QC

**Automated Checks**:
- Currently none - opportunity for improvement
- Consider adding pytest for utility functions

### Deployment

**Local Development**:
- Run notebooks in Jupyter or VSCode
- Data stored locally in `../data/`

**Cloud Deployment**:
- Upload processed data to S3 (see `S3-bucket.ipynb`)
- Share station inventories as GeoJSON
- Host interactive maps (e.g., `remote-data.html`)

See [agents_docs/aws_s3_integration.md](agents_docs/aws_s3_integration.md) for S3 deployment details.

## Detailed Documentation

The `agents_docs/` directory contains detailed reference documentation:

- **[architecture.md](agents_docs/architecture.md)**: System architecture, design patterns, and technical decisions
- **[data_sources.md](agents_docs/data_sources.md)**: Comprehensive guide to all data sources, APIs, and data formats
- **[notebook_reference.md](agents_docs/notebook_reference.md)**: Detailed documentation for each notebook
- **[data_processing.md](agents_docs/data_processing.md)**: Common data processing patterns, coordinate transformations, and optimization techniques
- **[environment_setup.md](agents_docs/environment_setup.md)**: Dependency management, troubleshooting, and environment configuration
- **[aws_s3_integration.md](agents_docs/aws_s3_integration.md)**: Cloud storage integration, credentials, and data sharing workflows

## Key File Paths

**Configuration**:
- `/Users/lsetiawan/Repos/SSEC/gaia-data-downloaders/pixi.toml` - Pixi package config
- `/Users/lsetiawan/Repos/SSEC/gaia-data-downloaders/environment.yml` - Conda environment
- `/Users/lsetiawan/Repos/SSEC/gaia-data-downloaders/.gitignore` - Git ignore patterns

**Notebooks** (18 total):
- Core downloaders: `CONUS404_Downloader.ipynb`, `HRRR_Downloader.ipynb`, `PRISM_Downloader.ipynb`, etc.
- USGS series: `USGS_Stream_Flow_Bulk_Downloader.ipynb`, `USGS_Stream_Gage_Site_Metadata_Downloader.ipynb`
- Inventory: `merged-inventory.ipynb`, `select_stations_by_watershed_nc.ipynb`
- S3 integration: `S3-bucket.ipynb`

**Data** (not in repository):
- `../data/GIS/SkagitBoundary.json` - AOI boundary file
- `../data/weather_data/` - Downloaded weather/climate data
- `../data/stageIV/` - NOAA Stage IV precipitation

**Output**:
- `/Users/lsetiawan/Repos/SSEC/gaia-data-downloaders/USGS_Stream_Gage/` - Stream gauge data (743 stations)
- `/Users/lsetiawan/Repos/SSEC/gaia-data-downloaders/cache/` - Download cache and failed IDs
- `stations_by_basin.csv`, `stations_by_basin_with_gages.csv` - Station inventories

## Contributing

When modifying this codebase:

1. **Document changes**: Update relevant `agents_docs/*.md` files
2. **Follow patterns**: Maintain consistency with existing notebooks
3. **Test thoroughly**: Run notebooks end-to-end with test data
4. **Commit selectively**: Exclude large data files from git
5. **Update dependencies**: Keep `pixi.toml` and `environment.yml` in sync

## Contact and Resources

**Project Resources**:
- AWS S3 Bucket: `s3://cresst` (requires credentials)
- Station inventory map: `remote-data.html` (interactive Folium map)
- QC visualization: `qc_basins_stations.png`

**Key Dependencies**:
- xarray - multi-dimensional data analysis
- geopandas - geospatial vector data
- rioxarray - raster I/O and clipping
- obspy - seismic data access
- pygeohydro - hydrological data utilities
- herbie-data - weather model access
- boto3 / obstore - S3 integration

## Conventions and Standards

**Naming Conventions**:
- Notebooks: `{Source}_Downloader.ipynb` or `{Purpose}.ipynb`
- Data variables: Use source-specific standard names (e.g., WRF uses `T2`, PRISM uses `tmean`)
- Output files: `{source}_{daterange}_{processing}.zarr`

**Coordinate Conventions**:
- Always use (longitude, latitude) order for bounds tuples
- Default CRS: EPSG:4326 (WGS84)
- Use `rio.clip()` for spatial subsetting

**Temporal Conventions**:
- Use pandas date_range for generating date sequences
- Use ISO 8601 format for timestamps: `YYYY-MM-DD HH:MM:SS`
- Time zones: Most data in UTC; USGS data may be local time

**Code Style**:
- Follow PEP 8 for Python code
- Use type hints in function signatures
- Document API keys and credentials requirements
- Handle exceptions gracefully

---

**Last Updated**: 2026-02-10
**Documentation Version**: 1.0
**Maintained By**: SSEC Team
