# Environment Setup and Configuration

## Table of Contents

1. [Package Managers](#package-managers)
2. [Environment Configuration](#environment-configuration)
3. [Dependencies](#dependencies)
4. [System Requirements](#system-requirements)
5. [Troubleshooting](#troubleshooting)
6. [Platform-Specific Notes](#platform-specific-notes)

## Package Managers

### Pixi (Recommended)

**What is Pixi?**
- Modern, fast package manager built on Conda
- Cross-platform (macOS, Linux, Windows)
- Automatic environment management
- Reproducible builds with lock file

**Installation**:

Visit https://pixi.sh/dev/installation/ for latest instructions.

macOS/Linux:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Windows:
```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

**Project Setup**:

```bash
# Navigate to repository
cd /Users/lsetiawan/Repos/SSEC/gaia-data-downloaders

# Install dependencies (reads pixi.toml)
pixi install

# Activate environment (optional, pixi run handles this automatically)
pixi shell

# Run notebook
pixi run jupyter notebook
```

**Configuration File**: `pixi.toml`

```toml
[workspace]
authors = ["Scott Henderson <3924836+scottyhq@users.noreply.github.com>"]
channels = ["conda-forge"]
name = "gaia-data-downloaders"
platforms = ["osx-arm64", "linux-64"]
version = "0.1.0"

[dependencies]
geopandas = ">=1.1.2,<2"
obspy = ">=1.4.2,<2"
matplotlib = ">=3.10.8,<4"
pandas = ">=2.2.3,<3"
rioxarray = ">=0.20.0,<0.21"
pygeohydro = ">=0.19.4,<0.20"
ipykernel = ">=7.1.0,<8"
obstore = ">=0.8.2,<0.9"
boto3 = ">=1.42.34,<2"
libgdal-netcdf = ">=3.12.1,<4"
contextily = ">=1.7.0,<2"
matplotlib-scalebar = ">=0.9.0,<0.10"
cartopy = ">=0.25.0,<0.26"
```

**Lock File**: `pixi.lock` (282 KB)
- Pins exact versions of all dependencies
- Ensures reproducibility across machines
- Automatically generated, don't edit manually

**VSCode Integration**:
1. Install Pixi
2. Run `pixi install` in terminal
3. In VSCode: Command Palette (Cmd+Shift+P) → "Python: Select Interpreter"
4. Choose interpreter at `.pixi/envs/default/bin/python`

---

### Conda (Alternative)

**Installation**:

Install Miniconda or Anaconda from https://docs.conda.io/

**Environment Setup**:

```bash
# Create environment from file
conda env create -f environment.yml

# Or manually
conda create -n gaia-hazlab python=3.11
conda activate gaia-hazlab
conda install -c conda-forge geopandas obspy matplotlib pandas rioxarray
pip install pygeohydro
```

**Configuration File**: `environment.yml`

```yaml
name: gaia-hazlab
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy
  - xarray
  - NetCDF4
  - geopandas
  - obspy
  - matplotlib
  - cartopy
  - pandas
  - elevation
  - regionmask
  - rasterio
  - rioxarray
  - ipykernel
  - pip
  - pip:
    - pygeohydro
```

**Activation**:

```bash
conda activate gaia-hazlab
jupyter notebook
```

**Updating**:

```bash
# Update all packages
conda update --all

# Update specific package
conda update xarray

# Update from environment file
conda env update -f environment.yml --prune
```

---

### pip (Minimal Setup)

**Prerequisites**:
- Python 3.11+
- System libraries: GDAL, PROJ, GEOS

**macOS**:

```bash
# Install GDAL via Homebrew
brew install gdal

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install packages
pip install geopandas obspy pygeohydro matplotlib pandas \
    xarray netcdf4 rasterio rioxarray cartopy jupyter
```

**Linux (Ubuntu/Debian)**:

```bash
# Install system dependencies
sudo apt-get install gdal-bin libgdal-dev python3-dev

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install packages
pip install geopandas obspy pygeohydro matplotlib pandas \
    xarray netcdf4 rasterio rioxarray cartopy jupyter
```

**Limitations**:
- More manual setup
- System dependencies required
- No automatic conflict resolution
- Not recommended for complex geospatial environments

---

## Environment Configuration

### Environment Variables

**Synoptic API Token**:

```bash
# Add to ~/.bashrc or ~/.zshrc
export SYNOPTIC_TOKEN='your_token_here'
```

**AWS Credentials**:

```bash
# ~/.aws/credentials
[cresst-user]
aws_access_key_id = AKIAXXXXXXXXXXXXXX
aws_secret_access_key = XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# ~/.bashrc or ~/.zshrc
export AWS_PROFILE=cresst-user
```

**NASA Earthdata**:

```bash
# ~/.netrc
machine urs.earthdata.nasa.gov
login your_username
password your_password

# Set permissions
chmod 600 ~/.netrc
```

**wgrib2 Path** (for HRRR):

```bash
# If using pixi
export PATH="$PATH:/path/to/gaia-data-downloaders/.pixi/envs/default/bin"

# Check
which wgrib2
```

---

### Jupyter Kernel Setup

**Register Conda Environment**:

```bash
conda activate gaia-hazlab
python -m ipykernel install --user --name=gaia-hazlab --display-name="Python (gaia-hazlab)"
```

**Register Pixi Environment**:

```bash
# Pixi handles this automatically
# Kernel appears as "default" in VSCode/JupyterLab
```

**List Available Kernels**:

```bash
jupyter kernelspec list
```

**Remove Kernel**:

```bash
jupyter kernelspec uninstall gaia-hazlab
```

---

### VSCode Configuration

**Recommended Extensions**:
- Python (Microsoft)
- Jupyter (Microsoft)
- Pylance (Microsoft)
- Remote - SSH (for remote work)

**Settings** (`settings.json`):

```json
{
  "python.defaultInterpreterPath": ".pixi/envs/default/bin/python",
  "jupyter.notebookFileRoot": "${workspaceFolder}",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "[python]": {
    "editor.formatOnSave": true,
    "editor.rulers": [88]
  }
}
```

---

## Dependencies

### Core Scientific Stack

| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | Latest | Numerical arrays |
| **pandas** | ≥2.2.3 | Tabular data |
| **xarray** | Latest | Multi-dimensional arrays |
| **dask** | Latest | Parallel computing |

### Geospatial

| Package | Version | Purpose |
|---------|---------|---------|
| **geopandas** | ≥1.1.2 | Vector data |
| **rasterio** | Latest | Raster I/O |
| **rioxarray** | ≥0.20.0 | Xarray raster extension |
| **shapely** | Latest | Geometric operations |
| **pyproj** | Latest | Coordinate transformations |
| **regionmask** | Latest | Spatial masking |
| **cartopy** | ≥0.25.0 | Cartographic plotting |
| **contextily** | ≥1.7.0 | Basemaps |

### Domain-Specific

| Package | Version | Purpose |
|---------|---------|---------|
| **obspy** | ≥1.4.2 | Seismic data |
| **pygeohydro** | ≥0.19.4 | Hydrological data |
| **herbie-data** | Latest | HRRR model access |
| **pyPRISMClimate** | Latest | PRISM data |
| **elevation** | Latest | DEM data |

### Cloud/Storage

| Package | Version | Purpose |
|---------|---------|---------|
| **obstore** | ≥0.8.2 | Object store (S3) |
| **boto3** | ≥1.42.34 | AWS SDK |
| **s3fs** | Latest | S3 filesystem |
| **zarr** | Latest | Chunked arrays |

### Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| **matplotlib** | ≥3.10.8 | Core plotting |
| **cartopy** | ≥0.25.0 | Maps |
| **contextily** | ≥1.7.0 | Basemaps |
| **folium** | Latest | Interactive maps |
| **matplotlib-scalebar** | ≥0.9.0 | Scale bars |

### Data Formats

| Package | Version | Purpose |
|---------|---------|---------|
| **netcdf4** | Latest | NetCDF I/O |
| **h5netcdf** | Latest | HDF5/NetCDF |
| **cfgrib** | Latest | GRIB2 format |
| **libgdal-netcdf** | ≥3.12.1 | GDAL NetCDF driver |

### Jupyter

| Package | Version | Purpose |
|---------|---------|---------|
| **ipykernel** | ≥7.1.0 | Jupyter kernel |
| **jupyterlab** | Latest | Notebook interface |

---

## System Requirements

### Minimum Requirements

- **CPU**: 2+ cores
- **RAM**: 8 GB
- **Disk**: 50 GB free space
- **OS**: macOS 10.15+, Ubuntu 20.04+, Windows 10+
- **Python**: 3.11+

### Recommended

- **CPU**: 4+ cores (for parallel downloads)
- **RAM**: 16+ GB (for large datasets)
- **Disk**: 500+ GB (for data storage)
- **SSD**: Strongly recommended (faster I/O)

### External Tools

**wgrib2** (required for HRRR):
- Installed automatically by pixi/conda
- Manual: https://www.cpc.ncep.noaa.gov/products/wesley/wgrib2/

**GDAL** (required for raster operations):
- Installed by pixi/conda
- Manual macOS: `brew install gdal`
- Manual Linux: `sudo apt-get install gdal-bin libgdal-dev`

---

## Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError: No module named 'geopandas'"

**Cause**: Environment not activated or package not installed

**Solution**:
```bash
# Pixi
pixi install
pixi shell

# Conda
conda activate gaia-hazlab
conda install -c conda-forge geopandas
```

---

#### Issue: "wgrib2: command not found"

**Cause**: wgrib2 not in PATH (required for HRRR notebook)

**Solution**:
```bash
# Check if installed
pixi list | grep wgrib2

# Add to PATH in notebook
import os
os.environ['PATH'] += ':/path/to/.pixi/envs/default/bin'

# Or install manually
conda install -c conda-forge wgrib2
```

---

#### Issue: "OSError: Could not find GDAL library"

**Cause**: GDAL not properly installed or linked

**Solution**:
```bash
# Pixi/Conda (reinstall)
pixi install --force-reinstall libgdal
conda install -c conda-forge libgdal

# macOS (manual)
brew install gdal
export GDAL_DATA=$(brew --prefix gdal)/share/gdal
export PROJ_LIB=$(brew --prefix proj)/share/proj
```

---

#### Issue: "PermissionError: [Errno 13] Permission denied: '~/.netrc'"

**Cause**: Incorrect file permissions for .netrc

**Solution**:
```bash
chmod 600 ~/.netrc
```

---

#### Issue: "RuntimeError: NetCDF: HDF error"

**Cause**: Corrupted download or incompatible NetCDF version

**Solution**:
```bash
# Delete corrupted file and re-download

# Update NetCDF libraries
pixi update netcdf4 h5netcdf
# Or
conda update netcdf4 h5netcdf
```

---

#### Issue: "KeyError: 'SYNOPTIC_TOKEN'"

**Cause**: Environment variable not set

**Solution**:
```bash
# Temporary
export SYNOPTIC_TOKEN='your_token'

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export SYNOPTIC_TOKEN="your_token"' >> ~/.bashrc
source ~/.bashrc
```

---

#### Issue: "Memory Error" or "Killed"

**Cause**: Insufficient RAM for dataset

**Solution**:
```python
# Use dask and chunking
ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Subset before loading
ds = ds.sel(time=slice('2020-01-01', '2020-01-31'))

# Process in batches
for year in range(2000, 2020):
    ds_year = ds.sel(time=ds.time.dt.year == year)
    # Process and save
```

---

#### Issue: "SSL: CERTIFICATE_VERIFY_FAILED"

**Cause**: SSL certificate issues (common on macOS)

**Solution**:
```bash
# Python 3.11+ on macOS
/Applications/Python\ 3.11/Install\ Certificates.command

# Or set environment variable (not recommended for production)
export PYTHONHTTPSVERIFY=0
```

---

### Dependency Conflicts

**Symptom**: Pixi/conda solver cannot find compatible versions

**Solution**:

1. Update lock file:
   ```bash
   pixi update
   ```

2. Relax version constraints in `pixi.toml`:
   ```toml
   # Change from:
   geopandas = "1.1.2"
   # To:
   geopandas = ">=1.1.2"
   ```

3. Create fresh environment:
   ```bash
   # Pixi
   rm -rf .pixi
   pixi install

   # Conda
   conda env remove -n gaia-hazlab
   conda env create -f environment.yml
   ```

---

### Performance Issues

**Symptom**: Slow notebook execution

**Diagnosis**:
```python
# Check dask configuration
import dask
print(dask.config.config)

# Monitor memory
import psutil
print(f"RAM: {psutil.virtual_memory().percent}% used")
```

**Solutions**:

1. Increase dask workers:
   ```python
   from dask.distributed import Client
   client = Client(n_workers=4, threads_per_worker=2)
   ```

2. Reduce chunk size:
   ```python
   ds = ds.chunk({'time': 10, 'lat': 50, 'lon': 50})
   ```

3. Clear cache:
   ```bash
   rm -rf ~/.cache/pyproj
   rm -rf ~/.cache/elevation
   ```

---

## Platform-Specific Notes

### macOS (Apple Silicon)

**Advantages**:
- Pixi has native ARM support
- Faster than Rosetta emulation

**Considerations**:
- Some packages (older GDAL versions) may need Rosetta
- Use `osx-arm64` platform in pixi.toml

**Rosetta Installation** (if needed):
```bash
softwareupdate --install-rosetta
```

---

### macOS (Intel)

**Platform**: `osx-64`

**Standard Setup**: Works with all packages

---

### Linux

**Platform**: `linux-64`

**System Dependencies**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential libgdal-dev libproj-dev libgeos-dev

# CentOS/RHEL
sudo yum install gcc gcc-c++ gdal-devel proj-devel geos-devel
```

**HPC Clusters**:
- Use Conda over Pixi (better support)
- Load system modules: `module load gdal proj geos`
- Request adequate resources (16+ GB RAM)

---

### Windows

**Platform**: Windows not currently supported in `pixi.toml`

**To Add Support**:
```toml
[workspace]
platforms = ["osx-arm64", "linux-64", "win-64"]
```

**Considerations**:
- wgrib2 requires Windows Subsystem for Linux (WSL) or Cygwin
- GDAL installation more complex (use OSGeo4W)
- Recommend using WSL2 with Linux setup

---

### Cloud Environments

**JupyterHub/JupyterLab**:
```bash
# Clone repo
git clone https://github.com/SSEC/gaia-data-downloaders.git
cd gaia-data-downloaders

# Install with pixi
pixi install

# Register kernel
pixi run python -m ipykernel install --user --name=gaia-data
```

**Google Colab**:
```python
# Install packages
!pip install geopandas obspy pygeohydro rioxarray xarray

# Clone repo
!git clone https://github.com/SSEC/gaia-data-downloaders.git
%cd gaia-data-downloaders
```

**AWS SageMaker**:
- Use conda_python3 kernel
- Install packages in notebook:
  ```python
  !conda install -c conda-forge geopandas obspy -y
  !pip install pygeohydro
  ```

---

## Best Practices

1. **Use Pixi**: Faster, more reliable than conda
2. **Pin Major Versions**: Allow minor updates (e.g., `>=1.1.2,<2`)
3. **Separate Environments**: Don't mix projects
4. **Update Regularly**: `pixi update` or `conda update --all`
5. **Version Control**: Commit `pixi.toml` and `pixi.lock`
6. **Document Requirements**: Add system deps to README
7. **Test Fresh Install**: Periodically test from scratch
8. **Use Environments**: Don't install globally
9. **Check Compatibility**: Test on target platform
10. **Monitor Resources**: Watch RAM and disk usage

---

**Last Updated**: 2026-02-10
