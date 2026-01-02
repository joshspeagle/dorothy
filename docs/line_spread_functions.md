# Line Spread Functions (LSF) for DOROTHY Surveys

This document describes the spectral resolution and line spread function (LSF) characteristics for each survey supported by DOROTHY. This information is essential for:

1. Convolving synthetic model spectra to match observations
2. Understanding the effective resolution at different wavelengths
3. Cross-survey comparisons and resolution matching

## Overview

| Survey | Wavelength Range | Resolution (R) | LSF Representation |
|--------|------------------|----------------|-------------------|
| DESI | 3600-9800 Å | 2000-5000 | Resolution matrix |
| BOSS | 3600-10400 Å | 1560-2650 | Gaussian σ (wdisp) |
| LAMOST LRS | 3700-9000 Å | ~1800 | Approximate Gaussian |
| LAMOST MRS | 4950-5350 Å (B), 6300-6800 Å (R) | 7500 | Approximate Gaussian |

## Fundamental Relationships

The spectral resolution R, FWHM, and Gaussian σ are related by:

```
R = λ / Δλ_FWHM

Δλ_FWHM = λ / R

σ_λ = Δλ_FWHM / (2 * sqrt(2 * ln(2))) ≈ Δλ_FWHM / 2.355

σ_velocity = c / (R * 2.355) ≈ 127,400 / R  [km/s]
```

For a Gaussian LSF, the 1σ width in velocity space is approximately `c / (R * 2.355)`.

---

## DESI (Dark Energy Spectroscopic Instrument)

### Wavelength Channels

DESI uses three spectral channels with wavelength-dependent resolution:

| Channel | Wavelength Range | Resolution Range | Reference λ |
|---------|------------------|------------------|-------------|
| Blue (B) | 3600-5930 Å | R = 2000-3200 | ~4500 Å |
| Red (R) | 5660-7720 Å | R = 3200-4100 | ~6500 Å |
| NIR (Z) | 7470-9800 Å | R = 4100-5100 | ~8500 Å |

### Resolution Matrix

DESI provides a **resolution matrix** R for each spectrum, which is the most rigorous LSF representation. This matrix is stored in the extracted spectra files as a band-diagonal array.

**Accessing the resolution matrix:**

```python
import numpy as np
from scipy.sparse import dia_matrix

def load_desi_resolution_matrix(resolution_data, spectrum_index):
    """
    Load DESI resolution matrix from extracted spectra.

    Parameters
    ----------
    resolution_data : ndarray
        3D array from RESOLUTION HDU with shape (n_spectra, n_diagonals, n_wavelengths)
    spectrum_index : int
        Index of the spectrum

    Returns
    -------
    R : scipy.sparse.dia_matrix
        Resolution matrix for applying to model spectra
    """
    nwave = resolution_data.shape[2]
    nband = resolution_data.shape[1]
    offsets = range(nband // 2, -nband // 2, -1)
    R = dia_matrix((resolution_data[spectrum_index], offsets), shape=(nwave, nwave))
    return R

# Apply resolution to model spectrum:
# observed = R @ model
```

### Approximate Gaussian Parameterization

For practical use when the full resolution matrix is not available, DESI can be approximated with a wavelength-dependent Gaussian:

```python
def desi_approximate_fwhm(wavelength):
    """
    Approximate FWHM in Angstroms for DESI spectra.

    Parameters
    ----------
    wavelength : array-like
        Wavelength in Angstroms

    Returns
    -------
    fwhm : array-like
        FWHM in Angstroms
    """
    wavelength = np.asarray(wavelength)

    # Piecewise linear R(λ) approximation
    R = np.where(wavelength < 5930,
                 2000 + (wavelength - 3600) * (3200 - 2000) / (5930 - 3600),
                 np.where(wavelength < 7720,
                          3200 + (wavelength - 5660) * (4100 - 3200) / (7720 - 5660),
                          4100 + (wavelength - 7470) * (5100 - 4100) / (9800 - 7470)))

    return wavelength / R

def desi_approximate_sigma(wavelength):
    """Approximate Gaussian sigma in Angstroms."""
    return desi_approximate_fwhm(wavelength) / 2.355
```

### References

- [DESI Spectrograph](https://www.desi.lbl.gov/spectrograph/)
- [DESI Data Processing Pipeline (arXiv:2209.14482)](https://arxiv.org/abs/2209.14482)
- [Specter Documentation](https://desi-specter.readthedocs.io/en/latest/)
- [NOIRLab Data Lab - DESI](https://datalab.noirlab.edu/data/desi)

---

## BOSS (Baryon Oscillation Spectroscopic Survey)

### Wavelength Channels

BOSS uses two spectral channels:

| Channel | Wavelength Range | Resolution Range |
|---------|------------------|------------------|
| Blue | 3600-6350 Å | R = 1560 (3700 Å) → 2270 (6000 Å) |
| Red | 5650-10000 Å | R = 1850 (6000 Å) → 2650 (9000 Å) |

The dichroic splits light at approximately 6000 Å.

### LSF Representation: WDISP

BOSS provides the LSF as a Gaussian σ in the `wdisp` array (wavelength dispersion), stored in units of **pixels** in log-wavelength space.

**Converting WDISP to physical units:**

```python
def boss_wdisp_to_sigma_angstrom(wdisp, wavelength):
    """
    Convert BOSS wdisp (pixels) to Gaussian sigma in Angstroms.

    Parameters
    ----------
    wdisp : array-like
        Wavelength dispersion in pixels (from spCFrame or spPlate files)
    wavelength : array-like
        Wavelength in Angstroms

    Returns
    -------
    sigma : array-like
        Gaussian sigma in Angstroms
    """
    # BOSS uses log10 wavelength grid with spacing 0.0001
    # Pixel size in wavelength units: d(λ)/d(pixel) = ln(10) * λ * 0.0001
    pixel_size_angstrom = np.log(10) * wavelength * 0.0001
    return wdisp * pixel_size_angstrom

def boss_wdisp_to_sigma_velocity(wdisp):
    """
    Convert BOSS wdisp to Gaussian sigma in km/s.

    The velocity dispersion is independent of wavelength for log-wavelength grids.

    Parameters
    ----------
    wdisp : array-like
        Wavelength dispersion in pixels

    Returns
    -------
    sigma_v : array-like
        Gaussian sigma in km/s
    """
    # Velocity per pixel: c * ln(10) * 0.0001 ≈ 69.08 km/s
    velocity_per_pixel = 299792.458 * np.log(10) * 0.0001  # km/s
    return wdisp * velocity_per_pixel
```

### Approximate Gaussian Parameterization

When wdisp is not available, use this wavelength-dependent approximation:

```python
def boss_approximate_fwhm(wavelength):
    """
    Approximate FWHM in Angstroms for BOSS spectra.

    Parameters
    ----------
    wavelength : array-like
        Wavelength in Angstroms

    Returns
    -------
    fwhm : array-like
        FWHM in Angstroms
    """
    wavelength = np.asarray(wavelength)

    # Piecewise linear R(λ) approximation based on SDSS documentation
    R = np.where(wavelength < 6000,
                 1560 + (wavelength - 3700) * (2270 - 1560) / (6000 - 3700),
                 1850 + (wavelength - 6000) * (2650 - 1850) / (9000 - 6000))

    return wavelength / R

def boss_approximate_sigma(wavelength):
    """Approximate Gaussian sigma in Angstroms."""
    return boss_approximate_fwhm(wavelength) / 2.355
```

### Typical Values

| Wavelength | R | FWHM | σ (Å) | σ (km/s) |
|------------|---|------|-------|----------|
| 4000 Å | 1690 | 2.37 Å | 1.01 Å | 75 km/s |
| 5000 Å | 1900 | 2.63 Å | 1.12 Å | 67 km/s |
| 6500 Å | 2080 | 3.13 Å | 1.33 Å | 61 km/s |
| 8000 Å | 2430 | 3.29 Å | 1.40 Å | 52 km/s |

### References

- [BOSS Spectrograph (SDSS-IV)](https://www.sdss4.org/instruments/boss_spectrograph/)
- [SDSS Spectroscopic Data Basics](https://www.sdss4.org/dr17/spectro/spectro_basics/)
- [spCFrame Data Model](https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/PLATE4/spCFrame.html)
- [MaNGA LSF Paper (arXiv:2011.04675)](https://arxiv.org/abs/2011.04675)

---

## LAMOST LRS (Low-Resolution Spectroscopy)

### Specifications

| Parameter | Value |
|-----------|-------|
| Wavelength Range | 3700-9100 Å |
| Resolution | R ≈ 1800 at 5500 Å |
| Dispersion | log10 spacing, COEFF1 = 0.0001 |
| Detector | 2048 × 2048 CCD |
| Fibers | 250 per spectrograph, 3.3" diameter |

### Approximate Gaussian Parameterization

LAMOST LRS has approximately constant resolution across the wavelength range:

```python
def lamost_lrs_approximate_fwhm(wavelength, R=1800):
    """
    Approximate FWHM in Angstroms for LAMOST LRS spectra.

    Parameters
    ----------
    wavelength : array-like
        Wavelength in Angstroms
    R : float
        Spectral resolution (default 1800)

    Returns
    -------
    fwhm : array-like
        FWHM in Angstroms
    """
    return np.asarray(wavelength) / R

def lamost_lrs_approximate_sigma(wavelength, R=1800):
    """Approximate Gaussian sigma in Angstroms."""
    return lamost_lrs_approximate_fwhm(wavelength, R) / 2.355
```

### Typical Values (R = 1800)

| Wavelength | FWHM | σ (Å) | σ (km/s) |
|------------|------|-------|----------|
| 4000 Å | 2.22 Å | 0.94 Å | 71 km/s |
| 5500 Å | 3.06 Å | 1.30 Å | 71 km/s |
| 7000 Å | 3.89 Å | 1.65 Å | 71 km/s |
| 8500 Å | 4.72 Å | 2.01 Å | 71 km/s |

### References

- [LAMOST Spectrograph](https://www.lamost.org/public/instrument/spectragraph?locale=en)
- [LAMOST DR9 Data Description](https://www.lamost.org/dr9/v2.0/doc/lr-data-production-description)

---

## LAMOST MRS (Medium-Resolution Spectroscopy)

### Specifications

LAMOST MRS uses two wavelength bands with higher resolution:

| Band | Wavelength Range | Resolution | Reference λ |
|------|------------------|------------|-------------|
| Blue (B) | 4950-5350 Å | R = 7500 | 5163 Å |
| Red (R) | 6300-6800 Å | R = 7500 | 6593 Å |

### Approximate Gaussian Parameterization

```python
def lamost_mrs_approximate_fwhm(wavelength, R=7500):
    """
    Approximate FWHM in Angstroms for LAMOST MRS spectra.

    Parameters
    ----------
    wavelength : array-like
        Wavelength in Angstroms
    R : float
        Spectral resolution (default 7500)

    Returns
    -------
    fwhm : array-like
        FWHM in Angstroms
    """
    return np.asarray(wavelength) / R

def lamost_mrs_approximate_sigma(wavelength, R=7500):
    """Approximate Gaussian sigma in Angstroms."""
    return lamost_mrs_approximate_fwhm(wavelength, R) / 2.355
```

### Typical Values (R = 7500)

| Band | Wavelength | FWHM | σ (Å) | σ (km/s) |
|------|------------|------|-------|----------|
| Blue | 5163 Å | 0.69 Å | 0.29 Å | 17 km/s |
| Red | 6593 Å | 0.88 Å | 0.37 Å | 17 km/s |

### References

- [LAMOST MRS Data Description](https://dr7.lamost.org/v2.0/doc/mr-data-production-description)

---

## Cross-Survey Resolution Comparison

To match spectra between surveys, you may need to convolve higher-resolution data to match lower resolution. The effective kernel σ for degrading resolution is:

```python
def resolution_matching_kernel(sigma_target, sigma_source):
    """
    Compute Gaussian kernel sigma needed to match resolutions.

    Parameters
    ----------
    sigma_target : float
        Target LSF sigma (lower resolution)
    sigma_source : float
        Source LSF sigma (higher resolution)

    Returns
    -------
    sigma_kernel : float
        Kernel sigma for convolution (in same units as inputs)
        Returns NaN if target resolution is higher than source.
    """
    if sigma_target <= sigma_source:
        return np.nan  # Cannot increase resolution
    return np.sqrt(sigma_target**2 - sigma_source**2)
```

### Resolution Hierarchy (lowest to highest)

1. **LAMOST LRS** (R ≈ 1800) - lowest resolution
2. **BOSS** (R ≈ 1560-2650) - overlapping with LAMOST LRS
3. **DESI** (R ≈ 2000-5000) - higher than BOSS
4. **LAMOST MRS** (R = 7500) - highest resolution

---

## Implementation Notes for DOROTHY

### Planned Functionality

1. **LSF Convolution Module**: Apply survey-specific LSF to model spectra
2. **Resolution Matching**: Degrade high-resolution spectra to match survey resolution
3. **Per-Spectrum LSF**: Load resolution matrices (DESI) or wdisp (BOSS) from data files

### Key Considerations

1. **LSF Shape**: Most fiber-fed spectrographs have approximately Gaussian LSFs, but real LSFs may have extended wings. The resolution matrix (DESI) or measured wdisp (BOSS) should be preferred when available.

2. **Wavelength Dependence**: Resolution varies with wavelength for all surveys. Use the wavelength-dependent functions rather than single R values.

3. **Sampling**: These surveys typically sample at ~3 pixels per FWHM. When convolving model spectra, ensure adequate sampling before convolution.

4. **Vacuum vs Air Wavelengths**: DESI, BOSS, and LAMOST all use vacuum wavelengths.

---

## Data Sources

For the most accurate LSF characterization:

| Survey | LSF Data Location |
|--------|-------------------|
| DESI | RESOLUTION HDU in coadd/spectra files |
| BOSS | WAVEDISP HDU (HDU #4) in spCFrame files |
| LAMOST LRS | Not provided; use approximate R=1800 |
| LAMOST MRS | Not provided; use approximate R=7500 |
