# This program is to compare the LW heating difference between CNTL and NCRF
# %% Section 1
# Import package
import numpy as np;
import xarray as xr;

from scipy import fft;
from matplotlib import pyplot as plt;
from matplotlib.colors import TwoSlopeNorm;

# functions
def kel_filter(data, axes=(None, None)):
    data_shape: tuple(int) = data.shape;

    wn: np.ndarray[int]   = fft.fftfreq(data_shape[axes[0]], d=1/data_shape[axes[0]]).astype(int);
    fr: np.ndarray[float] = fft.fftfreq(data_shape[axes[1]], d=1/4);

    wnm, frm = np.meshgrid(wn, fr);

    kel_curve = lambda wn, eq: wn * np.sqrt(9.81*eq) * 86400 / (2*np.pi*6.371e6);

    pos_cond = (
        (wnm >= 1) & (wnm <= 14) &
        (frm >= 1/20) & (frm <= 1/2.5) &
        (frm >= kel_curve(wnm, 8)) & (frm <= kel_curve(wnm, 90))
    );

    neg_cond = (
        (wnm <= -1) & (wnm >= -14) &
        (frm <= -1/20) & (frm >= -1/2.5) &
        (frm <= kel_curve(wnm, 8)) & (frm >= kel_curve(wnm, 90))
    );

    kel_mask: np.ndarray  = np.where(pos_cond | neg_cond, 1, 0);

    data_fft = fft.fft(data, axis=axes[0]);
    data_fft = fft.ifft(data_fft, axis=axes[1]) * data.shape[axes[1]];

    data_mask: np.ndarray = data_fft * kel_mask[:, None, :];

    return wn, fr, data_mask;

def reconstruct(data, axes=(None, None)):

    data_shape: tuple(int) = data.shape;

    data_ifft = fft.ifft(data, axis=axes[0]);
    data_ifft = fft.fft(data_ifft, axis=axes[1]) / data.shape[axes[1]];

    return data_ifft;

# vertical integral
def vint(data, lev, init_lev, term_lev, axis=1):
    lev_rng = np.where((lev >= init_lev) & (lev <= term_lev))[0];

    lev_sel = lev[lev_rng];
    data_sel = data[lev_rng];

    data_ave = (data_sel[1:] + data_sel[:-1]) / 2.;
    data_vint = np.nansum(data_ave * lev_sel.diff(dim="lev")) / np.nansum(lev_sel.diff(dim="lev"));

    return data_vint;

# %% Section 2
# Load data
# # Load CNTL and NCRF data
path: str = "/work/b11209013/2024_Research/MPAS/merged_data/"

dims: dict[str, np.ndarray] = dict();
data: dict[str, np.ndarray] = dict();

with xr.open_dataset(f"{path}CNTL/qv.nc") as f:
    f = f.sel(lat=slice(-5, 5));

    cntl_dims = f.coords;

    data["cntl"] = f["qv"] * 1000.;

with xr.open_dataset(f"{path}NCRF/qv.nc") as f:
    f = f.sel(lat=slice(-5, 5));

    ncrf_dims = f.coords;

    data["ncrf"] = f["qv"] * 1000.;

# Acquire anomalous data (rm cli and zonal)
data_ano : dict[str, np.ndarray] = dict(
    (key, data[key] - data[key].mean(dim={"time", "lon"}))
    for key in data.keys()
);

del data;

# Symmetric data
data_sym: dict[str, np.ndarray] = dict(
    (key, ((data_ano[key] + data_ano[key][:, :, :, ::-1])/2).mean(dim={"lat"}))
    for key in data_ano.keys()
);

# Bandpass filter
cntl_filt = kel_filter(data_sym["cntl"], axes=(-1, 0));
ncrf_filt = kel_filter(data_sym["ncrf"], axes=(-1, 0));

wn: dict[str, np.ndarray] = dict(
    cntl = cntl_filt[0],
    ncrf = ncrf_filt[0],
);

fr: dict[str, np.ndarray] = dict(
    cntl = cntl_filt[1],
    ncrf = ncrf_filt[1],
);

data_masked: dict[str, np.ndarray] = dict(
    cntl = cntl_filt[2],
    ncrf = ncrf_filt[2],
);

del cntl_filt, ncrf_filt;

data_recon: dict[str, np.ndarray] = dict(
    (key, reconstruct(data_masked[key], axes=(-1, 0)))
    for key in data_masked.keys()
);

print(data_recon["cntl"].shape);

# vertically integrate over specific levels
data_vint_400_600: dict[str, np.ndarray] = dict(
    cntl = vint(np.var(data_recon["cntl"], axis=(0, -1)), cntl_dims["lev"]*100., 40000, 60000),
    ncrf = vint(np.var(data_recon["ncrf"], axis=(0, -1)), ncrf_dims["lev"]*100., 40000, 60000),
);
print("vertically integrate over 400~600 hPa:");
print(data_vint_400_600);

data_vint_300_700: dict[str, np.ndarray] = dict(
    cntl = vint(np.var(data_recon["cntl"], axis=(0, -1)), cntl_dims["lev"]*100., 30000, 70000),
    ncrf = vint(np.var(data_recon["ncrf"], axis=(0, -1)), ncrf_dims["lev"]*100., 30000, 70000),
);
print("vertically integrate over 300~700 hPa:");
print(data_vint_300_700);
