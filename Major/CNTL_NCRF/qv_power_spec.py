# This program is to compute the powr spefctrum of temperature anomaly of CNTL and NCRF
# %% Section 1
# Import packages
import sys
import numpy as np;
import netCDF4 as nc;
import matplotlib.pyplot as plt;
from scipy.ndimage import convolve1d;

sys.path.append("/home/b11209013/Package");
import Theory as th; #type: ignore

# %% Section 2
# Load the data
path: str = "/work/b11209013/2024_Research/MPAS/merged_data/";

dims: dict[str, np.ndarray] = dict();
data: dict[str, np.ndarray] = dict();


# # CNTL
with nc.Dataset(f"{path}CNTL/qv.nc", "r") as cntl:
    for key in cntl.dimensions.keys():
        dims[key] = cntl.variables[key][:];
    
    lat_lim: tuple[int] = np.where((dims["lat"] >= -5) & (dims["lat"] <= 5))[0];  
    
    data["cntl"] = cntl.variables["qv"][:, :, lat_lim, :] * 1000.;
    
# # NCRF
with nc.Dataset(f"{path}NCRF/qv.nc", "r") as nsc:
    data["ncrf"] = nsc.variables["qv"][:, :, lat_lim, :] * 1000.;

ltime, llev, llat, llon = data["cntl"].shape;

# %% Section 3
# Processing data
# # Remove climatology and zonal mean
data_rm_cli: dict[str, np.ndarray] = dict(
    (exp, data[exp] - np.mean(data[exp], axis=(0, 3), keepdims=True))
    for exp in data.keys()
);

# # Construct symmetric data
data_sym: dict[str, np.ndarray] = dict(
    (exp, ((data_rm_cli[exp] + np.flip(data_rm_cli[exp], axis=2)) / 2))
    for exp in data_rm_cli.keys()
);

data_asy: dict[str, np.ndarray] = dict(
    (exp, ((data_rm_cli[exp] - np.flip(data_rm_cli[exp], axis=2)) / 2))
    for exp in data_rm_cli.keys()
);

# # Windowing
lsec: int = 120;
hanning: np.ndarray = np.hanning(lsec)[:, None, None, None];

sym_window: dict[str, np.ndarray] = dict(
    (exp, np.array([
        data_sym[exp][i*60:i*60+lsec] * hanning
        for i in range(5)
    ]))
    for exp in data_sym.keys()
);

asy_window: dict[str, np.ndarray] = dict(
    (exp, np.array([
        data_asy[exp][i*60:i*60+lsec] * hanning
        for i in range(5)
    ]))
    for exp in data_asy.keys()
);

# %% Section 4
# Compute power spectrum
def power_spec(
        data: np.ndarray,
) -> np.ndarray:
    fft: np.ndarray = np.fft.fft(data, axis=1);
    fft: np.ndarray = np.fft.ifft(fft, axis=4) * data.shape[4];

    ps : np.ndarray = (fft * fft.conj()) / (data.shape[1] * data.shape[4])**2;

    return ps.mean(axis=0).real;

sym_ps: dict[str, np.ndarray] = dict(
    (exp, power_spec(sym_window[exp]).sum(axis=(2)))
    for exp in data.keys()
);

asy_ps: dict[str, np.ndarray] = dict(
    (exp, power_spec(asy_window[exp]).sum(axis=2))
    for exp in data.keys()
);

# # Vertical average with mass wighted
def vertical_avg(
        data: np.ndarray,
        lev : np.ndarray,
) -> np.ndarray:
    data_ave : np.ndarray = (data[:, 1:] + data[:, :-1]) /2.;
    data_vint: np.ndarray = -np.sum(data_ave * np.diff(lev*100.)[None, :, None], axis=1) / -np.sum(np.diff(lev*100.));

    return data_vint;

sym_ps_weight: dict[str, np.ndarray] = dict(
    (exp, vertical_avg(sym_ps[exp], dims["lev"]))
    for exp in data.keys()
);
asy_ps_weight: dict[str, np.ndarray] = dict(
    (exp, vertical_avg(asy_ps[exp], dims["lev"]))
    for exp in data.keys()
);

 # Compute background

def background(data, nsmooth=20):
    kernel = np.array([1, 2, 1])
    kernel = kernel / kernel.sum()

    for _ in range(10):
        data = convolve1d(data, kernel, mode='nearest')

    data_low  = data[:data.shape[0]//2]
    data_high = data[data.shape[0]//2:]

    for _ in range(10):
        data_low = convolve1d(data_low, kernel, mode='nearest')

    for _ in range(40):
        data_high = convolve1d(data_high, kernel, mode='nearest')

    data = np.concatenate([data_low, data_high], axis=0)

    return data

bg: np.ndarray = background(
    (sym_ps_weight["cntl"] + asy_ps_weight["cntl"])/2
);

sym_peak: dict[str, np.ndarray] = dict(
    (exp, sym_ps_weight[exp] / bg)
    for exp in data.keys()
);

wn: np.ndarray = (np.fft.fftfreq(llon, d=1/llon).astype(int));
fr: np.ndarray = (np.fft.fftfreq(lsec, d=1/4));

wn_v: np.ndarray = np.fft.fftshift(wn);
fr_v: np.ndarray = np.fft.fftshift(fr);

fr_ana, wn_ana = th.genDispersionCurves(Ahe=[8, 25, 90]);
e_cond = np.where(wn_ana[3, 0] <= 0)[0];


# Compute dominant frequency and wavenumber
wnm, frm = np.meshgrid(wn, fr);

kelvin = lambda wn, eq: wn * np.sqrt(9.81*eq) * 86400 / (2*np.pi*6.371e6);
kelvin_cond: tuple[int] = np.where(
    (wnm >= 1) & (wnm <= 14) & (frm >= 1/20) & (frm <= 1/2.5) &
    (frm >= kelvin(wnm, 8)) & (frm <= kelvin(wnm, 90))
);

wn_cntl: float = np.sum(wnm[kelvin_cond] * sym_ps_weight["cntl"][kelvin_cond]) / np.sum(sym_ps_weight["cntl"][kelvin_cond]);
fr_cntl: float = np.sum(frm[kelvin_cond] * sym_ps_weight["cntl"][kelvin_cond]) / np.sum(sym_ps_weight["cntl"][kelvin_cond]);

wn_ncrf: float = np.sum(wnm[kelvin_cond] * sym_ps_weight["ncrf"][kelvin_cond]) / np.sum(sym_ps_weight["ncrf"][kelvin_cond]);
fr_ncrf: float = np.sum(frm[kelvin_cond] * sym_ps_weight["ncrf"][kelvin_cond]) / np.sum(sym_ps_weight["ncrf"][kelvin_cond]);

phase_speed = lambda wn, fr: fr / wn * (2*np.pi*6.371e6) / 86400;

cntl_speed: float = phase_speed(wn_cntl, fr_cntl);
ncrf_speed: float = phase_speed(wn_ncrf, fr_ncrf);

# Figure
plt.rcParams["font.family"] = "serif";

kelvin = lambda wn, eq: wn * np.sqrt(9.81*eq) * 86400 / (2*np.pi*6.371e6);
kelvin_inv = lambda fr, eq: fr * (2*np.pi*6.371e6) / (86400 * np.sqrt(9.81*eq));

def plot_lines(
    ax,
    wn_ana: np.ndarray,
    fr_ana: np.ndarray,
) -> None:
    for i in range(3):
        ax.plot(wn_ana[3, i, e_cond], fr_ana[3, i, e_cond], color="black", linewidth=1);
        ax.plot(wn_ana[4, i], fr_ana[4, i], color="black", linewidth=1);
        ax.plot(wn_ana[3, i], fr_ana[5, i], color="black", linewidth=1);

    ax.set_xticks(np.linspace(-14, 14, 8, dtype=int));
    ax.set_yticks(np.linspace(0, 0.5, 6));
    ax.hlines(y=1/20, xmin=1, xmax=kelvin_inv(1/20, 8), linestyle="-", color="red", linewidth=5);
    ax.hlines(y=1/2.5, xmin=kelvin_inv(1/2.5, 90), xmax=14, linestyle="-", color="red", linewidth=5);
    ax.vlines(x=1, ymin=1/20, ymax=kelvin(1, 90), linestyle="-", color="red", linewidth=5);
    ax.vlines(x=14, ymin=kelvin(14, 8), ymax=1/2.5, linestyle="-", color="red", linewidth=5);
    ax.plot(np.linspace(kelvin_inv(1/20, 90), kelvin_inv(1/2.5, 90), 100), kelvin(np.linspace(kelvin_inv(1/20, 90), kelvin_inv(1/2.5, 90), 100), 90), color="red", linewidth=5);
    ax.plot(np.linspace(kelvin_inv(1/20, 8), 14, 100), kelvin(np.linspace(kelvin_inv(1/20, 8), 14, 100), 8), color="red", linewidth=5);
    ax.axvline(0, linestyle="--", color="black")
    ax.axhline(1/3 , linestyle="--", color="black");
    ax.axhline(1/8 , linestyle="--", color="black");
    ax.axhline(1/20, linestyle="--", color="black");
    ax.text(15, 1/3 , "3 Days", ha="right", va="bottom");
    ax.text(15, 1/8 , "8 Days", ha="right", va="bottom");
    ax.text(15, 1/20, "20 Days", ha="right", va="bottom");


fig, ax = plt.subplots(1, 2, figsize=(12, 7), sharey=True);
plt.subplots_adjust(left=0.08, right=0.96, bottom=0.03, top=0.9);
cntl_ps = ax[0].contourf(
    wn_v, fr_v[fr_v>0],
    np.fft.fftshift(sym_peak["cntl"])[fr_v>0],
    cmap="Blues",
    levels=np.linspace(1, 10, 19),
    extend="max",
);
plot_lines(ax[0], wn_ana, fr_ana);
ax[0].plot(wn_cntl, fr_cntl, "ro", markersize=10);
ax[0].text(15, 0, f"Phase Speed: {cntl_speed:.2f} [m/s]", ha="right", va="bottom");
ax[0].text(0, -0.06, "Zonal Wavenumber", ha="center", fontsize=14);
ax[0].text(-20, 0.25, "Freqquency [CPD]", va="center", rotation=90, fontsize=14);
ax[0].set_xlim(-15, 15);
ax[0].set_ylim(0, 1/2);
ax[0].text(0, 0.52, "CNTL", ha="center", fontsize=16)

nsc_ps = ax[1].contourf(
    wn_v, fr_v[fr_v>0],
    np.fft.fftshift(sym_peak["ncrf"])[fr_v>0],
    cmap="Blues",
    levels=np.linspace(1, 10, 19),
    extend="max",
);
plot_lines(ax[1], wn_ana, fr_ana);
ax[1].text(15, 0, f"Phase Speed: {ncrf_speed:.2f} [m/s]", ha="right", va="bottom");
ax[1].plot(wn_ncrf, fr_ncrf, "ro", markersize=10);
ax[1].text(0, -0.06, "Zonal Wavenumber", ha="center", fontsize=14);
ax[1].set_xlim(-15, 15);
ax[1].set_ylim(0, 1/2);
ax[1].text(0, 0.52, "NCRF", ha="center", fontsize=16)

cbar = plt.colorbar(nsc_ps, ax=ax, orientation="horizontal", aspect=40, shrink=0.7)
cbar.set_label("Normalized Power", fontsize=14);

plt.savefig("/home/b11209013/Bachelor_Thesis/Major/Figure/Appendix01.png", dpi=300);
plt.show();

plt.contourf(
    wn_v, fr_v[fr_v>0],
    np.log(np.fft.fftshift(sym_ps_weight["cntl"])[fr_v>0]),
    cmap="Blues",
    #levels=np.linspace(1, 10, 19),
    extend="max",
    )
plt.xlim(-15, 15);
plt.ylim(0, 1/2);
plt.colorbar();
plt.show();

plt.contourf(
    wn_v, fr_v[fr_v>0],
    np.log(np.fft.fftshift(sym_ps_weight["ncrf"])[fr_v>0]),
    cmap="Blues",
#    levels=np.linspace(1, 10, 19),
    extend="max",
    )
plt.xlim(-15, 15);
plt.ylim(0, 1/2);
plt.colorbar();
plt.show();

