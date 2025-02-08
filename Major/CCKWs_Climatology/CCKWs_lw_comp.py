# This program is to composite the Q1, temperature, and Qv
# %% Section 1
# Import packages
import numpy as np;
import netCDF4 as nc;
import pickle as pkl;
from matplotlib import pyplot as plt;
from matplotlib.colors import TwoSlopeNorm;

# %% Section 2
# Load file
path: str = "/work/b11209013/2024_Research/MPAS/merged_data/CNTL/";

# # Load MPAS data
dims: dict[str, np.ndarray] = dict();
data: dict[str, np.ndarray] = dict();

with nc.Dataset(f"{path}rthratenlw.nc", "r") as f:
    for key in f.dimensions.keys():
        dims[key] = f.variables[key][:];

    lat_lim: tuple[int] = np.where((dims["lat"] >= -5) & (dims["lat"] <= 5))[0];
    dims["lat"] = dims["lat"][lat_lim];

    data["lw"] = f.variables["rthratenlw"][:, :, lat_lim, :] * (1000/dims["lev"])[None, :, None, None] ** (-0.286) * 86400;

with nc.Dataset(f"{path}rthratensw.nc", "r") as f:
    data["sw"] = f.variables["rthratensw"][:, :, lat_lim, :] * (1000/dims["lev"])[None, :, None, None] ** (-0.286) * 86400;


with nc.Dataset(f"{path}theta.nc", "r") as f:
    data["t"] = f.variables["theta"][:, :, lat_lim, :] * (1000/dims["lev"])[None, :, None, None] ** (-0.286);

with nc.Dataset(f"{path}qv.nc", "r") as f:
    data["qv"] = f.variables["qv"][:, :, lat_lim, :] * 1000;

# # Load Event
with open("/home/b11209013/Bachelor_Thesis/Major/CCKWs_Selection/CNTL_comp.pkl", "rb") as f:
    sel_kel = pkl.load(f);

sel_lon : np.ndarray = np.array(sel_kel["sel_lon"]);
sel_time: np.ndarray = np.array(sel_kel["sel_time"]);

# %% Section 3
# Processing data
# # Remove climatology and Zonal mean
data_rm_cli: dict[str, np.ndarray] = dict(
    (var, np.squeeze(data[var].mean(axis=2, keepdims=True) - data[var].mean(axis=(0, 2, 3), keepdims=True)))
    for var in data.keys()
);

# # Select Events
data_sel: dict[str, np.ndarray] = dict();

for var in data_rm_cli.keys():
    data_sel[var] = np.zeros((38, 24));

    for i in range(sel_lon.size):
        data_sel[var] += data_rm_cli[var][sel_time[i]-12:sel_time[i]+12, :, sel_lon[i]].T;

    data_sel[var] /= sel_lon.size;

# # Daily averaging LW and SW 

lw_daily = np.array([
    data_sel["lw"][:, i*4:(i+1)*4].mean(axis=1)
    for i in range(6)
]).T;
sw_daily = np.array([
    data_sel["sw"][:, i*4:(i+1)*4].mean(axis=1)
    for i in range(6)
]).T;

# %% Section 4
# Plot
plt.rcParams["font.family"] = "serif";
plt.rcParams["mathtext.fontset"] = "cm";
plt.rcParams["xtick.labelsize"] = 12;
plt.rcParams["ytick.labelsize"] = 12;

fig, ax = plt.subplots(2, 1, figsize=(16, 12), sharex=True);
lw_cf = ax[0].pcolormesh(
    np.linspace(-2.5, 2.5, 6), dims["lev"],
    lw_daily,
    cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0),
);

t_cf = ax[0].contour(
    np.linspace(-3, 2.75, 24), dims["lev"],
    data_sel["t"],
    colors="k", levels=np.arange(-2.5, 2.6, 0.5),
);
plt.subplots_adjust(left=0.1, bottom=0.1, right=1.03, top=0.92);
ax[0].set_yscale("log");
ax[0].set_xticks(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7, dtype=int), fontsize=12);
ax[0].set_yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10, dtype=int), fontsize=12);
ax[0].text(3.4, 350, "Level [hPa]", va="center", ha="center", rotation=90, fontsize=16);
ax[0].text(-4, 350, "LW Heating [K/day]", va="center", ha="center", rotation=90, fontsize=16);
ax[0].text(3, 95, "(a)\nShading: CNTL LW Heating [K/day]\nContour: CNTL Temperature [K]", va="bottom", ha="left", fontsize=16);
ax[0].set_xlim(3, -3);
ax[0].set_ylim(1000, 100);
plt.colorbar(lw_cf);
plt.clabel(t_cf, fmt="%1.2f", inline=True, fontsize=10);

sw_cf = ax[1].pcolormesh(
    np.linspace(-2.5, 2.5, 6), dims["lev"],
    sw_daily,
    cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0),
);

t_cf = ax[1].contour(
    np.linspace(-3, 2.75, 24), dims["lev"],
    data_sel["t"],
    colors="k", levels=np.arange(-2.5, 2.6, 0.5),
);
ax[1].set_yscale("log");
ax[1].set_xticks(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7, dtype=int), fontsize=12);
ax[1].set_yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10, dtype=int), fontsize=12);
ax[1].text(3.4, 350, "Level [hPa]", va="center", ha="center", rotation=90, fontsize=16);
ax[1].text(-4, 350, "SW Heating [K/day]", va="center", ha="center", rotation=90, fontsize=16);
ax[1].text(1.5, 1300, "Day After", va="center", ha="center", fontsize=16);
ax[1].text(-1.5, 1300, "Day Before", va="center", ha="center", fontsize=16);
ax[1].text(3, 95, "(b)\nShading: CNTL SW Heating [K/day]\n"+r"Contour: CNTL $q_v$ [g/kg]", va="bottom", ha="left", fontsize=16);
ax[1].set_xlim(-3, 3);
ax[1].set_ylim(100, 1000);
plt.gca().invert_xaxis();
plt.gca().invert_yaxis();
plt.colorbar(sw_cf);
plt.clabel(t_cf, fmt="%1.2f", inline=True, fontsize=10);

plt.savefig("/home/b11209013/Bachelor_Thesis/Major/Figure/Figure02.png", dpi=300);
plt.show();


