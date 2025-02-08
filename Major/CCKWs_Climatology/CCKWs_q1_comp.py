# This program is to composite the Q1, temperature, and Qv
# %% Section 1
# Import packages
import numpy as np;
import netCDF4 as nc;
import pickle as pkl;
import matplotlib

from matplotlib import pyplot as plt;
from matplotlib.colors import TwoSlopeNorm;

matplotlib.use('Agg')  # or 'Qt5Agg'
# %% Section 2
# Load file
path: str = "/work/b11209013/2024_Research/MPAS/merged_data/CNTL/";

# # Load MPAS data
dims: dict[str, np.ndarray] = dict();
data: dict[str, np.ndarray] = dict();

with nc.Dataset(f"{path}q1.nc", "r") as f:
    for key in f.dimensions.keys():
        dims[key] = f.variables[key][:];

    lat_lim: tuple[int] = np.where((dims["lat"] >= -5) & (dims["lat"] <= 5))[0];
    dims["lat"] = dims["lat"][lat_lim];

    data["q1"] = f.variables["q1"][:, :, lat_lim, :] * 86400 / 1004.5;

with nc.Dataset(f"{path}theta.nc", "r") as f:
    data["t"] = f.variables["theta"][:, :, lat_lim, :] * (1000/dims["lev"])[None, :, None, None] ** (-0.286);

with nc.Dataset(f"{path}qv.nc", "r") as f:
    data["qv"] = f.variables["qv"][:, :, lat_lim, :] * 1000;

# # Load Event
with open("/home/b11209013/Bachelor_Thesis/Major/CCKWs_Selection/CNTL_comp.pkl", "rb") as f:
    sel_kel = pkl.load(f);

sel_lon : np.ndarray = np.array(sel_kel["sel_lon"]);
sel_time: np.ndarray = np.array(sel_kel["sel_time"]);
print("sel_lon:", sel_lon);
print("sel_time:", sel_time);
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

# %% Section 4
# Plot
plt.rcParams["font.family"] = "serif";
plt.rcParams["mathtext.fontset"] = "cm";

plt.figure(figsize=(16, 7));
q1_cf = plt.pcolormesh(
    np.linspace(-3, 2.75, 24), dims["lev"],
    data_sel["q1"],
    cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0),
);
t_cf = plt.contour(
    np.linspace(-3, 2.75, 24), dims["lev"],
    data_sel["t"],
    colors="k", levels=np.arange(-2.5, 2.6, 0.5),
);
qv_cf = plt.contour(
    np.linspace(-3, 2.75, 24), dims["lev"],
    data_sel["qv"],
    colors="g", levels=np.arange(-0.8, 0.86, 0.25)
);
plt.subplots_adjust(left=0.1, bottom=0.15, right=1.03, top=0.9);
plt.yscale("log");
plt.xticks(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7, dtype=int), fontsize=12);
plt.yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10, dtype=int), fontsize=12);
plt.text(3.5, 350, "Level [hPa]", va="center", ha="center", rotation=90, fontsize=14);
plt.text(-3.8, 350, r"$Q_1$ [K/day]", va="center", ha="center", rotation=90, fontsize=14);
plt.text(1.5, 1300, "Day After", va="center", ha="center", fontsize=14);
plt.text(-1.5, 1300, "Day Before", va="center", ha="center", fontsize=14);
plt.text(3, 95, r"Shading: $Q_1$ [K/day]"+"\nContour: Temperature [K]; $q_v$ [g/kg]", va="bottom", ha="left", fontsize=14);
plt.xlim(-3, 3);
plt.ylim(100, 1000);
plt.gca().invert_xaxis();
plt.gca().invert_yaxis();
plt.colorbar(q1_cf);
plt.clabel(t_cf, fmt="%1.2f", inline=True, fontsize=10);
plt.clabel(qv_cf, fmt="%1.2f", inline=True, fontsize=10);
plt.savefig("/home/b11209013/Bachelor_Thesis/Major/Figure/Figure01.png", dpi=500);
plt.show();
