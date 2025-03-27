# This program is to compare the LW heating difference between CNTL and NCRF
# %% Section 1
# Import package
import numpy as np;
import netCDF4 as nc;

import pickle as pkl;
from matplotlib import pyplot as plt;
from matplotlib.colors import TwoSlopeNorm;

# %% Section 2
# Load data
# # Load CNTL and NCRF data
path: str = "/work/b11209013/2024_Research/MPAS/merged_data/"

dims: dict[str, np.ndarray] = dict();
data: dict[str, np.ndarray] = dict();

with nc.Dataset(f"{path}CNTL/rthratenlw.nc", "r") as f:
    for key in f.dimensions.keys():
        dims[key] = f.variables[key][...];

    lat_lim: tuple[int] = np.where((dims["lat"] >= -5) & (dims["lat"] <= 5))[0];
    dims["lat"] = dims["lat"][lat_lim];

    data["cntl_lw"] = f.variables["rthratenlw"][:, :, lat_lim, :] * (1000/dims["lev"])[None, :, None, None]**(-0.286) * 86400.;

with nc.Dataset(f"{path}CNTL/theta.nc", "r") as f:
    data["t"] = f.variables["theta"][:, :, lat_lim, :] * (1000/dims["lev"])[None, :, None, None]**(-0.286);

with nc.Dataset(f"{path}NCRF/rthratenlw.nc", "r") as f:
    data["ncrf_lw"] = f.variables["rthratenlw"][:, :, lat_lim, :] * (1000/dims["lev"])[None, :, None, None]**(-0.286) * 86400.;

with nc.Dataset(f"{path}CNTL/qv.nc", "r") as f:
    data["cntl_qv"] = f.variables["qv"][:, :, lat_lim, :] * 1000.

with nc.Dataset(f"{path}NCRF/qv.nc", "r") as f:
    data["ncrf_qv"] = f.variables["qv"][:, :, lat_lim, :] * 1000.;

# # Load Selected Events
with open("/home/b11209013/Bachelor_Thesis/CCKWs_Selection/CNTL_comp.pkl", "rb") as f:
    cntl_comp = pkl.load(f);

with open("/home/b11209013/Bachelor_Thesis/CCKWs_Selection/NCRF_comp.pkl", "rb") as f:
    ncrf_comp = pkl.load(f);

sel_lon: dict[str, np.ndarray] = dict(
    cntl=np.array(cntl_comp["sel_lon"]),
    ncrf=np.array(ncrf_comp["sel_lon"]),
);

sel_time: dict[str, np.ndarray] = dict(
    cntl=np.array(cntl_comp["sel_time"]),
    ncrf=np.array(ncrf_comp["sel_time"]),
);

# %% Section 3
# Processing data
# # Remove climatology and zonal mean
data_rm_cli: dict[str, np.ndarray] = dict(
    (key, data[key] - np.mean(data[key], axis=(0, 3), keepdims=True))
    for key in data.keys()
);

data_rm_cli: dict[str, np.ndarray] = dict(
    (key, data_rm_cli[key].mean(axis=2))
    for key in data_rm_cli.keys()
);

# # Select Events
data_sel: dict[str, np.ndarray] = dict(
    cntl_lw = np.array([
        data_rm_cli["cntl_lw"][i-12:i+12, :, j]
            for i, j in zip(sel_time["cntl"], sel_lon["cntl"])
    ]).mean(axis=0).T,
    ncrf_lw = np.array([
        data_rm_cli["ncrf_lw"][i-12:i+12, :, j]
            for i, j in zip(sel_time["ncrf"], sel_lon["ncrf"])
    ]).mean(axis=0).T,
    t = np.array([
        data_rm_cli["t"][i-12:i+12, :, j]
            for i, j in zip(sel_time["cntl"], sel_lon["cntl"])
    ]).mean(axis=0).T,
    cntl_qv = np.array([
        data_rm_cli["cntl_qv"][i-12:i+12, :, j]
            for i, j in zip(sel_time["cntl"], sel_lon["cntl"])
    ]).mean(axis=0).T,
    ncrf_qv = np.array([
        data_rm_cli["ncrf_qv"][i-12:i+12, :, j]
            for i, j in zip(sel_time["ncrf"], sel_lon["ncrf"])
    ]).mean(axis=0).T,
);

# %% Section 4
# Plot
plt.rcParams["font.family"] = "serif";
plt.rcParams["mathtext.fontset"] = "cm";
plt.rcParams["xtick.labelsize"] = 12;
plt.rcParams["ytick.labelsize"] = 12;

fig, ax = plt.subplots(2, figsize=(16, 12), sharex=True);

lw_diff = ax[0].pcolormesh(
    np.linspace(-3, 2.75, 24), dims["lev"],
    data_sel["cntl_lw"] - data_sel["ncrf_lw"],
    cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0),
);

t_c = ax[0].contour(
    np.linspace(-3, 2.75, 24), dims["lev"],
    data_sel["t"],
    colors="k", norm=TwoSlopeNorm(vcenter=0)
);
plt.subplots_adjust(left=0.1, bottom=0.1, right=1.03, top=0.92);
ax[0].set_yscale("log");
ax[0].set_xticks(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7, dtype=int), fontsize=12);
ax[0].set_yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10, dtype=int), fontsize=12);
ax[0].text(3.5, 350, "Level [hPa]", va="center", ha="center", rotation=90, fontsize=14);
ax[0].text(-4, 350, "LW Heating Difference [K/day]", va="center", ha="center", rotation=90, fontsize=14);
ax[0].text(3, 95, "(a)\nShading: LW heating Difference (CNTL - NCRF)\nContour: CNTL Temperature [K]", va="bottom", ha="left", fontsize=14);
ax[0].set_xlim(3, -3);
ax[0].set_ylim(1000, 100);
plt.clabel(t_c, fmt="%1.2f", inline=True, fontsize=10);

plt.colorbar(lw_diff);

qv_diff = ax[1].pcolormesh(
    np.linspace(-3, 2.75, 24), dims["lev"],
    data_sel["cntl_qv"] - data_sel["ncrf_qv"],
    cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0),
);

qv_c = ax[1].contour(
    np.linspace(-3, 2.75, 24), dims["lev"],
    data_sel["cntl_qv"],
    colors="k",
);
ax[1].set_yscale("log");
ax[1].set_xticks(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7, dtype=int), fontsize=12);
ax[1].set_yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10, dtype=int), fontsize=12);
ax[1].text(3.4, 350, "Level [hPa]", va="center", ha="center", rotation=90, fontsize=14);
ax[1].text(-4, 350, r"$q_v$ Difference [g/kg]", va="center", ha="center", rotation=90, fontsize=14);
ax[1].text(3, 95, "(b)\n"+r"Shading: $q_v$"+"Difference (CNTL - NCRF)\n"+r"Contour: CNTL $q_v$ [g/kg]", va="bottom", ha="left", fontsize=14);

ax[1].set_xlim(3, -3);
ax[1].set_ylim(1000, 100);
plt.clabel(qv_c, fmt="%1.2f", inline=True, fontsize=10);
plt.colorbar(qv_diff);

plt.savefig("/home/b11209013/Bachelor_Thesis/Figure/NCRF_LW_qv_diff.png", dpi=300);
plt.show();
