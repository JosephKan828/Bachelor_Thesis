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

with nc.Dataset(f"{path}CNTL/w.nc", "r") as f:
    for key in f.dimensions.keys():
        dims[key] = f.variables[key][...];

    lat_lim: tuple[int] = np.where((dims["lat"] >= -5) & (dims["lat"] <= 5))[0];
    dims["lat"] = dims["lat"][lat_lim];

    data["cntl_w"] = f.variables["w"][:, :, lat_lim, :];

with nc.Dataset(f"{path}CNTL/qv.nc", "r") as f:
    data["cntl_qv"] = f.variables["qv"][:, :, lat_lim, :] * 1000.;

with nc.Dataset(f"{path}NCRF/w.nc", "r") as f:
    data["ncrf_w"] = f.variables["w"][:, :, lat_lim, :];

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
qv_mean = dict(
    cntl = np.mean(data["cntl_qv"], axis=(0, 2, 3)),
    ncrf = np.mean(data["ncrf_qv"], axis=(0, 2, 3)),
)

w_rm_cli: dict[str, np.ndarray] = dict(
    cntl = (data["cntl_w"] - np.mean(data["cntl_w"], axis=(0, 2, 3), keepdims=True)).mean(axis=2),
    ncrf = (data["ncrf_w"] - np.mean(data["ncrf_w"], axis=(0, 2, 3), keepdims=True)).mean(axis=2),
);

# # Select Events
data_sel: dict[str, np.ndarray] = dict(
    cntl_w = np.array([
        w_rm_cli["cntl"][i-12:i+12, :, j]
            for i, j in zip(sel_time["cntl"], sel_lon["cntl"])
    ]).mean(axis=0).T,
    ncrf_w = np.array([
        w_rm_cli["ncrf"][i-12:i+12, :, j]
            for i, j in zip(sel_time["ncrf"], sel_lon["ncrf"])
    ]).mean(axis=0).T,
);

# compute vertical gradient in CCKWs
qv_grad = dict(
    cntl = np.gradient(qv_mean["cntl"], dims["lev"]*100.),
    ncrf = np.gradient(qv_mean["ncrf"], dims["lev"]*100.)
)

vertical_conv = dict(
    cntl = data_sel["cntl_w"]*qv_grad["cntl"][:, None],
    ncrf = data_sel["ncrf_w"]*qv_grad["ncrf"][:, None]
) 

daily_vconv = dict(
    cntl = np.array([vertical_conv["cntl"][:, i*4:(i+1)*4].mean(axis=1) for i in range(6)]).T,
    ncrf = np.array([vertical_conv["ncrf"][:, i*4:(i+1)*4].mean(axis=1) for i in range(6)]).T
)
plt.rcParams["font.family"]="serif"


xticks_1 = np.linspace(0, 17, 9, dtype=int);
xticks_2 = np.linspace(0,  5, 6, dtype=int);
yticks   = np.linspace(1000, 400, 7, dtype=int);

fig, ax = plt.subplots(1, 2, figsize=(19, 10), sharey=True)

ax[0].plot(qv_mean["cntl"], dims["lev"], color="blue", label="CNTL")
ax[0].plot(qv_mean["ncrf"], dims["lev"], color="red", label="NCRF")
ax[0].set_yscale("log")
ax[0].set_xticks(xticks_1);
ax[0].set_yticks(yticks);

ax[0].set_xticklabels([f"{tick}" for tick in xticks_1], fontsize=16)
ax[0].set_yticklabels([f"{tick}" for tick in yticks], fontsize=16)

ax[0].set_xlim(0, 17)
ax[0].set_ylim(1000, 400)
ax[0].set_xlabel(r"$q_v$ [g/kg]", fontsize=18)
ax[0].set_ylabel(r"Level [hPa]", fontsize=18)
ax[0].set_title("(a) Mean Moisture Profile of CNTL(b) and NCRF(r)", fontsize=18)
ax[0].legend(fontsize=18)

ax[1].plot(qv_grad["cntl"]*1e4, dims["lev"], color="blue", label="CNTL")
ax[1].plot(qv_grad["ncrf"]*1e4, dims["lev"], color="red", label="NCRF")

ax[1].set_xticks(xticks_2);
ax[1].set_xticklabels([f"{tick}" for tick in xticks_2], fontsize=16)
ax[1].set_xlim(0, 5)
ax[1].set_xlabel(r"$\frac{\partial q_v}{\partial p} [\times 10^{4} g/kg/Pa]$", fontsize=18)
ax[1].set_title("(b) Vertical Gradient of Moisture Profile of CNTL(b) and NCRF(r)", fontsize=18)
ax[1].legend(fontsize=18)
plt.savefig("/home/b11209013/Bachelor_Thesis/Figure/NCRF_moist.png", dpi=600)
plt.show()


def vert_int(data):
    data_ave = (data[1:] + data[:-1]) / 2.;
    data_vint = -np.sum(data_ave * np.diff(dims["lev"]*100.)[:, None], axis=0) / -np.sum(np.diff(dims["lev"]*100.));
    return data_vint;

diff = vert_int(daily_vconv["ncrf"]) - vert_int(daily_vconv["cntl"])
pos = np.where(diff >= 0)
neg = np.where(diff <0)

plt.figure(figsize=(12, 8))

xtick = np.linspace(-3, 3, 7, dtype=int);
ytick = np.linspace(-8e-7, 8e-7, 9);

plt.bar(np.linspace(-2.5, 2.5, 6)[pos], diff[pos], color="red", width=0.2)
plt.bar(np.linspace(-2.5, 2.5, 6)[neg], diff[neg], color="blue", width=0.2)
plt.axhline(0, linestyle="--", color="black")
plt.xticks(xtick, fontsize=16);
plt.yticks(ytick, fontsize=16);
plt.xlim(-3, 3)
plt.ylim(-8e-7, 8e-7)
plt.xlabel("Lag days", fontsize=18)
plt.ylabel("Vertical Moisture Advection (K/day)", fontsize=18)
plt.title(r"Difference Vertical Moisture Advection (NCRF - CNTL) ($w^\prime \frac{\partial \overline{q_v}}{\partial p}$)", fontsize=18)
plt.savefig("/home/b11209013/Bachelor_Thesis/Figure/NCRF_vadv.png", dpi=600)
plt.show()

print((vert_int(daily_vconv["ncrf"]) - vert_int(daily_vconv["cntl"])).sum())
