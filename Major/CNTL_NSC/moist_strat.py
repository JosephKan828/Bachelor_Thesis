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

with nc.Dataset(f"{path}NSC/w.nc", "r") as f:
    data["nsc_w"] = f.variables["w"][:, :, lat_lim, :];

with nc.Dataset(f"{path}NSC/qv.nc", "r") as f:
    data["nsc_qv"] = f.variables["qv"][:, :, lat_lim, :] * 1000.;

# # Load Selected Events
with open("/home/b11209013/Bachelor_Thesis/Major/CCKWs_Selection/CNTL_comp.pkl", "rb") as f:
    cntl_comp = pkl.load(f);

with open("/home/b11209013/Bachelor_Thesis/Major/CCKWs_Selection/NSC_comp.pkl", "rb") as f:
    nsc_comp = pkl.load(f);

sel_lon: dict[str, np.ndarray] = dict(
    cntl=np.array(cntl_comp["sel_lon"]),
    nsc =np.array(nsc_comp["sel_lon"]),
);

sel_time: dict[str, np.ndarray] = dict(
    cntl=np.array(cntl_comp["sel_time"]),
    nsc =np.array(nsc_comp["sel_time"]),
);

# %% Section 3
# Processing data
# # Remove climatology and zonal mean
qv_mean = dict(
    cntl = np.mean(data["cntl_qv"], axis=(0, 2, 3)),
    nsc  = np.mean(data["nsc_qv"], axis=(0, 2, 3)),
)

w_rm_cli: dict[str, np.ndarray] = dict(
    cntl = (data["cntl_w"] - np.mean(data["cntl_w"], axis=(0, 2, 3), keepdims=True)).mean(axis=2),
    nsc  = (data["nsc_w"]  - np.mean(data["nsc_w"], axis=(0, 2, 3), keepdims=True)).mean(axis=2),
);

# # Select Events
data_sel: dict[str, np.ndarray] = dict(
    cntl_w = np.array([
        w_rm_cli["cntl"][i-12:i+12, :, j]
            for i, j in zip(sel_time["cntl"], sel_lon["cntl"])
    ]).mean(axis=0).T,
    nsc_w = np.array([
        w_rm_cli["nsc"][i-12:i+12, :, j]
            for i, j in zip(sel_time["nsc"], sel_lon["nsc"])
    ]).mean(axis=0).T,
);

# compute vertical gradient in CCKWs
qv_grad = dict(
    cntl = np.gradient(qv_mean["cntl"], dims["lev"]*100.),
    nsc  = np.gradient(qv_mean["nsc"] , dims["lev"]*100.)
)

vertical_conv = dict(
    cntl = data_sel["cntl_w"]*qv_grad["cntl"][:, None],
    nsc  = data_sel["nsc_w"]*qv_grad["nsc"][:, None]
) 

daily_vconv = dict(
    cntl = np.array([vertical_conv["cntl"][:, i*4:(i+1)*4].mean(axis=1) for i in range(6)]).T,
    nsc = np.array([vertical_conv["nsc"][:, i*4:(i+1)*4].mean(axis=1) for i in range(6)]).T
)
plt.rcParams["font.family"]="serif"

plt.plot(qv_mean["cntl"], dims["lev"], color="blue", label="CNTL")
plt.plot(qv_mean["nsc"], dims["lev"], color="red", label="NSC")
plt.xlim(10, 17)
plt.yscale("log")
plt.yticks(np.linspace(800, 1000, 11), np.linspace(800, 1000, 11, dtype=int))
plt.ylim(1000, 800)
plt.xlabel(r"$q_v$ [g/kg]", fontsize=14)
plt.ylabel(r"Level [hPa]", fontsize=14)
plt.legend()
plt.savefig("/home/b11209013/Bachelor_Thesis/Major/Figure/Appendix04.png", dpi=300)
plt.show()


def vert_int(data):
    data_ave = (data[1:] + data[:-1]) / 2.;
    data_vint = -np.sum(data_ave * np.diff(dims["lev"]*100.)[:, None], axis=0) / -np.sum(np.diff(dims["lev"]*100.));
    return data_vint;

diff = vert_int(daily_vconv["nsc"]) - vert_int(daily_vconv["cntl"])
pos = np.where(diff >= 0)
neg = np.where(diff <0)

plt.bar(np.linspace(-2.5, 2.5, 6)[pos], diff[pos], color="red", width=0.2)
plt.bar(np.linspace(-2.5, 2.5, 6)[neg], diff[neg], color="blue", width=0.2)
plt.axhline(0, linestyle="--", color="black")
plt.xlim(-3, 3)
plt.ylim(-6e-7, 6e-7)
plt.xlabel("Lag days")
plt.ylabel("Vertical Moisture Convergence")
plt.savefig("/home/b11209013/Bachelor_Thesis/Major/Figure/Appendix05.png", dpi=300)
plt.show()

print((vert_int(daily_vconv["nsc"]) - vert_int(daily_vconv["cntl"])).sum())