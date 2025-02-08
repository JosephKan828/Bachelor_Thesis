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

    data["lw"] = f.variables["rthratenlw"][:, :, lat_lim, :] * (1000/dims["lev"])[None, :, None, None]**(-0.286) * 86400.;

with nc.Dataset(f"{path}CNTL/rthratensw.nc", "r") as f:
    data["sw"] = f.variables["rthratensw"][:, :, lat_lim, :] * (1000/dims["lev"])[None, :, None, None]**(-0.286) * 86400.;

with nc.Dataset(f"{path}CNTL/theta.nc", "r") as f:
    data["t"] = f.variables["theta"][:, :, lat_lim, :] * (1000/dims["lev"])[None, :, None, None]**(-0.286);

# # Load Selected Events
with open("/home/b11209013/Bachelor_Thesis/Major/CCKWs_Selection/CNTL_comp.pkl", "rb") as f:
    cntl_comp = pkl.load(f);

sel_lon : list[int] = cntl_comp["sel_lon"];
sel_time: list[int] = cntl_comp["sel_time"];

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
    (exp, np.array([
        data_rm_cli[exp][i-12:i+12, :, j]
        for i, j in zip(sel_time, sel_lon) 
    ]).mean(axis=0))
    for exp in data_rm_cli.keys()
    );

# %% Section 4
# Shape into daily
data_daily: dict[str, np.ndarray] = dict(
    (exp, np.array([
        data_sel[exp][i*4:(i+1)*4].mean(axis=0)
        for i in range(6)
    ]).T)
    for exp in data_sel.keys()
);

lw_eape: np.ndarray = 2*(data_daily["lw"] * data_daily["t"])/(data_daily["t"] * data_daily["t"]);
sw_eape: np.ndarray = 2*(data_daily["sw"] * data_daily["t"])/(data_daily["t"] * data_daily["t"]);

def vint(
    data: np.ndarray,
    lev : np.ndarray,
) -> np.ndarray:
    data_ave: np.ndarray = (data[1:] + data[:-1]) /2.;
    data_vint: np.ndarray = -np.sum(data_ave * np.diff(lev*100.)[:, None], axis=0) / 9.81 / -np.sum(np.diff(lev*100.));
    
    return data_vint;

lw_eape_vint: np.ndarray = vint(lw_eape, dims["lev"]);
sw_eape_vint: np.ndarray = vint(sw_eape, dims["lev"]);

plt.plot(np.linspace(-2.5, 2.5, 6), lw_eape_vint, label="LW")
plt.plot(np.linspace(-2.5, 2.5, 6), sw_eape_vint, label="SW")
plt.xticks(np.linspace(-3, 3, 7))
plt.xlim(-3, 3);
plt.gca().invert_xaxis();
plt.text(-3, -0.3, f"LW EAPE:{lw_eape_vint.mean():.2f}", va="center", ha="right", fontsize=12);
plt.text(-3, -0.4, f"SW EAPE:{sw_eape_vint.mean():.2f}", va="center", ha="right", fontsize=12);
plt.legend();
plt.show();