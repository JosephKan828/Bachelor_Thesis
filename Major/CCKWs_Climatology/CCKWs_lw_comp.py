# This program is to composite the Q1, temperature, and Qv
# Section 1
# Import packages
import numpy as np;
from netCDF4 import Dataset;
import pickle as pkl;
from matplotlib import pyplot as plt;
from matplotlib.colors import TwoSlopeNorm;
import matplotlib.gridspec as gridspec;
from mpl_toolkits.axes_grid1.inset_locator import inset_axes;
from scipy.integrate import trapezoid;

def load_data()-> dict[str, np.ndarray]:
    # path
    path = "/work/b11209013/2024_Research/MPAS/merged_data/CNTL/";

    data = dict();
    
    with Dataset(f"{path}/rthratenlw.nc".format(path), "r") as f:
        dims        = dict((var, f.variables[var][:]) for var in f.dimensions.keys());
        
        lat_lim     = np.where((dims["lat"] >= -5) & (dims["lat"] <= 5))[0];
        dims["lat"] = dims["lat"][lat_lim];
        
        convert     = (1000. / dims["lev"][None, :, None, None]) ** (-0.286);
        lat_slices  = (slice(None), slice(None), lat_lim, slice(None))

        data["lw"]  = f.variables["rthratenlw"][lat_slices] * convert * 86400;

    with Dataset(f"{path}/rthratensw.nc".format(path), "r") as f:
        data["sw"]  = f.variables["rthratensw"][lat_slices] * convert * 86400;

    with Dataset(f"{path}/theta.nc".format(path), "r") as f:
        data["theta"] = f.variables["theta"][lat_slices];
        data["t"]     = data["theta"] * convert;
        
    var_list = list(data.keys());

    return var_list, dims, data;

def main():
    # Load MPAS data
    var_list, dims, data = load_data();

    # Load events
    with open("/home/b11209013/Bachelor_Thesis/Major/CCKWs_Selection/CNTL_comp.pkl", "rb") as f:
        sel_kel = pkl.load(f);

    sel_lon : np.ndarray = np.array(sel_kel["sel_lon"]);
    sel_time: np.ndarray = np.array(sel_kel["sel_time"]);

    # Compute specific volume
    data["alpha"] = 287.5 * data["t"] / (dims["lev"][None, :, None, None]*100.);

    var_list.append("alpha");

    # Compute mean and anomaly field
    data_mean = dict(
        (var, data[var].mean(axis=(0, 2, 3), keepdims=True))
        for var in var_list
    );
    
    data_ano = dict(
        (var, (data[var] - data_mean[var]).mean(axis=2))
        for var in var_list
    );
    
    print("Mean field shape: ", data_mean["lw"].shape);
    
    sigma = -data["alpha"].mean(axis=(2, 3)) / data["theta"].mean(axis=(2, 3)) * np.gradient(data["theta"].mean(axis=(2, 3)), dims["lev"]*100., axis=1);
    
    # Select data
    data_sel: dict[str, np.ndarray] = dict();

    for var in data_ano.keys():

        sel_data = np.array([
            data_ano[var][sel_time[i]-12:sel_time[i]+12, :, sel_lon[i]]
            for i in range(sel_lon.size)
        ]);

        data_sel[var] = sel_data.mean(axis=0).T;
        
    sigma_sel = np.array([
            sigma[sel_time[i]-12:sel_time[i]+12]
            for i in range(sel_lon.size)
        ]);

    print("Selected sigma shape:", sigma_sel.shape);

    # Compute parameters
    
    # Compute daily mean variable
    a_daily = np.array([
        data_sel["alpha"][:, i*4:(i+1)*4].mean(axis=1)
        for i in range(6)
    ]).T;
    
    lw_daily = np.array([
        data_sel["lw"][:, i*4:(i+1)*4].mean(axis=1)
        for i in range(6)
    ]).T;
    sw_daily = np.array([
        data_sel["sw"][:, i*4:(i+1)*4].mean(axis=1)
        for i in range(6)
    ]).T;

    # Compute growth rate
    lw_growth_rate = 287.5/(1004.5*sigma*dims["lev"]*100.)[:, None] * (a_daily * lw_daily);
    sw_growth_rate = 287.5/(1004.5*sigma*dims["lev"]*100.)[:, None] * (a_daily * sw_daily);

    # Vertically integral
    def vert_int(data):
        data_ave = (data[1:] + data[:-1]) / 2.;
        data_vint = -np.sum(data_ave * np.diff(dims["lev"]*100.)[:, None], axis=0) / -np.sum(np.diff(dims["lev"]*100.));
        return data_vint;

    lw_eape_vint = vert_int(lw_growth_rate)*1e3;
    sw_eape_vint = vert_int(sw_growth_rate)*1e3;

    print(lw_eape_vint.sum());
    print(sw_eape_vint.sum());

    # Plot
    fig = plt.figure(figsize=(10, 6));

    # Plot
    # Set global font settings
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    # Create figure and GridSpec layout with increased hspace
    fig = plt.figure(figsize=(16, 15))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 0.2, 1, 0.2], hspace=0.35)

    # Upper panel (pcolormesh & contour)
    ax0 = fig.add_subplot(gs[0])
    lw_cf = ax0.pcolormesh(
        np.linspace(-2.5, 2.5, 6), dims["lev"],
        lw_daily, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0),
    )
    t_cf = ax0.contour(
        np.linspace(-3, 2.75, 24), dims["lev"],
        data_sel["t"], colors="k", levels=np.arange(-2.5, 2.6, 0.5),
    )
    ax0.set_yscale("log")
    ax0.set_yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10, dtype=int))
    ax0.set_xlim(3, -3)
    ax0.set_ylim(1000, 100)

    ax0.set_title("(a) CNTL LW Heating (Shading) & Temperature (Contour) & EAPE growth rate (Bar)", loc="left", fontsize=16)

    # Add colorbar for ax0
    cax0 = inset_axes(ax0, width="1.5%", height="80%", loc="center right",
                    bbox_to_anchor=(0.04, 0, 1, 1),
                    bbox_transform=ax0.transAxes, borderpad=0)
    fig.colorbar(lw_cf, cax=cax0)
    plt.clabel(t_cf, fmt="%1.2f", inline=True, fontsize=10)

    # Lower panel (LW bar plot)
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    lw_positive_cond = np.where(lw_eape_vint > 0)[0]
    lw_negative_cond = np.where(lw_eape_vint < 0)[0]

    ax1.bar(np.linspace(-2.5, 2.5, 6)[lw_positive_cond], lw_eape_vint[lw_positive_cond], width=0.2, color="r")
    ax1.bar(np.linspace(-2.5, 2.5, 6)[lw_negative_cond], lw_eape_vint[lw_negative_cond], width=0.2, color="b")
    ax1.axhline(0, color="k")
    ax1.set_yticks(np.linspace(-2.5, 2.5, 3))
    ax1.set_xlim(3, -3)
    ax1.set_ylim(-2.8, 2.8)

    # Align ax1 with ax0
    pos0 = ax0.get_position()
    pos1 = ax1.get_position()
    ax1.set_position([pos1.x0, pos1.y0+0.01, pos0.width, pos1.height])

    # Middle panel (SW pcolormesh & contour)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    sw_cf = ax2.pcolormesh(
        np.linspace(-2.5, 2.5, 6), dims["lev"],
        sw_daily, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0),
    )
    t_cf = ax2.contour(
        np.linspace(-3, 2.75, 24), dims["lev"],
        data_sel["t"], colors="k", levels=np.arange(-2.5, 2.6, 0.5),
    )
    ax2.set_yscale("log")
    ax2.set_yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10, dtype=int))

    ax2.set_ylim(1000, 100)
    ax2.set_title("(b) CNTL SW Heating (Shading) & Temperature (Contour) & EAPE growth rate (Bar)", loc="left", fontsize=16)

    # Add colorbar for ax2
    cax2 = inset_axes(ax2, width="1.5%", height="80%", loc="center right",
                    bbox_to_anchor=(0.04, 0, 1, 1),
                    bbox_transform=ax2.transAxes, borderpad=0)
    fig.colorbar(sw_cf, cax=cax2)
    plt.clabel(t_cf, fmt="%1.2f", inline=True, fontsize=10)

    # Move ax2 downward (adjusting GridSpec hspace)
    pos2 = ax2.get_position()
    ax2.set_position([pos2.x0, pos2.y0, pos2.width, pos2.height])  # Move down slightly

    # Lower panel (SW bar plot)
    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    sw_positive_cond = np.where(sw_eape_vint > 0)[0]
    sw_negative_cond = np.where(sw_eape_vint < 0)[0]

    ax3.bar(np.linspace(-2.5, 2.5, 6)[sw_positive_cond], sw_eape_vint[sw_positive_cond], color="r", width=0.2)
    ax3.bar(np.linspace(-2.5, 2.5, 6)[sw_negative_cond], sw_eape_vint[sw_negative_cond], color="b", width=0.2)

    ax3.axhline(0, color="k")
    ax3.set_yticks(np.linspace(-1, 1, 3))
    ax3.set_xlim(3, -3)

    # Align ax3 with ax2
    pos2 = ax2.get_position()
    pos3 = ax3.get_position()
    ax3.set_position([pos3.x0, pos3.y0+0.01, pos2.width, pos3.height])
    plt.savefig("/home/b11209013/Bachelor_Thesis/Major/Figure/Figure02.png", dpi=300);
    plt.show()

if __name__ == "__main__":
    main();
"""
# Section 2
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
    data["theta"] = f.variables["theta"][:, :, lat_lim, :] * (1000/dims["lev"])[None, :, None, None] ** (-0.286);
    data["t"] = data["theta"] * (1000/dims["lev"])[None, :, None, None] ** (-0.286);

# # Load Event
with open("/home/b11209013/Bachelor_Thesis/Major/CCKWs_Selection/CNTL_comp.pkl", "rb") as f:
    sel_kel = pkl.load(f);

sel_lon : np.ndarray = np.array(sel_kel["sel_lon"]);
sel_time: np.ndarray = np.array(sel_kel["sel_time"]);

#  Section 3
# Processing data
# # Remove climatology and Zonal mean
mean_field: dict[str, np.ndarray] = dict(
    (var, data[var].mean(axis=(0, 2, 3), keepdims=True))
    for var in data.keys()
)
print

data_rm_cli: dict[str, np.ndarray] = dict(
    (var, np.squeeze(data[var].mean(axis=2, keepdims=True) - data[var].mean(axis=(0, 2, 3), keepdims=True)))
    for var in data.keys()
);

# # Select Events
data_sel: dict[str, np.ndarray] = dict();

for var in data_rm_cli.keys():
    data_sel[var] = np.zeros((38, 24));

    for i in range(sel_lon.size):
        data_sel[var] += data[var][sel_time[i]-12:sel_time[i]+12, :, sel_lon[i]].T;

    data_sel[var] /= sel_lon.size;

# # Daily averaging LW and SW 
t_daily = np.array([
    data_sel["t"][:, i*4:(i+1)*4].mean(axis=1)
    for i in range(6)
]).T;
print(t_daily.shape)

lw_daily = np.array([
    data_sel["lw"][:, i*4:(i+1)*4].mean(axis=1)
    for i in range(6)
]).T;
sw_daily = np.array([
    data_sel["sw"][:, i*4:(i+1)*4].mean(axis=1)
    for i in range(6)
]).T;

#  Section 4
# Compute EAPE generation
# # Compute vertical integral
def vint(
    data: np.ndarray,
    lev : np.ndarray,
) -> np.ndarray:
    data_ave: np.ndarray = (data[1:] + data[:-1]) /2.;
    data_vint: np.ndarray = -np.sum(data_ave * np.diff(lev*100.)[:, None], axis=0);
    
    return data_vint;

lw_t_vint: np.ndarray = trapezoid(lw_daily*t_daily, x=dims["lev"]*100., axis=0);
sw_t_vint: np.ndarray = trapezoid(sw_daily*t_daily, x=dims["lev"]*100., axis=0);
t_t_vint : np.ndarray = trapezoid(t_daily *t_daily, x=dims["lev"]*100., axis=0);

lw_eape_vint: np.ndarray = 2*lw_t_vint / t_t_vint;
sw_eape_vint: np.ndarray = 2*sw_t_vint / t_t_vint;

print(lw_eape_vint.sum())
print(sw_eape_vint.sum())

# Section 4
# Plot
# Set global font settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

# Create figure and GridSpec layout with increased hspace
fig = plt.figure(figsize=(16, 15))
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 0.2, 1, 0.2], hspace=0.35)

# Upper panel (pcolormesh & contour)
ax0 = fig.add_subplot(gs[0])
lw_cf = ax0.pcolormesh(
    np.linspace(-2.5, 2.5, 6), dims["lev"],
    lw_daily, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0),
)
t_cf = ax0.contour(
    np.linspace(-3, 2.75, 24), dims["lev"],
    data_sel["t"], colors="k", levels=np.arange(-2.5, 2.6, 0.5),
)
ax0.set_yscale("log")
ax0.set_yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10, dtype=int))
ax0.set_xlim(3, -3)
ax0.set_ylim(1000, 100)

ax0.set_title("(a) CNTL LW Heating (Shading) & Temperature (Contour) & EAPE growth rate (Bar)", loc="left", fontsize=16)

# Add colorbar for ax0
cax0 = inset_axes(ax0, width="1.5%", height="80%", loc="center right",
                bbox_to_anchor=(0.04, 0, 1, 1),
                bbox_transform=ax0.transAxes, borderpad=0)
fig.colorbar(lw_cf, cax=cax0)
plt.clabel(t_cf, fmt="%1.2f", inline=True, fontsize=10)

# Lower panel (LW bar plot)
ax1 = fig.add_subplot(gs[1], sharex=ax0)
lw_positive_cond = np.where(lw_eape_vint > 0)[0]
lw_negative_cond = np.where(lw_eape_vint < 0)[0]

ax1.bar(np.linspace(-2.5, 2.5, 6), lw_eape_vint, width=0.1)
ax1.axhline(0, color="k")
ax1.set_yticks(np.linspace(-1.5, 1.5, 3))
ax1.set_xlim(3, -3)

# Align ax1 with ax0
pos0 = ax0.get_position()
pos1 = ax1.get_position()
ax1.set_position([pos1.x0, pos1.y0+0.01, pos0.width, pos1.height])

# Middle panel (SW pcolormesh & contour)
ax2 = fig.add_subplot(gs[2], sharex=ax0)
sw_cf = ax2.pcolormesh(
    np.linspace(-2.5, 2.5, 6), dims["lev"],
    sw_daily, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0),
)
t_cf = ax2.contour(
    np.linspace(-3, 2.75, 24), dims["lev"],
    data_sel["t"], colors="k", levels=np.arange(-2.5, 2.6, 0.5),
)
ax2.set_yscale("log")
ax2.set_yticks(np.linspace(100, 1000, 10), np.linspace(100, 1000, 10, dtype=int))

ax2.set_ylim(1000, 100)
ax2.set_title("(b) CNTL SW Heating (Shading) & Temperature (Contour) & EAPE growth rate (Bar)", loc="left", fontsize=16)

# Add colorbar for ax2
cax2 = inset_axes(ax2, width="1.5%", height="80%", loc="center right",
                bbox_to_anchor=(0.04, 0, 1, 1),
                bbox_transform=ax2.transAxes, borderpad=0)
fig.colorbar(sw_cf, cax=cax2)
plt.clabel(t_cf, fmt="%1.2f", inline=True, fontsize=10)

# Move ax2 downward (adjusting GridSpec hspace)
pos2 = ax2.get_position()
ax2.set_position([pos2.x0, pos2.y0, pos2.width, pos2.height])  # Move down slightly

# Lower panel (SW bar plot)
ax3 = fig.add_subplot(gs[3], sharex=ax0)
sw_positive_cond = np.where(sw_eape_vint > 0)[0]
sw_negative_cond = np.where(sw_eape_vint < 0)[0]
print(sw_positive_cond)
print(sw_negative_cond)

#ax3.bar(np.linspace(-3, 2.75, 24)[sw_positive_cond], sw_eape_vint[sw_positive_cond], color="r", label="LW")
#ax3.bar(np.linspace(-3, 2.75, 24)[sw_negative_cond], sw_eape_vint[sw_negative_cond], color="b", label="SW")
ax3.bar(np.linspace(-2.5, 2.5, 6), sw_eape_vint, width=0.1)
ax3.axhline(0, color="k")
ax3.set_yticks(np.linspace(-0.25, 0.25, 3))
ax3.set_xlim(3, -3)

# Align ax3 with ax2
pos2 = ax2.get_position()
pos3 = ax3.get_position()
ax3.set_position([pos3.x0, pos3.y0+0.01, pos2.width, pos3.height])
plt.savefig("/home/b11209013/Bachelor_Thesis/Major/Figure/Figure02.png", dpi=300);
plt.show()
"""