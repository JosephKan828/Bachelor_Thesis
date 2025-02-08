#!/usr/bin/env python
# coding: utf-8

# # CNTL CCKWs event selection

# ## Import packages

# In[57]:


import sys
import numpy as np
import pandas as pd
import netCDF4 as nc
import pickle as pkl

from matplotlib import pyplot as plt

sys.path.append('/home/b11209013/Package')
import DataProcess as dp   #type: ignore
import SignalProcess as sp #type: ignore


# ## Functions

# In[58]:


def is_within_region(wnm, frm, wnm_min, wnm_max, frm_min, frm_max, kel_sign=1):
    kel_curves = lambda ed, k: (86400/(2*np.pi*6.371e6))*np.sqrt(9.81*ed)*k
    
    return (
        (wnm > wnm_min) & (wnm < wnm_max) &
        (frm > kel_sign * frm_min) & (frm < kel_sign * frm_max) &
        (frm < kel_sign * kel_curves(90, wnm)) &
        (frm > kel_sign * kel_curves(8, wnm))
    )


# ## Load data

# In[59]:


with nc.Dataset('/work/b11209013/2024_Research/MPAS/merged_data/CNTL/q1.nc', 'r') as f:
    lon : np.ndarray = f.variables['lon'][:]
    lat : np.ndarray = f.variables['lat'][:]
    lev : np.ndarray = f.variables['lev'][:]
    time: np.ndarray = f.variables['time'][:]
    lat_lim = np.where((lat >= -5) & (lat <= 5))[0]
    
    lat : np.ndarray = lat[lat_lim]
    q1       = f.variables['q1'][:, :, lat_lim, :]

ltime, llev, llat, llon = q1.shape


# ## Vertical integrated Q1

# In[60]:


q1_ave = (q1[:, 1:] + q1[:, :-1]) / 2
q1_int = -np.sum(q1_ave*np.diff(lev)[None, :, None, None]*100, axis=1) *86400/9.8/2.5e6


# ## Process Data

# ### Symmetrize data

# In[61]:


fmt = dp.Format(lat)

q1_sym = fmt.sym(q1_int)

print(q1_sym.shape)


# ### bandpass filter

# In[62]:


wn = np.fft.fftfreq(llon, d = 1/llon).astype(int) # unit: None
fr = np.fft.fftfreq(ltime, d = 1/4) # unit: CPD

wnm, frm = np.meshgrid(wn, fr)

cond1 = is_within_region(
    wnm, frm,
    wnm_min=1, wnm_max=14,
    frm_min=1/20, frm_max=1/2.5
)

cond2 = is_within_region(
    wnm, frm,
    wnm_min=-14, wnm_max=-1,
    frm_min=1/20, frm_max=1/2.5,
    kel_sign=-1
)


# ### FFT on q1_sym

# In[63]:


q1_fft = np.array([np.fft.fft(q1_sym[i]) for i in range(ltime)])
q1_fft = np.array([np.fft.ifft(q1_fft[:, i]) for i in range(llon)]).T


# ### Filt with the filter

# In[64]:


mask = np.where(cond1 | cond2, 1, 0)
q1_filted = mask * q1_fft


# ### reconstruct

# In[65]:


q1_recon = np.array([np.fft.fft(q1_filted[:, i]) for i in range(llon)]).T
q1_recon = np.array([np.fft.ifft(q1_recon[i]) for i in range(ltime)])

q1_recon = np.real(q1_recon)


# ## Select events

# In[66]:


g99 = q1_recon.mean() + 1.96*q1_recon.std() # 97.5% z-test, single tailed

plt.figure(figsize=(8, 6))
c = plt.contourf(lon, time, q1_recon, levels=30, cmap='RdBu_r')
plt.contour(lon, time, q1_recon, levels=[g99], colors='k')

plt.scatter(-20, time[45], c="g")
plt.scatter(-33, time[80], c="g")
plt.scatter(50, time[154], c="g")
plt.scatter(150, time[220], c="g")
plt.scatter(120, time[252], c="g")
plt.scatter(10, time[286], c="g")
plt.scatter(115, time[340], c="g")
plt.scatter(-100, time[324], c="g")
plt.scatter(-158, time[325], c="g")
plt.scatter(-110, time[351], c="g")
plt.scatter(-105, time[20], c="g")
plt.scatter(-179, time[20], c="g")
plt.scatter(125, time[149], c="g")
plt.scatter(-179, time[227], c="g")
plt.scatter(-135, time[360], c="g")


plt.xlabel('Longitude')
plt.ylabel('Time')
plt.title('CNTL Reconstructed Q1')
plt.colorbar(c)
plt.show()

# ## Save to file

# ### Time and longitude selected

# In[67]:


lon_sel = np.array([-20, -33, 50, 150, 120, 10, 115, -100, -158, -110, -105, -179, 125, -179, -135])
time_sel = np.array([45, 80, 154, 220, 252, 286, 340, 324, 325, 351, 20, 20, 149, 227, 360])

lon_ref = np.array([np.argmin(np.abs(lon - l)) for l in lon_sel])
time_ref = time_sel
print("Longitude of events:", len(lon_ref))
output_list = [lon_ref, time_ref]

# ### Save file

# In[69]:
output_list = {
    "sel_lon": lon_ref,
    "sel_time": time_ref
}

pkl.dump(output_list, open("../CNTL_comp.pkl", "wb"))
