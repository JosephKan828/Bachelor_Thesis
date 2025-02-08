#!/usr/bin/env python
# coding: utf-8

# # CNTL CCKWs event selection

# ## Import packages

# In[1]:


import sys
import numpy as np
import pandas as pd
import netCDF4 as nc

from matplotlib import pyplot as plt

sys.path.append('/home/b11209013/Package')
import DataProcess as dp   #type: ignore
import SignalProcess as sp #type: ignore


# ## Functions

# In[2]:


def is_within_region(wnm, frm, wnm_min, wnm_max, frm_min, frm_max, kel_sign=1):
    kel_curves = lambda ed, k: (86400/(2*np.pi*6.371e6))*np.sqrt(9.81*ed)*k
    
    return (
        (wnm > wnm_min) & (wnm < wnm_max) &
        (frm > kel_sign * frm_min) & (frm < kel_sign * frm_max) &
        (frm < kel_sign * kel_curves(90, wnm)) &
        (frm > kel_sign * kel_curves(8, wnm))
    )


# ## Load data

# In[3]:


with nc.Dataset('/work/b11209013/2024_Research/MPAS/merged_data/NSC/q1.nc', 'r') as f:
    lon : np.ndarray = f.variables['lon'][:]
    lat : np.ndarray = f.variables['lat'][:]
    lev : np.ndarray = f.variables['lev'][:]
    time: np.ndarray = f.variables['time'][:]
    lat_lim = np.where((lat >= -5) & (lat <= 5))[0]
    
    lat : np.ndarray = lat[lat_lim]
    q1       = f.variables['q1'][:, :, lat_lim, :]

ltime, llev, llat, llon = q1.shape


# ## Vertical integrated Q1

# In[4]:


q1_ave = (q1[:, 1:] + q1[:, :-1]) / 2
q1_int = -np.sum(q1_ave*np.diff(lev)[None, :, None, None]*100, axis=1) *86400/9.8/2.5e6


# ## Process Data

# ### Symmetrize data

# In[5]:


fmt = dp.Format(lat)

q1_sym = fmt.sym(q1_int)

print(q1_sym.shape)


# ### bandpass filter

# In[6]:


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

# In[7]:


q1_fft = np.array([np.fft.fft(q1_sym[i]) for i in range(ltime)])
q1_fft = np.array([np.fft.ifft(q1_fft[:, i]) for i in range(llon)]).T


# ### Filt with the filter

# In[8]:


mask = np.where(cond1 | cond2, 1, 0)
q1_filted = mask * q1_fft


# ### reconstruct

# In[9]:


q1_recon = np.array([np.fft.fft(q1_filted[:, i]) for i in range(llon)]).T
q1_recon = np.array([np.fft.ifft(q1_recon[i]) for i in range(ltime)])

q1_recon = np.real(q1_recon)


# ## Select events

# In[10]:


g99 = q1_recon.mean() + 1.96*q1_recon.std() # 97.5% z-test, single tailed

plt.figure(figsize=(8, 6))
c = plt.contourf(lon, time, q1_recon, levels=30, cmap='RdBu_r')
plt.contour(lon, time, q1_recon, levels=[g99], colors='k')

plt.scatter(-179, time[18], c="g")
plt.scatter(93, time[22], c="g")
plt.scatter(-130, time[38], c="g")
plt.scatter(40, time[82], c="g")
plt.scatter(-170, time[52], c="g")
plt.scatter(140, time[97], c="g")
plt.scatter(140, time[129], c="g")
plt.scatter(60, time[112], c="g")
plt.scatter(-105, time[95], c="g")
plt.scatter(-75, time[135], c="g")
plt.scatter(-25, time[147], c="g")
plt.scatter(-100, time[180], c="g")
plt.scatter(-170, time[142], c="g")
plt.scatter(60, time[228], c="g")
plt.scatter(130, time[286], c="g")
plt.scatter(140, time[318], c="g")
plt.scatter(30, time[318], c="g")
plt.scatter(-120, time[258], c="g")
plt.scatter(-45, time[331], c="g")


plt.xlabel('Longitude')
plt.ylabel('Time')
plt.title('NSC Reconstructed Q1')
plt.colorbar(c)
plt.savefig('sel_image/NSC_Q1_recon.png')
plt.show()


# ## Save to file

# ### Time and longitude selected

# In[11]:


lon_sel = np.array([-179, 93, -130, 40, -170, 140, 140, 60, -105, -75, -25, -100, -170, 60, 130, 140, 30, -120, -45])
time_sel = np.array([18, 22,  38, 82,  52,  97, 129, 112,  95, 135, 147, 180, 142, 228, 286, 318, 318, 258, 331])

lon_ref = np.array([np.argmin(np.abs(lon - l)) for l in lon_sel])
time_ref = time_sel
print("Longitude of events", len(lon_ref))
output_list = [lon_ref, time_ref]


# ## Test of compositing points

# In[12]:


q1_test = q1.mean(axis=2) - q1.mean()


q1_comp = np.zeros((llev, 25))
for i in range(len(lon_ref)):
    q1_comp += q1_test[time_ref[i]-12:time_ref[i]+13, :, lon_ref[i]].T

q1_comp /= len(lon_ref)

plt.contourf(np.linspace(-3, 3, 25), lev, q1_comp, cmap='RdBu_r', levels=np.linspace(-0.1, 0.1, 21), extend='both')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.yscale('log')
plt.colorbar()


# ### Save file

# In[13]:


np.save('Q1_sel/NSC.npy', output_list)

