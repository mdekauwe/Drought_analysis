import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
fname = "AWAP.Rainf.3hr.2000.nc"
ds = xr.open_dataset(fname)
sec_2_3hour = 60. * 60. * 3
ppt = ds.Rainf * sec_2_3hour

ppt = ppt.sum(axis=0)
ppt = np.where(ppt == 0.0, np.nan, ppt)


plt.imshow(ppt[100:350,550:841])
plt.colorbar()
plt.show()
