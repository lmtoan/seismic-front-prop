import numpy as np
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from matplotlib import pyplot as plt
import gstools as gs
import pandas as pd
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

col_Names=["StationName", "Latitude", "Longitude"]

df=pd.read_csv('StationCode.txt', delimiter=' ', names=col_Names)

df.head()
#print(df)

Latitude_list = df["Latitude"].tolist()
Longitude_list = df["Longitude"].tolist()
StationName_list = df["StationName"].tolist()

BBox=(min(Longitude_list),max(Longitude_list),min(Latitude_list),max(Latitude_list),)
#print(BBox)

BBox1=(-73.7265,-73.6372,3.8797,4)

ruh_m = plt.imread('map_BBox1.png')
#https://www.openstreetmap.org/export#map=12/3.9382/-73.6781

fig, ax = plt.subplots(figsize = (8,12))
ax.scatter(Longitude_list, Latitude_list, zorder=1, alpha= 1, c='b', s=10)

for i, txt in enumerate(StationName_list):
    ax.annotate(txt, (Longitude_list[i], Latitude_list[i]))

#ax.set_title('Spatial Data')
ax.set_xlim(BBox1[0],BBox1[1])
ax.set_ylim(BBox1[2],BBox1[3])
ax.imshow(ruh_m, zorder=0, extent = BBox1, aspect= 'equal')
#plt.ticklabel_format(axis='x', style='sci')
#plt.show()

##


df['Porosity']=[0.2,0.3,0.5,0.8,0.6,0.7,0.5,0.7,0.8,0.56,0.89,0.85,0.88]
df['Permeability']=[0.8,0.6,0.7,0.8,0,0,0,0,0,0,0,0,0]


data=df.loc[:,['Latitude','Longitude','Porosity']].values
# conditioning data
print(data)
# grid definition for output field
gridx = np.linspace(min(Longitude_list),max(Longitude_list), 20)
print(gridx)
gridy = np.linspace(min(Latitude_list),max(Latitude_list), 20)
print(gridy)

# a GSTools based covariance model
cov_model = gs.Gaussian(dim=2, len_scale=4, anis=0.2, angles=-0.5, var=0.5, nugget=0.1)
# ordinary kriging with pykrige
OK1 = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], cov_model)
z1, ss1 = OK1.execute("grid", gridx, gridy)
#plt.imshow(z1, origin="lower")
fig, bx = plt.subplots(figsize=(8,12))
bx1=bx.imshow(z1,  origin="lower", extent=[BBox[0],BBox[1],BBox[2],BBox[3]])
bx.set_aspect(2) 


UK = UniversalKriging(data[:, 0],data[:, 1],data[:, 2],variogram_model="gaussian", drift_terms=["regional_linear"])

z, ss = UK.execute("grid", gridx, gridy)

fig, bx = plt.subplots(figsize=(8,12))
bx2=bx.imshow(z, origin="lower", extent=[BBox[0],BBox[1],BBox[2],BBox[3]]) # origin="lower", extent=[BBox[0],BBox[1],BBox[2],BBox[3]])
bx.set_aspect(2) 
cb_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
cbar = fig.colorbar(bx2, cax=cb_ax)
plt.show()

