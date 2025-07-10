import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib
matplotlib.rc('font',family='MicroSoft YaHei')
from matplotlib import cm

LU_attr = gpd.read_file('./data/mapping_shp/GSR.dbf',encoding='utf-8')
LU_colors = {
    'Transportation': (240 / 255, 255 / 255, 255 / 255),
    'Education and science': (255 / 255, 165 / 255, 0),
    'Specially-designed land': (128 / 255, 128 / 255, 128 / 255),
    'Commercial area': (255 / 255, 42 / 255, 42 / 255),
    'Farmland and forest': (85 / 255, 107 / 255, 47 / 255),
    'Residential area': (255 / 255, 255 / 255, 0),
    'Other non-construction land': (139 / 255, 69 / 255, 19 / 255),
    'Industrial area': (222 / 255, 184 / 255, 135 / 255),
    'Sports': (189 / 255, 183 / 255, 107 / 255),
    'Park': (143 / 255, 188 / 255, 143 / 255),
    'Water': (65 / 255, 105 / 255, 225 / 255),
    'Square': (211 / 255, 211 / 255, 211 / 255),
    'Cultural facilities': (240 / 255, 128 / 255, 128 / 255),
    'Health care': (255 / 255, 192 / 255, 203 / 255),
    'Administration': (255 / 255, 0, 255 / 255),
    'Municipal utilities': (255 / 255, 244 / 255, 221 / 255),
    'Green buffer': (0 / 255, 180 / 255, 39 / 255),
    'invalid': (1,1,1),
}

fig, ax = plt.subplots(figsize=(16, 12))
pmarks = []
for ctype, data in LU_attr.groupby(by='NGLU'):
    data.plot(color=LU_colors[ctype],
              ax=ax,
              linewidth=0.5,
              edgecolor='black',
              )
    pmarks.append(Patch(facecolor=LU_colors[ctype], label=ctype,edgecolor='black'))

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=[*handles, *pmarks], fontsize=15, title='Legend', bbox_to_anchor=(1.3, 0.8))

leg = ax.get_legend()
leg.get_title().set_fontsize(fontsize=15)

ax.set_axis_off()
ori_minx, ori_maxx = ax.get_xlim()
ax.set_xlim(ori_minx - 0.12 * (ori_maxx - ori_minx), ori_maxx + 0.12 * (ori_maxx - ori_minx))
plt.show()
