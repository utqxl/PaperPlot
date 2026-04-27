import os
import glob
import xarray as xr
import matplotlib as mpl
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Settings
start_date_filter = dt.datetime(2024, 6, 10)
end_date_filter = dt.datetime(2024, 6, 12, 12, 0, 0)
day_start_hour = 17
day_end_hour = 3

# Load ACSM data
acsm_ds = xr.open_mfdataset(sorted(glob.glob(os.path.join("ACSM", "*.nc"))), 
                            combine="by_coords", compat='override', coords="all")
acsm_ds = acsm_ds.sel(time=(acsm_ds["time"] >= np.datetime64(start_date_filter)) & 
                      (acsm_ds["time"] <= np.datetime64(end_date_filter)))

species_to_plot = ["total_organics", "nitrate", "ammonium", "sulfate", "chloride"]
colors = {"total_organics": "green", "nitrate": "blue", "ammonium": "gold", 
          "sulfate": "red", "chloride": "pink"}

# Load CCN data
ccn_ds = xr.open_mfdataset(sorted(glob.glob(os.path.join("CCN", "*.nc"))), 
                           combine="by_coords", compat='override', coords="all")
ccn_ds = ccn_ds.sel(time=(ccn_ds["time"] >= np.datetime64(start_date_filter)) & 
                    (ccn_ds["time"] <= np.datetime64(end_date_filter)))
ccn_concentration = ccn_ds['N_CCN'].values
ccn_ss = ccn_ds['CCN_supersaturation_set_point'].values

# Load and process wind data
met_ds = xr.open_mfdataset(sorted(glob.glob(os.path.join("Met", "*.nc"))), 
                           combine="by_coords", compat='override', coords="all")
met_ds = met_ds.sel(time=(met_ds["time"] >= np.datetime64(start_date_filter)) & 
                    (met_ds["time"] <= np.datetime64(end_date_filter)))

wind_series = met_ds['wind_speed'].to_series().rolling(window='30min', center=True).mean()
wind_dir_rad = np.deg2rad(met_ds['wind_direction'])
wind_dir_u_avg = np.cos(wind_dir_rad).to_series().rolling(window='30min', center=True).mean()
wind_dir_v_avg = np.sin(wind_dir_rad).to_series().rolling(window='30min', center=True).mean()
wind_dir_data_avg = np.rad2deg(np.arctan2(wind_dir_v_avg, wind_dir_u_avg)) % 360
wind_time_avg = wind_series.index.to_numpy()
wind_speed_data_avg = wind_series.values

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
fig.subplots_adjust(left=0.08, right=0.85, top=0.95, bottom=0.08, hspace=0.12)
ax1, ax2, ax3 = axes

fig.canvas.draw()
ax1_pos_before = ax1.get_position()
ax2_pos_before = ax2.get_position()
ax3_pos_before = ax3.get_position()

for ax, label in zip(axes, ['(a)', '(b)', '(c)']):
    ax.text(0.01, 0.95, label, transform=ax.transAxes, fontsize=24, va='top', ha='left')

# Day/night shading function
def add_day_night_shading(ax, time_min, time_max):
    time_min = pd.to_datetime(time_min)
    time_max = pd.to_datetime(time_max)
    current_date = time_min.date() - dt.timedelta(days=1)
    while current_date <= time_max.date() + dt.timedelta(days=1):
        day_start = dt.datetime.combine(current_date, dt.time(day_start_hour, 0))
        day_end = dt.datetime.combine(current_date + dt.timedelta(days=1), dt.time(day_end_hour, 0))
        night_start = dt.datetime.combine(current_date, dt.time(day_end_hour, 0))
        night_end = dt.datetime.combine(current_date, dt.time(day_start_hour, 0))
        if day_start < time_max and day_end > time_min:
            ax.axvspan(max(day_start, time_min), min(day_end, time_max), alpha=0.2, color='yellow', zorder=1)
        if night_start < time_max and night_end > time_min:
            ax.axvspan(max(night_start, time_min), min(night_end, time_max), alpha=0.1, color='grey', zorder=1)
        current_date += dt.timedelta(days=1)

# Panel 1: ACSM species
add_day_night_shading(ax1, start_date_filter, end_date_filter)
for species in species_to_plot:
    ax1.plot(acsm_ds["time"], acsm_ds[species], 
             label=species.replace("_", " ").title(), 
             color=colors.get(species, "black"), linewidth=2, zorder=3)
ax1.set_ylim(0, 2.5)
ax1.set_ylabel("Aerosol Chemical \n Speciation Monitor Mass \n Concentration ($\mu$g m$^{-3}$)", fontsize=24)
ax1.tick_params(axis='both', labelsize=24, length=8)
ax1.minorticks_off()
ax1.legend(loc="upper right", fontsize=24, frameon=True, fancybox=True, shadow=True, bbox_to_anchor=(0.935, 1))

# Panel 2: CCN concentration
add_day_night_shading(ax2, start_date_filter, end_date_filter)
norm = mcolors.Normalize(vmin=np.nanmin(ccn_ss) * 100, vmax=np.nanmax(ccn_ss) * 100)
scatter = ax2.scatter(ccn_ds["time"], ccn_concentration, c=ccn_ss * 100, cmap=plt.cm.plasma, 
                      norm=norm, s=10, zorder=3)
ax2.set_yscale("log")
ax2.set_ylim(1e1, 5e3)
ax2.set_ylabel("CCN Concentration\n(cm$^{-3}$)", fontsize=24)
ax2.tick_params(axis='both', labelsize=24, length=8)
ax2.minorticks_off()

cbar_ax = fig.add_axes([0.9, ax2_pos_before.y0, 0.015, ax2_pos_before.height])
cbar = fig.colorbar(scatter, cax=cbar_ax)
cbar.ax.tick_params(labelsize=24, length=8)
cbar.ax.minorticks_off()
ax2.set_position(ax2_pos_before)

# Panel 3: Wind speed and direction
ax3_dir = ax3.twinx()
add_day_night_shading(ax3, start_date_filter, end_date_filter)
ax3_dir.patch.set_visible(False)
# Baseline sector
ax3_dir.axhspan(190, 280, facecolor='none', edgecolor='red', linewidth=2, 
                alpha=0.7, hatch='////', zorder=1, label="Baseline Sector")
ax3_dir.plot(pd.to_datetime(wind_time_avg), wind_dir_data_avg, color="red", linewidth=2, zorder=2)
ax3.set_zorder(ax3_dir.get_zorder() + 1)
ax3.patch.set_visible(False)
ax3.plot(pd.to_datetime(wind_time_avg), wind_speed_data_avg, color="blue", linewidth=2, zorder=5)
ax3.set_ylabel("Wind Speed\n(m s$^{-1}$)", fontsize=24)
ax3.set_ylim(0, 27)
ax3.tick_params(axis='both', labelsize=24, length=8)
ax3.minorticks_off()
ax3_dir.set_ylabel("Wind Direction (°)", fontsize=24, labelpad=45)
ax3_dir.tick_params(axis='y', labelsize=24, length=8)
ax3_dir.minorticks_off()
ax3_dir.set_ylim(0, 360)

legend_elements = [
    Line2D([0], [0], color='blue', linewidth=2, label='Wind Speed'),
    Line2D([0], [0], color='red', linewidth=2, label='Wind Direction'),
    Patch(facecolor='none', edgecolor='red', alpha=0.7, hatch='////', label='Baseline Sector')
]
ax3.legend(handles=legend_elements, loc="lower right", fontsize=24, frameon=True, bbox_to_anchor=(0.95, 0.01))

# Position colorbar and supersaturation label
fig.canvas.draw()
ax3_dir_ylabel = ax3_dir.yaxis.get_label()
bbox_dir = ax3_dir_ylabel.get_window_extent(renderer=fig.canvas.get_renderer())
dir_label_x_fig = fig.transFigure.inverted().transform((bbox_dir.x0 + bbox_dir.width/2, 0))[0]
cbar_pos = cbar_ax.get_position()
cbar_ax.set_position([dir_label_x_fig - cbar_pos.width - 0.05, ax2_pos_before.y0, cbar_pos.width, ax2_pos_before.height])
fig.text(dir_label_x_fig, ax2_pos_before.y0 + ax2_pos_before.height/2, 'Supersaturation (%)', 
         fontsize=24, rotation=90, va='center', ha='center')

# Set custom x axis to show dates and time
ax3.xaxis.set_major_locator(mdates.HourLocator(interval=12))
def custom_formatter(x, pos=None):
    date = mdates.num2date(x)
    if date.hour == 0:
        return f"00\n{date.strftime('June %d 2024')}"
    elif date.hour == 12:
        return "12"
    return ""
ax3.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
ax3.tick_params(axis='x', pad=12, labelsize=24, length=8)
ax1.set_xlim(start_date_filter, end_date_filter)

fig.align_ylabels(axes)
plt.savefig('ARM.png', dpi=300, bbox_inches='tight')
plt.savefig('ARM.svg', format="svg", bbox_inches='tight')
plt.show()