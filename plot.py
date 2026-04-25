import os
import glob
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import datetime as dt
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

# =============================================================================
# COMMON SETTINGS
# =============================================================================

# Date range
start_date_filter = dt.datetime(2024, 6, 10)
end_date_filter = dt.datetime(2024, 6, 12, 12, 0, 0)

# Day/night settings (UTC)
day_start_hour = 17  # 18:00 UTC (8:00 AM local time + 10 hours)
day_end_hour = 3     # 3:00 UTC (17:00 PM local time + 10 hours)

# =============================================================================
# ACSM DATA PROCESSING
# =============================================================================

acsm_data_path = r"ACSM"
acsm_filelist = sorted(glob.glob(os.path.join(acsm_data_path, "*.nc")))
acsm_ds = xr.open_mfdataset(acsm_filelist, combine="by_coords", compat='override', coords="all")
acsm_ds = acsm_ds.sel(time=(acsm_ds["time"] >= np.datetime64(start_date_filter)) & 
                      (acsm_ds["time"] <= np.datetime64(end_date_filter)))

species_to_plot = ["total_organics", "nitrate", "ammonium", "sulfate", "chloride"]
colors = {
    "total_organics": "green",
    "nitrate": "blue", 
    "ammonium": "gold",
    "sulfate": "red",
    "chloride": "pink"
}

# =============================================================================
# CCN DATA PROCESSING
# =============================================================================

ccn_data_path = r"CCN"
ccn_filelist = sorted(glob.glob(os.path.join(ccn_data_path, "*.nc")))
ccn_ds = xr.open_mfdataset(ccn_filelist, combine="by_coords", compat='override', coords="all")
ccn_ds = ccn_ds.sel(time=(ccn_ds["time"] >= np.datetime64(start_date_filter)) & 
                    (ccn_ds["time"] <= np.datetime64(end_date_filter)))

ccn_concentration = ccn_ds['N_CCN'].values
ccn_time = ccn_ds["time"].values
ccn_ss = ccn_ds['CCN_supersaturation_set_point'].values

# =============================================================================
# WIND SPEED DATA PROCESSING (10-MINUTE AVERAGE)
# =============================================================================

met_data_path = r"Met"
met_filelist = sorted(glob.glob(os.path.join(met_data_path, "*.nc")))
met_ds = xr.open_mfdataset(met_filelist, combine="by_coords", compat='override', coords="all")
met_ds = met_ds.sel(time=(met_ds["time"] >= np.datetime64(start_date_filter)) & 
                    (met_ds["time"] <= np.datetime64(end_date_filter)))

# Wind speed
wind_speed_data = met_ds['wind_speed']
wind_series = wind_speed_data.to_series()
wind_speed_avg = wind_series.rolling(window='10min', center=True).mean()

# Wind direction (circular mean via vector components)
wind_dir = met_ds['wind_direction']
wind_dir_rad = np.deg2rad(wind_dir)

wind_dir_u = np.cos(wind_dir_rad)
wind_dir_v = np.sin(wind_dir_rad)

wind_dir_u_avg = wind_dir_u.to_series().rolling(window='10min', center=True).mean()
wind_dir_v_avg = wind_dir_v.to_series().rolling(window='10min', center=True).mean()

wind_dir_avg = np.rad2deg(np.arctan2(wind_dir_v_avg, wind_dir_u_avg)) % 360

# time alignment
wind_time_avg = wind_speed_avg.index.to_numpy()
wind_speed_data_avg = wind_speed_avg.values
wind_dir_data_avg = wind_dir_avg.values


# =============================================================================
# CREATE MULTI-PANEL PLOT 
# =============================================================================

fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True, 
                        gridspec_kw={'height_ratios': [1.3, 1, 1]}, 
                        constrained_layout=True)
ax1, ax2, ax3 = axes 

# Add panel labels
labels = ['(a)', '(b)', '(c)']
for ax, label in zip(axes, labels):
    ax.text(0.01, 0.95, label,
            transform=ax.transAxes,
            fontsize=24,
            va='top', ha='left')


def add_day_night_shading(ax, time_min, time_max):
    """Add day/night background shading to a plot"""
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

# =============================================================================
# PANEL 1: ACSM SPECIES
# =============================================================================

add_day_night_shading(ax1, start_date_filter, end_date_filter)

for species in species_to_plot:
    label = species.replace("_", " ").title()
    ax1.plot(acsm_ds["time"], acsm_ds[species], label=label, 
            color=colors.get(species, "black"), linewidth=2, zorder=3)

ax1.set_ylim(0, 2.5)
ax1.set_ylabel("ACSM Mass Concentration \n ($\mu g\ m^{-3}$)", fontsize=24)
ax1.tick_params(axis='y', labelsize=24)
ax1.grid(True, alpha=0.3, zorder=2)
ax1.legend(loc="upper right", fontsize=24, frameon=True, fancybox=True, shadow=True)

# =============================================================================
# PANEL 2: CCN CONCENTRATION
# =============================================================================

add_day_night_shading(ax2, start_date_filter, end_date_filter)

ss_min = np.nanmin(ccn_ss)
ss_max = np.nanmax(ccn_ss)
norm = mcolors.Normalize(vmin=ss_min * 100, vmax=ss_max * 100)
cmap = plt.cm.plasma

scatter = ax2.scatter(ccn_ds["time"], ccn_concentration, c=ccn_ss * 100, cmap=cmap, norm=norm, 
                      s=10, zorder=3)

ax2.set_yscale("log")
ax2.set_ylabel("CCN Concentration \n ($cm^{-3}$)", fontsize=24)
ax2.tick_params(axis='y', labelsize=24)
ax2.grid(True, alpha=0.3, zorder=2)
ax2.set_ylim(1e1, 1e4) 

cbar = fig.colorbar(scatter, ax=ax2, orientation='vertical', pad=0.02)
cbar.set_label('Supersaturation (%)', fontsize=24)
cbar.ax.tick_params(labelsize=24)

# =============================================================================
# PANEL 3: WIND SPEED + DIRECTION (10-MINUTE AVERAGE)
# =============================================================================

add_day_night_shading(ax3, start_date_filter, end_date_filter)

ax3.plot(pd.to_datetime(wind_time_avg),
          wind_speed_data_avg, 
          color="blue", 
          label="Wind Speed", 
          linewidth=2, 
          zorder=4)

ax3.set_ylabel("Wind Speed \n ($\mathrm{m\ s^{-1}}$)", fontsize=24)
ax3.tick_params(axis='y', labelsize=24)
ax3.grid(True, alpha=0.3, zorder=2)

max_wind = np.nanmax(wind_speed_data_avg)

ax3.set_ylim(0, np.ceil(max_wind * 1.1))
ax3_dir = ax3.twinx()

ax3_dir.plot(pd.to_datetime(wind_time_avg),
             wind_dir_data_avg,
             color="red",
             linewidth=2,
             label="Wind Direction",
             zorder=3)

ax3_dir.set_ylabel("Wind Direction (°)", fontsize=24)
ax3_dir.tick_params(axis='y', labelsize=24)

lines_1, labels_1 = ax3.get_legend_handles_labels()
lines_2, labels_2 = ax3_dir.get_legend_handles_labels()

ax3.legend(lines_1 + lines_2,
           labels_1 + labels_2,
           loc="lower right",
           fontsize=24,
           frameon=True)


ax3_dir.set_ylim(0, 360)
ax3_dir.axhspan(190, 270, color='grey', alpha=0.2, zorder = 1)
# =============================================================================
# FINAL PLOT FORMATTING
# =============================================================================

ax3.xaxis.set_major_locator(mdates.HourLocator(interval=12))

def custom_formatter(x, pos=None):
    date = mdates.num2date(x)
    if date.hour == 0:
        return f"00\n{date.strftime('June %d 2024')}"
    elif date.hour == 12:
        return "12"
    else:
        return ""

ax3.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
ax3.tick_params(axis='x', pad=12)
ax3.tick_params(axis='x', labelsize=24)

ax1.set_xlim(start_date_filter, end_date_filter)

fig.align_ylabels(axes)
plt.savefig('ARM.png', dpi=300, bbox_inches='tight')
plt.savefig('ARM.svg', format="svg", bbox_inches='tight')
plt.show()