# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:50:41 2022

this is a code example used for climate related analysis

@author: Yuting Chen

"""

from datetime import datetime, date
import argparse
import os
import pandas as pd
import numpy as np
import xarray as xr
from metpy.units import units
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.geometry import Polygon
import warnings

class ModelInfo:
    """ This class contains all infomation of climate model we are working on.
    In this demo code, it works on UKCP18 Convection-permitting ensemble data
    (daily/hourly precipitation)
    """

    def __init__(
        self,
        model_collection: str = "UKCP18",
        product: str = None,
        emission_scenario: str = None,
        area: str = None,
        resolution: str = "hourly",
        products_path: str = "",
    ):
        self.model_collection = model_collection
        if product == "cpm_uk_2.2km":
            self.product = product
            self.ensemble_list = ( 1, 4, 5, )  # TO DO: only select three ensemble here, will be expanded to use all ensembles
            # 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15
        else:
            raise Exception("This product is not available")
        self.emission_scenario = emission_scenario
        self.area = area
        self.resolution = resolution
        self.products_path = products_path
        self.summary_nc_fp = products_path
        self.summary_figs_fp = products_path
        self.summary_nc_fn = {}

    def set_time_period(self, time_period):
        """Define year range of the time period of interest"""
        self.time_period = time_period
        if self.time_period in ("current", "control"):
            self.year_range = np.arange(1980, 2000)
        elif self.time_period in ("future", "projected"):
            self.year_range = np.arange(2060, 2080)
        else:
            raise Exception("This time_period is not found")

    def set_task_info(self, task_description):
        """Set task info"""
        if task_description == "rainfall-extreme_analysis":
            self.task_info = "extreme analysis"
            self.variable = "pr"
        elif task_description == "rainfall-export_summary_stats":
            self.task_info = "export_summary_stats"
            self.variable = "pr"
        else:
            raise Exception("Analysis related to this task has not been finished")

    def set_summary_fn(self, fn):
        """Accessor method"""
        self.summary_nc_fn[self.time_period] = fn

    def get_summary_fn(self, time_period):
        """Accessor method"""
        if time_period in self.summary_nc_fn.keys():
            return self.summary_nc_fn[time_period]
        else:
            raise Exception("Summary file for this time period not found")

    def get_summary_figs_fp(self):
        """Accessor method"""
        return self.summary_figs_fp

    def get_year_range(self):
        """Accessor method"""
        return self.year_range

    def get_ensemble_list(self):
        """Accessor method"""
        return self.ensemble_list

    def get_model_fn(self, ensNo, year):
        """Get file name of this dataset"""
        if self.product == "cpm_uk_2.2km":
            cm_fn_pat = (
                self.products_path + "%02d/%s_%s_land-%s_%02d_day_%04d1201-%04d1130.nc"
            )
            cm_fn = (cm_fn_pat) % (
                ensNo,
                self.variable,
                self.emission_scenario,
                self.product,
                ensNo,
                year,
                year + 1,
            )
        else:
            raise Exception("This fn path has not been defined")
        return cm_fn

    def compute_summary_data(self, fn: str = "", save_file: bool = True):
        """compute summary data for each defined model"""

        (ds_mean, ds_max) = (None, None)

        for ensNo in self.get_ensemble_list():
            for year in self.get_year_range():
                
                cm_fn = self.get_model_fn(ensNo, year)
                ds_temp = load_netcdf(cm_fn, show_info=False)
                (ds_mean_this, ds_max_this) = get_key_data(ds_temp, self.variable)
                
                ds_mean = (
                    ds_mean_this
                    if ds_mean is None
                    else xr.concat([ds_mean, ds_mean_this], "time")
                )
                ds_mean = ds_mean.astype("f4")
                ds_max = (
                    ds_max_this
                    if ds_max is None
                    else xr.concat([ds_max, ds_max_this], "time")
                )
                ds_max = ds_max.astype("f4")
                print("ensemble-" + str(ensNo) + " year-" + str(year) + " finished.")
        ds_summary = xr.merge([ds_mean, ds_max])

        if save_file:
            ds_summary.to_netcdf(fn)
        return ds_summary

    def do_this_task(
        self,
        task_description: str = "rainfall-export_summary_stats",
        recompute_from_beginning: bool = False,
    ):

        if task_description == "rainfall-export_summary_stats":
            self.set_task_info(task_description)

            # processing climate data
            self.set_time_period("current")
            fn = self.summary_nc_fp + "summary_baseline.nc"
            self.set_summary_fn(fn)
            ds_summary_baseline = (
                self.compute_summary_data(fn)
                if recompute_from_beginning
                else xr.open_dataset(fn)
            )

            self.set_time_period("future")
            fn = self.summary_nc_fp + "summary_future.nc"
            self.set_summary_fn(fn)
            ds_summary_future = (
                self.compute_summary_data(fn)
                if recompute_from_beginning
                else xr.open_dataset(fn)
            )
        else:
            raise Exception("unspecified actions to the task: " + task_description)
        return (ds_summary_baseline, ds_summary_future)


class RainfallData:
    """This calss contains properties requird for rainfall-related analysis from
    different data source (such as climate model output, radar observation, re-analysis data, rain gauge data)
    In this demo code, it only works for climate model output (summary data in netCDF format)
    """

    def __init__(self, model_info: ModelInfo, datatype: str = "climate model output"):
        self.datatype = datatype
        self.ds_current = self.__load_file(model_info, "current")
        self.ds_future = self.__load_file(model_info, "future")

    def __load_file(self, data_info: ModelInfo, time_period: str = "current"):
        fn = data_info.get_summary_fn(time_period)
        ds = load_netcdf(fn, show_info=False)
        return ds

    def get_base_ensemble_median(self, stats_str):
        ds_median = (self.ds_current[stats_str].mean("time")).median("ensemble_member")
        return ds_median

    def get_diff_ensemble_median(self, stats_str, rel_diff=False):
        ds_median = (
            self.ds_future[stats_str].mean("time")
            - self.ds_current[stats_str].mean("time", skipna=True)
        ).median("ensemble_member")
        if rel_diff:
            # calculate relative change in this
            ds_median = ds_median / self.get_base_ensemble_median(stats_str)
        return ds_median

    def get_diff_ensemble_range(self, stats_str):
        ds_upper = (
            self.ds_future[stats_str].mean("time")
            - self.ds_current[stats_str].mean("time")
        ).max("ensemble_member")
        ds_lower = (
            self.ds_future[stats_str].mean("time")
            - self.ds_current[stats_str].mean("time")
        ).min("ensemble_member")
        return (ds_upper, ds_lower)

    def close_data(self):
        print("this job ends, starts to close dataset")
        self.ds_current.close()
        self.ds_future.close()

# load a netcdf file
def load_netcdf(cm_fn: str = None, show_info: bool = True):
    ds = xr.open_dataset(cm_fn)
    if show_info:
        print("This data include those dims: " + str(ds.dims))
        print(ds.variables)
    return ds

def get_key_data(ds, varname):
    """function for processing raw cpm output and get key properties

    Parameters
    ----------
    ds: dataset
        from xarray
    varname: str
        one of variable names in $ds
    """
    ds_annual_mean = ds[varname].mean("time")
    ds_annual_mean["year"] = ds["time"]["year"][-1]
    ds_annual_mean = ds_annual_mean.rename(varname + "_mean")

    ds_annual_max = ds[varname].max("time")
    ds_annual_max["year"] = ds["time"]["year"][-1]
    ds_annual_max = ds_annual_max.rename(varname + "_max")

    return (ds_annual_mean, ds_annual_max)

def add_shp_mask(area, ax, proj, plot_proj=None):
    """add a shapefile based mask to current ax
    (this is used after plotting data)

    Parameters
    ----------
    area : str
        area which want to be visualized
        (currently set to be visualize UK)
        TO DO: enable a selection of more countries which can be further used in GCM output
    ax: axes

    proj: PlateCarree
        Projection of Data

    plot_proj: PlateCarree
        Projection of Data

    """
    def __rect_from_bound(xmin, xmax, ymin, ymax):
        """Returns list of (x,y)'s for a rectangle"""
        xs = [xmax, xmin, xmin, xmax, xmax]
        ys = [ymax, ymax, ymin, ymin, ymax]
        return [(x, y) for x, y in zip(xs, ys)]

    if plot_proj is None:
        plot_proj = proj
    if area in {"UK", "United Kingdom"}:

        (resolution, category, name) = ("10m", "cultural", "admin_0_countries")
        shpfilename = shapereader.natural_earth(resolution, category, name)
        df = geopandas.read_file(shpfilename)

        # get geometry of a country
        poly = [df.loc[df["ADMIN"] == "United Kingdom"]["geometry"].values[0]]

        ax.add_geometries(poly, crs=proj, facecolor="none", edgecolor="black")

        pad1 = 2  # padding, degrees unit
        exts = [
            poly[0].bounds[0] - pad1,
            poly[0].bounds[2] + pad1,
            poly[0].bounds[1] - pad1,
            poly[0].bounds[3] + pad1,
        ]
        ax.set_extent(exts, crs=proj)

        msk = Polygon(__rect_from_bound(*exts)).difference(poly[0].simplify(0.01))
        msk_stm = proj.project_geometry(msk, proj)

        # plot the mask using semi-transparency on the masked-out portion
        ax.add_geometries(
            msk_stm,
            plot_proj,
            zorder=12,
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
        )
        return exts
    else:
        print("code for this area has not been finished. the whole dataset is plotted.")
        return None

def get_variable_customized_property(area, variable_specification):
    """get customized plotting-related property for this variable

    Parameters
    ----------
    area: str
        area to be visualized
    variable_specification
        variable to be visualized
    """
    if variable_specification == "abs_diff in extreme rainfall":
        colorbar_range = np.arange(-5, 31, 5)
        colorbar_cmap = "YlGn"
        colorbar_ylabel = "[mm/day]"
    elif variable_specification == "rel_diff in mean rainfall":
        colorbar_range = np.arange(-0.25, 0.26, 0.05)
        colorbar_cmap = "coolwarm"
        colorbar_ylabel = "[-]"
    else:
        raise Exception("this variable specification is not defined")
    if area in ("UK", "United Kingdom"):
        figure_extent = [-9, 2, 60, 49]
    else:
        figure_extent = None
    return (colorbar_range, colorbar_cmap, figure_extent, colorbar_ylabel)

def plot_this_for_an_country(ds, area, plotting_aim):

    fig = plt.figure(dpi=100, figsize=(8, 6))
    (proj, plot_proj) = (
        ccrs.PlateCarree(),
        ccrs.PlateCarree(),
    )  # Set Projection of Data and Projection of Plot
    ax = plt.axes(projection=proj)

    (colorbar_range, colorbar_cmap, figure_extent, colorbar_ylabel) = get_variable_customized_property(
        area, plotting_aim
    )

    c11 = ax.contourf(
        ds["longitude"],
        ds["latitude"],
        ds.values,
        colorbar_range,
        extend="both",
        transform=plot_proj,
        cmap=colorbar_cmap,
    )

    exts = add_shp_mask(area, ax, proj, plot_proj)

    # Add country borders to plot
    country_borders = cfeature.NaturalEarthFeature(
        category="cultural", name="admin_0_countries", scale="50m", facecolor="none"
    )
    ax.add_feature(country_borders, edgecolor="black", linewidth=1)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"))
    ax.add_feature(cfeature.LAND.with_scale("50m"))
    ax.add_feature(cfeature.LAKES.with_scale("50m"))

    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.gridlines(draw_labels=True)

    ax.set_extent(exts if figure_extent is None else figure_extent)

    cbar_ax = fig.add_axes([0.89, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(
        c11, shrink=0.99, aspect=20, fraction=0.03, pad=0.01, cax=cbar_ax
    )
    cbar.ax.set_ylabel(colorbar_ylabel, rotation=270)

    return (fig, ax)

def parse_args(args=None):

    parser = argparse.ArgumentParser(
        description="sample script written by Yuting Chen (Imperial College London)"
    )
    parser.add_argument(
        "model_collection",
        help="climate projection name which will be used (UKCP18, etc)",
    )
    parser.add_argument("emission_scenario", help="emission_scenario (rcp85, etc)")
    parser.add_argument(
        "product", help="product name which will be used (cpm_uk_2.2km, etc)"
    )
    parser.add_argument(
        "resolution",
        help="resolution of the climate data product of interest (daily, etc)",
    )
    parser.add_argument(
        "study_area", help="study area which will be analysed and plotted (UK, etc)"
    )
    parser.add_argument(
        "job_aim",
        help="data analaysis description (rel_diff in mean rainfall, abs_diff in max rainfall, etc)",
    )
    parser.add_argument(
        "products_path",
        help="Path climate data will be read from, expects filenames of format like: ./10/pr_rcp85_land-cpm_uk_2.2km_10_day....nc (F:/demo/cpm/)'",
    )
    parser.add_argument(
        "--recompute_from_beginning",
        action="store_true",
        help="compute summary data from climate model output available in $products_path",
    )
    parser.add_argument(
        "--plot_tag", action="store_true", help="plot anlysed results (default:False)"
    )

    args = parser.parse_args(args)

    return args

def main(args=None):
    """Main handles arguments and calls cell tracking functions in order"""

    args = parse_args(args)

    model_collection = args.model_collection  # "UKCP18"
    emission_scenario = args.emission_scenario  # "rcp85"
    product = args.product  # "cpm_uk_2.2km"
    resolution = args.resolution  # "daily"
    area = args.study_area  # "UK"
    job_aim = args.job_aim  # "rel_diff in mean rainfall"
    products_path = args.products_path  # "F:/demo/cpm/"
    recompute_from_beginning = args.recompute_from_beginning
    plot_tag = args.plot_tag

    # configuration -- set up climate model info
    model_info = ModelInfo(
        model_collection, product, emission_scenario, area, resolution, products_path
    )

    # pre-processing -- convection permitting model output
    model_info.do_this_task(
        "rainfall-export_summary_stats",
        recompute_from_beginning=recompute_from_beginning,
    )

    if plot_tag:
        plt.ioff()
        warnings.filterwarnings("ignore")

        # specify RainfallData which will be processed/plotted
        ra = RainfallData(model_info, datatype="climate model output")
        figpath = model_info.get_summary_figs_fp()
        if job_aim == "abs_diff in extreme rainfall":
            # TO DO: This is a demo code showing some stats related to extremes
            # Time series at each point hasn't been fit to gumbel distribution here
            # process data
            ds = ra.get_diff_ensemble_median("pr_max")
            # plot data
            (fig, ax) = plot_this_for_an_country(ds, area, job_aim)

            ax.set_title(
                "absolute difference in annual maxima %s rainfall \n %s (%s, ensemble average, 1980-2000 vs 2060-2080)"
                % (resolution, model_collection, area)
            )
            fig.savefig(figpath + job_aim + ".png")
        elif job_aim == "rel_diff in mean rainfall":
            # process data
            ds = ra.get_diff_ensemble_median("pr_mean", rel_diff=True)
            # plot data
            (fig, ax) = plot_this_for_an_country(ds, area, job_aim)
            ax.set_title(
                "relative change in annual mean %s rainfall \n %s (%s, ensemble average, 1980-2000 vs 2060-2080)"
                % (resolution, model_collection, area)
            )
            fig.savefig(figpath + job_aim + ".png")
        else:
            raise Exception("actions to this plotting_job_aim has not been specified")
    if plot_tag:
        ra.close_data()

if __name__ == "__main__":
    main()