# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:13:39 2021

@author: ruizca
"""
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5
from astropy.table import Table, unique, join
from astropy.utils.console import color_print
from astropy_healpix import HEALPix
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
from mocpy import MOC
from mocpy.mocpy import flatten_pixels
from scipy.stats import median_abs_deviation
from tqdm.auto import tqdm

from .. import rapidxmm
from .ecf import ECF


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
#plt.rc('text', usetex=True)
plt.rcParams['mathtext.fontset'] = "stix"
plt.rcParams['mathtext.rm'] = "STIXGeneral"
plt.rcParams['font.family'] = "STIXGeneral"
plt.rcParams["axes.formatter.use_mathtext"] = True

# Numpy random  number generator
rng = np.random.default_rng()


def get_neighbours(npixel, hp, level=5):
    # The central pixel is the first one
    # The output of hp.neighbours always follows the
    # same order, starting SW and rotating clockwise
    neighbours_level = [None] * (level + 1)
    neighbours_level[0] = [npixel]
    npixel_neighbours = [npixel]

    for i in range(1, level + 1):
        neighbours_level[i] = hp.neighbours(neighbours_level[i - 1]).flatten()
        npixel_neighbours += list(neighbours_level[i])

    sorted_neighbours = Table()
    sorted_neighbours["npixel"] = npixel_neighbours
    sorted_neighbours["order"] = range(len(npixel_neighbours))

    sorted_neighbours = unique(sorted_neighbours, keys=["npixel"])
    sorted_neighbours.sort("order")

    return sorted_neighbours


def get_bkg_npixels(src_center, nside, npixels=100):
    order = np.log2(nside).astype(int)
    bkg_moc_outer = MOC.from_cone(src_center.ra, src_center.dec, 120*u.arcsec, order)
    bkg_moc_inner = MOC.from_cone(src_center.ra, src_center.dec, 60*u.arcsec, order)
    bkg_moc = bkg_moc_outer.difference(bkg_moc_inner)
    bkg_npixels = flatten_pixels(bkg_moc._interval_set._intervals, order)

    return rng.choice(bkg_npixels, size=npixels, replace=False).tolist()


def get_bkg_data(npixel, obsid, hp):
    src_center = hp.healpix_to_skycoord(npixel)
    bkg_npixels = get_bkg_npixels(src_center, hp.nside, npixels=100)

    bkg_data = rapidxmm.query_npixels(
        bkg_npixels, obstype="pointed", instrum="PN"
    )
    mask = bkg_data["obsid"] == obsid
    bkg_data = bkg_data[mask]

    if len(bkg_data) < 15:
        bkg_data = None

    return bkg_data


def stats_bootstrap(src, bkg, exp, eef, ecf, ac=None, nbkg=None, nsim=1000):
    # Calculate median and MAD for the stack using bootstraping
    nstack, npixels, nbands = src.shape
    cr = np.zeros((nsim, npixels, nbands))
    cr_err = np.zeros((nsim, npixels, nbands))
    snr = np.zeros((nsim, npixels, nbands))
    texp = np.zeros((nsim, npixels, nbands))
    ecf_sample = np.zeros((nsim, nbands))
    # msrc = np.zeros((nsim, npixels, nbands))
    # mbkg = np.zeros((nsim, npixels, nbands))
    # mexp = np.zeros((nsim, npixels, nbands))

    for i in range(nsim):
        idx_sample = np.random.randint(nstack, size=nstack)

        S = np.sum(src[idx_sample, :, :], axis=0)
        B = np.sum(bkg[idx_sample, :, :], axis=0)
        t = np.sum(exp[idx_sample, :, :], axis=0)

        if ac is None:
            Bcorr = np.sum(bkg[idx_sample, :, :] / nbkg[idx_sample, :, :], axis=0)
            ac = np.ones_like(bkg)
        else:
            Bcorr = np.sum(ac[idx_sample, :, :] * bkg[idx_sample, :, :], axis=0)

        cr[i, :, :] = (
            np.sum(src[idx_sample, :, :] / eef[idx_sample, :, :], axis=0) -
            np.sum(bkg[idx_sample, :, :] / eef[idx_sample, :, :], axis=0)
        ) / t
        cr_err[i, :, :] = np.sqrt(
            np.sum(src[idx_sample, :, :] / eef[idx_sample, :, :]**2, axis=0) +
            np.sum(ac[idx_sample, :, :] * bkg[idx_sample, :, :] / eef[idx_sample, :, :]**2, axis=0)
        ) / t
        snr[i, :, :] = (S - B) / np.sqrt(S + Bcorr)
        #snr[i, :, :] = cr[i, :, :] / cr_err[i, :, :]
        ecf_sample[i, :] = np.mean(ecf[idx_sample, :], axis=0)
        # msrc[i, :, :] = np.sum(src[idx_sample, :, :], axis=0)
        # mbkg[i, :, :] = np.sum(bkg[idx_sample, :, :], axis=0)
        # mexp[i, :, :] = np.sum(exp[idx_sample, :, :], axis=0)
        texp[i, :, :] = t

    cr_median = np.nanmedian(cr, axis=0)
    snr_median = np.nanmedian(snr, axis=0)
    ecf_median = np.nanmedian(ecf_sample, axis=0)
    texp_median = np.nanmedian(texp, axis=0)
    #cr_median = np.mean(cr, axis=0)
    #snr_median = np.mean(snr, axis=0)

    # src_median = np.median(msrc, axis=0)
    # bkg_median = np.median(mbkg, axis=0)
    # exp_median = np.median(mexp, axis=0)

    # kk1 = (src_median - bkg_median) / exp_median
    # kk2 = np.sqrt(src_median + bkg_median) / exp_median
    # kk3 = (src_median - bkg_median) / np.sqrt(src_median)

    cr_mad = np.zeros((npixels, nbands))
    snr_mad = np.zeros((npixels, nbands))
    for i in range(nbands):
        cr_mad[:, i] = median_abs_deviation(cr[:, :, i], axis=0, nan_policy="omit", scale="normal")
        snr_mad[:, i] = median_abs_deviation(snr[:, :, i], axis=0, nan_policy="omit", scale="normal")

    return cr_median, cr_mad, snr_median, snr_mad, ecf_median, texp_median


def flux_bootstrap(src_flux, src_flux_err, bkg_flux, bkg_flux_err, nsim=1000):
    nstack, nbands = src_flux.shape
    flux = np.zeros((nsim, nbands))
    flux_err = np.zeros((nsim, nbands))

    for i in range(nsim):
        idx_sample = np.random.randint(nstack, size=nstack)

        ngood = np.zeros(nbands, dtype=int)
        for j in range(nbands):
            good_idx = np.where(np.isfinite(src_flux[idx_sample, j]))
            ngood[j] = len(good_idx[0])

        flux[i, :] = (
            np.nansum(src_flux[idx_sample, :], axis=0) -
            np.nansum(bkg_flux[idx_sample, :], axis=0)
        )  / ngood

        flux_err[i, :] = np.sqrt(
            np.nansum(
                src_flux_err[idx_sample, :]**2 +
                bkg_flux_err[idx_sample, :]**2,
                axis=0
            )
        ) / ngood

    flux_median = np.median(flux, axis=0)
    flux_err_median = np.median(flux_err, axis=0)
    flux_mad = median_abs_deviation(flux, axis=0, scale="normal")

    return flux_median, flux_mad


def print_stats(cr, cr_err, snr, snr_err, texp, flux, flux_err, ebands=["6", "7", "8"]):
    color_print("\nStatistics", "yellow")
    color_print("----------", "yellow")

    for i, eband in enumerate(ebands):
        idx_max = np.argmax(cr[:, i])
        cr_peak = cr[idx_max, i]
        cr_peak_mad = cr_err[idx_max, i]
        
        texp_peak = texp[idx_max, i]

        idx_max = np.argmax(snr[:, i])
        snr_peak = snr[idx_max, i]
        snr_peak_mad = snr_err[idx_max, i]

        color_print(f"Energy band {eband}:", "white")
        print(f"Median net CR at peak: {cr_peak:.01e} ?? {cr_peak_mad:.01e} counts/s")
        print(f"Median exposure time at peak: {texp_peak:.01e} s")

        if flux is not None:
            f, ferr = flux[i], flux_err[i]
            print(f"Median flux: {f:.01e} ?? {ferr:.01e} erg/s/cm-2")

        print(f"Median SNR at peak: {snr_peak:.01f} ?? {snr_peak_mad:.01f}\n")


def print_params(parnames, params):
    color_print("\nAverage parameters", "yellow")
    color_print("------------------", "yellow")
    color_print("Weighted by number of repetitions in the stack")

    average_params = np.median(params, axis=0)
    for name, par in zip(parnames, average_params):
        color_print(f"{name}: {par:.04f}", "white")

    return average_params


def plot_stack(npixels, hp, cr, snr, filename=None, scale=None):
    lon, lat = hp.healpix_to_lonlat(npixels)
    boundaries = hp.boundaries_lonlat(npixels, 1)

    patches = []
    for blon, blat in zip(*boundaries):
        patches.append(Polygon(np.array([blon.value, blat.value]).T, closed=True))

    if not scale:
        vmin_cr, vmax_cr = cr.flatten().min(), cr.flatten().max()
        vmin_snr, vmax_snr = snr.flatten().min(), snr.flatten().max()
        scale = [vmin_cr, vmax_cr, vmin_snr, vmax_snr]
    else:
        vmin_cr, vmax_cr = scale[0], scale[1]
        vmin_snr, vmax_snr = scale[2], scale[3]

    norm_cr = Normalize(vmin=vmin_cr, vmax=vmax_cr)
    norm_snr = Normalize(vmin=vmin_snr, vmax=vmax_snr)

    fig, axs = plt.subplots(2, 3, constrained_layout=False, figsize=(5.5, 4))
    for i, eband in enumerate(["6", "7", "8"]):
        # Count-rate "images"
        pcm_cr = axs[0, i].scatter(
            lon, lat, c=cr[:, i], s=1, vmin=vmin_cr, vmax=vmax_cr
        )

        p = PatchCollection(patches, alpha=1)
        p.set_array(cr[:, i])
        p.set_norm(norm_cr)
        axs[0, i].add_collection(p)

        axs[0, i].set_title(f"Energy band {eband}")
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])

        # signal-to-noise ratio "images"
        pcm_snr = axs[1, i].scatter(
            lon, lat, c=snr[:, i], s=1, vmin=vmin_snr, vmax=vmax_snr
        )

        p = PatchCollection(patches, alpha=1)
        p.set_array(snr[:, i])
        p.set_norm(norm_snr)
        axs[1, i].add_collection(p)

        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

        if i == 0:
            axs[0, i].set_ylabel("Stack net CR (median)")
            axs[1, i].set_ylabel("Stack SNR (median)")

    plt.tight_layout()

    fig.colorbar(pcm_cr, ax=axs[0, :], shrink=0.6, location='bottom', pad=0.02)
    fig.colorbar(pcm_snr, ax=axs[1, :], shrink=0.6, location='bottom', pad=0.02)

    if filename:
        fig.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

    return scale


def plot_radial(npixels, level, hp, cr, cr_err, snr, snr_err, filename=None):
    radius = list(range(level + 1))
    cr_radial = np.zeros((level + 1, 3))
    cr_err_radial = np.zeros((level + 1, 3))
    snr_radial = np.zeros((level + 1, 3))
    snr_err_radial = np.zeros((level + 1, 3))

    cr_radial[0, :] = cr[0, :]
    cr_err_radial[0, :] = cr_err[0, :]
    snr_radial[0, :] = snr[0, :]
    snr_err_radial[0, :] = snr_err[0, :]

    npixel_neighbours = [npixels[0]]
    for i in range(1, level + 1):
        npixel_neighbours = list(set(hp.neighbours(npixel_neighbours).flatten()))
        mask = [p in npixel_neighbours for p in npixels]

        cr_radial[i] = np.sum(cr[mask], axis=0) / len(npixels[mask])
        cr_err_radial[i] = np.sqrt(np.sum(cr_err[mask]**2, axis=0)) / len(npixels[mask])
        snr_radial[i] = np.sum(snr[mask], axis=0) / len(npixels[mask])
        snr_err_radial[i] = np.sqrt(np.sum(snr_err[mask]**2, axis=0)) / len(npixels[mask])

    cr_min = np.nanmin(cr_radial - 1.1*cr_err_radial)
    cr_max = np.nanmax(cr_radial + 1.1*cr_err_radial)
    snr_min = np.nanmin(snr_radial - 1.1*snr_err_radial)
    snr_max = np.nanmax(snr_radial + 1.1*snr_err_radial)

    # filename_npz = filename.parent.joinpath(filename.stem + "_radial.npz")
    # np.savez(
    #     filename_npz,
    #     cr_radial=cr_radial,
    #     cr_err_radial=cr_err_radial,
    #     snr_radial=snr_radial,
    #     snr_err_radial=snr_err_radial,
    # )

    fig, axs = plt.subplots(
        2, 3, sharex=True, constrained_layout=False, figsize=(5.5, 3.5)
    )
    for i, eband in enumerate(["6", "7", "8"]):
        axs[0, i].errorbar(
            radius, cr_radial[:, i], yerr=cr_err_radial[:, i], fmt="o", capsize=2
        )
        axs[0, i].set_title(f"Energy band {eband}", size="x-small")
        axs[0, i].set_ylim(cr_min, cr_max)
        axs[0, i].ticklabel_format(
            axis="y", style="sci", scilimits=(0,0), useMathText=True
        )
        axs[0, i].xaxis.offsetText.set_fontsize(8)
        axs[0, i].grid(color='gray', linestyle=':')

        axs[1, i].errorbar(
            radius, snr_radial[:, i], yerr=snr_err_radial[:, i], fmt="o", capsize=2
        )
        axs[1, i].set_ylim(snr_min, snr_max)
        axs[1, i].grid(color='gray', linestyle=':')

        if i == 0:
            axs[0, i].set_ylabel("net counts / s / pixel")
            axs[1, i].set_ylabel("SNR / pixel")

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Distance to central npixel")

    plt.tight_layout()

    if filename:
        filename = filename.parent.joinpath(filename.stem + "_radial" +  filename.suffix)
        fig.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


def stack_npixels(
    npixels,
    level_neighbours=5,
    params=None,
    max_data=1000,
    calc_flux=True,
    use_flagged_pixels=False,
    skip_detections=False,
    custom_bkg=False,
    moc_masked_sources=None,
    order=16,
    with_plots=False,
    plotfile=None,
    scale=None,
):
    ecf_pn = {
        "6": ECF.ecf_det_eband("PN", "6"),
        "7": ECF.ecf_det_eband("PN", "7"),
        "8": ECF.ecf_det_eband("PN", "8"),
    }

    num_neighbours = sum([8*k for k in range(level_neighbours + 1)]) + 1

    ebands = ["6", "7", "8"]
    src_stack = np.zeros((max_data, num_neighbours, len(ebands)))
    bkg_stack = np.zeros((max_data, num_neighbours, len(ebands)))
    exp_stack = np.zeros((max_data, num_neighbours, len(ebands)))
    eef_stack = np.ones((max_data, num_neighbours, len(ebands)))
    ac_stack = np.zeros((max_data, num_neighbours, len(ebands)))
    npixels_bkg_stack = np.ones((max_data, num_neighbours, len(ebands)))
    ecf_stack = np.zeros((max_data, len(ebands)))

    if calc_flux:
        src_flux_center = np.full((max_data, len(ebands)), np.nan)
        bkg_flux_center = np.full((max_data, len(ebands)), np.nan)
        src_flux_err_center = np.full((max_data, len(ebands)), np.nan)
        bkg_flux_err_center = np.full((max_data, len(ebands)), np.nan)

    if params:
        params_stack = np.zeros((max_data, len(params.colnames)))

    hp = HEALPix(nside=2 ** order, order="nested", frame=FK5())

    n, nsrc = 0, 0
    for j, npixel in enumerate(tqdm(npixels)):
        sorted_neighbours = get_neighbours(npixel, hp, level=level_neighbours)
        data = rapidxmm.query_npixels(
            sorted_neighbours["npixel"], obstype="pointed", instrum="PN"
        )

        if len(data) == 0:
            continue

        nsrc += 1
        data = data.group_by(["obsid", "instrum"])

        for group in data.groups:
            data_obs_order = join(
                sorted_neighbours, group, keys=["npixel"], join_type="left"
            )
            data_obs_order.sort("order")

            if skip_detections:
                if np.any(data_obs_order["band8_flags"] >= 8):
                    continue

            if custom_bkg:
                bkg_data = get_bkg_data(npixel, group["obsid"][0], hp)

                if bkg_data is None:
                    # We couldn't find a good background region for this npixel,
                    # so it's rejected from the stack
                    continue

            for i, eband in enumerate(ebands):
                if use_flagged_pixels:
                    mask = [True] * len(sorted_neighbours)
                else:
                    mask = data_obs_order[f"band{eband}_flags"] == 0

                src_stack[n, mask, i] = data_obs_order[f"band{eband}_src_counts"][mask]
                exp_stack[n, mask, i] = data_obs_order[f"band{eband}_exposure"][mask]
                eef_stack[n, mask, i] = data_obs_order["eef"][mask]
                ac_stack[n, mask, i] = data_obs_order["area_ratio"][mask]

                if custom_bkg:
                    mask_bkg = bkg_data[f"band{eband}_flags"] == 0

                    # The same average bkg value is assigned to all npixels in the detection
                    bkg_counts = bkg_data[f"band{eband}_bck_counts"][mask_bkg]
                    bkg_stack[n, mask, i] = np.mean(bkg_counts)
                    npixels_bkg_stack[n, mask, i] = len(bkg_counts)
                else:
                    bkg_stack[n, mask, i] = data_obs_order[f"band{eband}_bck_counts"][mask]

                if calc_flux and np.any(mask):
                    ecf_stack[n, i] = ecf_pn[eband][group["filt"][0]].get_ecf(params["NHGAL"][j], 1.9)
                    exp = np.mean(exp_stack[n, mask, i])
                    ngood = len(exp_stack[n, mask, i])

                    src_flux_center[n, i] = (
                        np.sum(src_stack[n, mask, i])
                        / exp / ecf_stack[n, i] / 1e11 / ngood
                    )
                    src_flux_err_center[n, i] = (
                        np.sqrt(np.sum(src_stack[n, mask, i]))
                        / exp / ecf_stack[n, i] / 1e11 / ngood
                    )

                    if custom_bkg:
                        exp_bkg = np.mean(bkg_data[f"band{eband}_exposure"][mask_bkg])
                        ngood_bkg = len(bkg_data[f"band{eband}_exposure"][mask_bkg])

                        bkg_flux_center[n, i] = (
                            np.sum(bkg_counts)
                            / exp_bkg / ecf_stack[n, i] / 1e11 / ngood_bkg
                        )
                        bkg_flux_err_center[n, i] = (
                            np.sqrt(np.sum(bkg_counts))
                            / exp_bkg / ecf_stack[n, i] / 1e11 / ngood_bkg
                        )
                    else:
                        bkg_flux_center[n, i] = (
                            np.sum(bkg_stack[n, mask, i])
                            / exp / ecf_stack[n, i] / 1e11 / ngood
                        )
                        bkg_flux_err_center[n, i] = (
                            np.sqrt(np.sum(bkg_stack[n, mask, i]))
                            / exp / ecf_stack[n, i] / 1e11 / ngood
                        )

            if params:
                for i, col in enumerate(params.colnames):
                    params_stack[n, i] = params[col][j]

            n += 1

    src_stack = src_stack[:n, :, :]
    bkg_stack = bkg_stack[:n, :, :]
    exp_stack = exp_stack[:n, :, :]
    ecf_stack = ecf_stack[:n, :]

    if custom_bkg:
        # No need to take into account the area correction when using custom
        # backgrounds, since counts are extracted in regions with the same size
        ac_stack = None
        npixels_bkg_stack = npixels_bkg_stack[:n, :]
    else:
        ac_stack = ac_stack[:n, :]
        npixels_bkg_stack = None

    if n < 2:
        return None, None, None, None, None, None, None

    cr, cr_mad, snr, snr_mad, ecf, texp = stats_bootstrap(
        src_stack, bkg_stack, exp_stack, eef_stack, ecf_stack, ac_stack, npixels_bkg_stack, nsim=1000
    )

    flux, flux_mad = None, None
    flux2, flux2_mad = None, None
    if calc_flux:
        src_flux_center = src_flux_center[:n, :]
        src_flux_err_center = src_flux_err_center[:n, :]
        bkg_flux_center = bkg_flux_center[:n, :]
        bkg_flux_err_center = bkg_flux_err_center[:n, :]

        flux, flux_mad = flux_bootstrap(
            src_flux_center,
            src_flux_err_center,
            bkg_flux_center,
            bkg_flux_err_center,
            nsim=1000
        )

        flux2 = np.mean(cr, axis=0) / ecf / 1e11
        flux2_mad = np.sqrt(np.mean(cr_mad**2, axis=0)) / ecf / 1e11

    if with_plots:
        scale = plot_stack(
            sorted_neighbours["npixel"], hp, cr, snr, plotfile, scale
        )

        plot_radial(
            sorted_neighbours["npixel"],
            level_neighbours,
            hp,
            cr,
            cr_mad,
            snr,
            snr_mad,
            plotfile
        )

    print_stats(
        cr, cr_mad, snr, snr_mad, texp, flux, flux_mad
    )

    if params:
        average_params = print_params(params.colnames, params_stack[:n, :])
    else:
        average_params = None

    return flux, flux_mad, flux2, flux2_mad, average_params, scale, n, nsrc
