# -*- coding: utf-8 -*-
"""
API for querying the XMM-Newton Upper Limit Database using the XSA interface

@author: ruizca
"""
import requests
from astropy.coordinates import ICRS
from astropy.table import Table
from astropy_healpix import HEALPix
from pandas import DataFrame as df


def _check_kwargs(obstype, instrum):
    if obstype not in ["slew", "pointed", None]:
        raise ValueError(f"Unknown obstype: {obstype}")

    if instrum not in ["PN", "M1", "M2", None]:
        raise ValueError(f"Unknown instrum: {instrum}")


def _filter_uls(uls, obstype, instrum):
    if obstype:
        mask = uls["obstype"] == obstype
        uls = uls[mask]

    if instrum:
        mask = uls["instrum"] == instrum
        uls = uls[mask]

    return uls


def _get_level(obstype):
    if obstype == "pointed":
        return 16
    else:
        return 15


def _npixels_to_coords(npixels, level):
    hp = HEALPix(nside=2**level, order="nested", frame=ICRS())
    return hp.healpix_to_skycoord(npixels)


def _query_xsa_uls(payload, obstype=None, instrum=None):
    _check_kwargs(obstype, instrum)

    r = requests.get(
        "http://nxsa.esac.esa.int/nxsa-sl/servlet/get-uls", params=payload
    )
    r.raise_for_status()

    try:
        uls = Table.from_pandas(df.from_records(r.json()))
    except Exception:
        print(r.text)

    if uls:
        uls = _filter_uls(uls, obstype, instrum)

    return uls


def query_radec(ra, dec, **kwargs):
    """
    Search for upper limits at the coordinates defined by ra, dec.

    Parameters
    ----------
    ra : List
        List-like of Right Ascension values, in degrees.
    dec : List
        List-like of Declination values, in degrees.
    obstype : str, optional
        Filter the results by observation type ("pointed" or "slew").
    instrum : str, optional
        Filter the results by instrument ("PN, "M1" or"M2"). The default is None.

    Returns
    -------
    Astropy Table
        An Astropy Table with the upper limit data for each observation
        containing the positions for the list of npixels.

    """
    if len(ra) != len(dec):
        raise ValueError("ra and dec lengths don't match!")

    payload = {
        "ra": ";".join(str(round(c, 10)) for c in ra),
        "dec": ";".join(str(round(c, 10)) for c in dec),
    }

    return _query_xsa_uls(payload, **kwargs)


def query_coords(coords, **kwargs):
    """
    Search for upper limits at the coordinates defined by "coords"

    Parameters
    ----------
    coords : Astropy SkyCoord
    obstype : str, optional
        Filter the results by observation type ("pointed" or "slew").
    instrum : str, optional
        Filter the results by instrument ("PN, "M1" or"M2"). The default is None.

    Returns
    -------
    Astropy Table
        An Astropy Table with the upper limit data for each observation
        containing the positions for the list of npixels.

    """
    payload = {
        "ra": ";".join(str(round(c.ra.deg, 10)) for c in coords),
        "dec": ";".join(str(round(c.dec.deg, 10)) for c in coords),
    }

    return _query_xsa_uls(payload, **kwargs)


def query_npixels(npixels, obstype="pointed", instrum=None):
    """
    Search for upper limits at the coordinates corresponding to "npixels", a
    list-like of integers corresponding to npixel values in the nested ordering
    scheme.

    Parameters
    ----------
    npixels : TYPE
        DESCRIPTION.
    obstype : str, optional
        The observation type ("pointed" or "slew"). For pointed observations
        npixels are assumed to use order=16, and order=15 for slew observations.
        The default is "pointed".
    instrum : str, optional
        Filter the results by instrument ("PN, "M1" or"M2"). The default is None.

    Returns
    -------
    Astropy Table
        An Astropy Table with the upper limit data for each observation
        containing the positions for the list of npixels.
    """
    level = _get_level(obstype)
    coords = _npixels_to_coords(npixels, level)

    return query_coords(coords, obstype=obstype, instrum=instrum)
