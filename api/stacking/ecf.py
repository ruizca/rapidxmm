# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:52:33 2020

@author: ruizca
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline


_ecf = np.load("api/stacking/ecf.npy", allow_pickle="TRUE").item()


class ECF:
    def __init__(self, detector, filt, eband):
        filt = self._parse_filter(filt)
        eband = self._parse_eband(eband)

        self._lognh_grid = _ecf[filt][eband][detector]["lognh"]
        self._gamma_grid = _ecf[filt][eband][detector]["gamma"]
        self._interpolator = self._set_interpolator(detector, filt, eband)

    def get_ecf(self, nh, gamma):
        lognh = np.log10(nh)

        # Keep values of lognh and gamma between interpolation limits
        lognh = np.maximum(lognh, self._lognh_grid[0])
        lognh = np.minimum(lognh, self._lognh_grid[-1])

        gamma = np.maximum(gamma, self._gamma_grid[0])
        gamma = np.minimum(gamma, self._gamma_grid[-1])

        return self._interpolator.ev(lognh, gamma) / 1e11

    @classmethod
    def ecf_det_eband(cls, detector, eband):
        return {
            "Thin1": cls(detector, "Thin", eband),
            "Thin2": cls(detector, "Thin", eband),
            "Medium": cls(detector, "Medium", eband),
            "Thick": cls(detector, "Thick", eband),
        }

    def _set_interpolator(self, detector, filt, eband):
        return RectBivariateSpline(
            self._lognh_grid,
            self._gamma_grid,
            np.array(_ecf[filt][eband][detector]["ecf"]),
            kx=1,
            ky=1,
        )

    @staticmethod
    def _parse_filter(obsfilter):
        if obsfilter == "Thin1" or obsfilter == "Thin2" or obsfilter == "Thin":
            obsfilter = "thin"

        if obsfilter == "Medium":
            obsfilter = "med"

        return obsfilter.lower()

    @staticmethod
    def _parse_eband(eband):
        if eband == "6":
            eband = "SOFT"

        elif eband == "7":
            eband = "HARD"

        elif eband == "8":
            eband = "FULL"

        else:
            raise ValueError

        return eband.lower()
