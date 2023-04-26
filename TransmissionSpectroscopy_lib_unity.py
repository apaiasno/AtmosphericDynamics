# TransmissionSpectroscopy_lib_unity.py
# Library of routines for transmission spectroscopy on the Unity cluster (no petitRADTRANS).
 
# Imports
import os

import numpy as np

import scipy
import scipy.ndimage
import scipy.interpolate
 
import astropy.stats
import astropy.units as u
import astropy.constants as const

import matplotlib.pyplot as plt
 
import pandas as pd

import urllib.request
 
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

### Basic data manipulation routines ###

def get_indices(x, x_low, x_up):
    ''' Finds lower and upper index of specific range in array.
        
	Parameters
	----------
        x : 1-D numpy array 
            Spectrum wavelengths
        x_low : float
            Lower bound of range of interest
        x_up: float
            Upper bound of range of interest
        
        Returns
        -------
        i_low : float
            Lower bound index
        i_up : float
            Upper bound index
    '''
    i_low = np.where(x >= x_low)[0][0]
    i_up = np.where(x <= x_up)[0][-1]
    return i_low, i_up


def interpolate_xy(x, y, x_new, fill_value='extrapolate'):
    ''' Interpolate y values to new grid of x values.

        Parameters
        ----------
        x : 1-D numpy array 
            Current x values
        y : 1-D numpy array 
            Current y values
        x_new : 1-D numpy array 
            New x values to interpolate onto

        Returns
        -------
        y_new : 1-D numpy array 
            Interpolated y values
    '''
    quadinterp = scipy.interpolate.interp1d(x, y, kind='slinear', bounds_error=False, fill_value=fill_value)
    return quadinterp(x_new)

### Time-related routines ###

def extrapolate_mid_transit_linear(time_mid_0, P, t_start, t_end):
    ''' Extrapolates mid-transit time to epoch of observation based on orbital period.
        
        Parameters
        ----------
        time_mid_0 : tuple of floats (1 x 3) 
            Mid-transit time ephemeris and its errors in JD (or some day units)
        P : tuple of floats (1 x 3) 
            Period and its errors in days
        t_start : float 
            Observation start time in same units as ephemeris
        t_end : float 
            Observation end time in same units as ephemeris
        
        Returns
        -------
        time_mid : tuple of floats (1 x 3) 
            Mid-transit time during epoch of observation and its error in same units as ephemeris
    '''
    E = 0
    time_mid = np.copy(time_mid_0[0])

    # Take maximum of errors on period (as representative error)
    P_err = np.max(P[1:])

    # If ephemeris is later than epoch of observations
    if time_mid > t_end:
        while time_mid > t_end:
            E -= 1
            time_mid = time_mid_0[0] + E*P[0]
            time_mid_err = np.sqrt(time_mid_0[-1]**2 + (E * P_err)**2)

    # If ephemeris is earlier than epoch of observations
    elif time_mid < t_start:
        while time_mid < t_start:
            E += 1
            time_mid = time_mid_0[0] + E*P[0]
            time_mid_err = np.sqrt(time_mid_0[1]**2 + (E * P_err)**2)

    # If ephemeris is within the epoch of observations
    else:
        time_mid, time_mid_err = np.copy(time_mid_0[0])*u.day, np.max(time_mid_0[1:])

    return time_mid, time_mid_err

def extrapolate_mid_transit_quadratic(time_mid_0, P, c, t_start, t_end):
    ''' Extrapolates mid-transit time to epoch of observation based on orbital period.
        
        Parameters
        ----------
        time_mid_0 : tuple of floats (1 x 3) 
            Mid-transit time ephemeris and its error in JD (or some day units)
        P : tuple of floats (1 x 3) 
            Period and its error in days
        c : tuple of floats (1 x 3) 
            Quadratic coefficient and its error
        t_start : float 
            Observation start time in same units as ephemeris
        t_end : float
            Observation end time in same units as ephemeris
        
        Returns
        -------
        time_mid : tuple of floats (1 x 3)
            Mid-transit time during epoch of observation and its error in same units as ephemeris
    '''
    E = 0
    time_mid = np.copy(time_mid_0[0])

    # Take maximum of errors on period and quadratic coefficient (as representative error)
    P_err = np.max(P[1:])
    c_err = np.max(c[1:])

    # If ephemeris is later than epoch of observations
    if time_mid > t_end:
        while time_mid > t_end:
            E -= 1
            time_mid = time_mid_0[0] + E*P[0] + c[0]*E**2.
            time_mid_err = np.sqrt(time_mid_0[-1]**2 + (E * P_err)**2 + ((E**2)*c_err)**2)

    # If ephemeris is earlier than epoch of observations
    elif time_mid < t_start:
        while time_mid < t_start:
            E += 1
            time_mid = time_mid_0[0] + E*P[0] + c[0]*E**2.
            time_mid_err = np.sqrt(time_mid_0[1]**2 + (E * P_err)**2 + ((E**2)*c_err)**2)

    # If ephemeris is within the epoch of observations
    else:
        time_mid, time_mid_err = np.copy(time_mid_0[0]), np.max(time_mid_0[1:])

    return time_mid, time_mid_err

### Wavelength related routines ###

def air2vacuum(wav_air):
    ''' Converts wavelengths from air to vacuum.
        
        Parameters
        ----------
        wav_air : 1-D numpy array
            List of wavelengths in air
        
        Returns
        -------
        wav_vacuum : 1-D numpy array
            List of wavelengths in vacuum
    '''
    s = (10.**4)/wav_air
    n = 1 + 0.00008336624212083 + (0.02408926869968/(130.1065924522 - s**2)) + (0.0001599740894897/(38.92568793293 - s**2))
    wav_vacuum = n * wav_air
    return wav_vacuum

def vacuum2air(wav_vacuum):
    ''' Converts wavelengths from vacuum to air.
        
        Parameters
        ----------
        wav_vacuum : 1-D numpy array
            List of wavelengths in vacuum

        Returns
        -------
        wav_air : 1-D numpy array
            List of wavelengths in air
    '''
    s = (10.**4)/wav_vacuum
    n = 1 + 0.0000834254 + (0.02406147/(130 - s**2)) + (0.00015998/(38.9 - s**2))
    wav_air = wav_vacuum/n
    return wav_air


### Velocity-related routines ###

def doppler_velocity(lambda_obs, lambda_0):
    ''' Computes Doppler velocity of an observed wavelength relative to a reference wavelength.
        
        Parameters
        ----------
        lambda_obs : float or numpy array
            Observed wavelengths in units of length
        lambda_ref : float or numpy array 
            Reference wavelengths with matching units as target wavelength array

        Returns
        -------
        vel : float or numpy array
            Doppler velocities with same shape as observed wavelength array
    '''
    c = const.c.to(u.km/u.s).value
    vel = ((lambda_obs - lambda_0)/lambda_0) * c
    return vel

def RV_semiAmplitude(M_star, m_p, a_p, e, i_orbit):
    ''' Computes stellar RV semi-amplitude of a planet-hosting star.

        Parameters
        ----------
        M_star : float in units of mass
            Stellar mass
        m_p : float in units of mass
            Planetary mass
        a_p : float in units of distance
            Planetary semi-major axis
        e : float, unitless
            Planetary orbital eccentricity
        i_orbit : float in units of angle
            Planetary orbital inclination
    
        Returns
        -------
        K_star : float in units of velocity
            Stellar RV semi-amplitude

    '''
    K_star = np.sqrt(const.G/(1-e**2.)) * m_p.to(u.kg)*np.sin(i_orbit.to(u.deg)) * (M_star.to(u.kg)+m_p.to(u.kg))**(-1/2) * (a_p.to(u.m))**(-1/2)
    return K_star

def RV_circular_orbit(x, K, v_sys, ephemeris=None):
    ''' Computes the radial velocity of a circular orbit given orbital parameters.
        
        Parameters
        ----------
        x : float or 1-D numpy array
            Time or phase
        K : float
            RV semi-amplitude (in velocity units)
        v_sys : float
            Systemic velocity (in velocity units)
        ephemeris: None or tuple
            None if x is in phase; else tuple of mid-transit time and orbital period (in same units as x)
        Outputs:
        RV : float or 1-D numpy array
            Radial velocity (in velocity units)
    '''
    if ephemeris == None:
        RV = K * np.sin(2*np.pi*x) + v_sys
        return RV
    else:
        t_mid, P = ephemeris
        arg = 2 * np.pi * (x-t_mid)/P * u.rad
        RV = K * np.sin(arg) + v_sys     
        return RV

### Transmission signal modelling routines ###

def dopplerShadow_RV(phases_in, slope, offset):
    ''' Computes RV of Doppler Shadow as a function of orbital phase assuming RV varies linearly with phase.
        
        Parameters
        ----------
        phases_in : 1-D numpy array
            In-transit orbital phases over which to calculate RV
        slope : float
            Slope of function for determining how RV changes with orbital phase
        offset : float
            Offset in function for determining how RV changes with orbital phase
        
        Returns
        -------
        RVs : 1-D numpy array
            RVs of Doppler shadow feature
    '''
    RVs_shadow = slope * phases_in + offset
    return RVs_shadow 

def model_planet_transit(params, vels, phases_in):
    ''' Simultaneously models in-transit planet absorption and Doppler shadow.
        
        Parameters
        ----------
        params: 1-D numpy array
            Planet absorption model parameters --> K_p, v_offset, amp, width
            Doppler shadow model parameters --> slope, offset, amp_shadow, amp_width
        vels : 1-D numpy array
            Velocities associated with data
        phases_in: 1-D numpy array 
            In-transit phases associated with data        
        
        Returns
        -------
        transit_signal: 2-D numpy array
            Modelled planet absorption and Doppler shadow
    '''
    K_p, v_offset, amp, width, slope, offset, amp_shadow, width_shadow = params
    vels_diff = np.mean(np.diff(vels))
    planet_vels = RV_circular_orbit(phases_in, K_p, v_offset)
    shadow_vels = dopplerShadow_RV(phases_in, slope, offset)
    transit_signal = np.ones((len(phases_in), len(vels)))
    for i in range(len(transit_signal)):
        absorption = astropy.modeling.functional_models.Gaussian1D(amplitude=amp, mean=planet_vels[i],
                                                                   stddev=width)
        shadow = astropy.modeling.functional_models.Gaussian1D(amplitude=amp_shadow, mean=shadow_vels[i],
                                                                   stddev=width_shadow)
        transit_signal[i] = shadow(vels) - absorption(vels)
    return transit_signal

### MCMC sampling routines ###

def logPrior(theta, priors, priors_bool):
    ''' Calculates gaussian/uniform priors for MCMC fitting.
        
        Parameters
        ----------
        theta : 1-D numpy array
            Model parameters
        priors : 2-D numpy array 
            Pairs of values representing prior ranges for each model parameter
        priors_bool : 1-D boolean array
            True if uniform prior, False if Gaussian prior
        
        Returns
        -------
        prior_norm : float 
            Prior probability
    '''
    priors_uniform = priors[priors_bool]
    theta_uniform = theta[priors_bool]
    prior_norm = 0
    # Uniform priors
    for i, prior in enumerate(priors_uniform):
        lower, upper = prior
        if not (lower < theta_uniform[i] < upper):
            prior_norm += -np.inf
        else:
            prior_norm += np.log(1./(upper-lower))
    # Gaussian priors
    priors_gauss = priors[~priors_bool]
    priors_gauss_center, priors_gauss_width = priors_gauss[:,0], priors_gauss[:,1]
    theta_gauss = theta[~priors_bool]
    coef = np.sum(-np.log(priors_gauss_width*np.sqrt(2.*np.pi)))
    chi2 = np.sum(((theta_gauss-priors_gauss_center)/priors_gauss_width)**2.)
    exponent = -chi2/2.
    prior_norm += coef + exponent
    return prior_norm#/len(theta)

def logLikelihood(theta, x, data, data_err, model_func, y=None, args=None):
    ''' Calculates chi-squared likelihood for MCMC fitting.

        Parameters
        ----------
        theta : 1-D numpy array 
            Model parameters
        x : 1-D numpy array
            Data x coordinates
        data : 1-D/2-D numpy array
            Data values
        data_err : 1-D/2-D numpy array
            Errors associated with data
        model_func : function 
            Model function
        y : 1-D numpy array
            Data y coordinates; None if data is 1-D
        args : 1-D numpy array
            Additional arguments taken in by model; None if model does not take in additional arguments

        Returns
        -------
        likelihood : float 
            Likelihood probability
    '''
    scale = np.product(np.shape(data))
    if y is None:
        if args is None:
            model = model_func(theta, x)
        else:
            model = model_func(theta, x, *args)
    else:
        if args is None:
            model = model_func(theta, x, y)
        else:
            model = model_func(theta, x, y, *args)
    chi2 = np.sum(((model - data)/data_err)**2)
    if np.isnan(chi2):
        likelihood = -np.inf
    else:
        likelihood = -0.5 * (scale*np.log(2*np.pi) + np.sum(2*np.log(data_err)) + chi2)
#        likelihood = np.sum(-np.log(data_err*np.sqrt(2.*np.pi)))-chi2/2.
#         likelihood = -chi2/2.
    return likelihood    
    
def logPosterior(theta, priors, priors_bool, x, data, data_err, model_func, y=None, args=None):
    ''' Calculates posterior for MCMC fitting.
        
        Parameters
        ----------
        theta : 1-D numpy array
            Model parameters
        priors : 2-D numpy array 
            Pairs of values representing prior ranges for each model parameter
        priors_bool : 1-D boolean array
            True if uniform prior, False if Gaussian prior
        x : 1-D numpy array
            Data x coordinates
        data : 1-D/2-D numpy array
            Data values
        data_err : 1-D/2-D numpy array
            Errors associated with data
        model_func : function 
            Model function
        y : 1-D numpy array
            Data y coordinates; None if data is 1-D
        args : 1-D numpy array
            Additional arguments taken in by model; None if model does not take in additional arguments
            
       Returns
       ------- 
       post : float
            Posterior probability
    '''
    scale = np.product(np.shape(data))
    prior = logPrior(theta, priors, priors_bool)
    likelihood = logLikelihood(theta, x, data, data_err, model_func, y, args)
    post = scale*prior + likelihood        
#    post = prior + likelihood
    return post

### class: FluxMap ###

class FluxMap:

    ## Initialization ##
    
    def __init__(self, config_filename):
        ''' Reads in planet config file of system parameters and initializes FluxMap. Tuples are in the order of (value, positive error, negative error).
            
            Parameters
            ----------
            config_filename : string 
                Filename of planet config file for reading in system parameters.
            
            Returns
            -------
            Initializes a FluxMap object with system properties
        '''
        df = pd.read_csv(config_filename, delim_whitespace=True, comment='#', header=None)
        
        # Read in parameters from planet config file
        self.M_star = np.array(df.loc[df[0]=='M_star', 1:3].values[0], dtype=float) * u.M_sun
        self.R_star = np.array(df.loc[df[0]=='R_star', 1:3].values[0], dtype=float) * u.R_sun
        self.T_eff = np.array(df.loc[df[0]=='T_eff', 1:3].values[0], dtype=float) * u.K
        self.vsini = np.array(df.loc[df[0]=='vsini', 1:3].values[0], dtype=float) * u.km/u.s
        self.M_p = np.array(df.loc[df[0]=='M_p', 1:3].values[0], dtype=float) * u.M_jup
        self.R_p = np.array(df.loc[df[0]=='R_p', 1:3].values[0], dtype=float) * u.R_jup
        self.i_p = np.array(df.loc[df[0]=='i_p', 1:3].values[0], dtype=float) * u.deg
        self.l_p = np.array(df.loc[df[0]=='l_p', 1:3].values[0], dtype=float) * u.deg
        self.P = np.array(df.loc[df[0]=='P', 1:3].values[0], dtype=float) * u.day
        self.tau = np.array(df.loc[df[0]=='tau', 1:3].values[0], dtype=float) * u.day
        self.T_14 = np.array(df.loc[df[0]=='T_14', 1:3].values[0], dtype=float) * u.day
        self.T_0 = np.array(df.loc[df[0]=='T_0', 1:3].values[0], dtype=float) * u.day
        self.time_system = df.loc[df[0]=='time_system', 1].values[0]
        self.ra = float(df.loc[df[0]=='ra', 1].values[0]) * u.deg
        self.dec = float(df.loc[df[0]=='dec', 1].values[0]) * u.deg

    @property
    def M_star(self):
        '''
        Type : float 

        Mass of the host star.
        '''
        return self._M_star

    @M_star.setter
    def M_star(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'M_star' must be an astropy Quantity. Got: %s" % (type(value)))
        elif np.sum(value<0):
            raise ValueError("Attribute 'M_star' must be a positive value. Got: %s" % (value))
        elif value.unit != u.M_sun:
            raise ValueError("Attribute 'M_star' must be in units of solMass. Got: %s" % (value.unit))
        else:
            self._M_star = value

    @property
    def R_star(self):
        '''
        Type : float 

        Radius of the host star.
        '''
        return self._R_star

    @R_star.setter
    def R_star(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'R_star' must be an astropy Quantity. Got: %s" % (type(value)))
        elif np.sum(value<0):
            raise ValueError("Attribute 'R_star' must be a positive value. Got: %s" % (value))
        elif value.unit != u.R_sun:
            raise ValueError("Attribute 'R_star' must be in units of solRad. Got: %s" % (value.unit))
        else:
            self._R_star = value
    
    @property
    def T_eff(self):
        '''
        Type : float 

        Effective temperature of the host star.
        '''
        return self._T_eff

    @T_eff.setter
    def T_eff(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'T_eff' must be an astropy Quantity. Got: %s" % (type(value)))
        elif np.sum(value<0):
            raise ValueError("Attribute 'T_eff' must be a positive value. Got: %s" % (value))
        elif value.unit != u.K:
            raise ValueError("Attribute 'T_eff' must be in units of solMass. Got: %s" % (value.unit))
        else:
            self._T_eff = value
    
    @property
    def vsini(self):
        '''
        Type : float 

        Projected rotational velocity of the host star.
        '''
        return self._vsini

    @vsini.setter
    def vsini(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'vsini' must be an astropy Quantity. Got: %s" % (type(value)))
        elif np.sum(value<0):
            raise ValueError("Attribute 'vsini' must be a positive value. Got: %s" % (value))
        elif value.unit != u.km/u.s:
            raise ValueError("Attribute 'vsini' must be in units of km/s. Got: %s" % (value.unit))
        else:
            self._vsini = value
    
    @property
    def M_p(self):
        '''
        Type : float 

        Mass of the planet.
        '''
        return self._M_p

    @M_p.setter
    def M_p(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'M_p' must be an astropy Quantity. Got: %s" % (type(value)))
        elif np.sum(value<0):
            raise ValueError("Attribute 'M_p' must be a positive value. Got: %s" % (value))
        elif value.unit != u.M_jup:
            raise ValueError("Attribute 'M_p' must be in units of jupiterMass. Got: %s" % (value.unit))
        else:
            self._M_p = value
    
    @property
    def R_p(self):
        '''
        Type : float 

        Radius of the planet.
        '''
        return self._R_p

    @R_p.setter
    def R_p(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'R_p' must be an astropy Quantity. Got: %s" % (type(value)))
        elif np.sum(value<0):
            raise ValueError("Attribute 'R_p' must be a positive value. Got: %s" % (value))
        elif value.unit != u.R_jup:
            raise ValueError("Attribute 'R_p' must be in units of jupiterRad. Got: %s" % (value.unit))
        else:
            self._R_p = value
    
    @property
    def i_p(self):
        '''
        Type : float 

        Orbital inclination of the planet.
        '''
        return self._i_p

    @i_p.setter
    def i_p(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'i_p' must be an astropy Quantity. Got: %s" % (type(value)))
        elif value.unit != u.deg:
            raise ValueError("Attribute 'i_p' must be in units of deg. Got: %s" % (value.unit))
        else:
            self._i_p = value
    
    @property
    def l_p(self):
        '''
        Type : float 

        Spin-orbit alignment between host star's rotation and planet's orbit.
        '''
        return self._l_p

    @l_p.setter
    def l_p(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'l_p' must be an astropy Quantity. Got: %s" % (type(value)))
        elif value.unit != u.deg:
            raise ValueError("Attribute 'l_p' must be in units of deg. Got: %s" % (value.unit))
        else:
            self._l_p = value
    
    @property
    def P(self):
        '''
        Type : float 

        Orbital period of the planet.
        '''
        return self._P

    @P.setter
    def P(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'P' must be an astropy Quantity. Got: %s" % (type(value)))
        elif np.sum(value<0):
            raise ValueError("Attribute 'P' must be a positive value. Got: %s" % (value))
        elif value.unit != u.day:
            raise ValueError("Attribute 'P' must be in units of day. Got: %s" % (value.unit))
        else:
            self._P = value
    
    @property
    def tau(self):
        '''
        Type : float 

        Ingress/Egress duration of the planet's transit.
        '''
        return self._tau

    @tau.setter
    def tau(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'tau' must be an astropy Quantity. Got: %s" % (type(value)))
        elif np.sum(value<0):
            raise ValueError("Attribute 'tau' must be a positive value. Got: %s" % (value))
        elif value.unit != u.day:
            raise ValueError("Attribute 'tau' must be in units of day. Got: %s" % (value.unit))
        else:
            self._tau = value
    
    @property
    def T_14(self):
        '''
        Type : float 

        Total duration from 1st to 4th contact of the planet's transit.
        '''
        return self._T_14

    @T_14.setter
    def T_14(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'T_14' must be an astropy Quantity. Got: %s" % (type(value)))
        elif np.sum(value<0):
            raise ValueError("Attribute 'T_14' must be a positive value. Got: %s" % (value))
        elif value.unit != u.day:
            raise ValueError("Attribute 'T_14' must be in units of day. Got: %s" % (value.unit))
        else:
            self._T_14 = value
    
    @property
    def T_0(self):
        '''
        Type : float 

        Mid-transit time of the planet's transit.
        '''
        return self._T_0

    @T_0.setter
    def T_0(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'T_0' must be an astropy Quantity. Got: %s" % (type(value)))
        elif np.sum(value<0):
            raise ValueError("Attribute 'T_0' must be a positive value. Got: %s" % (value))
        elif value.unit != u.day:
            raise ValueError("Attribute 'T_0' must be in units of day. Got: %s" % (value.unit))
        else:
            self._T_0 = value
    
    @property
    def time_system(self):
        '''
        Type : float 

        Mass of the host star.
        '''
        return self._time_system

    @time_system.setter
    def time_system(self, value):
        if not isinstance(value, str):
            raise TypeError("Attribute 'time_system' must be a string. Got: %s" % (type(value)))
        elif not ((value == 'BJD_TDB') or (value == 'JD_UTC')):
            raise ValueError("Attribute 'time_system' must be either 'BJD_TDB' or 'JD_UTC'. Got: %s" % (value))
        else:
            self._time_system = value
   
    @property
    def ra(self):
        '''
        Type : float 

        Right ascension of target system.
        '''
        return self._ra

    @ra.setter
    def ra(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'ra' must be an astropy Quantity. Got: %s" % (type(value)))
        elif np.sum(value<0):
            raise ValueError("Attribute 'ra' must be a positive value. Got: %s" % (value))
        elif value.unit != u.deg:
            raise ValueError("Attribute 'ra' must be in units of deg. Got: %s" % (value.unit))
        else:
            self._ra = value
    
    @property
    def dec(self):
        '''
        Type : float 

        Declination of target system.
        '''
        return self._dec

    @dec.setter
    def dec(self, value):
        if not isinstance(value, u.quantity.Quantity):
            raise TypeError("Attribute 'dec' must be an astropy Quantity. Got: %s" % (type(value)))
        elif np.sum(value<0):
            raise ValueError("Attribute 'dec' must be a positive value. Got: %s" % (value))
        elif value.unit != u.deg:
            raise ValueError("Attribute 'dec' must be in units of day. Got: %s" % (value.unit))
        else:
            self._dec = value

    ## Static methods ##
    
    @staticmethod
    def convertTimingSystem(times, ra, dec, time_system_current='JD_UTC', time_system_new='BJD_TDB'):
        ''' Converts date between JD_UTC and BJD_TDB timing systems.

            Parameters
            ----------
            times : list-like array of floats
                Date in either JD_UTC or BJD_TDB
            ra: float
                Right ascension of observation in degrees
            dec: float
                Declination of observation in degrees
            time_system_current: string
                Timing system of observation, either JD_UTC or BJD_TDB
            time_system_new: string
                Timing system to convert to

            Returns
            -------
            t_convert : float
                Converted date
        '''
        if time_system_current not in ['BJD_TDB', 'JD_UTC']:
            raise ValueError("Input 'time_system_current' must be either 'BJD_TDB' or 'JD_UTC'. Got: %s" % (time_system_current))
        elif time_system_new not in ['BJD_TDB', 'JD_UTC']: 
            raise ValueError("Input 'time_system_new' must be either 'BJD_TDB' or 'JD_UTC'. Got: %s" % (time_system_new))
        elif time_system_current == time_system_new:
            raise ValueError("Inputs 'time_system_current' and 'time_system_new' are the same and do not require conversion.")
        else:
            times = [str(t) for t in times]
            times_str = ','.join(times)
            if time_system_current == 'JD_UTC':
                conversion_url = 'https://astroutils.astronomy.osu.edu/time/convert.php?JDS='+times_str+'&RA='+str(ra)+'&DEC='+str(dec)+'&FUNCTION=utc2bjd'
            else:
                conversion_url = 'https://astroutils.astronomy.osu.edu/time/convert.php?JDS='+times_str+'&RA='+str(ra)+'&DEC='+str(dec)+'&FUNCTION=bjd2utc'
            page = urllib.request.urlopen(conversion_url)
            times_convert = page.read()
            times_convert = list(filter(None, times_convert.split(b'\n')))
            times_convert = np.array([float(t.decode("utf-8")) for t in times_convert])
        return times_convert
   
    @staticmethod
    def phase_fold(times, T_0, P):
        ''' Phase-fold observation times.
            
            Parameters
            ----------
            times : 1-D numpy array
                Times associated with data (make sure it's the same time system as the ephemeris).
            T_0 : float 
                Reference mid-transit time for the system.
            P : float
                Orbital period of the system (make sure units are consistent with T_0 and times).
            
            Returns
            -------
            phases : 1-D numpy array
                Phases (same size as times) corresponding to the data
        '''
        phases = (times-T_0) % P # wrap times
        phases = phases/P # convert to phase
        phases[phases > 0.5] = phases[phases > 0.5] - 1. # center phases about 0 = mid-transit
        return phases
    
    @staticmethod
    def norm_spectra(wav, flx, flx_err=None, method='mean_filt', cutoff=0.1, exclude=None, filt_size=None, smooth=None, plot_on=True):
        ''' Normalizes a spectrum.

            Parameters
            ----------
            wav : 1-D numpy array
                Spectral wavelengths
            flx : 1-D numpy array
                Spectral fluxes
            flx_err: 1-D numpy array or None
                Spectral flux errors; None if template spectrum
            method : string
                Method of continuum identification; options are: 1) mean filter ('mean_filt'), 
                2) median filter ('med_filt'), and 3) spline fitting ('spline')
            cutoff : float between 0 and 1
                Cutoff for fraction of lowest points to include for continuum identification
            exclude : 2-D numpy array
                Pairs of wavelength ranges to exclude from continuum fitting
            filt_size: odd integer 
                Filter size for filter-based continuum identification
            smooth : float
                Smoothing factor for spline-based continuum identification
            plot_on : boolean
                Plots spectrum with continuum identified as well as normalized spectrum
            
            Returns
            -------
            flx_norm: 1-D numpy arry
                Spectral fluxes normalized 
        '''
        if type(method) is not str:
            raise ValueError("Input 'method' must be a string. Got: %s"%type(method))

        cutoff_ind = int(len(flx)*cutoff)
        ind = np.argpartition(flx, -cutoff_ind)[-cutoff_ind:] 
        ind = ind[np.argsort(ind)]
        wav_top = np.copy(wav[ind])
        flx_top = np.copy(flx[ind])
        wav_stitch = []
        flx_stitch = []
        if exclude is not None:
            include = np.hstack([wav_top[0], exclude.flatten(), wav_top[-1]])
            include = include.reshape((int(len(include)/2), 2))
        else:
            include = np.array([[wav_top[0], wav_top[-1]]])
        for wav_range in include:
            i_low, i_up = get_indices(wav_top, wav_range[0], wav_range[1])
            wav_stitch = np.hstack([wav_stitch, wav_top[i_low:i_up]])            
            flx_stitch = np.hstack([flx_stitch, flx_top[i_low:i_up]])            
        if method == 'med_filt':
            if type(filt_size) is not int:
                raise ValueError("Input 'filt_size' must be an integer. Got: %s"%type(filt_size))
            elif not filt_size%2:
                raise ValueError("Input 'filt_size' must be odd. Got: %d"%filt_size)
            else:
                flx_cont = scipy.signal.medfilt(flx_stitch, filt_size) # Keeps median values within filter range
                flx_cont = interpolate_xy(wav_stitch, flx_cont, wav)
        elif method == 'mean_filt':
            if type(filt_size) is not int:
                raise ValueError("Input 'filt_size' must be an integer. Got: %s"%type(filt_size))
            elif not filt_size%2:
                raise ValueError("Input 'filt_size' must be odd. Got: %d"%filt_size)
            else:
                flx_cont = scipy.ndimage.uniform_filter(flx_stitch, filt_size) # Keeps median values within filter range
                flx_cont = interpolate_xy(wav_stitch, flx_cont, wav)
        elif method == 'spline':
            spl = scipy.interpolate.UnivariateSpline(wav_stitch, flx_stitch)
            spl.set_smoothing_factor(smooth)
            flx_cont = spl(wav)
        else:
            raise ValueError("Input 'method' must be one of the following: 'med_filt', 'mean_filt', or 'spline'. Got: %s"%method)

        if plot_on:
            fig = plt.figure(figsize=(10,6))
            plt.plot(wav, flx)
            plt.plot(wav, flx_cont, 'o')
            plt.show()

        flx_norm = flx / flx_cont
        if flx_err is not None:
            flx_norm_err = flx_err / flx_cont                
            return flx_norm, flx_norm_err 
        else:
            return flx_norm
    
    @staticmethod 
    def shift4CCF(wav_temp, flx_temp, index_shift_low, index_shift_up):
        ''' Shift template spectrum uniformly in velocity for cross-correlation. 
            
            Parameters
            ----------
            wav_temp : 1-D numpy array 
                Logarithmically-spaced wavelengths
            flx_temp : 1-D numpy array 
                Template spectrum to be shifted
            index_shift_low : integer 
                Maximal leftwards index shift for cross-correlation
            index_shift_up : integer 
                Maximal rightwards index shfit for cross-correlation
            
            Returns
            -------
            vels : 1-D numpy array
                Radial velocities
            flx_temp_shifted : 2-D numpy array
                Shifted spectra
        '''
        del_lambda, lambda_0 = wav_temp[1]-wav_temp[0], wav_temp[0]
        flx_temp_shifted = []
        shift_range = np.arange(-index_shift_low, index_shift_up+1, 1)
        for i in shift_range:
            flx_temp_shifted.append(scipy.ndimage.shift(flx_temp, i, order=1, mode='nearest'))
        flx_temp_shifted = np.array(flx_temp_shifted)
        c = const.c.to(u.km/u.s)
        vels = c * shift_range * del_lambda/lambda_0
        return vels, flx_temp_shifted

    @staticmethod
    def cross_corr(wav, flxs, flxs_err, flx_temp, index_shift_low, index_shift_up):
        ''' Performs cross-correlation between a spectrum and a reference template spectrum. Both should share
            the same wavelength grid.
            
            Parameters
            ----------
            wav : 1-D numpy array
                Logarithmically-spaced wavelengths for spectrum of interest AND template spectrum
            flxs : 2-D numpy array
                Spectrum of interest for cross-correlation with a template; normalized to continuum = 0
            flxs_err : 2-D numpy
                Spectral errors
            flx_temp : 1-D numpy array 
                Template spectrum to be shifted; normalized to continuum = 0
            index_shift_low : integer
                Maximal leftwards index shift for cross-correlation
            index_shift_up : integer
                Maximal rightwards index shift for cross-correlation
            
            Returns
            -------
            vels : 1-D numpy array
                Velocities corresponding to CCF
            CCFs : 2-D numpy array
                Cross-correlation power spectrum (CCF)
            CCFs_err : 2-D numpy array
                Propagated errors on CCF
        '''
        vels, flx_temps = FluxMap.shift4CCF(wav, flx_temp, index_shift_low, index_shift_up)
        # Calculate CCFs
        CCFs = []
        A = np.array([temp/np.linalg.norm(temp) for temp in flx_temps]) # normalize templates
        for flx in flxs:
            if np.linalg.norm(flx) == 0:
                B = flx
            else:
                B = flx/np.linalg.norm(flx) # normalize data
#         CCF = np.dot(A, B)
            CCF = np.dot(flx_temps, flx) # compute CCF shifted templates
            CCFs.append(CCF)
        CCFs = np.array(CCFs)
        # Calculate CCF errors
        CCFs_err = []
        for flx in flxs_err:
            if np.linalg.norm(flx) == 0:
                B = flx
            else:
                B = flx/np.linalg.norm(flx) # normalize data
#         CCF = np.dot(A, B)
            product = flx_temps * flx
            CCF_err = np.sqrt(np.sum(product**2., axis=1)) # compute CCF shifted templates
            CCFs_err.append(CCF_err)
        CCFs_err = np.array(CCFs_err)
        return vels, CCFs, CCFs_err

    @staticmethod
    def SYSREM(wav, flxs, flxs_err, num_iter=10, num_sys=5):
	''' Performs SYSREM algorithm for removing linearly related systematics from spectra. 
	    Adapted from PySysRem by Dr. Stephanie T. Douglas (https://github.com/stephtdouglas) 
            and Dr. Marshall Johnson's code (https://github.com/captain-exoplanet).

	    Parameters
            ----------
	    wav : 1-D numpy array 
                Wavelengths
	    flxs : 2-D numpy array 
                Spectral fluxes, continuum-normalized with baseline at 1
	    flxs_err : 2-D numpy array 
                Spectral flux errors
	    num_iter : integer 
                Number of iterations over which to minimize a and c for a given systematic
	    num_sys : integer 
                Number of systematics over which to perform systematic correction

	    Returns
            -------
	    flxs_SYSREM : 2-D numpy array
                Spectral fluxes corrected for systematics
	'''
	c = np.zeros(np.shape(flxs)[1]) # defined for each wavelength
	a = np.ones(np.shape(flxs)[0]) # defined for each observation
	flxs_SYSREM = np.copy(flxs) # set baseline around 0
	flxs_err_sq = flxs_err ** 2.

	print('Beginning SYSREM...')
	for sys in range(num_sys): # iterate of number of systematics
	    for iter in range(num_iter): # iterate of number of minimization iterations
		# Compute c (~extinction coefficient for each wavelength)
		c_num = np.sum(a*flxs_SYSREM.T/flxs_err_sq.T, axis=1)
		c_denom = np.sum(a**2/flxs_err_sq.T, axis=1)
		c = c_num/c_denom
		# Compute a (~airmass across observations)
		a_num = np.sum(c*flxs_SYSREM/flxs_err_sq, axis=1)
		a_denom = np.sum(c**2/flxs_err_sq, axis=1)
		a = a_num/a_denom

	    # Compute systemic errors
	    syserr = np.ones_like(flxs)
	    syserr *= c
	    syserr = (syserr.T * a).T
	    flxs_SYSREM -= syserr
	    print('Systematic removed:', sys)
	    print('Mean error:', np.mean(syserr))
	#flxs_SYSREM += med # Return baseline back to 1
	return flxs_SYSREM, syserr

    ## Routines for adding and cleaning observations ##
        
    def buildWavGrid(self, wav):
        ''' Builds log-spaced wavelength grid for all subsequent observations.
        
            Parameters 
            ----------
            wav : 1-D list-like object
                Wavelength array of an observed spectrum

            Returns
            -------
            FluxMap object updated with "wav" attribute
        '''
        # Use input wavelength for limits and length of wavelength grid
        wav_geom = np.geomspace(wav.min(), wav.max(), int(1.5*len(wav)))
        self.wav = wav_geom
    
    def addNewObservation(self, ra, dec, t_obs, wav_obs, flux_obs, flux_obs_err, time_system_obs, spec_type):
        ''' Adds a new observation to FluxMap object.

            Parameters
            ----------
            ra : float
                Right ascension in header of observation in degrees
            dec : float
                Declination in header of observation in degrees
            t_obs : float
                Time of new observation (make sure it's the same time system as the ephemeris)
            wav_obs : 1-D numpy array
                Wavelengths of new observation
            flux_obs : 1-D numpy array
                Fluxes of new observation
            flux_obs_err : 1-D numpy array
                Flux errors of new observation
            time_system_obs : string
                Timing system of observations (needs to be either 'JD_UTC' or 'BJD_TDB')
            spec_type : boolean
                True if observation being added is a transmission spectrum; False of observation is a stellar spectrum

            Returns
            -------
            FluxMap object with updated "times", "fluxes", and "fluxes_err" attributes
        '''
        
        if (len(wav_obs) != len(flux_obs)) or (len(flux_obs) != len(flux_obs_err)):
            raise ValueError("Dimension mismatch between wavelength, flux, and error arrays.") 
        elif time_system_obs not in ['BJD_TDB', 'JD_UTC']:
            raise ValueError("Input 'time_system_new' must be either 'BJD_TDB' or 'JD_UTC'. Got: %s" % (time_system_obs))
        elif abs(ra - self.ra) > 0.5*u.deg:
            raise ValueError("Input 'ra' deviates by more the 0.5 degrees from nominal value. Expected %s degrees, got: %s degrees"%(self.ra.value, ra.value))
        elif abs(dec - self.dec) > 0.5*u.deg:
            raise ValueError("Input 'dec' deviates by more the 0.5 degrees from nominal value. Expected %s degrees, got: %s degrees"%(self.dec.value, dec.value)) 
        else:
            # Convert timing system if mismatch with ephemeris
            if time_system_obs != self.time_system:
                t_obs = self.convertTimingSystem([t_obs], ra.value, dec.value, time_system_current=time_system_obs, time_system_new=self.time_system)[0]

            # Add observations to object
            if not hasattr(self, 'times'): # First observation; create spectrum attributes
                # Establish wavelength grid
                self.buildWavGrid(wav_obs)
                
                # Interpolate onto wavelength grid
                flux_interp = interpolate_xy(wav_obs, flux_obs, self.wav)
                flux_interp_err = interpolate_xy(wav_obs, flux_obs_err,
                                               self.wav)*np.sqrt(len(self.wav)/len(wav_obs))
                # Add to object
                self.times = np.array([t_obs]) * u.day
                self.fluxes = np.array([flux_interp])
                self.fluxes_err = np.array([flux_interp_err])
                self.spec_type = spec_type 
            else: # Append Nth observation
                if spec_type != self.spec_type:
                    spec_types = ['stellar', 'transmission']
                    raise ValueError("Input 'spec_typ' must match FluxMap object attribute 'spec_type' (%s). Got: %s" % (spec_types[self.spec_type], spec_types[spec_type]))
                if (max(wav_obs) < min(self.wav)) or (min(wav_obs) > max(self.wav)):
                    raise ValueError("Input 'wav_obs' does not fall within range of FluxMap object attribute 'wav'.") 
                # Interpolate onto wavelength grid
                flux_interp = interpolate_xy(wav_obs, flux_obs, self.wav)
                flux_interp_err = interpolate_xy(wav_obs, flux_obs_err,
                                               self.wav)*np.sqrt(len(self.wav)/len(wav_obs))
                # Add to object
                self.times = np.hstack([self.times.value, t_obs]) * u.day
                self.fluxes = np.vstack([self.fluxes, flux_interp])
                self.fluxes_err = np.vstack([self.fluxes_err, flux_interp_err])

    def __len__(self):
        ''' Returns number of observations contained in FluxMap object.

            Parameters
            ----------

            Returns
            -------
            num_obs : integer
                Number of observations
        '''
        num_obs = len(self.fluxes)
        if num_obs != len(self.fluxes_err):
            raise ValueError("Number of flux observations must match number of flux error observations.")
        elif num_obs != len(self.times):
            raise ValueError("Number of flux observations must match number of observation times.")
        return num_obs

    def cropSpectra(self, wav_low, wav_up):
        ''' Restricts wavelength of spectra.
        
            Parameters
	    ----------	
            wav_low: float
                Lower bound wavelength
            wav_up: float
                Upper bound wavelengths

            Returns
            -------
            FluxMap object with cropped spectra
        '''
        i_low, i_up = get_indices(self.wav, wav_low, wav_up)
        self.wav = self.wav[i_low:i_up]
        self.fluxes = self.fluxes[:, i_low:i_up]
        self.fluxes_err = self.fluxes_err[:, i_low:i_up]
        
    def sigmaClipSpectra(self, sigma_up=3, sigma_down=None):
        ''' Removes outliers from spectra.
            
            Parameters
            ----------
            sigma_up: float 
                Number of standard deviations to use for both the lower and upper clipping limit if 'sigma_down'
                isn't defined; else used for upper limit only.
            sigma_down: float or None
                Number of standard deviations to use for lower clipping limit if not None

            Returns
            -------
            FluxMap object with outliers removed from spectra
        '''
        for i, flx in enumerate(self.fluxes):
            flx_copy = np.copy(flx)
            flx_up = np.copy(flx)
            flx_up[flx_up<1] = np.ones_like(flx_up[flx_up<1]) # ignore absorption features
            mask_up = astropy.stats.sigma_clip(flx_up, sigma=sigma_up, masked=True)
            if sigma_down:
                mask_down = astropy.stats.sigma_clip(flx, sigma=sigma_down, masked=True)            
                self.fluxes[i][mask_down.mask] = np.ones_like(self.fluxes[i][mask_down.mask])
            
            self.fluxes[i][mask_up.mask] = np.ones_like(self.fluxes[i][mask_up.mask])  
            
            fig = plt.figure(figsize=(10,6))
            plt.plot(self.wav, flx_copy)
            plt.plot(self.wav[mask_up.mask], flx_copy[mask_up.mask], 'o')
            if sigma_down:
                plt.plot(self.wav[mask_down.mask], flx_copy[mask_down.mask], 'o')
            plt.show()

    def normSpectra(self, method='mean_filt', cutoff=0.1, filt_size=None, smooth=None, exclude=None, plot_on=True):
        ''' Normalizes spectra contained in FluxMap object.
            
            Parameters
            ----------
            
            Returns
            -------
            FluxMap object with spectra normalized
        '''
        for i in range(len(self)):
            self.fluxes[i], self.fluxes_err[i] = self.norm_spectra(self.wav, self.fluxes[i], self.fluxes_err[i], method=method, cutoff=cutoff, filt_size=filt_size, smooth=smooth, exclude=exclude, plot_on=plot_on)    

    ### Time-related routines ###

    def phaseFold(self):
        ''' Dynamically determines phases of observations.

            Parameters
            ----------

            Returns
            -------
            phases : 1-D numpy array
                Phases corresponding to each observation in current order of observations
        '''
        phases = self.phase_fold(self.times, self.T_0[0], self.P[0])
        return phases
    
    def sortByOrder(self, order):
        ''' Sorts observations according to an array specifying order.
            
            Parameters
            ----------
            order : 1-D numpy array
                Order of indices for sorting

            Returns
            -------
            FluxMap object with updated "times", "fluxes", "fluxes_err" according to specified sorting.
        '''
        self.times = self.times[order]
        self.fluxes = self.fluxes[order]
        self.fluxes_err = self.fluxes_err[order]
    
    def sortByTime(self):
        ''' Sorts observations in order of observation times.
            
            Parameters
            ----------

            Returns
            -------
            FluxMap object with updated "times", "fluxes", "fluxes_err" in order of observation times.
        '''
        order = np.argsort(self.times)
        self.sortByOrder(order)
    
    def sortByPhase(self):
        ''' Sorts observations in order orbital phase.
            
            Parameters
            ----------

            Returns
            -------
            FluxMap object with updated "times", "fluxes", "fluxes_err" in order of orbital phase.
        '''
        order = np.argsort(self.phaseFold())
        self.sortByOrder(order)
    
    ### Construct transmission spectra ###
    def isOut(self):
        ''' Labels observations as fully out-of-transit (True) or not (False).

            Parameters
            ----------

            Returns
            -------
            is_out : boolean array
                Array in order of observations; True if fully out-of-transit, False if not
        '''
        is_out = abs(self.phaseFold().value) > self.T_14[0]/(2*self.P[0])
        return is_out

    def isIn(self):
        ''' Labels observations as fully in-transit (True) or not (False).

            Parameters
            ----------

            Returns
            -------
            is_in : boolean array
                Array in order of observations; True if fully in-transit, False if not
        '''
        is_in = abs(self.phaseFold().value) < ((self.T_14[0]/2) - self.tau[0])/self.P[0]
        return is_in
    
    def makeTransSpectra(self):
        ''' Divide out stellar component from FluxMap fluxes to recover transmission spectra.
            
            Parameters
            ----------
            
            Returns
            -------
            FluxMap object with "spec_type" attribute set to True and "fluxes" converted to transmission spectra)
        '''
        if not self.spec_type:
            weights = 1./(self.fluxes_err[self.isOut()]**2) # weighted by inverse squared errors (inverse of variance)
            coefs = weights/np.sum(weights, axis=0)
            flux_stellar = np.sum(self.fluxes[self.isOut()]*coefs, axis=0) # stellar component = weighted mean of out-of-transit observations
            flux_stellar_err = np.sqrt(np.sum((coefs*self.fluxes_err[self.isOut()])**2, axis=0)) # propagate errors
            fluxes_trans = self.fluxes/flux_stellar
            fluxes_trans_err = fluxes_trans * np.sqrt((self.fluxes_err/self.fluxes)**2 + (flux_stellar_err/flux_stellar)**2)
            self.fluxes = fluxes_trans - 1.
            self.fluxes_err = fluxes_trans_err
            self.flux_stellar = flux_stellar
            self.flux_stellar_err = flux_stellar_err
            self.spec_type = True
        else:
            print('Spectrum type is already transmission.')
    
    ### Data analysis routines ###
    def crossCorr(self, wav_temp, flx_temp, index_shift_low, index_shift_up):
        ''' Cross correlate spectra in FluxMap object with a template spectrum.

            Parameters
            ----------
            wav_temp : 1-D array-like
                Wavelengths of template spectrum
            flx_temp : 1-D array-like
                Fluxes of template spectrum
            index_shift_low : integer 
                Maximal leftwards index shift for cross-correlation
            index_shift_up : integer 
                Maximal rightwards index shift for cross-correlation
                
            Returns
            -------
            vels : 1-D numpy array
                Velocities corresponding to CCF
            CCFs : 2-D numpy array
                Cross-correlation power spectrum (CCF)
            CCFs_err : 2-D numpy array
                Propagated errors on CCF
        '''
        flx_temp = interpolate_xy(wav_temp, flx_temp, FM.wav)        
        vels, CCFs, CCFs_err = self.cross_corr(FM.wav, FM.fluxes, FM.fluxes_err, flx_temp, 
                                            index_shift_low, index_shift_up)
        return vels, CCFs, CCFs_err

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
