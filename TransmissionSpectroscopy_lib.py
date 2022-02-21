# TransmissionSpectroscopy_lib.py
# Library of routines for transmission spectroscopy.

# Imports
import numpy as np
import scipy
import scipy.interpolate
import scipy.signal

import astropy.stats
import astropy.units as u
import astropy.constants as const
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve

import matplotlib.pyplot as plt

import pandas as pd

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


### Wavelength-related routines ###

### Time-related routines ###

def phase_fold(times, t0, P):
    ''' Phase-fold observation times.
        Inputs:
        - times: 1-D numpy array of times associated with data (make sure it's the same time system as the ephemeris)
        - t0: float representing reference mid-transit time for the system 
        - P: float representing orbital period of the system (make sure units are consistent with t0 and times)
        Return:
        - phases: array of phases (same size as times) with the folded phases corresponding to the data
    '''
    phases = (times-t0) % P # wrap times
    phases = phases/P # convert to phase
    phases[phases > 0.5] = phases[phases > 0.5] - 1. # center phases about 0 = mid-transit
    return phases

def extrapolate_mid_transit_linear(time_mid_0, P, t_start, t_end):
    ''' Extrapolates mid-transit time to epoch of observation based on orbital period.
        Inputs:
        - time_mid_0: tuple of floats (1 x 3) representing mid-transit time ephemeris and its error in JD (or some day units)
        - P: tuple of floats (1 x 3) representing period and its error in days
        - t_start: float for observation start time in same units as ephemeris
        - t_end: float for observation end time in same units as ephemeris
        Outputs:
        - time_mid: tuple of floats (1 x 3) represeting mid-transit time during epoch of observation and its error in same units as ephemeris
    '''
    E = 0
    time_mid = np.copy(time_mid_0[0])
    
    # Take maximum of errors on period (as representative error
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
        Inputs:
        - time_mid_0: tuple of floats (1 x 3) representing mid-transit time ephemeris and its error in JD (or some day units)
        - P: tuple of floats (1 x 3) representing period and its error in days
        - c: tuple of floats (1 x 3) representing quadratic coefficient and its error
        - t_start: float for observation start time in same units as ephemeris
        - t_end: float for observation end time in same units as ephemeris
        Outputs:
        - time_mid: tuple of floats represeting mid-transit time during epoch of observation and its error in same units as ephemeris
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

### Spectrum manipulation routines ###

def velocity_shift(index_shift, del_lambda, lambda_0 ):
    c = 299792.458
    return c * index_shift * del_lambda/lambda_0

def interpolate_spec(wav, flux, wav_new):
    ''' Interpolate spectrum to new wavelength grid
    Inputs:
    - wav: 1-D numpy array of current wavelengths
    - flux: 1-D numpy array of fluxes
    - wav_new: 1-D numpy array of wavelength grid to interpolate to
    Outputs:
    - flux_new: 1-D numpy array of flux array interpolated to wavelength grid
    '''
    quadinterp = scipy.interpolate.interp1d(wav, flux, kind='slinear', bounds_error=False, fill_value="extrapolate")
    return quadinterp(wav_new)

def shift_spec(wav, v):
    ''' Shifts spectra in velocity space.
        Inputs:
        - wav: 1-D numpy array of spectrum wavelengths
        - v: float representing velocity to shift spectrum by in km/s
        Outputs:
        - wav_new: 1-D numpy List of new shifted wavelengths
    '''
    c = const.c.to(u.km/u.s)
    wav_new = (v/c)*wav + wav
    return wav_new  

def get_indices(x, x_low, x_up):
    ''' Finds lower and upper index of specific range in array.
        Inputs:
        - x: 1-D numpy array of spectrum wavelengths
        - x_low, x_up: floats representing lower/upper bounds of range of interest
        Outputs:
        - i_low, i_up: floats representing lower/upper bound indices
    '''
    i_low = np.where(x >= x_low)[0][0]
    i_up = np.where(x <= x_up)[0][-1]
    return i_low, i_up


def norm_spectra(wav, flx, flx_err,  method='mean_filt', cutoff=0.1, filt_size=None, smooth=None, exclude=None):
        ''' Normalizes a spectrum.
            Inputs:
            - wav, flx: 1-D numpy arrays representing wavelength and flux of spectrum
            - method: string representing method of normalization; default is to apply a mean filter
            - cutoff: float (0-1) representing cutoff for lowest points to include for continuum identification
            - filt_size: odd integer representing filter size for filter-based continuum identification
            - smooth: float representing smoothing factor for spline-based continuum identification
            - exclude: 2-D, two-column  numpy array with pairs of wavelength ranges to exclude from continuum fitting.
            Outputs:
            - flx_norm: FluxMap object with spectra normalized
        '''
        cutoff_ind = int(len(flx)*cutoff)
        ind = np.argpartition(flx, -cutoff_ind)[-cutoff_ind:] 
        ind = ind[np.argsort(ind)]
        wav_top = np.copy(wav[ind])
        flx_top = np.copy(flx[ind])
        wav_stitch = []
        flx_stitch = []
        include = np.hstack([wav_top[0], exclude.flatten(), wav_top[-1]])
        include = include.reshape((int(len(include)/2), 2))
        for wav_range in include:
            i_low, i_up = get_indices(wav_top, wav_range[0], wav_range[1])
            wav_stitch = np.hstack([wav_stitch, wav_top[i_low:i_up]])            
            flx_stitch = np.hstack([flx_stitch, flx_top[i_low:i_up]])            
        if method == 'med_filt':
            flx_cont = scipy.signal.medfilt(flx_stitch, filt_size) # Keeps median values within filter range
            flx_cont = interpolate_spec(wav_stitch, flx_cont, wav)

            fig = plt.figure(figsize=(10,6))
            plt.plot(wav, flx)
            plt.plot(wav, flx_cont, 'o')
            plt.show()

            flx_norm = flx / flx_cont
            flx_norm_err = flx_err / flx_cont                
        elif method == 'mean_filt':
            flx_cont = scipy.ndimage.uniform_filter(flx_stitch, filt_size) # Keeps median values within filter range
            flx_cont = interpolate_spec(wav_stitch, flx_cont, wav)

            fig = plt.figure(figsize=(10,6))
            plt.plot(wav, flx)
            plt.plot(wav, flx_cont, 'o')
            plt.show()

            flx_norm = flx / flx_cont  
            flx_norm_err = flx_err / flx_cont                
        elif method == 'spline':
            spl = scipy.interpolate.UnivariateSpline(wav_stitch, flx_stitch)
            spl.set_smoothing_factor(smooth)
            flx_cont = spl(wav)

            fig = plt.figure(figsize=(10,6))
            plt.plot(wav, flx)
            plt.plot(wav, flx_cont, 'o')
            plt.show()

            flx_norm = flx / flx_cont
            flx_norm_err = flx_err / flx_cont                
        return flx_norm, flx_norm_err

### Dynamical routines ##

def RV_semiAmplitude(M_star, m_p, a_p, e, i_orbit):
    return np.sqrt(const.G/(1-e**2.)) * m_p*np.sin(i_orbit) * (M_star+m_p)**(-1/2) * a_p**(-1/2)

def RV_circular_orbit(x, K, v_sys, ephemeris=None):
    ''' Computes the radial velocity equation given orbital parameters.
        Inputs:
        - x: float or 1-D numpy array representing either time or phase
        - K: float representing RV semi-amplitude (in km/s)
        - v_sys: systemic velocity (in km/s)
        - ephemeris: None if x is in phase; else tuple of mid-transit time and orbital period (in same units as x)
        Outputs:
        - Radial velocity (in km/s)
    '''
    if ephemeris == None:
        return K * np.sin(2*np.pi*x) + v_sys
    else:
        t_mid, P = ephemeris
        arg = 2 * np.pi * (x-t_mid)/P * u.rad
        return K * np.sin(arg) + v_sys        

### Least-squares deconvolution routines ###

def make_M(wav, temp, n, filename_M=None, filename_temp=None):
    ''' Generates 2D line mask matrix M (n x m) out of 1D template spectrum (1 x m) for LSD. Saves template and line mask to numpy array when flags are strings.
        Inputs:
        - wav: 1-D numpy array of wavelengths corresponding to OBSERVED spectrum
        - temp: 2-D numpy array (2 x length of spectrum) of wavelength and template spectrum
        - n: integer representing number of velocities in output line profile (must be odd for symmetry)
        Outputs:
        - M: 2-D numpy array of template spectrum expanded into a Toeplitz matrix of the form used in LSD
        - Numpy array files when flags are set to strings of filenames
    '''
    temp_wav, temp_flux = temp
    temp_flux = interpolate_spec(temp_wav, temp_flux, wav)
    M = np.roll(scipy.linalg.toeplitz(temp_flux)[:, 0:n], int(-n/2), axis = 0) # Ji's
    if filename_M != None:
        np.save(filename_M, M)
        df = pd.DataFrame(np.vstack([wav, temp_flux]).T)
        df.to_csv(filename_temp, header=['wav', 'flux'], index=False)
    return M

def make_R(m):
    ''' Generates first-order Tikhonov regularization matrix (m x m) of specified dimension for LSD.
        Inputs:
        - m: integer representing length of template/observed spectrum
        Outputs:
        - R: 2-D numpy array of first-order Tikhonov regularization
    '''
    R = np.zeros((m,m))
    for i in range(m):
        if i == 0:
            R[i,0:2] = [1, -1]
        elif i == m-1:
            R[i, -2:] = [-1, 1]
        else:
            R[i,i-1] = -1
            R[i,i] = 2
            R[i,i+1] = -1
    return R

def leastSqDeconvolution(wav, Y, n, Lambda, filename_root=None, temp=None):
    ''' Performs LSD for a given observed spectrum (1 x m), template spectrum (1 x m, same log-spaced wavelength sampling as observed spectrum), and regularization parameter.
        Inputs:
        - wav: 1-D numpy array of wavelengths corresponding to OBSERVED spectrum 
        - Y: 1-D numpy array of observed spectrum
        - n: integer representing number of velocities in output line profile (must be odd for symmetry)
        - Lambda: float representing coefficient of regularization
        - filename_root: either 1) string representing base prefix of files to read from or write to or 2) None (does not read or save M matrix and template array)
        - temp: either 1) 2-D numpy array of synthetic template wavelength AND spectrum (does not reading from existing file unless filename_root is provided and file exists) or 2) None (reads from existing file)
        Outputs:
        - Z: 1-D numpy array (1 x n) of deconvolved line profile that when convolved with template spectrum yields observed spectrum
          Z = (M^T * M + Lambda R)^-1 * M^T * Y 
    '''
    if filename_root != None:
        try:
            M = np.load('LSD/'+filename_root+'_stellar_M.npy')
        except FileNotFoundError:
#            try:
            M = make_M(wav, temp, n, filename_M='LSD/'+filename_root+'_stellar_M.npy', 
                            filename_temp='LSD/'+filename_root+'_stellar_temp.csv')
#            except:
#                print('Error: template spectrum or n are not valid. Try again.')
#                return
    else:
        try:
            M = make_M(wav, temp, n)
        except:
            print('Error: template spectrum or n are not valid. Try again.')
            return
    autocorr = np.linalg.inv(np.dot(M.T, M) + Lambda * make_R(n))
    CCF = np.dot(M.T, Y)
    Z = np.dot(autocorr, CCF)
    return Z

def rot_kernel(center, epsilon, scale, DC, vsini, vels):
    ''' Generates rotational broadening line profile.
        Inputs:
        - center: float representing center of line profile
        - epsilon: float representing linear limb-darkening coefficient
        - scale: float representing scaling factor
        - DC: float representing vertical offset
        - vsini: float representing sky-projected rotational velocity
        - vels: numpy 1-D array of velocities 
        Outputs:
        - G: numpy 1-D array of synthetic rotational broadening line profile
    '''
    denom = np.pi * vsini * (1. - epsilon/3.)
    c1 = 2. * (1.-epsilon)/denom
    c2 = 0.5 * np.pi * epsilon/denom
    G = c1*np.sqrt(1. - ((vels-center)/vsini)**2) + c2*(1. - ((vels-center)/vsini)**2) 
    G[np.isnan(G) == True] = 0
    G = G*scale + DC
    return G

def rot_kernel_res(params, vels, G_data):
    ''' Computes empirical rotational kernel residuals compared to fit
        Inputs:
        - params: numpy 1-D array of free parameters (center, epsilon, scale, DC, vsini)
        - vels: numpy 1-D array of velocities
        - G_data: numpy 1-D array of observed line profile
        Outputs:
        - G_res: residuals between model rotationally-broadened line profile and observed line profile
    '''
    G_model = rot_kernel(*params, vels)
    G_res = G_data - G_model
    return G_res

def rot_gauss_kernel(params, R, vels):
    ''' Generates rotational + Gaussian (e.g. instrumental) broadening line profile.
        Inputs:     
        - params: numpy 1-D array of free parameters (center, epsilon, scale, DC, vsini)
        - R: float representing resolving power
        - vels: numpy 1-D array of velocities 
        Outputs:
        - G: numpy 1-D array of synthetic rotational broadening line profile
    '''
    G_rot = rot_kernel(*params, vels) 
    gauss_res = const.c.to(u.km/u.s).value/R
    kernel_gauss = Gaussian1DKernel(stddev=gauss_res)
    LP = convolve(G_rot, kernel_gauss, normalize_kernel=True, boundary='extend')
    return LP

def rot_gauss_kernel_res(params, R, vels, LP_data):
    ''' Computes empirical rotational kernel residuals compared to fit.
        Inputs:
        - params: numpy 1-D array of free parameters (center, epsilon, scale, DC, vsini)
        - R: float representing resolving power
        - vels: numpy 1-D array of velocities
        - G_data: numpy 1-D array of observed line profile
        Outputs:
        - G_res: residuals between model rotationally-broadened line profile and observed line profile
    '''
    LP_model = rot_gauss_kernel(params, R, vels)
    LP_res = LP_data - LP_model
    return LP_res

def rot_gauss_kernel_global_res(params, R, vels, LPs_data, phases):
    ''' Computes empirical rotational kernel residuals compared to fit across all observations.
        Inputs:
        - params: numpy 1-D array of free parameters (epsilon, scale, DC, vsini, K_star, v_sys)
        - R: float representing resolving power
        - vels: numpy 1-D array of velocities
        - LPs_data: numpy 2-D array of observed line profile
        Outputs:
        - LPs_res: residuals between model rotationally-broadened line profile and observed line profile
    '''
    LPs_model = np.ones_like(LPs_data)
    LPs_model[:] = np.nan
    centers = RV_circular_orbit(phases, params[-2], params[-1], ephemeris=None)
    for i, center in enumerate(centers):
        params_test = np.ones(5)
        params_test[:] = np.nan
        params_test[0] = center
        params_test[1:] = params[:-2]
        LPs_model[i] = rot_gauss_kernel(params_test, R, vels)
    LPs_res = LPs_data - LPs_model
    return LPs_res.flatten()

### Stellar template construction routines ###
# Grid in units of stellar radius. Star oriented such that rotational axis is along y-axis.

def in_circle(points):
    ''' Determines if a list of points falls within a unit circle at origin.
        Inputs:
        - points: 2-D numpy array of coordinate points
        Outputs:
        - points_in: 2-D numpy array of coordinate points that fall within the circle
    '''
    points_in = points[np.hypot(points[:,0], points[:,1]) <= 1]
    return points_in

def out_circle(points):
    ''' Determines if a list of points falls outside a circle of a given center and radius.
        Inputs:
        - points: 2-D numpy array of coordinate points
        Outputs:
        - points_out: 2-D numpy array of coordinate points that fall within the circle
    ''' 
    points_out = points[np.hypot(points[:,0], points[:,1]) > 1]
    return points_out

def planet_pos(phases, a, i, l):
    ''' Calculates the sky-projected coordinates of the planet's center at a given time.
        Inputs:
        - phases: 1-D numpy array of orbital phases
        - a: float representing semi-major axis of the planet in units of stellar radius
        - i: float representing inclination of the orbital angular momentum vector to the line of sight in radians
        - l: float representing spin-orbit misalignment in radians
        Outputs:
        - points_pl: 1-D numpy array of coordinate points of planet's center at time t
    '''
    xp = a * np.sin(2*np.pi*phases)
    yp = a * np.cos(2*np.pi*phases) * np.cos(i)
    x = xp * np.cos(l) + yp * np.sin(l)
    y = -xp * np.sin(l) + yp * np.cos(l)
    points_pl = np.vstack([x, y]).T
    return points_pl


def cell_velocity(points, vsini):
    ''' Calculates the line-of-sight velocity of cells of given coordinates.
        Inputs:
        - points: 2-D numpy array of coordinate points
        - vsini: float representing projected rotational velocity of star
        Outputs:
        - v: 1-D numpy array of line-of-sight velocities of cells of interest
    '''
    v = points[:,0]*vsini
    return v

def cell_mu(points):
    ''' Calculates the limb mu angle of cells of given coordinates.
        Inputs:
        - Array of coordinate points of cells of interest
        Outputs:
        - Array of limb mu angles of cells of interest
    '''
    mu = np.sqrt(1 - points[:,0]**2 - points[:,1]**2)
    return mu

### class: FluxMap ###

class FluxMap:
    
    def __init__(self, wav, t_start, t_end, flux, flux_err, t0, P, T14, tau, 
                    time_system='JD_UTC', spec_type=False, c=False):
        ''' Generates flux map from initial spectrum. Tuples are in the order (value, error) or (value, error_positive, error_negative).
            Inputs:
            - wav: 1-D numpy array of wavelengths
            - t_start: float representing time of first observation (make sure it's the same time system as the ephemeris)
            - t_end: float representing time of last observation (make sure it's the same time system as the ephemeris)
            - flux: 1-D numpy array of fluxes
            - flux_err: 1-D numpy array of flux errors
            - t0: tuple of floats representing mid-transit time ephemeris and error for system
            - P: tuple of floats representing period and error for system (make sure units are consistent with t0 and t)
            - T14: tuple of floats representing transit duration between 1st and 4th contact (make sure units are consistent with t0 and t) and its error
            - tau: tuple of floats representing ingress/egress duration and its error (make sure units are consistent with t0 and t)
            - K_star: tuple of floats representing stellar RV semi-amplitude and its error (in km/s)
            - v_sys: tuple of floats representing systemic velocity and its error (in km/s)
            - time_system: string representing timing system used by observations and ephemeris
            - spec_type: bool representing type of spectrum (True = transmission, False = stellar) 
            - c: tuple of floats representing quadratic coefficient of ephemeris propagation; False if linear
            Outputs:
            - self: FluxMap object
        '''       
        self.times = np.array([t_start.value])*u.day
        if c == False:
            self.t0 = extrapolate_mid_transit_linear(t0, P, t_start, t_end)
        else:
            self.t0 = extrapolate_mid_transit_quadratic(t0, P, c, t_start, t_end)
        self.P = P
        self.T14 = T14
        self.tau = tau
        self.phases = np.array([phase_fold(self.times, self.t0[0], self.P[0])]) * u.rad
        self.is_out = abs(self.phases.value) > self.T14[0]/(2*self.P[0]) # fully out-of-transit
        self.is_in = abs(self.phases.value) < ((self.T14[0]/2) - self.tau[0])/self.P[0] # fully in-transit
        wav_geom = np.geomspace(wav.min(), wav.max(), int(len(wav)*1.5))
        self.wav = wav_geom
        self.fluxes = interpolate_spec(wav, flux, self.wav)
        self.fluxes_err = interpolate_spec(wav, flux_err, self.wav)*np.sqrt(len(self.wav)/len(wav))
        self.time_system = time_system
        self.spec_type = spec_type
        self.num_obs = np.shape(self.fluxes)[0]        
       
    def addNewObservation(self, wav_new, t_new, flux_new, flux_new_err):
        ''' Adds a new observation to flux map.
            Inputs:
            - wav_new: 1-D numpy array of wavelengths of new observation
            - t_new: float representing time of new observation (make sure it's the same time system as the ephemeris)
            - flux_new: 1-D numpy array of fluxes of new observation
            - flux_new_err: 1-D numpy array of flux errors of new observation
            Outputs:
            - self: FluxMap object updated with new observation
        '''
        self.times = np.hstack([self.times.value, t_new.value]); self.times *= u.day
        self.phases = phase_fold(self.times, self.t0[0], self.P[0]) * u.rad
        self.is_out = abs(self.phases.value) > self.T14[0]/(2*self.P[0]) # fully out-of-transit
        self.is_in = abs(self.phases.value) < ((self.T14[0]/2) - self.tau[0])/self.P[0] # fully in-transit
        flux_interp = interpolate_spec(wav_new, flux_new, self.wav)
        flux_interp_err = interpolate_spec(wav_new, flux_new_err,
                                           self.wav)*np.sqrt(len(self.wav)/len(wav_new))
        self.fluxes = np.vstack([self.fluxes, flux_interp])
        self.fluxes_err = np.vstack([self.fluxes_err, flux_interp_err])
        
        self.num_obs = len(self.fluxes)
        
    def cropSpectra(self, wav_low, wav_up):
        ''' Restricts wavelength of spectra.
            Inputs:
            - wav_low, wav_up: floats representing lower/upper bound wavelengths of range of interest
            Outputs:
            - self: FluxMap object with cropped spectra
        '''
        i_low, i_up = get_indices(self.wav, wav_low, wav_up)
        self.wav = self.wav[i_low:i_up]
        self.fluxes = self.fluxes[:, i_low:i_up]
        self.fluxes_err = self.fluxes_err[:, i_low:i_up]
        
    def sigmaClipSpectra(self, sigma=3):
        ''' Removes outliers from spectra.
            Inputs:
            - sigma: float representing number of standard deviations to use for both the lower and upper clipping limit
            Outputs:
            - self: FluxMap object with outliers removed from spectra.
        '''
        for i, flx in enumerate(self.fluxes):
            flx_up = np.copy(flx)
            flx_up[flx_up<1] = np.ones_like(flx_up[flx_up<1])
            mask = astropy.stats.sigma_clip(flx_up, sigma=sigma, masked=True)
            
            fig = plt.figure(figsize=(10,6))
            plt.plot(self.wav, flx)
            plt.plot(self.wav[mask.mask], flx[mask.mask], 'o')
            plt.show()
            
            self.fluxes[i][mask.mask] = np.ones_like(self.fluxes[i][mask.mask])
        
    def normSpectra(self, method='mean_filt', cutoff=0.1, filt_size=None, smooth=None, exclude=None):
        ''' Normalizes a spectrum.
            Inputs:
            - method: string representing method of normalization; default is to apply a mean filter
            - cutoff: float (0-1) representing cutoff for lowest points to include for continuum identification
            - filt_size: odd integer representing filter size for filter-based continuum identification
            - smooth: float representing smoothing factor for spline-based continuum identification
            - exclude: 2-D, two-column numpy array listing pairs of wavelength ranges to exclude from coninuum fitting.
            Outputs:
            - self: FluxMap object with spectra normalized
        '''
        for i, flx in enumerate(self.fluxes):
            self.fluxes[i], self.fluxes_err[i] = norm_spectra(self.wav, self.fluxes[i], self.fluxes_err[i], method=method, cutoff=cutoff, filt_size=filt_size, smooth=smooth, exclude=exclude)
    
    def shift2restframe(self, K, v_sys):
        ''' Shifts observations to restframe of object with circular orbit of given semiamplitude and offset.
            Inputs:
            - self: FluxMap object with spec_type attribute set to False( fluxes are raw stellar spectra)
            - K: tuple of floats representing orbital semiamplitude and its error
            - v_sys: tuple of floats representing RV offset and its error
            Outputs:
            - self: FluxMap with spectra shifted to restframe.
        '''
        RVs = RV_circular_orbit(self.phases, K[0], v_sys[0], ephemeris=None)
        for i in range(self.num_obs):
            wav_shift = shift_spec(self.wav, -RVs[i])
            self.fluxes[i] = interpolate_spec(wav_shift, self.fluxes[i], self.wav)

    def makeTransSpectra(self):
        ''' Divide out stellar component from FluxMap fluxes to recover transmission spectra.
            Inputs:
            - self: FluxMap object with spec_type attribute set to False (fluxes are raw stellar spectra)
            Outputs:
            - self: FluxMap object with spec_type attribute set to True (fluxes are reduced transmission spectra)
        '''
        if not self.spec_type:
            weights = 1./(self.fluxes_err[self.is_out]**2) # weighted by inverse squared errors (inverse of variance)
            coefs = weights/np.sum(weights, axis=0)
            flux_stellar = np.sum(self.fluxes[self.is_out]*coefs, axis=0) # stellar component = weighted mean of out-of-transit observations
            flux_stellar_err = np.sqrt(np.sum((coefs*self.fluxes_err[self.is_out])**2, axis=0)) # propagate errors
            fluxes_trans = self.fluxes/flux_stellar
            fluxes_trans_err = fluxes_trans * np.sqrt((self.fluxes_err/self.fluxes)**2 + (flux_stellar_err/flux_stellar)**2)
            self.fluxes = fluxes_trans
            self.fluxes_err = fluxes_trans_err
            self.flux_stellar = flux_stellar
            self.flux_stellar_err = flux_stellar_err
            self.spec_type = True
        else:
            print('Spectrum type is already transmission.')

    def makeLSDProfiles(self, n, Lambda, filename_root=None, temp=None):
        ''' Performs LSD for all observations.
            Inputs:
            - self: FluxMap object with spec_type attribute set to False (fluxes are raw stellar spectra)
            - n: integer representing number of velocities in output line profile (must be odd for symmetry)
            - Lambda: float representing coefficient of regularization
            - filename_root: either 1) string representing base prefix of files to read from or write to or 2) None (does not read or save M matrix and template array)
            - temp: either 1) 1-D numpy array of synthetic template spectrum (does not reading from existing file unless filename_root is provided and file exists) or 2) None (reads from existing file)
            Outputs:
            - LPs: 2-D numpy array of deconvolved line profiles for all observations. Saves to numpy array file if filename_root is provided.
        '''
        if not self.spec_type:
            LPs = np.empty((self.num_obs, n))
            LPs[:] = np.nan
            for i, flux in enumerate(self.fluxes):
                LP = leastSqDeconvolution(self.wav, flux, n, Lambda, filename_root=filename_root, temp=temp)
                LPs[i] = LP
            return LPs
        else:
            print('Spectrum type is not stellar.') 
            return
        
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
        
        
