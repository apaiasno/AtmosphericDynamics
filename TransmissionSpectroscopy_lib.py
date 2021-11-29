# TransmissionSpectroscopy_lib.py
# Library of routines for transmission spectroscopy.

# Imports
import numpy as np
import scipy
import scipy.interpolate

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

### Spectrum manipulation routines ###

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

### class: FluxMap ###

class FluxMap:
    
    def __init__(self, wav, t, flux, flux_err, t0, t0_err, P, P_err, T14, tau,
                 time_system='JD_UTC', spec_type=False):
        ''' Generates flux map from initial spectrum.
            Inputs:
            - wav: 1-D numpy array of wavelengths
            - t: float representing time of observation (make sure it's the same time system as the ephemeris)
            - flux: 1-D numpy array of fluxes
            - flux_err: 1-D numpy array of flux errors
            - t0: float representing mid-transit time ephemeris for system
            - P: float representing period for system (make sure units are consistent with t0 and t)
            - T14: transit duration between 1st and 4th contact (make sure units are consistent with t0 and t)
            - tau: ingress/egress duration (make sure units are consistent with t0 and t)
            - time_system: string representing timing system used by observations and ephemeris
            - spec_type: bool representing type of spectrum (True = transmission, False = stellar) 
            Outputs:
            - self: FluxMap object
        '''
            
        self.wav = np.geomspace(wav.min(), wav.max(), len(wav))        
        self.times = np.array([t])
        self.fluxes = flux
        self.fluxes_err = flux_err
        self.t0, self.t0_err = t0, t0_err
        self.P, self.P_err = P, P_err
        self.T14 = T14
        self.tau = tau
        self.phases = phase_fold(self.times, t0, P)
        self.time_system = time_system
        self.spec_type = spec_type
        self.num_obs = np.shape(self.fluxes)[0]
        self.is_out = abs(self.phases) > self.T14/(2*self.P) # fully out-of-transit
        self.is_in = abs(self.phases) < ((self.T14/2) - self.tau)/self.P # fully in-transit
       
    def addNewObservation(self, wav_new, t_new, flux_new, flux_new_err):
        ''' Adds a new observation to flux map.
            Inputs:
            - wav_new: 1-D numpy array of wavelengths of new observation
            - t_new: float representig time of new observation (make sure it's the same time system as the ephemeris)
            - flux_new: 1-D numpy array of fluxes of new observation
            - flux_new_err: 1-D numpy array of flux errors of new observation
            Outputs:
            - self: FluxMap object updated with new observation
        '''
        flux_interp = interpolate_spec(wav_new, flux_new, self.wav)
        flux_interp_err = interpolate_spec(wav_new, flux_new_err, self.wav)
        self.times = np.hstack([self.times, t_new])
        self.phases = phase_fold(self.times, self.t0, self.P)
        self.is_out = abs(self.phases) > self.T14/(2*self.P) # fully out-of-transit
        self.is_in = abs(self.phases) < ((self.T14/2) - self.tau)/self.P # fully in-transit
        self.fluxes = np.vstack([self.fluxes, flux_interp])
        self.fluxes_err = np.vstack([self.fluxes_err, flux_interp_err])*np.sqrt(len(self.wav)/len(wav_new))
        self.num_obs = len(self.fluxes)
    
    def makeTransSpectra(self):
            ''' Divide out stellar component from FluxMap fluxes to recover transmission spectra.
                Inputs:
                - self: FluxMap object with spec_type attribute set to False (fluxes are raw stellar spectra)
                Outputs:
                - self: FluxMap object with spec_type attribute set to True (fluxes are reduced transmission spectra)
            '''
            if not self.spec_type:
                weights = 1./(self.fluxes_err[self.is_out]**2) # weighted by inverse squared errors (inverse of variance)
                flux_stellar = np.sum(self.fluxes[self.is_out]*weights, axis=0)/np.sum(weights, axis=0) # stellar component = weighted mean of out-of-transit observations
                coefs = weights/np.sum(weights, axis=0)
                flux_stellar_err = np.sqrt(np.sum((coefs*self.fluxes_err[self.is_out])**2, axis=0)) # propagate errors
                fluxes_trans = self.fluxes/flux_stellar
                fluxes_trans_err = fluxes_trans * np.sqrt((self.fluxes_err/self.fluxes)**2 + (flux_stellar_err/flux_stellar)**2)
                self.fluxes = fluxes_trans
                self.fluxes_trans_err = fluxes_trans_err
                self.flux_stellar = flux_stellar
                self.spec_type = True
            else:
                print('Spectrum type is already transmission.')
        
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
        
        