# TransmissionSpectroscopy_lib.py
# Library of routines for transmission spectroscopy.

# Imports
import numpy as np
import scipy

### Wavelength-related routines ###

### Time-related routines ###

def phase_fold(times, t0, P):
    ''' Phase fold the times of lightcurve.
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
    
    def __init__(self, wav, t, flux, t0, t0_err, P, P_err, time_system='JD_UTC', spec_type=False):
        ''' Generates flux map from initial spectrum.
            Inputs:
            - wav: 1-D numpy array of wavelengths
            - t: float representing time of observation (make sure it's the same time system as the ephemeris)
            - flux: 1-D numpy array of fluxes
            - t0: float representing mid-transit time ephemeris for system
            - P: float representing period for system (make sure units are consistent with t0 and t)
            - time_system: string representing timing system used by observations and ephemeris
            - spec_type: bool representing type of spectrum (True = transmission, False = stellar) 
            Outputs:
            - self: initial FluxMap object
        '''
            
        self.wav = wav        
        self.times = np.array([t])
        self.fluxes = flux
        self.t0 = t0
        self.t0_err = t0_err
        self.P = P
        self.P_err = P_err
        self.phases = np.array([phase_fold(t, t0, P)])
        self.time_system = time_system
        self.spec_type = spec_type
        self.num_obs = np.shape(self.fluxes)[0]
       
    def addNewObservation(self, wav_new, t_new, flux_new):
        ''' Adds a new observation to flux map.
            Inputs:
            - wav_new: 1-D numpy array of wavelengths of new observation
            - t_new: float representig time of new observation (make sure it's the same time system as the ephemeris)
            - flux_new: 1-D numpy array of fluxes of new observation
            Outputs:
            - self: FluxMap object updated with new observation
        '''
        flux_interp = interpolate_spec(wav_new, flux_new, self.wav)
        self.fluxes = np.vstack([self.fluxes, flux_interp])
        self.num_obs = np.len(self.fluxes)
        
        
        