# PlanetModel_lib.py
# Library of routines for modelling planetary atmospheres.
 
# Imports
import os

import numpy as np
import pandas as pd
import scipy.signal
import astropy.units as u

import petitRADTRANS.nat_cst as nc
from petitRADTRANS import Radtrans

import TransmissionSpectroscopy_lib_unity as TSL


### petitRADTRANS routines ###

def rad2transSpec(wav_p, rad_p, R_p, R_star, normalize=True, cutoff=0.1, exclude=None, smooth=None):
    ''' Converts wavelength-dependent radii to transmission spectrum.

        Parameters
        ----------
        wav_p : 1-D numpy array
            List of wavelengths
        rad_p : 1-D numpy array
            List of planet radii corresponding to wavelength array
        R_p : float in Jupiter radii (unitless object)
            Planet radius
        R_star : float in solar radii (unitless object)
            Host star radius
        normalize : boolean
            True if applying normalization routine; else False
        cutoff : float between 0 and 1
            Cutoff for fraction of lowest points to include for continuum identification
        exclude : 2-D numpy array
            Pairs of wavelength ranges to exclude from continuum fitting
        smooth : float
            Smoothing factor for spline-based continuum identification
    '''
    # Calculate absorption depth
    flx_p = 1. - (rad_p**2. - R_p**2.)/(R_star**2.)
    # Normalize
    if normalize:
        flx_p = TSL.FluxMap.norm_spectra(wav_p, flx_p, cutoff=cutoff, exclude=exclude, smooth=smooth, method='spline')
    return flx_p

def make_transmission_template(planet_name, planet_config, petit_config, species, lambda_low, lambda_up, 
                               pressures, contribution=False, Pcloud=False, in_air=False, normalize=True, cutoff=0.1, exclude=None, smooth=None):
    ''' Computes model planet radius as a function of wavelength and resulting transmission spectrum
        using petitRADTRANS. Saves template spectrum and P-T profile.
 
        Parameters
        ----------
        planet_config : string
            Filename of planet config file for reading in system parameters
        petit_config : string
            Filename of petitRADTRANS config file for reading in template parameters, including
                - Mean molecular weight of atmosphere
                - Reference pressure
                - Atmospheric opacity in IR
                - Ratio between optical and IR opacity
                - Planetary internal temperature
                - Atmospheric equilibrium temperature
                - Species atomic weights
                - Species abundances
        species : list strings
            Species to model
        lambda_low : float
            Lower bound wavelength (Angstroms)
        lambda_up : float
            Upper bound wavelength (Angstroms)
        pressures : 1-D numpy array
            Pressure structure of model atmosphere
        contribution : boolean
            True if calculating contribution function; else False
        Pcloud : float or False
            Pressure level of cloud deck in bars; False if no cloud.
        in_air : boolean
            True if species opacities are in air (Fe+, Ti+, TiO); else False
        normalize : boolean
            True if applying normalization; else False
        cutoff : float between 0 and 1
            Cutoff for fraction of lowest points to include for continuum identification
        exclude : 2-D numpy array
            Pairs of wavelength ranges to exclude from continuum fitting
        smooth : float
            Smoothing factor for spline-based continuum identification

        Returns
        -------
        wav_pl : 1-D numpy array
            Model wavelengths
        rad_pl : 1-D numpy array
            Model planetary radii
        flux_pl : 1-D numpy array
            Model fluxes
        temperatures : 1-D numpy array
            Model temperatures
        contr_func : 2-D numpy array
            Contribution function at each pressure level as a function of wavelength; only returns if input 
            'contribution' is True
    '''

    # Read in system parameters
    df = pd.read_csv(planet_config, delim_whitespace=True, comment='#', header=None)
    M_pl = np.array(df.loc[df[0]=='M_p', 1].values[0], dtype=float) * nc.m_jup
    R_pl = np.array(df.loc[df[0]=='R_p', 1].values[0], dtype=float) * nc.r_jup
    R_star = np.array(df.loc[df[0]=='R_star', 1].values[0], dtype=float) * nc.r_sun
    gravity = nc.G * (M_pl)/(R_pl**2)

    # Read in template parameters
    df = pd.read_csv(petit_config, delim_whitespace=True, comment='#', header=None)
    P_0 = np.array(df.loc[df[0]=='P_0', 1].values[0], dtype=float)
    kappa_IR = np.array(df.loc[df[0]=='kappa_IR', 1].values[0], dtype=float)
    gamma = np.array(df.loc[df[0]=='gamma', 1].values[0], dtype=float)
    T_int = np.array(df.loc[df[0]=='T_int', 1].values[0], dtype=float)
    T_eq = np.array(df.loc[df[0]=='T_eq', 1].values[0], dtype=float)

    # Set-up atmosphere as radiative transfer object with varying pressure layers:
    atmosphere = Radtrans(line_species = species, \
                          rayleigh_species = ['H2', 'He'], \
                          continuum_opacities = ['H2-H2', 'H2-He'], \
                          wlen_bords_micron = [lambda_low/10.**4,lambda_up/10.**4], \
                          mode = 'lbl')
    # default rayleigh_species and continuum_opacities
    atmosphere.setup_opa_structure(pressures)

    # Define temperature and abundance vertical structure:
    temperatures = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_eq)

    # Make dictionary of abundances; solar abundances from Palme et al. 2014
    H2_mass_fraction = 2 * 1.008 * 1.
    He_mass_fraction = 4.00126 * (10.**(10.925-12))
    mass_fractions = {}
    mass_fractions['H2'] = H2_mass_fraction * np.ones_like(temperatures)
    mass_fractions['He'] = He_mass_fraction * np.ones_like(temperatures)
    total_mass_fraction = H2_mass_fraction + He_mass_fraction
    for s in species:
        s_mass = np.array(df.loc[df[0]==s+'_mass', 1].values[0], dtype=float)
        s_abundance = np.array(df.loc[df[0]==s+'_abundance', 1].values[0], dtype=float)
        s_mass_fraction = s_mass * (10.**(s_abundance-12))
        mass_fractions[s] = s_mass_fraction * np.ones_like(temperatures)
        total_mass_fraction += s_mass_fraction
    for key in mass_fractions:
        mass_fractions[key] /= total_mass_fraction
    MMW = np.array(df.loc[df[0]=='MMW', 1].values[0], dtype=float) * np.ones_like(temperatures)

    # Calculate apparent radius as a function of wavelength 
    if not Pcloud:
        atmosphere.calc_transm(temperatures, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P_0, contribution=contribution)
    else:
        atmosphere.calc_transm(temperatures, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P_0, Pcloud=Pcloud, contribution=contribution)
    wav_pl = nc.c/atmosphere.freq/1e-8
    rad_pl = atmosphere.transm_rad

    # Convert apparent radius to transmission spectrum
    flx_pl = rad2transSpec(wav_pl, rad_pl, R_pl, R_star, normalize=normalize, cutoff=cutoff, exclude=exclude, smooth=smooth)
    if not in_air:
        wav_pl = TSL.vacuum2air(wav_pl)

    # Set up save files
    species_str = '_'.join(species)
    lambda_str = str(lambda_low)+'_'+str(lambda_up)
    template_dir = 'planet_templates/'+planet_name+'/'
    template_filename = template_dir+planet_name+'_'+species_str+'_'+lambda_str+'_template.csv'
    PT_filename = template_dir+planet_name+'_'+species_str+'_'+lambda_str+'_PT.csv'
    contr_func_filename = template_dir+planet_name+'_'+species_str+'_'+lambda_str+'_contributionFunction'
    
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    if os.path.exists(template_filename):
        os.remove(template_filename)
    if os.path.exists(PT_filename):
        os.remove(PT_filename)
    
    # Make header for template file
    header_str = '# \n'
    header_str = header_str + '# Parameters \n'
    header_str = header_str + '# M_p: %f g \n'%M_pl
    header_str = header_str + '# R_p: %f cm \n'%R_pl
    header_str = header_str + '# R_star: %f cm \n'%R_star
    header_str = header_str + '# gravity: %f cm/s^2 \n'%gravity
    header_str = header_str + '# MMW: %f \n'%MMW[0]
    header_str = header_str + '# P_0: %f bars \n'%P_0
    header_str = header_str + '# P_low: %f bars \n'%min(pressures)
    header_str = header_str + '# P_up: %f bars \n'%max(pressures)
    header_str = header_str + '# kappa_IR: %f \n'%kappa_IR
    header_str = header_str + '# gamma: %f \n'%gamma
    header_str = header_str + '# T_int: %f K \n'%T_int
    header_str = header_str + '# T_eq: %f K \n'%T_eq
    header_str = header_str + '# P_low: %f bars \n'%min(pressures)
    header_str = header_str + '# P_up: %f bars \n'%max(pressures)
    for key in mass_fractions:
        header_str = header_str + '# %s mass fraction: %f \n'%(key, mass_fractions[key][0])
    header_str_template = '# '+template_filename+'\n' + header_str + 'Wavelength,Radius,Flux\n'
    
    # Save template spectrum
    df = pd.DataFrame(np.vstack([wav_pl, rad_pl, flx_pl]).T)
    df.head()
    with open(template_filename, 'a') as f:# Append
        f.write(header_str_template)
        df.to_csv(f, mode='a', header=False, index=False)
        f.close()
        
    # Save PT profile
    header_str_PT = '# '+PT_filename+'\n' + header_str + 'Pressure,Temperature\n'
    df = pd.DataFrame(np.vstack([pressures, temperatures]).T)
    df.head()
    with open(PT_filename, 'a') as f:# Append
        f.write(header_str_PT)
        df.to_csv(f, mode='a', header=False, index=False)
        f.close()

    # Save contribution function 
    if contribution:
        contr_func = atmosphere.contr_tr
        np.save(contr_func_filename, contr_func)
        return wav_pl, rad_pl, flx_pl, temperatures, contr_func
    else:
        return wav_pl, rad_pl, flx_pl, temperatures

def find_lines(pressures, contr_func, pressure_threshhold, order):
    ''' Identify absorption features and split into high vs. low altitude.

        Parameters
        ----------
        pressures : 1-D numpy array
            Model pressure levels (in bars)
        contr_func : 2-D numpy array
            Contribution function corresponding to contribution from each pressure level at a each wavelength
        pressure_threshhold : float
            Percentile of high altitude lines
        order : integer
            Input to finding relative minima; corresponds to how many points on each side for comparison 
            to find relative minima

        Returns
        -------
        pressures_eff : 1-D numpy array
            Effective pressure probed at each wavelength
        line_indices : 1-D numpy array
            Indices of wavelength array that correspond to a spectral line
        high_alt : 1-D boolean array
            True if index in line_indices corresponds to a high altitude line
    '''
    pressure_eff = np.dot(pressures, contr_func) # effective pressure probed at each wavelength
    line_indices, = scipy.signal.argrelextrema(pressure_eff, np.less, order=order)
    high_alt = pressure_eff[line_indices] <= np.percentile(pressure_eff[line_indices], pressure_threshhold)
    return pressure_eff, line_indices, high_alt

def make_altitude_mask(alt_indices, width, num_wav):
    ''' Builds a mask for specifying lines of a given alitude assuming a fixed line width.
    
        Parameters
        ----------
        alt_indices : 1-D numpy array
            Array of indices corresponding to spectral line centers for a given altitude bin
        width : integer
            Assumed width of lines
        num_wav : integer
            Number of elements in wavelength array
            
        Returns
        -------
        ind_lines_center : 1-D numpy array
            Array of indices corresponding to spectral line centers for a given altitude bin       
        ind_lines: list of 1-D numpy arrays
            Array of indices corresponding to spectral lines assuming a fixed line width
            
    '''
    ind_lines_center = np.array([], dtype=int)
    ind_lines = []
    for ind in alt_indices:
        ind_low = ind - width
        ind_up = ind + width + 1
        if ind_low < 0:
            ind_low = 0
        if ind_up > num_wav - 1:
            ind_up = num_wav - 1
        ind_lines_center = np.hstack([ind_lines_center, np.arange(ind_low, ind_up, dtype=int)])
        ind_lines.append(np.arange(ind_low, ind_up, dtype=int))
    return ind_lines_center, ind_lines 

def make_altitude_temp(flx_temp, rad_temp, alt_mask, R_p):
    ''' Builds template for masked lines.
        
        Parameters
        ----------
        flx_temp : 1-D numpy array
            Planet template spectrum
        rad_temp : 1-D numpy array
            Planet radius as a function of wavelength
        alt_mask : 1-D numpy array
            Indices corresponding to spectra lines
        R_p : float with units
            Radius of planet
        
        Returns
        -------
        flx_temp_alt : 1-D numpy array
            Planet template spectrum for given altitude bin
        rad_temp_alt : 1-D numpy array
            Planet radius as a function of wavelength for lines in a given altitude bin
    '''
    flx_temp_alt = np.ones_like(flx_temp)
    flx_temp_alt[alt_mask] = flx_temp[alt_mask]
    rad_temp_alt = np.ones_like(flx_temp) * R_p.to(u.cm).value
    rad_temp_alt[alt_mask] = rad_temp[alt_mask]
    return flx_temp_alt, rad_temp_alt
