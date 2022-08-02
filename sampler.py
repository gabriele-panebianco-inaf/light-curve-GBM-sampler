from time import time
EXECUTION_TIME_START = time()

import logging
import numpy as np
import os

from astropy.io import fits
from astropy.table import QTable
import astropy.units as u

from write_light_curve import *


# Put here some configuration parameters. Should I put them into a YAML file?
GBM_Catalog = "/home/gabriele/Documents/fermiGBM/light-curve-GBM-sampler/GBM_burst_archive/"
GBM_Catalog+= "GBM_bursts_flnc_band.fits"
Name_Transient = "GRB120817168" #"GRB160530667" #None
Random_seed = 37
Spectral_Model_Type = "flnc" # pflx or flnc
Spectral_Model_Name = "band" # plaw, comp, band, sbpl
Output_Directory = "/home/gabriele/Documents/fermiGBM/light-curve-GBM-sampler/Output/"

######################################################





if __name__ == '__main__':

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format ='%(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set Script Arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-f", "--configurationfile", help="name of the configuration YAML file")
    # args = parser.parse_args()

    # Load the GBM Burst Catalog
    with fits.open(GBM_Catalog) as hdulist:
        table_catalog = QTable.read(hdulist['CATALOG'])
        table_catalog.add_index('name')
    
    # Setup the random sampler
    rng = np.random.default_rng(Random_seed)

    # Choose a burst
    if Name_Transient is not None:
        try:
            transient = table_catalog.iloc[table_catalog['name'] == Name_Transient][0]
        except IndexError:
            logger.error(f"Requested {Name_Transient} was not found in the catalog.\n")
            raise
    else:
        random_index = rng.integers(0, high=len(table_catalog))
        transient = table_catalog[random_index]


    # Make Output Directories
    os.makedirs(os.path.dirname(Output_Directory+"Logs/"), exist_ok=True)

    # Define Logger for file too
    f_handler = logging.FileHandler(Output_Directory+f"Logs/{transient['name']}.log", mode='w')
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(logging.Formatter('%(asctime)s. %(levelname)s: %(message)s'))# -%(name)s
    logger.addHandler(f_handler)

    # Print info on what we have done until now.
    logger.info(f"GBM Bursts Catalog: {GBM_Catalog}")
    logger.info(f"Number of GRBs in the catalog: {len(table_catalog)}.")
    if Random_seed is not None:
        logger.info(f"Set Random seed: {Random_seed}.")
    else:
        logger.warning(f"No Random seed set: this run cannot be reproduced.")
    if Name_Transient is not None:
        logger.info(f"Transient {transient['name']} was requested.\n")
    else:
        logger.info(f"Transient {transient['name']} was randomly chosen.\n")

    # Check correct input
    if Spectral_Model_Name != "band":
        raise NotImplementedError("Only band spectral models allowed.")


    # Define Transient Parameters
    logger.info(f"{15*'='}TRANSIENT PARAMETERS{25*'='}")

    lii = transient['lii'].to("deg").value
    bii = transient['bii'].to("deg").value
    label = Spectral_Model_Type +'_' + Spectral_Model_Name + '_'
    ampl  = transient[label+'ampl' ]
    epeak = transient[label+'epeak']
    alpha = transient[label+'alpha']
    beta  = transient[label+'beta' ]
    epiv  = 100 * u.keV
    flux = transient[label+'phtflux']
    
    logger.info(f"Galactic coordinates: l={lii} deg, b={bii} deg.")
    logger.info(f"Spectral model: {Spectral_Model_Name}. Type: {Spectral_Model_Type}.")
    logger.info(f"{label}ampl : {ampl }")
    logger.info(f"{label}epeak: {epeak}")
    logger.info(f"{label}alpha: {alpha}")
    logger.info(f"{label}beta : {beta }")
    logger.info(f"{label}epiv : {epiv }")
    logger.info(f"{label}phtflux : {flux}")


    # Sample two random numbers for polarization
    polarization_ampli = rng.random()         # random number bewtween [0,1)
    polarization_phase = rng.random() * 180.0 # random number bewtween [0,180.0)

    logger.info(f"RANDOM Polarization amplitude: {polarization_ampli}")
    logger.info(f"RANDOM Polarization phase: {polarization_phase}")

    logger.info(f"{60*'='}\n")


    # Gaussian Light Curve
    light_curve_output_name = Write_Gaussian_light_curve(transient, logger, Output_Directory)

    # Empirical Light Curve
    LC_info = Empirical_Light_Curve(transient, logger, Output_Directory)


    # Write a source file
    source_file_output_name = Output_Directory + f"{transient['name']}.source"
    logger.info(f"Write Source file: {source_file_output_name}")
    with open(source_file_output_name, 'w') as f:

        # Introduction on the sampler
        f.write(f"# Source file template: https://github.com/zoglauer/megalib/blob/main/resource/examples/cosima/source/CrabOnly.source \n")
        
        if Random_seed is not None:
            f.write(f"# Random seed: {Random_seed}\n")
        else:
            f.write(f"# Random seed not set: run cannot be reproduced.\n")

        if Name_Transient is not None:
            f.write(f"# Input transient: {transient['name']} (explicitly requested by user).\n")
        else:
            f.write(f"# Input transient: {transient['name']} (randomly sampled).\n")
        
        f.write(f"# Spectral Model:    {Spectral_Model_Name}.\n")
        f.write(f"# Spectral Fit type: {Spectral_Model_Type}.\n")
        


        # General Parameters, Physics list, Output formats
        version = 1
        geometry_str = "$(MEGALIB)/resource/examples/geomega/mpesatellitebaseline/SatelliteWithACS.geo.setup"
        PhysicsListEM = "LivermorePol"
        StoreSimulationInfo = "init-only"

        f.write(f"\n# Global Parameters\n")
        f.write(f"Version                     {version}\n")
        f.write(f"Geometry                    {geometry_str}\n")
        f.write(f"\n# Physics list\n")     
        f.write(f"PhysicsListEM               {PhysicsListEM}\n")
        f.write(f"\n# Output formats\n")         
        f.write(f"StoreSimulationInfo         {StoreSimulationInfo}\n")
        


        # Run and Source Parameters
        RunName = "GRBSim"
        SourceName = transient['name']
        RunTime = 1000.0
        SourceName_Beam = "FarFieldPointSource" # This should work for GRBs
        SourceParticleType = 1                  # 1=photon
        
        SourceName_Spectrum = "Band"


        f.write(f"\n# Run and source parameters\n")
        f.write(f"Run                         {RunName}\n")
        f.write(f"{RunName}.FileName             {SourceName}\n")
        f.write(f"{RunName}.Time                 {RunTime}\n")
        f.write(f"{RunName}.Source               {SourceName}\n")
        f.write(f"{SourceName}.ParticleType   {SourceParticleType}\n")
        f.write(f"{SourceName}.Beam           {SourceName_Beam} {0} {0}\n")
        f.write(f"{SourceName}.Orientation    Galactic Fixed {lii} {bii}\n")
        
        f.write(f"\n# Band Spectrum parameters: Flux integration min and max energies, alpha, beta, epeak\n")
        f.write(f"{SourceName}.Spectrum       {SourceName_Spectrum} {transient['flu_low'].value} {transient['flu_high'].value} {alpha} {beta} {epeak.value}\n")
        
        f.write(f"\n# Average photon flux, in photon/cm2/s, for a Band function law fit to a single spectrum over the duration of the burst.\n")
        f.write(f"{SourceName}.Flux           {flux.value}\n")
        
        f.write(f"\n# Polarization: random numbers from constant distribution. 1st one in [0,1], 2nd one in [0,180]\n")
        f.write(f"{SourceName}.Polarization   RelativeX {polarization_ampli} {polarization_phase}\n")
        
        f.write(f"\n# GBM Light Curve.\n")
        f.write(f"{SourceName}.Lightcurve     File false {LC_info.output_name}\n")  # false: not repeating
        f.write(f"\n# Info on the Light curve.\n")
        f.write(f"# 1st column is the time point in [s]. GBM times are expressed wrt trigger time, we shifted them to start the simulation from 0.0.\n")
        f.write(f"# After the shift GBM Trigger time is at {LC_info.trigger:.5f} s. Light curve stops at {LC_info.stop:.5f} s. Time resolution is {LC_info.step:.5f} s. There are {LC_info.num} points.\n")
        f.write(f"# 2nd column is light curve value in [1/s]. Time integral of the light curve is normalized to 1 like a Probability Distribution Function.\n")
        f.write(f"# The light curve values are excess rates: observed data - best fit background model. Negative excess rates are manually set to 0.0.\n")
        f.write(f"# We selected the events from GBM detector {LC_info.det} with energy in [{LC_info.emin:.1f},{LC_info.emax:.1f}] keV.\n")



    # End
    logger.info(f"Total execution time: {np.round(time()-EXECUTION_TIME_START, 3)} s.")
