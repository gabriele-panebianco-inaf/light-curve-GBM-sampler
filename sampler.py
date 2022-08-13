from time import time
EXECUTION_TIME_START = time()

import logging
import numpy as np
import os

from argparse import ArgumentParser
from astropy.io import fits
from astropy.table import QTable
import astropy.units as u

from write_light_curve import *

SPECTRAL_TYPE_LIST = ["flnc", "pflx"]
SPECTRAL_TYPE_IMPL = ["flnc"]
SPECTRAL_FUNC_LIST = ["plaw", "comp", "band", "sbpl"]
SPECTRAL_FUNC_IMPL = ["band"]
DEFAULT_CATALOG = "./GBM_burst_archive/GBM_bursts_flnc_band.fits"
DEFAULT_OUTPUT = "./Output/"
DEFAULT_LCDIRE = "./Archive_Light_Curve/"

if __name__ == '__main__':

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format ='%(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set Script Arguments
    parser = ArgumentParser(description="Sample a GRB from GBM Burst Archive")
    parser.add_argument("-s"   , "--randomseed"  , help="Specify random seed."           , type=int, default=None)
    parser.add_argument("-t"   , "--transient"   , help="Name of the transient"          , type=str, default=None)
    parser.add_argument("-type", "--spectraltype", help="Spectral Model Type"            , type=str, default="flnc")
    parser.add_argument("-func", "--spectralfunc", help="Spectral Model Function"        , type=str, default="band")
    parser.add_argument("-c"   , "--catalogue"   , help="Input Transient Catalogue"      , type=str, default=DEFAULT_CATALOG)
    parser.add_argument("-o"   , "--outputdir"   , help="Output Directory"               , type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("-m"   , "--mode"        , help="0=Light curve from Local archive. 1=Light curve from downloaded TTE data, only BGO. 2=Light curve from downloaded TTE data, both NaI and BGO.", type=int, default=0)
    parser.add_argument("-l"   , "--lcurvedir"   , help="Local Directory with Light Curve Files.", type=str, default=DEFAULT_LCDIRE)
    args = parser.parse_args()

    Random_seed         = args.randomseed
    Name_Transient      = args.transient
    Spectral_Model_Type = args.spectraltype
    Spectral_Model_Name = args.spectralfunc
    GBM_Catalog         = args.catalogue
    Light_Curve_Archive = args.lcdirectory
    Output_Directory    = args.outputdir

    if not (Spectral_Model_Type in SPECTRAL_TYPE_LIST):
        raise ValueError(f"Spectral Fit type {Spectral_Model_Type} not supported. Supported types: {SPECTRAL_TYPE_LIST}.")
    if not (Spectral_Model_Type in SPECTRAL_TYPE_IMPL):
        raise NotImplementedError(f"Spectral Fit type {Spectral_Model_Type} not implemented. Currently implemented: {SPECTRAL_TYPE_IMPL}.")
    
    if not (Spectral_Model_Name in SPECTRAL_FUNC_LIST):
        raise ValueError(f"Spectral Model {Spectral_Model_Name} not supported. Supported models: {SPECTRAL_FUNC_LIST}.")
    if not (Spectral_Model_Name in SPECTRAL_FUNC_IMPL):
        raise NotImplementedError(f"Spectral Model type {Spectral_Model_Name} not implemented. Currently implemented: {SPECTRAL_FUNC_IMPL}.")
    
    if args.mode == 0:
        Download_Curve = False
        Use_Nai = False
    elif args.mode == 1:
        Download_Curve = True
        Use_Nai = False
    elif args.mode == 2:
        Download_Curve = True
        Use_Nai = True
    else:
        raise ValueError(f"Mode must be 0, 1 or 2. {args.mode} was given.")
    
    
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




    # Define Transient Parameters
    logger.info(f"{15*'='}TRANSIENT PARAMETERS{35*'='}")

    lii = transient['lii'].to("deg").value
    bii = transient['bii'].to("deg").value
    label = Spectral_Model_Type +'_' + Spectral_Model_Name + '_'
    ampl  = transient[label+'ampl' ]
    epeak = transient[label+'epeak']
    alpha = transient[label+'alpha']
    ebreak= epeak / (2+alpha)
    beta  = transient[label+'beta' ]
    epiv  = 100 * u.keV
    flux = transient[label+'phtflux']
    
    logger.info(f"Galactic coordinates: l={lii} deg, b={bii} deg.")
    logger.info(f"Spectral model: {Spectral_Model_Name}. Type: {Spectral_Model_Type}.")
    logger.info(f"{label}ampl  : {ampl.value:.3e} {ampl.unit}.")
    logger.info(f"{label}epeak : {epeak.value:.1f} {epeak.unit}.")
    logger.info(f"{label}ebreak: {ebreak.value:.1f} {ebreak.unit}.")
    logger.info(f"{label}alpha : {alpha:.3f}.")
    logger.info(f"{label}beta  : {beta:.3f}.")
    logger.info(f"{label}epiv  : {epiv.value:.0f} {epiv.unit}.")
    logger.info(f"{label}phtflux : {flux.value:.3f} {flux.unit}.")


    # Sample two random numbers for polarization
    polarization_ampli = rng.random()         # random number bewtween [0,1)
    polarization_phase = rng.random() * 180.0 # random number bewtween [0,180.0)

    logger.info(f"RANDOM Polarization amplitude: {polarization_ampli:.5f}")
    logger.info(f"RANDOM Polarization phase    : {polarization_phase:.5f}")

    logger.info(f"{70*'='}\n")

    if Download_Curve:
        try:
            # Gaussian Light Curve
            #light_curve_output_name = Write_Gaussian_light_curve(transient, logger, Output_Directory)

            # Empirical Light Curve
            os.makedirs(os.path.dirname(Output_Directory+".Temp/"), exist_ok=True)
            LC_info = Empirical_Light_Curve(transient, logger, Output_Directory, Use_Nai)
        except:
            logger.error("Delete temporary GBM files and directory")
            shutil.rmtree(Output_Directory+".Temp/")
            raise
    else:
        logger.info(f"{15*'='}SAMPLE LIGHT CURVE FROM LOCAL DIRECTORY{16*'='}")
        files = [f for f in os.listdir(Light_Curve_Archive) if isfile(join(Light_Curve_Archive, f))]
        random_index_lc = rng.integers(0, high=len(files))
        LC_info = Light_Curve_Info(output_name=files[random_index])
        logger.info(f"{70*'='}\n")



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
        f.write(f"{RunName}.OrientationSky       Galactic Fixed 180.0 90.0\n")
        f.write(f"{RunName}.Source               {SourceName}\n")
        f.write(f"{SourceName}.ParticleType   {SourceParticleType}\n")
        f.write(f"{SourceName}.Beam           {SourceName_Beam} {0} {0}\n")

        f.write(f"# Orientation: Latitude b [deg], Longitude l [deg]\n")
        f.write(f"{SourceName}.Orientation    Galactic Fixed {bii} {lii}\n")
        
        f.write(f"\n# Band Spectrum parameters: Flux integration min and max energies, alpha, beta, ebreak\n")
        f.write(f"{SourceName}.Spectrum       {SourceName_Spectrum} {transient['flu_low'].value} {transient['flu_high'].value} {alpha:.3f} {beta:.3f} {ebreak.value:.3f}\n")
        
        f.write(f"\n# Average photon flux, in photon/cm2/s, for a Band function law fit to a single spectrum over the duration of the burst.\n")
        f.write(f"{SourceName}.Flux           {flux.value:.4f}\n")
        
        f.write(f"\n# Polarization: random numbers from constant distribution. 1st one in [0,1], 2nd one in [0,180]\n")
        f.write(f"{SourceName}.Polarization   RelativeX {polarization_ampli:.5f} {polarization_phase:.5f}\n")
        
        f.write(f"\n# GBM Light Curve.\n")
        f.write(f"{SourceName}.Lightcurve     File false {LC_info.output_name}\n")  # false: not repeating
       


    # End
    logger.info(f"Total execution time: {np.round(time()-EXECUTION_TIME_START, 3)} s.")
