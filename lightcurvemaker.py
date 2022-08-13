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

DEFAULT_CATALOG = "./GBM_burst_archive/GBM_bursts_flnc_band_flux15.fits"
DEFAULT_OUTPUT = "./Archive_Light_Curve/"

if __name__ == '__main__':

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format ='%(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set Script Arguments
    parser = ArgumentParser(description="Sample a GRB from GBM Burst Archive")
    parser.add_argument("-i", "--index", help="GRB Index in the catalog.", type=int, default=0)
    parser.add_argument("-c", "--catalogue", help="Input Transient Catalogue", type=str, default=DEFAULT_CATALOG)
    parser.add_argument("-o"   , "--outputdir"   , help="Output Directory"               , type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("-nai" , "--usenai"      , help="Download and produce LC of NaIs", type=int, default=1)
    args = parser.parse_args()

    Transient_Index = args.index
    GBM_Catalog = args.catalogue
    Output_Directory = args.outputdir
    Use_Nai = bool(args.usenai)

    if Transient_Index < 0:
        raise ValueError(f"Index must be non negative, {Transient_Index} was given.")

    # Load the GBM Burst Catalog
    with fits.open(GBM_Catalog) as hdulist:
        table_catalog = QTable.read(hdulist['CATALOG'])
        table_catalog.add_index('name')


    transient = table_catalog[Transient_Index]

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
    logger.info(f"Transient selected: {transient['name']}. Index: {Transient_Index}.\n")
    
    # Transient Parameters
    logger.info(f"{15*'='}TRANSIENT PARAMETERS{35*'='}")

    label = "flnc_band_"
    ampl  = transient[label+'ampl' ]
    epeak = transient[label+'epeak']
    alpha = transient[label+'alpha']
    ebreak= epeak / (2+alpha)
    beta  = transient[label+'beta' ]
    flux  = transient[label+'phtflux']
    
    logger.info(f"Galactic coordinates: l={transient['lii'].value:.3f} deg, b={transient['bii'].value:.3f} deg.")
    logger.info(f"{label}phtflux : {flux.value:.3f} {flux.unit}.")
    logger.info(f"{label}ampl  : {ampl.value:.3e} {ampl.unit}.")
    logger.info(f"{label}epeak : {epeak.value:.1f} {epeak.unit}.")
    logger.info(f"{label}ebreak: {ebreak.value:.1f} {ebreak.unit}.")
    logger.info(f"{label}alpha : {alpha:.3f}.")
    logger.info(f"{label}beta  : {beta:.3f}.")

    logger.info(f"{70*'='}\n")

    # Empirical Light Curve
    try:        
        os.makedirs(os.path.dirname(Output_Directory+".Temp/"), exist_ok=True)
        LC_info = Empirical_Light_Curve(transient, logger, Output_Directory, Use_Nai)
    except:
        logger.error("Delete temporary GBM files and directory")
        shutil.rmtree(Output_Directory+".Temp/")
        raise

    # End
    logger.info(f"Total execution time: {np.round(time()-EXECUTION_TIME_START, 3)} s.\n")

    