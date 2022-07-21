from time import time
EXECUTION_TIME_START = time()
import logging
import numpy as np
import os

from astropy.io import fits
from astropy.table import QTable
from astropy.modeling.models import Gaussian1D
import astropy.units as u

from scipy.special import erfinv


# Put here some configuration parameters. Should I put them into a YAML file?
GBM_Catalog = "/home/gabriele/Documents/fermiGBM/light-curve-GBM-sampler/GBM_burst_archive/"
GBM_Catalog+= "GBM_bursts_flnc_band.fits"
Name_Transient = None # "GRB160530667"
Random_seed = 0
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
    os.makedirs(os.path.dirname(Output_Directory+"Sources/"), exist_ok=True)
    os.makedirs(os.path.dirname(Output_Directory+"LightCurves/"), exist_ok=True)

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

    # Write Coordinates in degree
    #ra  = transient['ra' ].to("deg").value
    #dec = transient['dec'].to("deg").value
    #logger.info(f"Ra, Dec in degrees. ra={ra}, dec={dec}.")

    lii = transient['lii'].to("deg").value
    bii = transient['bii'].to("deg").value

    logger.info(f"Galactic latitude and longitude in deg. l={lii}, b={bii}.")

    # Write spectral parameters

    if Spectral_Model_Name != "band":
        raise NotImplementedError("Only band spectral models allowed.")
    
    logger.info(f"Spectral model: {Spectral_Model_Name}. Type: {Spectral_Model_Type}.")
    label = Spectral_Model_Type +'_' + Spectral_Model_Name + '_'
    
    ampl  = transient[label+'ampl' ]
    epeak = transient[label+'epeak']
    alpha = transient[label+'alpha']
    beta  = transient[label+'beta' ]
    epiv  = 100 * u.keV

    logger.info(f"{label}ampl : {ampl }")
    logger.info(f"{label}epeak: {epeak}")
    logger.info(f"{label}alpha: {alpha}")
    logger.info(f"{label}beta : {beta }")
    logger.info(f"{label}epiv : {epiv }")

    flux = transient[label+'phtflux']
    #fluxb = transient[label+'phtfluxb']

    logger.info(f"{label}phtflux : {flux}")
    #logger.info(f"{label}phtfluxb : {fluxb}")

    # Now sample two random numbers for polarization

    polarization_ampli = rng.random()         # random number bewtween [0,1)
    polarization_phase = rng.random() * 180.0 # random number bewtween [0,180.0)

    logger.info(f"Polarization amplitude: {polarization_ampli}")
    logger.info(f"Polarization phase: {polarization_phase}")

    # Now read the gaussian temporal shape

    gauss_peak_time = (transient['pflx_spectrum_start']+transient['pflx_spectrum_stop']) / 2.0
    gauss_sigma = transient['t90'] / (2*np.sqrt(2)*erfinv(0.90))
    gauss_amplitude = 1 / (gauss_sigma * np.sqrt(2 * np.pi)) # Normalization: integral is 1.

    logger.info(f"Highest flux recorded in catalog at [{gauss_peak_time}] from trigger time.")
    logger.info(f"GBM t90 ={transient['t90']}.")
    logger.info(f"Gaussian Pulse sigma={gauss_sigma}.")

    light_curve = Gaussian1D(amplitude=gauss_amplitude, mean=gauss_peak_time, stddev=gauss_sigma)

    logging.info(light_curve)

    # Prepare the time array (x)
    time_start = transient['flnc_spectrum_start']   # 'back_interval_low_stop'
    time_stop  = transient['flnc_spectrum_stop']    # 'back_interval_high_start'

    time_num = 200
    time_step = (time_stop - time_start)/time_num

    logger.info(f"Evaluate {time_num} time points in [{time_start},{time_stop}]. Step = {time_step}.")

    time_array = np.linspace(time_start, time_stop, time_num)
    lc_values = light_curve(time_array)

    light_curve_table = QTable([time_array, lc_values], names=('time', 'curve'), meta={'name': 'Gaussian'})

    logger.info(light_curve_table)

    light_curve_output_name = Output_Directory+f"LightCurves/Gauss_{transient['name']}.dat"
    logger.info(f"Print Gaussian Light Curve: {light_curve_output_name}")

    with open(light_curve_output_name, 'w') as f:
        f.write(f"IP LINLIN\n")
        for row in light_curve_table:
            f.write(f"DP {row['time'].value} {row['curve'].value}\n")
        f.write(f"EN\n")
    #light_curve_table.write(light_curve_output_name, format='ascii', overwrite=True)


    # Write a source file
    source_file_output_name = Output_Directory + f"Sources/{transient['name']}.source"
    logger.info(f"Write Source file: {source_file_output_name}")
    with open(source_file_output_name, 'w') as f:
        f.write(f"# Example run for Cosima, based on {transient['name']}\n")
        f.write(f"#This file is similar to template https://github.com/zoglauer/megalib/blob/main/resource/examples/cosima/source/CrabOnly.source \n")
        
        
        f.write("\n# Global parameters !!! Should I print these? How do I choose them?\n")
        version = 1
        geometry_str = "$(MEGALIB)/resource/examples/geomega/mpesatellitebaseline/SatelliteWithACS.geo.setup"
        f.write(f"Version         {version}\n")
        f.write(f"Geometry        {geometry_str}\n")
        
        
        f.write(f"\n#Physics list !!! Should I print it? How do I choose it?\n")
        PhysicsListEM = "LivermorePol"
        f.write(f"PhysicsListEM               {PhysicsListEM}\n")
        
        
        f.write(f"\n#Output formats !!! Should I print it?\n")
        StoreSimulationInfo = "all"
        f.write(f"StoreSimulationInfo         {StoreSimulationInfo}\n")
        
        
        f.write(f"\n# Run & source parameters\n")
        RunName = "GRBSim"
        RunName_FileName = "GRBOnlyObservation"
        RunTime = 1000.0

        f.write(f"Run {RunName}\n")

        f.write("# !!! Should I print .FileName and .Time? How do I choose .Time?\n")
        f.write(f"{RunName}.FileName {RunName_FileName}\n")
        f.write(f"{RunName}.Time {RunTime}\n")

        SourceName = transient['name']
        SourceName_Beam = "FarFieldPointSource" # This should work for GRBs
        SourceParticleType = 1 # 1=photon
        SourceName_Spectrum = "Band"


        f.write(f"{RunName}.Source {SourceName}\n")
        f.write(f"{SourceName}.ParticleType           {SourceParticleType}\n")
        f.write(f"# !!! Is 0 0 always okay after FarFieldPointSource?\n")
        f.write(f"{SourceName}.Beam                   {SourceName_Beam} {0} {0}\n")
        f.write(f"{SourceName}.Orientation            Galactic Fixed {lii} {bii}\n")
        f.write(f"# I print: Flux integration min&max energies, alpha, beta, epeak\n")
        f.write(f"{SourceName}.Spectrum               {SourceName_Spectrum} {transient['flu_low'].value} {transient['flu_high'].value} {alpha} {beta} {epeak.value}\n")
        f.write(f"# I print field flnc_band_phtflux of the catalog, in cm-2 s-1\n")
        f.write(f"{SourceName}.Flux                   {flux.value}\n")
        f.write(f"# !!! Is RelativeX always okay? Then I print two random numbers from constant distribution. 1st one in [0,1], 2nd one in [0,180]\n")
        f.write(f"{SourceName}.Polarization           RelativeX {polarization_ampli} {polarization_phase}\n")
        f.write(f"# !!! Should I print the full path to the file? Is the format okay?\n")# false: not repeating
        f.write(f"# !!! This Lightcurve has columns time (s) and value (1/s), normalized to 1 like a Probability Distribution Function. Are these formats okay?\n")
        f.write(f"{SourceName}.Lightcurve             File false {light_curve_output_name}\n")



    # End
    logger.info(f"Total execution time: {np.round(time()-EXECUTION_TIME_START, 3)} s.")
