import os
from os.path import isfile, join

import numpy as np
import shutil

from gbm.binning.unbinned import bin_by_time
from gbm.data import TTE, GbmDetectorCollection
from gbm.finder import TriggerFtp


DETECTORS = np.array(['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1'])

def Empirical_Light_Curve(transient, logger, Output_Directory):
    """
    Given a transient name find its trigger data, download it, analyze it and get the light curve.

    Parameters
    ----------
    transient: `astropy.table.row.Row`
        Row of the chosen transient.
    logger : `logging.Logger`
        Logger from main.
    Output_Directory : str
        Path to the Output directory.

    Returns
    -------
    

    """

    logger.info(f"{15*'='}Get the Empirical GBM Light Curve")

    # Read the detector mask from Burst Catalog entry and turn it into a list of bools.
    det_mask = list(str(int(transient['scat_detector_mask'])))
    det_mask = [bool(int(d)) for d in det_mask]
    detectors = DETECTORS[det_mask].tolist()

    logger.info(f"List of detectors used by the GBM analysis: {detectors}")

    # Find the Burst data in the Online Archive
    trig_finder = TriggerFtp(transient['trigger_name'][2:])

    # Make a Directory to host the TTE Data and Download them
    try:
        Temp_Directory = Output_Directory+"Temp/"
        os.makedirs(os.path.dirname(Temp_Directory), exist_ok=True)
        # trig_finder.get_tte(Temp_Directory,dets=detectors)
    except:
        os.rmdir(Temp_Directory)
        logger.error("Could not get the GBM TTE files. Deleting temporary directory.")
        raise

    # Load the TTE files into the code
    tte_filenames = [f for f in os.listdir(Temp_Directory) if isfile(join(Temp_Directory, f))]
    ttes = [TTE.open(Temp_Directory+f) for f in tte_filenames]

    # Time binning: turn the TTE files into CSPEC files
    bintime = (transient['flnc_spectrum_stop'].value - transient['flnc_spectrum_start'].value)/200.0
    time_range = (transient['back_interval_low_start'].unmasked.value, transient['back_interval_high_stop'].unmasked.value)

    logger.info(f"Selecting time range: {time_range} s, binning {bintime} s.")

    cspecs = [t.to_phaii(bin_by_time, bintime, time_range = time_range, time_ref = 0.0) for t in ttes]
    cspecs = GbmDetectorCollection.from_list(cspecs)

    [logger.info(f"Det: {c.detector} | Energy: {c.energy_range} keV | Time: {c.time_range} s") for c in cspecs]


    # Remove GBM files and directory
    try:
        #shutil.rmtree(Temp_Directory)
        logger.info("Delete temporary GBM files and directory")
    except OSError as e:
        logger.error(f"Error: {Temp_Directory} : {e.strerror}")




    return None