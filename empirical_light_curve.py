import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from os.path import isfile, join

from gbm.background import BackgroundFitter
from gbm.background.binned import Polynomial
from gbm.binning.unbinned import bin_by_time
from gbm.data import TTE, GbmDetectorCollection
from gbm.data.primitives import TimeBins
from gbm.detectors import Detector
from gbm.finder import TriggerFtp

DETECTORS = np.array(['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1'])
FIGURE_FORMAT = ".pdf"

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
    logger.info(f"Connect to Database...")
    #trig_finder = TriggerFtp(transient['trigger_name'][2:])

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

    # Background Fitting to get the excess
    bkgd_range = [(transient['back_interval_low_start'].unmasked.value, transient['back_interval_low_stop'].unmasked.value), (transient['back_interval_high_start'].unmasked.value, transient['back_interval_high_stop'].unmasked.value)]
    backfitters = [BackgroundFitter.from_phaii(cspec, Polynomial, time_ranges = bkgd_range) for cspec in cspecs]
    backfitters = GbmDetectorCollection.from_list(backfitters, dets=cspecs.detector())
    logger.info(f"Fit Background in {bkgd_range} s.")
    backfitters.fit(order = 1)
    logger.info(f"Interpolate over source time range...")
    bkgds = backfitters.interpolate_bins(cspecs.data()[0].tstart, cspecs.data()[0].tstop)
    bkgds = GbmDetectorCollection.from_list(bkgds, dets=cspecs.detector())

    # Energy integration of data and background
    # Energy Ranges in keV
    erange_nai = ( 10.0,   900.0)
    erange_bgo = (250.0, 40000.0)

    # Energy intergration
    data_timebins = cspecs.to_lightcurve(nai_kwargs = {'energy_range':erange_nai},
                                            bgo_kwargs = {'energy_range':erange_bgo}
                                    )
    bkgd_timebins = bkgds.integrate_energy(nai_args = erange_nai,
                                            bgo_args = erange_bgo
                                      )
    
    # Save
    for data_t, bkgd_t, det in zip(data_timebins, bkgd_timebins, detectors):

        data = data_t.slice(transient['back_interval_low_stop'].unmasked.value,transient['back_interval_high_start'].unmasked.value)
        bkgd_lo_edges = bkgd_t.time_centroids - 0.5*bkgd_t.time_widths
        bkgd_hi_edges = bkgd_t.time_centroids + 0.5*bkgd_t.time_widths
        bkgd = TimeBins(bkgd_t.counts, bkgd_lo_edges, bkgd_hi_edges, bkgd_t.exposure)
        bkgd = bkgd.slice(transient['back_interval_low_stop'].unmasked.value,transient['back_interval_high_start'].unmasked.value)
        
        # Compute excess and time slice
        if not np.allclose(data.centroids, bkgd.centroids):
            raise
        widths = data.widths
        centroids = data.centroids

        excess = data.rates - bkgd.rates
        normalization = np.sum(excess*widths)
        pdf = excess / normalization

        light_curve_output_name = Output_Directory+f"LightCurves/{transient['name']}_{det}.dat"
        logger.info(f"Print Empirical Light Curve: {light_curve_output_name}")

        with open(light_curve_output_name, 'w') as f:
            f.write(f"IP LINLIN\n")
            for t,d in zip(centroids, pdf):
                f.write(f"DP {t} {d}\n")
            f.write(f"EN\n")

        # Save plots

        # Define pyplot Figure and Axes
        fig, axs = plt.subplots(1, figsize = (15,5) )
    
        axs.step(centroids, pdf, label = 'Excess rates', color = 'C0', where = 'mid')
        axs.set_xlabel('Time since trigger (s)', fontsize = 'large')
        axs.set_ylabel('Excess rates pdf (1/s)', fontsize = 'large')
        if Detector.from_str(det).is_nai():
            erange_det = erange_nai
        else:
            erange_det = erange_bgo
    
        plot_title = 'Lightcurve. Excess rates of detector: '+det+'. Energy range '+str(erange_det)+' keV.'
        axs.set_title(plot_title, fontsize = 'large')
        #axs.set_xlim(view_range[0], view_range[1])
        axs.grid()
        axs.legend()

        figure_name = Output_Directory+f"LightCurves/{transient['name']}_{det}"+FIGURE_FORMAT
        fig.savefig(figure_name, facecolor = 'white')
    

        

    # Remove GBM files and directory
    try:
        #shutil.rmtree(Temp_Directory)
        logger.info("Delete temporary GBM files and directory")
    except OSError as e:
        logger.error(f"Error: {Temp_Directory} : {e.strerror}")




    return None