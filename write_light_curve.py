from curses import erasechar
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from os.path import isfile, join
from scipy.special import erfinv
from astropy.modeling.models import Gaussian1D

from gbm.background import BackgroundFitter
from gbm.background.binned import Polynomial
from gbm.binning.unbinned import bin_by_time
from gbm.data import TTE, GbmDetectorCollection
from gbm.data.primitives import TimeBins
from gbm.detectors import Detector
from gbm.finder import TriggerFtp

DETECTORS = np.array(['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1'])
FIGURE_FORMAT = ".pdf"


class Light_Curve_Info:
    def __init__(self, output_name="", trigger=0.0, step=0.0, stop=0.0, num=0, det="", emin=0.0, emax=0.0):
        self.output_name = output_name
        self.trigger = trigger
        self.step = step
        self.stop = stop
        self.num = num
        self.det = det
        self.emin = emin
        self.emax = emax

def Write_Gaussian_light_curve(transient, logger, Output_Directory):
    """
    Write a simple lightcurve file: gaussian shape.
    
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
    light_curve_output_name : str
        Path to the written light curve.
    """

    logger.info(f"{15*'='}GAUSSIAN LIGHT CURVE{25*'='}")
    # Define the gaussian temporal shape

    gauss_peak_time = (transient['pflx_spectrum_start']+transient['pflx_spectrum_stop']) / 2.0
    gauss_sigma = transient['t90'] / (2*np.sqrt(2)*erfinv(0.90))
    gauss_amplitude = 1 / (gauss_sigma * np.sqrt(2 * np.pi)) # Normalization: integral is 1.
    light_curve = Gaussian1D(amplitude=gauss_amplitude, mean=gauss_peak_time, stddev=gauss_sigma)

    logger.info(f"Highest flux recorded in catalog at [{gauss_peak_time}] from trigger time.")
    logger.info(f"GBM t90: {transient['t90']}.")
    logger.info(f"Gaussian Pulse sigma: {gauss_sigma}.")
  

    # Evaluate the function in a given time array
    time_step = (transient['flnc_spectrum_stop'] - transient['flnc_spectrum_start']) / 200.0

    time_start= np.maximum(transient['flnc_spectrum_start'], transient['back_interval_low_stop'].unmasked)
    time_stop = np.minimum(transient['flnc_spectrum_stop' ], transient['back_interval_high_start'].unmasked)

    time_num = (time_stop - time_start)/time_step
    time_num = int(np.floor(time_num.unmasked.to("").value))
    
    time_array = np.linspace(time_start, time_stop, time_num)
    lc_values = light_curve(time_array)

    logger.info(f"Evaluate {time_num} time points in [{time_start},{time_stop}]. Step = {time_step}.")


    # Time offset shift
    time_shifted = time_array - time_start

    logger.info(f"Shift time axis by {time_start}. Trigger is at {-time_start}, Stop at {time_stop-time_start}.")


    # Write
    output = Output_Directory+"Light_Curves_Extra/"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    light_curve_output_name = output +f"{transient['name']}_gauss.dat"
    
    logger.info(f"Write Gaussian Light Curve: {light_curve_output_name}")

    with open(light_curve_output_name, 'w') as f:
        f.write(f"IP LINLIN\n")
        for t,v in zip(time_shifted.value, lc_values.value):
            f.write(f"DP {t:.6f} {v:.6e}\n")
        f.write(f"EN\n")


    # Plot and save figure
    plot_title = f"Gaussian Light Curve: mean={np.round(gauss_peak_time-time_start,3)}, "
    plot_title+= f"sigma={np.round(gauss_sigma,3)}. Trigger: {np.round(-time_start,3)}."
    
    fig, axs = plt.subplots(1, figsize = (15,5) )
    axs.step(time_shifted.value, lc_values.value, label = 'Source rates', color = 'C0', where = 'mid')
    axs.axvline(-time_start.value, color='C1', label=f"Trigger: {-time_start}.")
    axs.set_xlabel('Time [s]', fontsize = 'large')
    axs.set_ylabel('Excess rates pdf [1/s]', fontsize = 'large')    
    axs.set_title(plot_title, fontsize = 'large')
    axs.grid()
    axs.legend()

    figure_name = output+f"{transient['name']}_gauss"+FIGURE_FORMAT
    fig.savefig(figure_name, facecolor = 'white')

    logger.info(f"{60*'='}\n")

    return light_curve_output_name








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
    light_curve_output_name : str
        Name of the output light curve file.

    """
    logger.info(f"{15*'='}EMPIRICAL GBM LIGHT CURVE{20*'='}")


    # Read the detector mask from Burst Catalog entry and turn it into a list of bools.
    det_mask = list(transient['scat_detector_mask'])
    det_mask = [bool(int(d)) for d in det_mask]
    detectors = DETECTORS[det_mask].tolist()
    # Select only the BGO
    detectors = [detectors[-1]]

    logger.info(f"List of detectors used by the GBM analysis: {detectors}.")


    # Find the Burst data in the Online Archive
    # Make a Directory to host the TTE Data and Download them
    try:
        logger.info(f"Connect to database...")
        trig_finder = TriggerFtp(transient['trigger_name'][2:])
        Temp_Directory = Output_Directory+".Temp/"
        os.makedirs(os.path.dirname(Temp_Directory), exist_ok=True)
        trig_finder.get_tte(Temp_Directory, dets=detectors)
    except:
        os.rmdir(Temp_Directory)
        logger.error("Could not get the GBM TTE files. Deleting temporary directory.\n")
        raise


    # Load the TTE files
    tte_filenames = [f for f in os.listdir(Temp_Directory) if isfile(join(Temp_Directory, f))]
    ttes = [TTE.open(Temp_Directory+f) for f in tte_filenames]

    # Time binning: turn the TTE files into CSPEC files
    bintime = (transient['flnc_spectrum_stop'].value - transient['flnc_spectrum_start'].value)/200.0
    time_range = (transient['back_interval_low_start'].unmasked.value, transient['back_interval_high_stop'].unmasked.value)

    cspecs = [t.to_phaii(bin_by_time, bintime, time_range = time_range, time_ref = 0.0) for t in ttes]
    cspecs = GbmDetectorCollection.from_list(cspecs)
    [logger.info(f"Det: {c.detector} | Energy: {c.energy_range} keV | Time: {c.time_range} s | Binning: {bintime} s.") for c in cspecs]
    

    # Background Fitting to get the excess
    bkgd_range = [(transient['back_interval_low_start'].unmasked.value, transient['back_interval_low_stop'].unmasked.value), (transient['back_interval_high_start'].unmasked.value, transient['back_interval_high_stop'].unmasked.value)]
    backfitters = [BackgroundFitter.from_phaii(cspec, Polynomial, time_ranges = bkgd_range) for cspec in cspecs]
    backfitters = GbmDetectorCollection.from_list(backfitters, dets=cspecs.detector())

    logger.info(f"Fit Background in {bkgd_range} s.")
    logger.info(f"flnc_spectrum_start,stop: [{transient['flnc_spectrum_start'].value},{transient['flnc_spectrum_stop'].value}] s.")

    backfitters.fit(order = 1)

    #logger.info(f"Interpolate background over source time range...")
    bkgds = backfitters.interpolate_bins(cspecs.data()[0].tstart, cspecs.data()[0].tstop)
    bkgds = GbmDetectorCollection.from_list(bkgds, dets=cspecs.detector())

    # Energy integration (keV) of data and background
    erange_nai = ( 10.0,   900.0) #   8 keV -  1 MeV
    erange_bgo = ( 10.0, 50000.0) # 150 keV - 40 MeV

    data_timebins = cspecs.to_lightcurve(nai_kwargs = {'energy_range':erange_nai},
                                         bgo_kwargs = {'energy_range':erange_bgo}
                                        )
    bkgd_timebins = bkgds.integrate_energy(nai_args = erange_nai,
                                           bgo_args = erange_bgo
                                          )

    
    # Write the Lightcurves

    LC_time_start= np.maximum(transient['flnc_spectrum_start'].value, transient['back_interval_low_stop'].unmasked.value)
    LC_time_stop = np.minimum(transient['flnc_spectrum_stop' ].value, transient['back_interval_high_start'].unmasked.value)

    logger.info(f"Slice Light curves between [{LC_time_start},{LC_time_stop}] s.")

    for data_t, bkgd_t, det, cspec in zip(data_timebins, bkgd_timebins, detectors, cspecs):
        
        data = data_t.slice(LC_time_start,LC_time_stop)

        bkgd_lo_edges = bkgd_t.time_centroids - 0.5*bkgd_t.time_widths
        bkgd_hi_edges = bkgd_t.time_centroids + 0.5*bkgd_t.time_widths
        bkgd = TimeBins(bkgd_t.counts, bkgd_lo_edges, bkgd_hi_edges, bkgd_t.exposure)
        bkgd = bkgd.slice(LC_time_start, LC_time_stop)
        
        # Check we have the same time bins
        if not np.allclose(data.centroids, bkgd.centroids):
            raise

        centroids = data.centroids - LC_time_start
        logger.info(f"Shift Time Axis by {LC_time_start} s. Trigger at {-LC_time_start} s. End at {LC_time_stop-LC_time_start} s.")

        widths = data.widths
        excess = data.rates - bkgd.rates

        # Negative values to 0
        excess = np.where(excess < 0, 0.0, excess)
        
        # Normalize sum to 1
        normalization = np.sum(excess*widths)
        excess /= normalization
        

        if Detector.from_str(det).is_nai():
            erange_low  = np.maximum(cspec.energy_range[0], erange_nai[0])
            erange_high = np.minimum(cspec.energy_range[1], erange_nai[1])
            output = Output_Directory+"LightCurves_Extra/"
        else:
            erange_low  = np.maximum(cspec.energy_range[0], erange_bgo[0])
            erange_high = np.minimum(cspec.energy_range[1], erange_bgo[1])
            output = Output_Directory

        erange_det = (erange_low, erange_high)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        light_curve_output_name = output+f"{transient['name']}_{det}.dat"
        logger.info(f"Write GBM Light Curve: {light_curve_output_name}")

        with open(light_curve_output_name, 'w') as f:
            f.write(f"IP LINLIN\n")
            f.write(f"DP {0:.6f} {0:.6e}\n")
            for t,d in zip(centroids, excess):
                f.write(f"DP {t:.6f} {d:.6e}\n")
            f.write(f"EN\n")


        # Define pyplot Figure and Axes
        plot_title = f"Excess rates of detector: {det}. Energy range [{erange_det[0]:.1f}, {erange_det[1]:.1f}] keV."
        fig, axs = plt.subplots(1, figsize = (15,5) )
    
        axs.step(centroids, excess, label = 'Excess rates', color = 'C0', where = 'mid')
        axs.axvline(-LC_time_start, color='C1', label=f"Trigger: {-LC_time_start} s.")
        axs.set_xlabel('Time [s]', fontsize = 'large')
        axs.set_ylabel('Excess rates pdf [1/s]', fontsize = 'large')
        axs.set_title(plot_title, fontsize = 'large')
        axs.grid()
        axs.legend()

        figure_name = output+f"{transient['name']}_{det}"+FIGURE_FORMAT
        fig.savefig(figure_name, facecolor = 'white')
    

    # Remove GBM files and directory
    try:
        shutil.rmtree(Temp_Directory)
        logger.info("Delete temporary GBM files and directory")
    except OSError as e:
        logger.error(f"Error: {Temp_Directory} : {e.strerror}")

    
    logger.info(f"{60*'='}\n")

    LC_info = Light_Curve_Info(output_name="./"+f"{transient['name']}_{det}.dat",
                               trigger = -LC_time_start,
                               step = data.widths[0],
                               stop = LC_time_stop-LC_time_start,
                               num  = len(centroids)+1,
                               det  = det,
                               emin = erange_det[0],
                               emax = erange_det[1]
                              )

    return LC_info