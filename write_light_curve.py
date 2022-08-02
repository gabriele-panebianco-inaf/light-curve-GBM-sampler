import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from os.path import isfile, join
from scipy.special import erfinv
from scipy.interpolate import interp1d
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
ERANGE_NAI = ( 10.0,   900.0) #   8 keV -  1 MeV recommended
ERANGE_BGO = ( 10.0, 50000.0) # 150 keV - 40 MeV recommended
FLUX_THRESHOLD = 15.0 # ph/s/cm2

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
    LC_info : Light_Curve_Info
        Data of the output light curve file.
    """
    logger.info(f"{15*'='}EMPIRICAL GBM LIGHT CURVE{30*'='}")
    Temp_Directory = Output_Directory+".Temp/"
    LC_info = Light_Curve_Info()
    
    # bintime = (transient['flnc_spectrum_stop'].value - transient['flnc_spectrum_start'].value)/200.0

    if transient['flnc_band_phtflux'].value>FLUX_THRESHOLD:
        time_resolution = 100.0
    else:
        time_resolution = 20.0
    bintime = transient['t90'].unmasked.value/ time_resolution
    
    # Time range for first time slice (with background intervals)
    time_range = (transient['back_interval_low_start'].unmasked.value,
                  transient['back_interval_high_stop'].unmasked.value
                 )


    # Time range for background fit
    bkgd_range = [(transient['back_interval_low_start' ].unmasked.value,
                   transient['back_interval_low_stop'  ].unmasked.value),
                  (transient['back_interval_high_start'].unmasked.value,
                   transient['back_interval_high_stop' ].unmasked.value)
                 ]

    # Time range for second time slice (no background intervals)
    LC_time_start= np.maximum(transient['flnc_spectrum_start'].value, transient['back_interval_low_stop'  ].unmasked.value)
    LC_time_stop = np.minimum(transient['flnc_spectrum_stop' ].value, transient['back_interval_high_start'].unmasked.value)


    
    # Choose and download data
    try:
        # Read the Detector Mask, turn it into booleans
        det_mask = list(transient['scat_detector_mask'])
        det_mask = [bool(int(d)) for d in det_mask]
        detectors = DETECTORS[det_mask].tolist()
        # detectors = [detectors[-1]] # Select only the BGO

        logger.info(f"List of detectors used by the GBM analysis: {detectors}.")
        logger.info(f"Connect to database...")

        # Find the Burst Data in the Online Archive
        trig_finder = TriggerFtp(transient['trigger_name'][2:])

        # Make the download directory
        os.makedirs(os.path.dirname(Temp_Directory), exist_ok=True)
        
        # Download the TTE Data
        trig_finder.get_tte(Temp_Directory, dets=detectors)
    except:
        os.rmdir(Temp_Directory)
        logger.error("Could not get the GBM TTE files. Deleting temporary directory.\n")
        raise


    # Load the TTE files, perform time binning to turn them into CSPEC files
    tte_filenames = [f for f in os.listdir(Temp_Directory) if isfile(join(Temp_Directory, f))]
    ttes = [TTE.open(Temp_Directory+f) for f in tte_filenames]
    cspecs = [t.to_phaii(bin_by_time, bintime, time_range = time_range, time_ref = 0.0) for t in ttes]
    cspecs = GbmDetectorCollection.from_list(cspecs)

    [logger.info(f"Det: {c.detector} | Energy: {c.energy_range} keV | Time: {c.time_range} s.") for c in cspecs]
    

    # Fit Background
    backfitters = [BackgroundFitter.from_phaii(cspec, Polynomial, time_ranges = bkgd_range) for cspec in cspecs]
    backfitters = GbmDetectorCollection.from_list(backfitters, dets=cspecs.detector())

    logger.info(f"Fit Background in {bkgd_range} s with a 2-order polynomial in time.")
    logger.info(f"Fluency: start,stop: [{transient['flnc_spectrum_start'].value:.3f},{transient['flnc_spectrum_stop'].value:.3f}] s.")
    logger.info(f"Time Binning: {bintime:.3f} s. It is 1/{time_resolution:.0f} of T90: {transient['t90'].unmasked}.")

    backfitters.fit(order = 2)

    bkgds = backfitters.interpolate_bins(cspecs.data()[0].tstart, cspecs.data()[0].tstop)
    bkgds = GbmDetectorCollection.from_list(bkgds, dets=cspecs.detector())


    # Energy integration of data and background
    data_timebins = cspecs.to_lightcurve(nai_kwargs = {'energy_range':ERANGE_NAI},
                                         bgo_kwargs = {'energy_range':ERANGE_BGO}
                                        )
    bkgd_timebins = bkgds.integrate_energy(nai_args = ERANGE_NAI,
                                           bgo_args = ERANGE_BGO
                                          )


    # Slice the light curve data and define Offset
    light_curve_data = [d.slice(LC_time_start, LC_time_stop) for d in data_timebins]
    Time_Offset = light_curve_data[0].centroids[0] # All Light curves have the same binning.
    Trigger_shifted = - Time_Offset
    Stop_shifted = LC_time_stop-Time_Offset

    logger.info(f"Slice Light curves between [{LC_time_start},{LC_time_stop}] s.")
    logger.info(f"Shift Time Axis by {Time_Offset:.3f} s. End at {Stop_shifted:.2f} s.")
    
    # Write the light curves
    for data, bkgd_t, cspec in zip(light_curve_data, bkgd_timebins, cspecs):

        # Background from BackgroundRates (bugged) to TimeBins
        bkgd_t_lo_edges = bkgd_t.time_centroids - 0.5* bkgd_t.time_widths # data_t.lo_edges
        bkgd_t_hi_edges = bkgd_t.time_centroids + 0.5* bkgd_t.time_widths # data_t.hi_edges
        bkgd = TimeBins(bkgd_t.counts, bkgd_t_lo_edges, bkgd_t_hi_edges, bkgd_t.exposure)
        bkgd = bkgd.slice(LC_time_start, LC_time_stop)
        
        # Check we have the same time bins
        try:
            if len(data.centroids)!=bkgd.centroids:
                raise
            np.allclose(data.centroids, bkgd.centroids)
            Background_Rates = bkgd.rates
        except:
            logger.warning(f"Data and Background arrays not defined at the same time array. Interpolate background rates.")
            f = interp1d(bkgd.centroids, bkgd.rates)
            Background_Rates = f(data.centroids)
        

        # Time shift
        Centroids_shifted = data.centroids - Time_Offset
        
        # Compute Excess Rates
        Excess = data.rates - Background_Rates

        # Negative values to 0
        Excess = np.where(Excess < 0, 0.0, Excess)
        
        # Normalize sum to 1
        Excess /= np.sum(Excess*data.widths)

        # ######################################################
        
        # Define energy range and directory where to save the current light curve
        if Detector.from_str(cspec.detector).is_nai():
            erange_low  = np.maximum(cspec.energy_range[0], ERANGE_NAI[0])
            erange_high = np.minimum(cspec.energy_range[1], ERANGE_NAI[1])
            output = Output_Directory+"Light_Curves_Extra/"
        else:
            erange_low  = np.maximum(cspec.energy_range[0], ERANGE_BGO[0])
            erange_high = np.minimum(cspec.energy_range[1], ERANGE_BGO[1])
            output = Output_Directory
            LC_info = Light_Curve_Info(output_name="./"+f"{transient['name']}_{cspec.detector}.dat",
                               trigger = Trigger_shifted,
                               step = data.widths[0],
                               stop = Stop_shifted,
                               num  = len(Centroids_shifted),
                               det  = cspec.detector,
                               emin = erange_low,
                               emax = erange_high
                              )

        os.makedirs(os.path.dirname(output), exist_ok=True)
        light_curve_output_name = output+f"{transient['name']}_{cspec.detector}.dat"

        logger.info(f"{cspec.detector} | Write Light Curve: {light_curve_output_name}")

        # Write the Light Curve as a text file
        with open(light_curve_output_name, 'w') as f:
            f.write(f"IP LINLIN\n")
            for t,d in zip(Centroids_shifted, Excess):
                f.write(f"DP {t:.6f} {d:.6e}\n")
            f.write(f"EN\n")
        
        # ######################################################

        # Define pyplot Figure and Axes
        plot_title = f"Excess rates of detector: {cspec.detector}. Energy range [{erange_low:.1f}, {erange_high:.1f}] keV."
        figure_name = output+f"{transient['name']}_{cspec.detector}"+FIGURE_FORMAT
        
        fig, axs = plt.subplots(1, figsize = (15,5) )
        axs.step(Centroids_shifted, Excess, label = 'Excess rates', color = 'C0', where = 'mid')
        axs.axvline(Trigger_shifted, color='C1', label=f"Trigger: {Trigger_shifted} s.")

        axs.axvline(transient['pflx_spectrum_start'].unmasked.value-Time_Offset,color='C2',label="Peak bin.")
        axs.axvline(transient['pflx_spectrum_stop' ].unmasked.value-Time_Offset,color='C2')
        axs.axvline(transient['t90_start'].unmasked.value-Time_Offset, color='C3', label="T90.")
        axs.axvline(transient['t90_start'].unmasked.value+transient['t90'].unmasked.value-Time_Offset, color='C3')
        axs.axvline(transient['t50_start'].unmasked.value-Time_Offset, color='C4', label="T50.")
        axs.axvline(transient['t50_start'].unmasked.value+transient['t50'].unmasked.value-Time_Offset, color='C4')

        axs.set_xlabel('Time [s]', fontsize = 'large')
        axs.set_ylabel('Excess rates pdf [1/s]', fontsize = 'large')
        axs.set_title(plot_title, fontsize = 'large')
        axs.grid()
        axs.legend()
        fig.savefig(figure_name, facecolor = 'white')
    

    # Remove GBM files and directory
    try:
        shutil.rmtree(Temp_Directory)
        logger.info("Delete temporary GBM files and directory")
    except OSError as e:
        logger.error(f"Error: {Temp_Directory} : {e.strerror}")

    
    logger.info(f"{70*'='}\n")
    return LC_info




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
