import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

from matplotlib.backends.backend_pdf import PdfPages
from os.path import isfile, join
from scipy.special import erfinv
from scipy.interpolate import interp1d
from astropy.modeling.models import Gaussian1D
from astropy.table import QTable

from gbm.background import BackgroundFitter
from gbm.background.binned import Polynomial
from gbm.binning.unbinned import bin_by_time
from gbm.data import TTE, GbmDetectorCollection
from gbm.data.primitives import TimeBins
from gbm.detectors import Detector
from gbm.finder import TriggerFtp


DETECTORS = np.array(['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1'])
FIGURE_FORMAT = ".pdf"
ERANGE_NAI = (  8.0,   900.0) #   8 keV -  1 MeV recommended (8-900 by Eric)
ERANGE_BGO = (250.0, 40000.0) # 150 keV - 40 MeV recommended (250-40 by Eric)
FLUX_THRESHOLD = 15.0         # ph/s/cm2

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




def Empirical_Light_Curve(transient, logger, Output_Directory, Use_NaI):
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
    Use_Nai : bool
        Make Light Curves for NaIs.

    Returns
    -------
    LC_info : Light_Curve_Info
        Data of the output light curve file.
    """

    logger.info(f"{15*'='}EMPIRICAL GBM LIGHT CURVE{30*'='}")

    Temp_Directory = Output_Directory+".Temp/"
    LC_info = Light_Curve_Info()
    

    if transient['flnc_band_phtflux'].value>FLUX_THRESHOLD:
        time_resolution = 100.0
    else:
        time_resolution = 50.0
    bintime = transient['t90'].value/ time_resolution
    
    # Time range for first time slice (with background intervals)
    time_range = (transient['back_interval_low_start'].value,
                  transient['back_interval_high_stop'].value
                 )

    # Time range for background fit
    bkgd_range = [(transient['back_interval_low_start' ].value,
                   transient['back_interval_low_stop'  ].value),
                  (transient['back_interval_high_start'].value,
                   transient['back_interval_high_stop' ].value)
                 ]

    # Time range for second time slice (no background intervals)
    LC_time_start= np.maximum(transient['flnc_spectrum_start'].value, transient['back_interval_low_stop'  ].value)
    LC_time_stop = np.minimum(transient['flnc_spectrum_stop' ].value, transient['back_interval_high_start'].value)


    
    # Choose and download data
    try:
        # Read the Detector Mask, turn it into booleans
        det_mask = list(transient['scat_detector_mask'])
        det_mask = [bool(int(d)) for d in det_mask]
        detectors = DETECTORS[det_mask].tolist()
        if not Use_NaI:
            detectors = [detectors[-1]] # Select only the BGO

        logger.info(f"Retrieve TTE from GBM detectors {detectors}. Connect to database...")

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

    logger.info(f"Raw TTE Data Ranges.")
    [logger.info(f"Det: {t.detector} | Energy: [{t.energy_range[0]:.2f}, {t.energy_range[1]:.2f}] keV | Time: [{t.time_range[0]:.3f}, {t.time_range[1]:.3f}] s.") for t in ttes]


    #=============================================================================================
    # Energy Slice the TTEs
    # ttes_sliced = []
    # for tte in ttes:
    #     if Detector.from_str(tte.detector).is_nai():
    #         ERange = ERANGE_NAI
    #     else:
    #         ERange = ERANGE_BGO
    #     ttes_sliced.append(tte.slice_energy(ERange))

    # logger.info(f"Energy-sliced TTE Data Ranges.")
    # [logger.info(f"Det: {t.detector} | Energy: [{t.energy_range[0]:.2f}, {t.energy_range[1]:.2f}] keV | Time: [{t.time_range[0]:.3f}, {t.time_range[1]:.3f}] s.") for t in ttes_sliced]
    #=============================================================================================
    
    #=============================================================================================
    # Merge the NaIs.
    tte_nai_list = [tte for tte in ttes if Detector.from_str(tte.detector).is_nai()]
    tte_nai    = TTE.merge(tte_nai_list)

    tte_bgo      = [tte for tte in ttes if not Detector.from_str(tte.detector).is_nai()]
    if len(tte_bgo) == 1:
        tte_bgo = tte_bgo[0]
    else:
        raise ValueError(f"Downloaded {len(tte_bgo)} BGO TTE.")
    
    # tte_merged = TTE.merge(ttes_sliced)
    # if tte_nai.detector != 'n0':
    #     tte_merged.set_properties(detector='n0', trigtime=tte_merged.trigtime)
    # else:
    #     tte_merged.set_properties(detector='n1', trigtime=tte_merged.trigtime)
    
    ttes = [tte_nai, tte_bgo]
    # ttes = [tte_nai, tte_bgo, tte_merged]

    # Metadata: detector, energy range.
    meta = QTable(names = ('detector', 'min_energy', 'max_energy'),
                  dtype = ('str', 'float', 'float32')
                 )
    if len(tte_nai_list) ==1:
        Name_merged_NaIs = tte_nai_list[0].detector
    else:
        Name_merged_NaIs = "Merge"+''.join(["_"+t.detector for t in tte_nai_list])
    Name_merged_dets = "Sum"  +''.join(["_"+t.detector for t in tte_nai_list])+"_"+tte_bgo.detector
    # meta.add_row( (Name_merged_NaIs, ERANGE_NAI[0], ERANGE_NAI[1]) )
    # meta.add_row( (tte_bgo.detector, ERANGE_BGO[0], ERANGE_BGO[1]) )
    # meta.add_row( (Name_merged_dets, ERANGE_NAI[0], ERANGE_BGO[1]) )
 
    #=============================================================================================

    # Time Binning and Slice
    msg = f"T90: {transient['t90']}. "
    msg+= f"Range [{transient['t90_start'].value},"
    msg+= f"{transient['t90_start'].value+transient['t90'].value}] s."

    logger.info(msg)
    if transient['t90'] > 400*u.s:
        logger.warning(f"Careful! This is a very long GRB!")
    logger.info(f"Binning: T90/{time_resolution:.0f} = {bintime:.3f} s.")

    cspecs = [t.to_phaii(bin_by_time, bintime, time_range = time_range, time_ref = 0.0) for t in ttes]    
    cspecs = GbmDetectorCollection.from_list(cspecs)


    #[logger.info(f"Det: {c.detector} | Energy: [{c.energy_range[0]:.2f}, {c.energy_range[1]:.2f}] keV | Time: [{c.time_range[0]:.3f}, {c.time_range[1]:.3f}] s.") for c in cspecs]
    #[logger.info(f"Det: {m['detector']} | Energy: [{m['min_energy']:.2f}, {m['max_energy']:.2f}] keV | Time: [{c.time_range[0]:.3f}, {c.time_range[1]:.3f}] s.") for m, c in zip(meta, cspecs)]

    

    # Fit Background
    backfitters = [BackgroundFitter.from_phaii(cspec, Polynomial, time_ranges = bkgd_range) for cspec in cspecs]
    backfitters = GbmDetectorCollection.from_list(backfitters, dets=cspecs.detector())

    logger.info(f"Fluency time range: [{transient['flnc_spectrum_start'].value:.3f}, {transient['flnc_spectrum_stop'].value:.3f}] s.")
    logger.info(f"Try Fit Background in {bkgd_range} s with a 2-order polynomial in time.")

    try:
        backfitters.fit(order = 2)
    except np.linalg.LinAlgError:
        logger.error("Fit with order 2 failed. Try order 1")
        try:
            backfitters.fit(order = 1)
        except:
            logger.error("Fit with order 1 failed. Try running the code without NaIs.")
            shutil.rmtree(Temp_Directory)
            logger.error("Delete temporary GBM files and directory.")
            raise

    bkgds = backfitters.interpolate_bins(cspecs.data()[0].tstart, cspecs.data()[0].tstop)
    bkgds = GbmDetectorCollection.from_list(bkgds, dets=cspecs.detector())


    # Energy slice&integration of data and background
    data_timebins = cspecs.to_lightcurve(
        nai_kwargs = {'energy_range':ERANGE_NAI},
        bgo_kwargs = {'energy_range':ERANGE_BGO}
        )
    bkgd_timebins = []
    bkgd_backrates = bkgds.integrate_energy(
        nai_args = ERANGE_NAI,
        bgo_args = ERANGE_BGO
        )
    
    # Background from BackgroundRates (.slice_time is bugged) to TimeBins
    for bkgd_t in bkgd_backrates:

        bkgd_t_lo_edges = bkgd_t.time_centroids - 0.5* bkgd_t.time_widths
        bkgd_t_hi_edges = bkgd_t.time_centroids + 0.5* bkgd_t.time_widths
        bkgd = TimeBins(bkgd_t.counts, bkgd_t_lo_edges, bkgd_t_hi_edges, bkgd_t.exposure)
        bkgd = bkgd.slice(LC_time_start, LC_time_stop)
        bkgd_timebins.append(bkgd)

    # ========================================================================================
    # Metadata: detector, energy range.
    for c in cspecs:
        if Detector.from_str(c.detector).is_nai():
            erange_low  = ERANGE_NAI[0]
            erange_high = ERANGE_NAI[1]
            name = Name_merged_NaIs
        else:
            erange_low  = ERANGE_BGO[0]
            erange_high = ERANGE_BGO[1]
            name = c.detector
        meta.add_row( (name, np.maximum(c.energy_range[0], erange_low), np.minimum(c.energy_range[1], erange_high)) )

    # Now implement the sum
    data_timebins.append( TimeBins.sum(data_timebins) )
    bkgd_timebins.append( TimeBins.sum(bkgd_timebins) )
    #sum_detector = "Sum"+''.join(["_"+c.detector for c in cspecs])
    meta.add_row( (Name_merged_dets, np.amin(meta['min_energy']), np.amax(meta['max_energy'])) )
    # ========================================================================================


    # Slice the light curve data and define Offset
    light_curve_data = [d.slice(LC_time_start, LC_time_stop) for d in data_timebins]
    Time_Offset = light_curve_data[0].centroids[0] # All Light curves have the same binning.
    Trigger_shifted = - Time_Offset
    Stop_shifted = LC_time_stop-Time_Offset

    logger.info(f"Slice Light curves between [{LC_time_start:.3f}, {LC_time_stop:.3f}] s.")
    logger.info(f"Shift Time Axis by {Time_Offset:.3f} s. End at {Stop_shifted:.3f} s.")

    # Figure PDF
    figure_name = Output_Directory+f"{transient['name']}"+FIGURE_FORMAT
    pp = PdfPages(figure_name)
    
    # Write the light curves
    for data, bkgd, m in zip(light_curve_data, bkgd_timebins, meta):
        
        # Check we have the same time bins
        try:
            if len(data.centroids)!=len(bkgd.centroids):
                logger.error(f"Data length {len(data.centroids)}, background length {len(bkgd.centroids)}.")
                raise ValueError
            if not np.allclose(data.centroids, bkgd.centroids):
                logger.error("Data and Background arrays not defined at the same time array.")
                raise ValueError
            Background_Counts = np.round(bkgd.counts, 0)
            Background_Uncert = bkgd.count_uncertainty
        except ValueError:
            logger.warning(f"Interpolate background counts and uncertainties.")
            f = interp1d(bkgd.centroids, bkgd.counts)
            Background_Counts = np.round(f(data.centroids),0)
            f = interp1d(bkgd.centroids, bkgd.count_uncertainty)
            Background_Uncert = f(data.centroids)

        
        # Time shift
        Centroids_shifted = data.centroids - Time_Offset
        # Compute Excess Counts
        Excess = data.counts - Background_Counts
        # Clip Negative values to 0
        Excess = np.where(Excess < 0, 0.0, Excess)

        # Uncertainties
        Uncert = np.where(Excess > 0, data.count_uncertainty + Background_Uncert, 0.0)

        # ######################################################     
        
        LC_info = Light_Curve_Info(output_name="./"+f"{transient['name']}_{m['detector']}.dat",
                           trigger = Trigger_shifted,
                           step = data.widths[0],
                           stop = Stop_shifted,
                           num  = len(Centroids_shifted),
                           det  = m['detector'],
                           emin = m['min_energy'],
                           emax = m['max_energy']
                          )

        # Write the Light Curve as a text file
        os.makedirs(os.path.dirname(Output_Directory), exist_ok=True)
        light_curve_output_name = Output_Directory+f"{transient['name']}_{m['detector']}.dat"
        logger.info(f"{m['detector']} | Write Light Curve: {light_curve_output_name}")

        with open(light_curve_output_name, 'w') as f:
            f.write(f"# Info on the Light curve.\n")
            f.write(f"# DP | Time point in [s] | Excess counts.\n")
            f.write(f"# We shifted GBM times to start from 0.0. After the shift GBM Trigger time is at {LC_info.trigger:.5f} s. Light curve stops at {LC_info.stop:.5f} s. Time resolution is {LC_info.step:.5f} s. There are {LC_info.num} points.\n")
            f.write(f"# Light curve values are Excess Counts = Observed GBM Counts - Best fit Background Model. Negative excess counts are clipped to 0.\n")
            f.write(f"# We selected the events from GBM detector {LC_info.det} with energy in [{LC_info.emin:.1f},{LC_info.emax:.1f}] keV.\n")

            f.write(f"IP LINLIN\n")
            for t,d in zip(Centroids_shifted, Excess):
                f.write(f"DP {t:.6f} {d:.1f}\n")
            f.write(f"EN\n")

        # Define pyplot Figure and Axes
        fig, ax = plt.subplots(1, figsize = (15, 5), constrained_layout=True )
        plot_title = f"Excess counts of {transient['name']} from GBM detector: {m['detector']}. Energy range [{LC_info.emin:.1f}, {LC_info.emax:.1f}] keV."
        
        ax.axvline(Trigger_shifted, color='C1',lw=2, label=f"Trigger: {Trigger_shifted:.3f} s.")
        ax.axvline(transient['t90_start'].value-Time_Offset, color='C3', ls='-.', label=f"Range T90: {transient['t90'].value:.3f} s.")
        ax.axvline(transient['t90_start'].value+transient['t90'].value-Time_Offset, color='C3', ls='-.',)
        #ax.axvline(transient['pflx_spectrum_start'].value-Time_Offset,color='C2',label="Peak range.")
        #ax.axvline(transient['pflx_spectrum_stop' ].value-Time_Offset,color='C2')
        condition_peak = np.logical_and(
            Centroids_shifted>(transient['pflx_spectrum_start'].value-Time_Offset),
            Centroids_shifted<(transient['pflx_spectrum_stop' ].value-Time_Offset)
        )
        # condition_peak = np.full(Centroids_shifted.size, False)
        # idx_peak_start = np.argmin(np.abs(Centroids_shifted - (transient['pflx_spectrum_start'].value-Time_Offset)))
        # idx_peak_stop  = np.argmin(np.abs(Centroids_shifted - (transient['pflx_spectrum_stop' ].value-Time_Offset)))
        # condition_peak[idx_peak_start:idx_peak_stop] = True
        ax.fill_between(Centroids_shifted[condition_peak], 0.0, Excess[condition_peak], alpha=0.2, step='mid', color='C2', label='Peak')

        ax.bar(Centroids_shifted,
           height = 2.0 * Uncert,
           width = data.widths,
           bottom = Excess - Uncert,
           alpha=0.4, color='grey', label='Errors'
          )
        ax.step(Centroids_shifted, Excess, label = 'Excess counts', color = 'C0', where = 'mid')

        ax.set_xlabel('Time [s]', fontsize = 'large')
        ax.set_ylabel('Excess Counts', fontsize = 'large')
        ax.set_title(plot_title, fontsize = 'large')
        ax.grid()
        ax.legend()
        pp.savefig(fig)

    pp.close()
    
    # Remove GBM files and directory
    try:
        shutil.rmtree(Temp_Directory)
        logger.info("Delete temporary GBM files and directory")
    except OSError as e:
        logger.error(f"Error: {Temp_Directory} : {e.strerror}")

    
    logger.info(f"{70*'='}\n")
    return LC_info.output_name




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

    time_start= np.maximum(transient['flnc_spectrum_start'], transient['back_interval_low_stop'])
    time_stop = np.minimum(transient['flnc_spectrum_stop' ], transient['back_interval_high_start'])

    time_num = (time_stop - time_start)/time_step
    time_num = int(np.floor(time_num.to("").value))
    
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
