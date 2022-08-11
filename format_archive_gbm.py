import numpy as np
import pandas as pd
from astropy.table import QTable
import astropy.units as u

filename = "/home/gabriele/Documents/fermiGBM/light-curve-GBM-sampler/GBM_burst_archive/heasarc_fermigbrst.tdat"
columns = "trigger_name name ra dec lii bii error_radius trigger_time duration_energy_low duration_energy_high back_interval_low_start back_interval_low_stop back_interval_high_start back_interval_high_stop t50 t50_error t50_start t90 t90_error t90_start bcat_detector_mask flu_low flu_high fluence fluence_error fluence_batse fluence_batse_error flux_1024 flux_1024_error flux_1024_time flux_64 flux_64_error flux_64_time flux_256 flux_256_error flux_256_time flux_batse_1024 flux_batse_1024_error flux_batse_1024_time flux_batse_64 flux_batse_64_error flux_batse_64_time flux_batse_256 flux_batse_256_error flux_batse_256_time actual_64ms_interval actual_256ms_interval actual_1024ms_interval scat_detector_mask pflx_spectrum_start pflx_spectrum_stop pflx_plaw_ampl pflx_plaw_ampl_pos_err pflx_plaw_ampl_neg_err pflx_plaw_pivot pflx_plaw_pivot_pos_err pflx_plaw_pivot_neg_err pflx_plaw_index pflx_plaw_index_pos_err pflx_plaw_index_neg_err pflx_plaw_phtflux pflx_plaw_phtflux_error pflx_plaw_phtflnc pflx_plaw_phtflnc_error pflx_plaw_ergflux pflx_plaw_ergflux_error pflx_plaw_ergflnc pflx_plaw_ergflnc_error pflx_plaw_phtfluxb pflx_plaw_phtfluxb_error pflx_plaw_phtflncb pflx_plaw_phtflncb_error pflx_plaw_ergflncb pflx_plaw_ergflncb_error pflx_plaw_redchisq pflx_plaw_redfitstat pflx_plaw_dof pflx_plaw_statistic pflx_comp_ampl pflx_comp_ampl_pos_err pflx_comp_ampl_neg_err pflx_comp_epeak pflx_comp_epeak_pos_err pflx_comp_epeak_neg_err pflx_comp_index pflx_comp_index_pos_err pflx_comp_index_neg_err pflx_comp_pivot pflx_comp_pivot_pos_err pflx_comp_pivot_neg_err pflx_comp_phtflux pflx_comp_phtflux_error pflx_comp_phtflnc pflx_comp_phtflnc_error pflx_comp_ergflux pflx_comp_ergflux_error pflx_comp_ergflnc pflx_comp_ergflnc_error pflx_comp_phtfluxb pflx_comp_phtfluxb_error pflx_comp_phtflncb pflx_comp_phtflncb_error pflx_comp_ergflncb pflx_comp_ergflncb_error pflx_comp_redchisq pflx_comp_redfitstat pflx_comp_dof pflx_comp_statistic pflx_band_ampl pflx_band_ampl_pos_err pflx_band_ampl_neg_err pflx_band_epeak pflx_band_epeak_pos_err pflx_band_epeak_neg_err pflx_band_alpha pflx_band_alpha_pos_err pflx_band_alpha_neg_err pflx_band_beta pflx_band_beta_pos_err pflx_band_beta_neg_err pflx_band_phtflux pflx_band_phtflux_error pflx_band_phtflnc pflx_band_phtflnc_error pflx_band_ergflux pflx_band_ergflux_error pflx_band_ergflnc pflx_band_ergflnc_error pflx_band_phtfluxb pflx_band_phtfluxb_error pflx_band_phtflncb pflx_band_phtflncb_error pflx_band_ergflncb pflx_band_ergflncb_error pflx_band_redchisq pflx_band_redfitstat pflx_band_dof pflx_band_statistic pflx_sbpl_ampl pflx_sbpl_ampl_pos_err pflx_sbpl_ampl_neg_err pflx_sbpl_pivot pflx_sbpl_pivot_pos_err pflx_sbpl_pivot_neg_err pflx_sbpl_indx1 pflx_sbpl_indx1_pos_err pflx_sbpl_indx1_neg_err pflx_sbpl_brken pflx_sbpl_brken_pos_err pflx_sbpl_brken_neg_err pflx_sbpl_brksc pflx_sbpl_brksc_pos_err pflx_sbpl_brksc_neg_err pflx_sbpl_indx2 pflx_sbpl_indx2_pos_err pflx_sbpl_indx2_neg_err pflx_sbpl_phtflux pflx_sbpl_phtflux_error pflx_sbpl_phtflnc pflx_sbpl_phtflnc_error pflx_sbpl_ergflux pflx_sbpl_ergflux_error pflx_sbpl_ergflnc pflx_sbpl_ergflnc_error pflx_sbpl_phtfluxb pflx_sbpl_phtfluxb_error pflx_sbpl_phtflncb pflx_sbpl_phtflncb_error pflx_sbpl_ergflncb pflx_sbpl_ergflncb_error pflx_sbpl_redchisq pflx_sbpl_redfitstat pflx_sbpl_dof pflx_sbpl_statistic pflx_best_fitting_model pflx_best_model_redchisq flnc_spectrum_start flnc_spectrum_stop flnc_plaw_ampl flnc_plaw_ampl_pos_err flnc_plaw_ampl_neg_err flnc_plaw_pivot flnc_plaw_pivot_pos_err flnc_plaw_pivot_neg_err flnc_plaw_index flnc_plaw_index_pos_err flnc_plaw_index_neg_err flnc_plaw_phtflux flnc_plaw_phtflux_error flnc_plaw_phtflnc flnc_plaw_phtflnc_error flnc_plaw_ergflux flnc_plaw_ergflux_error flnc_plaw_ergflnc flnc_plaw_ergflnc_error flnc_plaw_phtfluxb flnc_plaw_phtfluxb_error flnc_plaw_phtflncb flnc_plaw_phtflncb_error flnc_plaw_ergflncb flnc_plaw_ergflncb_error flnc_plaw_redchisq flnc_plaw_redfitstat flnc_plaw_dof flnc_plaw_statistic flnc_comp_ampl flnc_comp_ampl_pos_err flnc_comp_ampl_neg_err flnc_comp_epeak flnc_comp_epeak_pos_err flnc_comp_epeak_neg_err flnc_comp_index flnc_comp_index_pos_err flnc_comp_index_neg_err flnc_comp_pivot flnc_comp_pivot_pos_err flnc_comp_pivot_neg_err flnc_comp_phtflux flnc_comp_phtflux_error flnc_comp_phtflnc flnc_comp_phtflnc_error flnc_comp_ergflux flnc_comp_ergflux_error flnc_comp_ergflnc flnc_comp_ergflnc_error flnc_comp_phtfluxb flnc_comp_phtfluxb_error flnc_comp_phtflncb flnc_comp_phtflncb_error flnc_comp_ergflncb flnc_comp_ergflncb_error flnc_comp_redchisq flnc_comp_redfitstat flnc_comp_dof flnc_comp_statistic flnc_band_ampl flnc_band_ampl_pos_err flnc_band_ampl_neg_err flnc_band_epeak flnc_band_epeak_pos_err flnc_band_epeak_neg_err flnc_band_alpha flnc_band_alpha_pos_err flnc_band_alpha_neg_err flnc_band_beta flnc_band_beta_pos_err flnc_band_beta_neg_err flnc_band_phtflux flnc_band_phtflux_error flnc_band_phtflnc flnc_band_phtflnc_error flnc_band_ergflux flnc_band_ergflux_error flnc_band_ergflnc flnc_band_ergflnc_error flnc_band_phtfluxb flnc_band_phtfluxb_error flnc_band_phtflncb flnc_band_phtflncb_error flnc_band_ergflncb flnc_band_ergflncb_error flnc_band_redchisq flnc_band_redfitstat flnc_band_dof flnc_band_statistic flnc_sbpl_ampl flnc_sbpl_ampl_pos_err flnc_sbpl_ampl_neg_err flnc_sbpl_pivot flnc_sbpl_pivot_pos_err flnc_sbpl_pivot_neg_err flnc_sbpl_indx1 flnc_sbpl_indx1_pos_err flnc_sbpl_indx1_neg_err flnc_sbpl_brken flnc_sbpl_brken_pos_err flnc_sbpl_brken_neg_err flnc_sbpl_brksc flnc_sbpl_brksc_pos_err flnc_sbpl_brksc_neg_err flnc_sbpl_indx2 flnc_sbpl_indx2_pos_err flnc_sbpl_indx2_neg_err flnc_sbpl_phtflux flnc_sbpl_phtflux_error flnc_sbpl_phtflnc flnc_sbpl_phtflnc_error flnc_sbpl_ergflux flnc_sbpl_ergflux_error flnc_sbpl_ergflnc flnc_sbpl_ergflnc_error flnc_sbpl_phtfluxb flnc_sbpl_phtfluxb_error flnc_sbpl_phtflncb flnc_sbpl_phtflncb_error flnc_sbpl_ergflncb flnc_sbpl_ergflncb_error flnc_sbpl_redchisq flnc_sbpl_redfitstat flnc_sbpl_dof flnc_sbpl_statistic flnc_best_fitting_model flnc_best_model_redchisq bcatalog scatalog last_modified".split()
Output = "./GBM_burst_archive/"



columns.append('final_space')
print('Len columns + final space column:',len(columns))

# list of column names that needs to be string
lst_str_cols = ["trigger_name", "name", "bcat_detector_mask", "scat_detector_mask"]
dict_dtypes = {x : 'str'  for x in lst_str_cols}
df = pd.read_table(filename,
                   sep = '|',
                   comment = '#',
                   names=columns,
                   skiprows=352,
                   skipfooter=1,
                   engine='python',
                   dtype=dict_dtypes
                  )
df = df.drop(labels=['final_space'],axis=1)
df = df.sort_values('trigger_name',ignore_index=True)

qt = QTable.from_pandas(df)
qt.meta={'EXTNAME': 'CATALOG'}


Columns_Adimensional = ['name','trigger_name','trigger_time','last_modified',
                        'bcat_detector_mask','bcatalog',
                        'scat_detector_mask','scatalog',
                        'pflx_best_fitting_model','pflx_best_model_redchisq',
                        'pflx_plaw_index'   ,'pflx_plaw_index_pos_err','pflx_plaw_index_neg_err',
                        'pflx_plaw_redchisq','pflx_plaw_redfitstat'   ,
                        'pflx_plaw_dof'     ,'pflx_plaw_statistic'    ,
                        'pflx_comp_index'   ,'pflx_comp_index_pos_err','pflx_comp_index_neg_err',
                        'pflx_comp_redchisq','pflx_comp_redfitstat'   ,
                        'pflx_comp_dof'     ,'pflx_comp_statistic'    ,
                        'pflx_band_alpha'   ,'pflx_band_alpha_pos_err','pflx_band_alpha_neg_err',
                        'pflx_band_beta'    ,'pflx_band_beta_pos_err' ,'pflx_band_beta_neg_err' ,
                        'pflx_band_redchisq','pflx_band_redfitstat'   ,
                        'pflx_band_dof'     ,'pflx_band_statistic'    ,
                        'pflx_sbpl_indx1'   ,'pflx_sbpl_indx1_pos_err','pflx_sbpl_indx1_neg_err',
                        'pflx_sbpl_indx2'   ,'pflx_sbpl_indx2_pos_err','pflx_sbpl_indx2_neg_err',
                        'pflx_sbpl_redchisq','pflx_sbpl_redfitstat'   ,
                        'pflx_sbpl_dof'     ,'pflx_sbpl_statistic'    ,
                        'flnc_best_fitting_model','flnc_best_model_redchisq',
                        'flnc_plaw_index'   ,'flnc_plaw_index_pos_err','flnc_plaw_index_neg_err',
                        'flnc_plaw_redchisq','flnc_plaw_redfitstat'   ,
                        'flnc_plaw_dof'     ,'flnc_plaw_statistic'    ,
                        'flnc_comp_index'   ,'flnc_comp_index_pos_err','flnc_comp_index_neg_err',
                        'flnc_comp_redchisq','flnc_comp_redfitstat'   ,
                        'flnc_comp_dof'     ,'flnc_comp_statistic'    ,
                        'flnc_band_alpha'   ,'flnc_band_alpha_pos_err','flnc_band_alpha_neg_err',
                        'flnc_band_beta'    ,'flnc_band_beta_pos_err' ,'flnc_band_beta_neg_err' ,
                        'flnc_band_redchisq','flnc_band_redfitstat'   ,
                        'flnc_band_dof'     ,'flnc_band_statistic'    ,
                        'flnc_sbpl_indx1'   ,'flnc_sbpl_indx1_pos_err','flnc_sbpl_indx1_neg_err',
                        'flnc_sbpl_indx2'   ,'flnc_sbpl_indx2_pos_err','flnc_sbpl_indx2_neg_err',
                        'flnc_sbpl_redchisq','flnc_sbpl_redfitstat'   ,
                        'flnc_sbpl_dof'     ,'flnc_sbpl_statistic'
                       ]

Columns_Angle = ['ra','dec','lii','bii','error_radius']

Columns_Time = ['t90', 't90_error', 't90_start',
                't50', 't50_error', 't50_start',
                'back_interval_low_start' ,'back_interval_low_stop',
                'back_interval_high_start','back_interval_high_stop',
                'pflx_spectrum_start','pflx_spectrum_stop',
                'flnc_spectrum_start','flnc_spectrum_stop',
                'flux_64_time'  ,'flux_batse_64_time'  ,'actual_64ms_interval'  ,
                'flux_256_time' ,'flux_batse_256_time' ,'actual_256ms_interval' ,
                'flux_1024_time','flux_batse_1024_time','actual_1024ms_interval'
               ]

Columns_Amplitude = ['pflx_plaw_ampl','pflx_plaw_ampl_pos_err','pflx_plaw_ampl_neg_err',
                     'pflx_comp_ampl','pflx_comp_ampl_pos_err','pflx_comp_ampl_neg_err',
                     'pflx_band_ampl','pflx_band_ampl_pos_err','pflx_band_ampl_neg_err',
                     'pflx_sbpl_ampl','pflx_sbpl_ampl_pos_err','pflx_sbpl_ampl_neg_err',
                     'flnc_plaw_ampl','flnc_plaw_ampl_pos_err','flnc_plaw_ampl_neg_err',
                     'flnc_comp_ampl','flnc_comp_ampl_pos_err','flnc_comp_ampl_neg_err',
                     'flnc_band_ampl','flnc_band_ampl_pos_err','flnc_band_ampl_neg_err',
                     'flnc_sbpl_ampl','flnc_sbpl_ampl_pos_err','flnc_sbpl_ampl_neg_err'
                    ]

Columns_Energy = ['duration_energy_low','duration_energy_high','flu_low','flu_high',
                  'pflx_plaw_pivot','pflx_plaw_pivot_pos_err','pflx_plaw_pivot_neg_err',
                  'pflx_comp_epeak','pflx_comp_epeak_pos_err','pflx_comp_epeak_neg_err',
                  'pflx_comp_pivot','pflx_comp_pivot_pos_err','pflx_comp_pivot_neg_err',
                  'pflx_band_epeak','pflx_band_epeak_pos_err','pflx_band_epeak_neg_err',
                  'pflx_sbpl_pivot','pflx_sbpl_pivot_pos_err','pflx_sbpl_pivot_neg_err',
                  'pflx_sbpl_brken','pflx_sbpl_brken_pos_err','pflx_sbpl_brken_neg_err',
                  'pflx_sbpl_brksc','pflx_sbpl_brksc_pos_err','pflx_sbpl_brksc_neg_err',
                  'flnc_plaw_pivot','flnc_plaw_pivot_pos_err','flnc_plaw_pivot_neg_err',
                  'flnc_comp_epeak','flnc_comp_epeak_pos_err','flnc_comp_epeak_neg_err',
                  'flnc_comp_pivot','flnc_comp_pivot_pos_err','flnc_comp_pivot_neg_err',
                  'flnc_band_epeak','flnc_band_epeak_pos_err','flnc_band_epeak_neg_err',
                  'flnc_sbpl_pivot','flnc_sbpl_pivot_pos_err','flnc_sbpl_pivot_neg_err',
                  'flnc_sbpl_brken','flnc_sbpl_brken_pos_err','flnc_sbpl_brken_neg_err',
                  'flnc_sbpl_brksc','flnc_sbpl_brksc_pos_err','flnc_sbpl_brksc_neg_err'
                 ]

Columns_Flux_p = ['flux_1024','flux_1024_error',
                  'flux_64','flux_64_error',
                  'flux_256','flux_256_error',
                  'flux_batse_1024','flux_batse_1024_error',
                  'flux_batse_64','flux_batse_64_error',
                  'flux_batse_256','flux_batse_256_error',
                  'pflx_plaw_phtflux' ,'pflx_plaw_phtflux_error',
                  'pflx_plaw_phtfluxb','pflx_plaw_phtfluxb_error',
                  'pflx_comp_phtflux' ,'pflx_comp_phtflux_error',
                  'pflx_comp_phtfluxb','pflx_comp_phtfluxb_error',
                  'pflx_band_phtflux' ,'pflx_band_phtflux_error',
                  'pflx_band_phtfluxb','pflx_band_phtfluxb_error',
                  'pflx_sbpl_phtflux' ,'pflx_sbpl_phtflux_error',
                  'pflx_sbpl_phtfluxb','pflx_sbpl_phtfluxb_error',
                  'flnc_plaw_phtflux' ,'flnc_plaw_phtflux_error',
                  'flnc_plaw_phtfluxb','flnc_plaw_phtfluxb_error',
                  'flnc_comp_phtflux' ,'flnc_comp_phtflux_error',
                  'flnc_comp_phtfluxb','flnc_comp_phtfluxb_error',
                  'flnc_band_phtflux' ,'flnc_band_phtflux_error',
                  'flnc_band_phtfluxb','flnc_band_phtfluxb_error',
                  'flnc_sbpl_phtflux' ,'flnc_sbpl_phtflux_error',
                  'flnc_sbpl_phtfluxb','flnc_sbpl_phtfluxb_error',
                 ]
Columns_Flux_E = ['pflx_plaw_ergflux','pflx_plaw_ergflux_error',
                  'pflx_comp_ergflux','pflx_comp_ergflux_error',
                  'pflx_band_ergflux','pflx_band_ergflux_error',
                  'pflx_sbpl_ergflux','pflx_sbpl_ergflux_error',
                  'flnc_plaw_ergflux','flnc_plaw_ergflux_error',
                  'flnc_comp_ergflux','flnc_comp_ergflux_error',
                  'flnc_band_ergflux','flnc_band_ergflux_error',
                  'flnc_sbpl_ergflux','flnc_sbpl_ergflux_error',
                 ]
Columns_Fluence_p = ['pflx_plaw_phtflnc' ,'pflx_plaw_phtflnc_error' ,
                     'pflx_plaw_phtflncb','pflx_plaw_phtflncb_error',
                     'pflx_comp_phtflnc' ,'pflx_comp_phtflnc_error' ,
                     'pflx_comp_phtflncb','pflx_comp_phtflncb_error',
                     'pflx_band_phtflnc' ,'pflx_band_phtflnc_error' ,
                     'pflx_band_phtflncb','pflx_band_phtflncb_error',
                     'pflx_sbpl_phtflnc' ,'pflx_sbpl_phtflnc_error' ,
                     'pflx_sbpl_phtflncb','pflx_sbpl_phtflncb_error',
                     'flnc_plaw_phtflnc' ,'flnc_plaw_phtflnc_error' ,
                     'flnc_plaw_phtflncb','flnc_plaw_phtflncb_error',
                     'flnc_comp_phtflnc' ,'flnc_comp_phtflnc_error' ,
                     'flnc_comp_phtflncb','flnc_comp_phtflncb_error',
                     'flnc_band_phtflnc' ,'flnc_band_phtflnc_error' ,
                     'flnc_band_phtflncb','flnc_band_phtflncb_error',
                     'flnc_sbpl_phtflnc' ,'flnc_sbpl_phtflnc_error' ,
                     'flnc_sbpl_phtflncb','flnc_sbpl_phtflncb_error',
                    ]
Columns_Fluence_E = ['fluence','fluence_error','fluence_batse','fluence_batse_error',
                     'pflx_plaw_ergflnc' ,'pflx_plaw_ergflnc_error' ,
                     'pflx_plaw_ergflncb','pflx_plaw_ergflncb_error',
                     'pflx_comp_ergflnc' ,'pflx_comp_ergflnc_error' ,
                     'pflx_comp_ergflncb','pflx_comp_ergflncb_error',
                     'pflx_band_ergflnc' ,'pflx_band_ergflnc_error' ,
                     'pflx_band_ergflncb','pflx_band_ergflncb_error',
                     'pflx_sbpl_ergflnc' ,'pflx_sbpl_ergflnc_error' ,
                     'pflx_sbpl_ergflncb','pflx_sbpl_ergflncb_error',
                     'flnc_plaw_ergflnc' ,'flnc_plaw_ergflnc_error' ,
                     'flnc_plaw_ergflncb','flnc_plaw_ergflncb_error',
                     'flnc_comp_ergflnc' ,'flnc_comp_ergflnc_error' ,
                     'flnc_comp_ergflncb','flnc_comp_ergflncb_error',
                     'flnc_band_ergflnc' ,'flnc_band_ergflnc_error' ,
                     'flnc_band_ergflncb','flnc_band_ergflncb_error',
                     'flnc_sbpl_ergflnc' ,'flnc_sbpl_ergflnc_error' ,
                     'flnc_sbpl_ergflncb','flnc_sbpl_ergflncb_error',
                    ]
#####################################################################################
units = {'Adim'     :u.Unit(""),
         'Angle'    :u.Unit("deg"),
         'Time'     :u.Unit("s"),
         'Energy'   :u.Unit("keV"),
         'Amplitude':u.Unit("cm-2 s-1 keV-1"),
         'Flux_p'   :u.Unit("cm-2 s-1"),
         'Fluence_p':u.Unit("cm-2"),
         'Flux_E'   :u.Unit("erg cm-2 s-1"),
         'Fluence_E':u.Unit("erg cm-2"),     
        }
#####################################################################################

for c in Columns_Angle:
    qt[c].unit = units['Angle']
for c in Columns_Time:
    qt[c].unit = units['Time']
for c in Columns_Energy:
    qt[c].unit = units['Energy']
    
for c in Columns_Amplitude:
    qt[c].unit = units['Amplitude']
    
for c in Columns_Flux_p:
    qt[c].unit = units['Flux_p']
for c in Columns_Flux_E:
    qt[c].unit = units['Flux_E']
    
for c in Columns_Fluence_p:
    qt[c].unit = units['Fluence_p']
for c in Columns_Fluence_E:
    qt[c].unit = units['Fluence_E']


print(f"Write general catalog: {len(qt)} entries.")
qt.write(Output+"GBM_bursts.fits", format='fits', overwrite=True)


# Prepare the masked catalog
columns_flnc_band = ['name', 'trigger_name', 'trigger_time', 'scat_detector_mask',
                    'ra','dec','lii','bii','error_radius',
                    't90', 't90_error', 't90_start',
                    't50', 't50_error', 't50_start',
                    'back_interval_low_start' ,'back_interval_low_stop',
                    'back_interval_high_start','back_interval_high_stop',
                    'pflx_spectrum_start','pflx_spectrum_stop',
                    'flnc_spectrum_start','flnc_spectrum_stop',
                    'duration_energy_low','duration_energy_high','flu_low','flu_high',
                    'flnc_band_ampl' , 'flnc_band_ampl_pos_err' , 'flnc_band_ampl_neg_err' ,
                    'flnc_band_alpha', 'flnc_band_alpha_pos_err', 'flnc_band_alpha_neg_err',
                    'flnc_band_beta' , 'flnc_band_beta_pos_err' , 'flnc_band_beta_neg_err' ,
                    'flnc_band_epeak', 'flnc_band_epeak_pos_err', 'flnc_band_epeak_neg_err',
                    'flnc_band_phtflux' ,'flnc_band_phtflux_error' ,
                    'flnc_band_ergflux' ,'flnc_band_ergflux_error' ,
                    'flnc_band_phtflnc' ,'flnc_band_phtflnc_error' ,
                    'flnc_band_ergflnc' ,'flnc_band_ergflnc_error' ,
                    'flnc_band_redchisq','flnc_band_redfitstat'    ,
                    'flnc_band_dof'     ,'flnc_band_statistic',
                    'flnc_best_fitting_model', 'flnc_best_model_redchisq'
                ]

qt = qt[columns_flnc_band]

# Quality cuts

condition = np.where(qt['flnc_band_ampl'].mask == True)[0]
print(f"Drop {len(condition)} rows for no BAND")
qt.remove_rows(condition)

# condition = np.where(qt['flnc_band_redchisq'] > 10.0)[0]
# print(f"Drop {len(condition)} for reduced chi2 > 10.")
# qt.remove_rows(condition)

# condition = np.where(qt['error_radius'] > 40.0*u.deg)[0]
# print(f"Drop {len(condition)} for error radius > 40 deg.")
# qt.remove_rows(condition)

condition = np.where(qt['t90'] > 500.0*u.s)[0]
print(f"Drop {len(condition)} for t90 > 500 s.")
qt.remove_rows(condition)

# E_break = qt['flnc_band_epeak'] / (2.0+qt['flnc_band_alpha'])
# condition = np.where(E_break < 10.0*u.keV)[0]
# print(f"Drop {len(condition)} for E_break < 10 keV.")
# qt.remove_rows(condition)

print(f"Write flnc band selection: {len(qt)} entries.")
qt.write(Output+"GBM_bursts_flnc_band.fits", format='fits', overwrite=True)


# Strong GRB selection
condition = np.where(qt['flnc_band_phtflux'] < 15.0*u.Unit("s-1 cm-2"))[0]
print(f"Drop {len(condition)} for flux < 15 ph/s/cm2.")
qt.remove_rows(condition)

print(f"Write strong GRB selection: {len(qt)} entries.")
qt.write(Output+"GBM_bursts_flnc_band_flux15.fits", format='fits', overwrite=True)