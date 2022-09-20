# light-curve-GBM-sampler
Sample random GRBs from GBM Burst Archive, write its spectral parameters and light curve into a .source file for MEGALib.

The sampler is a python script sampler.py, that works in the gbm conda environment.

Put some info and links on how to create the environment.

## Workflow: Prepare a formatted Catalog of GBM Bursts.
You can use the ***GBM_bursts_flnc_band.fits*** file in the repository.

I created it by formatting the online ASCII Catalog of GBM Bursts:
1. Go to NASA HEASARC's Archive [ASCII Catalogs](https://heasarc.gsfc.nasa.gov/FTP/heasarc/dbase/dump/)
2. Download ***heasarc_fermigbrst.tdat.gz***
3. Unzip it.
4. Run the script ***format_archive_gbm.py*** to create ***GBM_bursts_flnc_band.fits*** and ***GBM_bursts_flnc_band.fits_flux15***. It can run in any environment with pandas and astropy. Note: the gbm environment does not have pandas.

## Workflow: Create light curves from the data of the strongest transients.
1. Run ***lightcurvemaker.py*** in the gbm environment.

## Workflow: Run the script to create the source files.
1. Run ***sampler.py*** in the gbm environment.

## Some caveats
To produce the source files we use several independent random numbers from uniform distributions:

1. The first is a random integer in [0; catalog_length]; we use it as an index to select one entry of the GBM GRB catalog. Then we write down the GRB name, l, b, flux, low index, high index, break energy, but NOT the light curve (usually too noisy). P.S: instead of choosing a random GRB we can also request one, if we are interested in simulating a particular GRB.
2. The second is a random float in [0; 1], the third in [0; 180]. Those are the polarization values. They have nothing to do with GBM.
3. The fourth is a random integer in [1; N_light_curves]. It selects one of the pre-produced light curves. The GRB used for the spectrum and the one used for the light curve are different.
