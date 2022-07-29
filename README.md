# light-curve-GBM-sampler
Sample random GRBs from GBM Burst Archive, write its spectral parameters and light curve into a .source file for MEGALib.

The sampler is a python script sampler.py, that works in the gbm conda environment.

Put some info and links on how to create the environment.

## Workflow: Prepare a formatted Catalog of GBM Bursts.
You can use the ***GBM_bursts_flnc_band.fits*** file in the repository.

I created it by formatting the online ASCII Catalog of GBM Bursts:
1. Go to NASA HEASARC's Archive [ASCII Catalogs](https://heasarc.gsfc.nasa.gov/FTP/heasarc/dbase/dump/)
2. Download ***heasarc_fermigbrst.tdat.gz***
3. Unzip it
4. Run the notebook ***Read_GBM_tdat.ipynb*** to create ***GBM_bursts_flnc_band.fits***. It can run in any environment with pandas and astropy. Note: the gbm environment does not have pandas.

## Workflow: Run the script to create the source files.
1. Run ***sampler.py*** in the gbm environment.

