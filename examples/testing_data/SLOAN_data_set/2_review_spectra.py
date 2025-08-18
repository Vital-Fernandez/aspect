import lime
from pathlib import Path

fname = '/home/vital/Astrodata/SDSS_spectra/high_SN/spec-0266-51630-0307.fits'
spec = lime.Spectrum.from_file(fname, instrument='sdss')
spec.plot.spectrum(rest_frame=True, ax_cfg={'title':Path(fname).stem})

'''
We fit galaxies using an adaptation of the publicly available Gas AND Absorption Line Fitting (GANDALF, Sarzi et al. 2006)
 and penalised PiXel Fitting (pPXF, Cappellari & Emsellem 2004). Stellar population models for the continuum are from of 
 Maraston & Strömbäck (2011) and Thomas, Maraston & Johansson (2011).
'''