import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.constants import c
from skimage import color, data, restoration

from inst.readmultispec import readmultispec
from inst.airtovac import airtovac

from inst.FTS_resample import resample, FTSfits
from PyAstronomy import pyasl

import matplotlib.pyplot as plt
from gplot import *
gplot.colors('classic')
from pause import pause

import importlib
Inst = importlib.import_module('inst.inst_CRIRES')
Spectrum = Inst.Spectrum

c = 3e5

def atm_mask(w,f,bp,berv):
        # mask telluric lines
        # still more testing needed
        maskfile = 'inst/CRIRES_atm/mask_0.25.txt' 
        f_mask = np.genfromtxt(maskfile, dtype=None)[:,1]

        lmin = w[0]
        lmax = w[-1]
        
        #w_mask = airtovac(mask[:,0])/np.e**(berv/c)
        w_mask = mask[:,0]

        sm = slice(*np.searchsorted(w_mask, [lmin, lmax]))
        w_mask1 = w_mask[sm]
        f_mask1 = f_mask[sm]

        # correction of shift between model and observed data
        # problem in CRIRES - may can be removed if problem solved?!
        if np.min(f) <= 0:
            shift = np.argmax(np.correlate(f_mask1,f,'same'))-len(f)/2
        else:
            shift = np.argmin(f_mask1)-np.argmin(f)

        lmin += shift*(w[1]-w[0])
        lmax += shift*(w[1]-w[0])
        sm = slice(*np.searchsorted(w_mask, [lmin, lmax]))
        f_mask1 = f_mask[sm]
        w_mask1 = w_mask[sm]
   
        if not f_mask1.size:
            f_mask = np.ones(len(f))*f_mask[np.argmin(abs(w_mask-lmax))]
            w_mask = w
        else:
            f_mask = np.interp(w,w_mask1,f_mask1)

        mask_perc = (f_mask >= 0.1).sum()/len(f_mask)*100
        print("# masked data (telluric): %4d / %4d, percent: %2.2f " % ((f_mask >= 0.1).sum(),len(f_mask),mask_perc))

        if mask_perc >= 70:
            print("too much values in telluric mask")

        bp[f_mask > 0.1] |= 8

        plot_tell = 0
        if plot_tell:
            plt.plot(w,f/np.nanmean(f))
            plt.plot(w_tpl,f_tpl/np.nanmean(f_tpl))
            plt.plot(w,f_mask,'+')
            plt.show()
            exit()
        return bp, mask_perc

def flag_tell(w,f,w_tpl,f_tpl,bp,o):
        # another way of flagging tellurics
        # more testing needed
        x = np.arange(len(f_tpl))
        ba = (np.load("lib/CRIRES/blaze_K2166.npy"))[o-1]
        bandp = (np.poly1d(ba)(x))
    #    print(len(x),len(bandp))
     #   exit()
        good = np.where(f_tpl/np.nanmean(f_tpl) > 0.8*bandp/np.nanmean(bandp))
        bp[f_tpl/np.nanmean(f_tpl) < 0.8*bandp/np.nanmean(bandp)] |= 64

#        good = np.where((f_tpl/np.nanmean(f_tpl) > 0.9*bandp/np.nanmean(bandp)) & (f/np.nanmean(f) > 0.9*bandp/np.nanmean(bandp)))
 #       bp[(f_tpl/np.nanmean(f_tpl) < 0.9*bandp/np.nanmean(bandp)) & (f/np.nanmean(f) < 0.9*bandp/np.nanmean(bandp))] |= 64

        w1 = w*1.

        gplot(w,f/np.nanmean(f),'w l t "data flag",', w_tpl,f_tpl/np.nanmean(f_tpl),'w l t "tpl flag",',w1,bandp/np.mean(bandp)*0.85,'w l t "blaze"')
        pause()

        w = w[good]
        f = f[good]
        w_tpl = w_tpl[good]
        f_tpl = f_tpl[good]

   #     return w,f,w_tpl,f_tpl
        return bp

def atm_model(w,f,berv,o):
        # divides spectra by atmosphere model
        # much more testing needed !
        modelfile = 'inst/CRIRES_atm/stdAtmos_crires_airmass1.fits'
        hdu = fits.open(modelfile, ignore_blank=True)
        atm_model = hdu[1].data
        w_atm2 = atm_model.field(0).astype(np.float64)
        f_atm2 = atm_model.field(1).astype(np.float64)

 #       file_obs2 = "data/CRIRES/210220_PiOri/cr2res_obs_nodding_extracted_combined.fits"
        file_obs2 = "/data/jana/VIPER/210220_tetVir/K2166/cr2res_obs_nodding_extracted_combined.fits"
        x2, w_atm, f_atm, bp2, bjd2, berv2 = Spectrum(file_obs2, o=o)
        bw = (np.load('wave_solution_tetvir.npy'))[o-1]
        w_atm = np.poly1d(bw[::-1])(x2)

      #  bb = [-0.457193, 314.784913, 5675608.219445]
        ba = (np.load("lib/CRIRES/blaze_K2166.npy"))[o-1]
        bandp = np.poly1d(ba)(x2)

        lmin = w[0]
        lmax = w[-1]
        lmin = max(w[0], w_atm[0])
        lmax = min(w[-1], w_atm[-1])
       
        smo = slice(*np.searchsorted(w_atm, [lmin, lmax]))
        w_atm1 = w_atm[smo]
        f_atm1 = f_atm[smo]
        smo = slice(*np.searchsorted(w, [lmin, lmax]))
        w = w[smo]
        f = f[smo]
        bandp = bandp[smo]

        gplot(w_atm,f_atm/np.nanmean(f_atm),'w l t "tell",', w,f/np.nanmean(f),'w l t "data",',w,bandp/np.mean(bandp)*0.9,'w l t "bandp"')#, w_atm1,f_atm1/np.nanmean(f_atm1),'w l')
        pause()

        # correction of shift between model and observed data
        rv, ccc = pyasl.crosscorrRV(w, f,w_atm1, f_atm1, -5.,5.,0.005,skipedge=20)
        pol = (np.poly1d(np.polyfit(rv,ccc,15)))(rv)
        shift = rv[np.argmax(pol)]
        print('shift atmmod: ', shift)
        print(w,w_atm1)
        w_atm /= (1+shift/3e5)
        smo = slice(*np.searchsorted(w_atm, [lmin, lmax]))
        w_atm1 = w_atm[smo]
        f_atm1 = f_atm[smo]

        print(w,w_atm1)

        f_atm1 = np.interp(w,w_atm1,f_atm1)

    #    f_div = f/np.nanmean(f)/(f_atm1/np.mean(f_atm1))
        f_div = f/f_atm1
        print('mean', np.nanmean(abs(f/np.mean(f)-f_div/np.mean(f_div))))
        i_flag = np.where(abs(f_div) <= (np.median(f_div) + 2*np.std(abs(f_div))))[0]
  #      i_flag = np.where(abs(f/np.nanmean(f)-f_div/np.nanmean(f_div)) <= 0.2)[0]
    #    f_div[f_div<=0] = 1000
   #     i_flag = np.where(abs(f_div) <= (np.median(f_div)*1.5))[0]
        w1 = w*1.
        w = w[i_flag]
        f = f[i_flag]
        f_div = f_div[i_flag]#*np.median(f)

        plot_tell = 1
        if plot_tell:
             print('mean', np.nanmean(abs(f/np.mean(f)-f_div/np.mean(f_div))))
     #        gplot(w,f/np.mean(f),'w l,', w,f_atm1/np.mean(f_atm1),'w l')
             gplot(w,f/np.mean(f),'w l t "data",', w,f_div/np.mean(f_div),'w l t "div data",',w1,f_atm1/np.mean(f_atm1),'w l t "tell",',w,abs(f/np.nanmean(f)-f_div/np.nanmean(f_div)),'w p pt 7 ps 0.4 t "res"')
             pause()

        return w,f_div,i_flag    #f_atm


def deconv_RL(f_tpl, w_tpl,bb=4):
        # Deconvolution of f_tpl using Richardson Lucy algorithm
        # needed for CRIRES data
        psf = np.ones((1,bb))/bb
        f2 = f_tpl+0.
        f_tpl1 = np.reshape(f_tpl,(1,-1))

        deconv_RL = restoration.richardson_lucy(f_tpl1/np.max(f_tpl1)/2., psf, iterations=30)[0]
        f_tpl = deconv_RL*np.max(f_tpl1)*2 

        rv, ccc = pyasl.crosscorrRV(w_tpl[200:-200], f_tpl[200:-200],w_tpl[200:-200], f2[200:-200], -10.,10.,0.01,skipedge=20)
        pol = (np.poly1d(np.polyfit(rv,ccc,15)))(rv)
        shift = rv[np.argmax(pol)]

        print('shift RL:         ', shift, rv[np.argmax(ccc)])
        # there seems to appear a shift in the spectra after the deconvolution
        # why ???
        # correction ???
        # more testing needed, or another way of deconvolution

        plot_RL = 0
        if plot_RL:
            gplot(rv,ccc,'w l,',rv,pol,'w l')
          #  gplot(w_tpl, f2, 'w l,', w_tpl /(1+shift/3e5),f_tpl,'w l')
            pause()
   #     w_tpl /= (1+shift/3e5)

        '''
        bo = 10
        f_tpl = f_tpl[bo:-bo]
        w_tpl = w_tpl[bo:-bo]
        f_ok = f_ok[bo:-bo]
        w_ok = w_ok[bo:-bo]
        x_ok = x_ok[bo:-bo]
        '''

        return f_tpl, w_tpl

def flag_outlier(x_ok,w_ok,f_ok,resid,ts=5):
    # flag/ clip outliers in the spectra
           
    len1 = len(x_ok)
        
    i_flag = 1 * np.isnan(f_ok)    
    i_flag = np.where(abs(resid) <= (np.median((resid)) + ts*np.std(abs(resid))))[0]
 #   i_flag = np.where(abs(resid) <= 0.08)[0]
    x_ok = x_ok[i_flag]
    w_ok = w_ok[i_flag]
    f_ok = f_ok[i_flag]
  #  e_ok = e_ok[i_flag]
      
    print("Nr of flagged data   :",len1-len(x_ok),"/",len1)

    return x_ok,w_ok,f_ok, i_flag


def search_gap(w_ok, x_ok, f_ok):
    # search for large gaps in frequency range
    w2 = w_ok[1:-1]-w_ok[0:-2]
 
    try:
        gap = np.where(w2>10*np.median(w2))[0][0]+1
    except:
        gap = 0
    
    if gap > 1:
         #   gap2 = -1
        if chunk == 0:
            x_ok = x_ok[0:gap]
            w_ok = w_ok[0:gap]
            f_ok = f_ok[0:gap]
            #    x_ok = x_ok[gap:gap2]
             #   w_ok = w_ok[gap:gap2]
              #  f_ok = f_ok[gap:gap2]
        if chunk == 1:
            x_ok = x_ok[gap:-1]
            w_ok = w_ok[gap:-1]
            f_ok = f_ok[gap:-1]

    return w_ok, x_ok, f_ok

def calc_weighting(resid):
 #  ps = np.poly1d(np.polyfit(x_ok,abs(resid),2))

#    plt.plot(x_ok,abs(resid))
 #   plt.plot(x_ok,(ps(x_ok)))
  #  plt.show()
   # exit()
#    sig = resid 
 #   sig[np.abs(resid)<np.std(resid)]  = np.std(resid)
  #  sig[np.abs(resid)>np.std(resid)] = np.abs(resid)
#    sig = [1/max(abs(mm),0.001) for mm in resid]
 #   print("sig", sig)
#    plt.plot(sig)
 #   plt.show()
  #  exit()
  #  sig = 1./ps(x_ok)
    sig = 1./abs(resid - np.mean(resid))

    return sig

def IP_mgs(vk, s0=1.9, a1=0.2, a2=0.2, a3=0.1, a4=0.1):    

        s1 = 0.5 * s0   # width of second Gaussian with fixed relative width
        s3 = s0
    #    a1 = a1  # relative ampitude

        IP_k0 = np.exp(-(vk/(s0/1.))**2)   # Gauss IP

        IP_k1 = a1*np.exp(-((vk-s0/0.75)/s1)**2)   # Gauss IP
        IP_k2 = a2*np.exp(-((vk+s0/0.75)/s1)**2)   # Gauss IP

        IP_k3 = a3*np.exp(-((vk-s0/0.35)/s3)**2)   # Gauss IP
        IP_k4 = a4*np.exp(-((vk+s0/0.35)/s3)**2)   # Gauss IP

        IP_k = IP_k0 + IP_k1 +IP_k2 + IP_k3 + IP_k4

        IP_k = IP_k.clip(0,None)
        IP_k /= IP_k.sum()          # normalise IP
        return IP_k


def IP_simple(x, s):
        return  np.exp(-(x/s)**2/2)/(np.exp(-(x/s)**2/2).sum())
