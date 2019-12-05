import numpy as np
import exoplanet as xo

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from astropy.io import ascii
from scipy.signal import savgol_filter

import pymc3 as pm
import theano.tensor as tt
import astropy.units as u
from astropy.units import cds
from astropy import constants as c

import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astroquery.gaia import Gaia

import pickle
import os.path
from datetime import datetime
import requests

import glob

import warnings
warnings.filterwarnings("ignore")

NamastePymc3_path = os.path.dirname(os.path.abspath( __file__ ))
from stellar import starpars

id_dic={'TESS':'TIC','tess':'TIC','Kepler':'KIC','kepler':'KIC','KEPLER':'KIC','K2':'EPIC','k2':'EPIC'}

#goto='/Users/hosborn' if 'Users' in os.path.dirname(os.path.realpath(__file__)).split('/') else '/home/hosborn'

def K2_lc(epic):
    '''
    # Opens K2 lc
    '''
    df,_=starpars.GetExoFop(epic,"k2")
    lcs=[]
    print("K2 campaigns to search:",df['campaign'])
    for camp in df['campaign'].split(','):
        lcs+=[getK2lc(epic,camp)]
    lcs=lcStack(lcs)
    return lcs,df

def getK2lc(epic,camp,saveloc=None):
    '''
    Gets (or tries to get) all LCs from K2 sources. Order is Everest > Vanderburg > PDC.
    '''
    from urllib.request import urlopen
    import everest
    try:
        lc=openEverest(epic,camp)
    except:
        print("No everest")
        try:
            lc=openVand(epic,camp)
        except:
            print("No vand")
            try:
                lc=openPDC(epic,camp)
            except:
                print("No LCs at all")
    return lc

def openFits(f,fname):
    '''
    # opens and processes all lightcurve files (especially, but not only, fits files).
    # Processing involvesd iteratively masking anomlaous flux values
    '''
    #print(type(f),"opening ",fname,fname.find('everest')!=-1,f[1].data,f[0].header['TELESCOP']=='Kepler')
    if type(f)==fits.hdu.hdulist.HDUList or type(f)==fits.fitsrec.FITS_rec:
        if f[0].header['TELESCOP']=='Kepler' or fname.find('kepler')!=-1:
            if fname.find('k2sff')!=-1:
                lc={'time':f[1].data['T'],'flux':f[1].data['FCOR'],
                    'flux_err':np.tile(np.median(abs(np.diff(f[1].data['FCOR']))),len(f[1].data['T'])),
                    'flux_raw':f[1].data['FRAW'],
                    'bg_flux':f[1+np.argmax([f[n].header['NPIXSAP'] for n in range(1,len(f)-3)])].data['flux_raw']}
                    #'rawflux':,'rawflux_err':,}
            elif fname.find('everest')!=-1:
                #logging.debug('Everest file')#Everest (Luger et al) detrending:
                print("everest file")
                lc={'time':f[1].data['TIME'],'flux':f[1].data['FCOR'],'flux_err':f[1].data['RAW_FERR'],
                    'raw_flux':f[1].data['fraw'],'bg_flux':f[1].data['BKG'],'qual':f[1].data['QUALITY']}
            elif fname.find('k2sc')!=-1:
                print("K2SC file")
                #logging.debug('K2SC file')#K2SC (Aigraine et al) detrending:
                lc={'time':f[1].data['time'],'flux':f[1].data['flux'],'flux_err':f[1].data['error']}
            elif fname.find('kplr')!=-1 or fname.find('ktwo')!=-1:
                #logging.debug('kplr/ktwo file')
                if fname.find('llc')!=-1 or fname.find('slc')!=-1:
                    #logging.debug('NASA/Ames file')#NASA/Ames Detrending:
                    print("Kepler file")
                    lc={'time':f[1].data['TIME'],'flux':f[1].data['PDCSAP_FLUX'],
                        'flux_err':f[1].data['PDCSAP_FLUX_ERR'],'raw_flux':f[1].data['SAP_FLUX'],
                        'bg_flux':f[1].data['SAP_BKG']}
                    if ~np.isnan(np.nanmedian(f[1].data['PSF_CENTR2'])):
                        lc['cent_1']=f[1].data['PSF_CENTR1'];lc['cent_2']=f[1].data['PSF_CENTR2']
                    else:
                        lc['cent_1']=f[1].data['MOM_CENTR1'];lc['cent_2']=f[1].data['MOM_CENTR2']
                elif fname.find('XD')!=-1 or fname.find('X_D')!=-1:
                    #logging.debug('Armstrong file')#Armstrong detrending:
                    lc={'time':f[1].data['TIME'],'flux':f[1].data['DETFLUX'],
                        'flux_err':f[1].data['APTFLUX_ERR']/f[1].data['APTFLUX']}
            else:
                print("unidentified file type")
                #logging.debug("no file type for "+str(f))
                return None
        elif f[0].header['TELESCOP']=='TESS':
            print("TESS file")
            time = f[1].data['TIME']
            sap = f[1].data['SAP_FLUX']/np.nanmedian(f[1].data['SAP_FLUX'])
            pdcsap = f[1].data['PDCSAP_FLUX']/np.nanmedian(f[1].data['PDCSAP_FLUX'])
            pdcsap_err = f[1].data['PDCSAP_FLUX_ERR']/np.nanmedian(f[1].data['PDCSAP_FLUX'])
            lc={'time':time,'flux':pdcsap,'flux_err':pdcsap_err,'raw_flux':f[1].data['SAP_FLUX'],
                'bg_flux':f[1].data['SAP_BKG']}
            if ~np.isnan(np.nanmedian(f[1].data['PSF_CENTR2'])):
                lc['cent_1']=f[1].data['PSF_CENTR1'];lc['cent_2']=f[1].data['PSF_CENTR2']
            else:
                lc['cent_1']=f[1].data['MOM_CENTR1'];lc['cent_2']=f[1].data['MOM_CENTR2']
    elif type(f)==np.ndarray and np.shape(f)[1]==3:
        #Already opened lightcurve file
        lc={'time':lc[:,0],'flux':lc[:,1],'flux_err':lc[:,2]}
    elif type(f)==dict:
        lc=f
    else:
        print('cannot identify fits type to identify with')
        #logging.debug('Found fits file but cannot identify fits type to identify with')
        return None

    # Mask bad data (nans, infs and negatives) 
    lc['mask'] = np.isfinite(lc['flux']) & np.isfinite(lc['time']) & (lc['flux']>0.0) 
    # Mask data if it's 4.2-sigma from its points either side (repeating at 7-sigma to get any points missed)
    lc['mask'][lc['mask']]=CutAnomDiff(lc['flux'][lc['mask']],4.2)
    lc['mask'][lc['mask']]=CutAnomDiff(lc['flux'][lc['mask']],7.0)

    # Convert to parts per thousand
    mu = np.median(lc['flux'][lc['mask']])
    lc['flux'] = (lc['flux'] / mu - 1) * 1e3
    yerr = (lc['flux_err'] / mu)*1e3
    
    # Identify outliers
    m2 = lc['mask']
    
    for i in range(10):
        y_prime = np.interp(lc['time'], lc['time'][m2], lc['flux'][m2])
        smooth = savgol_filter(y_prime, 101, polyorder=3)
        resid = lc['flux'] - smooth
        sigma = np.sqrt(np.nanmean(resid**2))
        m0 = np.abs(resid) < 3*sigma
        if m2.sum() == m0.sum():
            m2 = m0
            break
        m2 = m0

    # Only discard positive outliers
    lc['mask']*=(resid < 3*sigma)
    '''
    # Make sure that the data type is consistent
    lc['time'] = np.ascontiguousarray(x[m2], dtype=np.float64)
    lc['flux'] = np.ascontiguousarray(y[m2], dtype=np.float64)
    lc['flux_err'] = np.ascontiguousarray(yerr[m2], dtype=np.float64)
    lc['trend_rem'] = np.ascontiguousarray(smooth[m2], dtype=np.float64)
    
    for key in lc:
        if key not in ['time','flux','flux_err','trend_rem']:
            lc[key]=np.ascontiguousarray(lc[key][m][m2], dtype=np.float64)
    '''
    return lc

def openPDC(epic,camp):
    if camp == '10':
    #https://archive.stsci.edu/missions/k2/lightcurves/c1/201500000/69000/ktwo201569901-c01_llc.fits
        urlfilename1='https://archive.stsci.edu/missions/k2/lightcurves/c102/'+str(epic)[:4]+'00000/'+str(epic)[4:6]+'000/ktwo'+str(epic)+'-c102_llc.fits'
    else:
        urlfilename1='https://archive.stsci.edu/missions/k2/lightcurves/c'+str(int(camp))+'/'+str(epic)[:4]+'00000/'+str(epic)[4:6]+'000/ktwo'+str(epic)+'-c'+str(camp).zfill(2)+'_llc.fits'
    if requests.get(urlfilename1, timeout=600).status_code==200:
        with fits.open(urlfilename1) as hdus:
            lc=openFits(hdus,urlfilename1)
        return lc
    else:
        return None

def lcStack(lcs):
    #Stacks multiple lcs together
    outlc={}
    for key in lcs[0]:
        outlc[key]=np.hstack([lcs[nlc][key] for nlc in range(len(lcs))])
    return outlc

def openVand(epic,camp,v=1):
    lcvand=[]
    #camp=camp.split(',')[0] if len(camp)>3
    if camp=='10':
        camp='102'
    elif camp=='et' or camp=='E':
        camp='e'
        #https://www.cfa.harvard.edu/~avanderb/k2/ep60023342alldiagnostics.csv
    else:
        camp=str(int(camp)).zfill(2)
    if camp in ['09','11']:
        #C91: https://archive.stsci.edu/missions/hlsp/k2sff/c91/226200000/35777/hlsp_k2sff_k2_lightcurve_226235777-c91_kepler_v1_llc.fits
        url1='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(int(camp))+'1/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(int(camp))+'1_kepler_v1_llc.fits'
        print("Vanderburg LC at ",url1)
        if requests.get(url1, timeout=600).status_code==200:
            with fits.open(url1) as hdus:
                lcvand+=[openFits(hdus,url1)]
        url2='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(int(camp))+'2/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(int(camp))+'2_kepler_v1_llc.fits'
        if requests.get(url1, timeout=600).status_code==200:
            with fits.open(url1) as hdus:
                lcvand+=[openFits(hdus,url2)]
    elif camp=='e':
        print("Engineering data")
        #https://www.cfa.harvard.edu/~avanderb/k2/ep60023342alldiagnostics.csv
        url='https://www.cfa.harvard.edu/~avanderb/k2/ep'+str(epic)+'alldiagnostics.csv'
        print("Vanderburg LC at ",url)
        df=pd.read_csv(url,index_col=False)
        lc={'time':df['BJD - 2454833'].values,
            'flux':df[' Corrected Flux'].values,
            'flux_err':np.tile(np.median(abs(np.diff(df[' Corrected Flux'].values))),df.shape[0])}
        lcvand+=[openFits(lc,url)]
    else:
        urlfitsname='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(camp)+'/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(camp)+'_kepler_v'+str(int(v))+'_llc.fits'.replace(' ','')
        print("Vanderburg LC at ",urlfitsname)
        if requests.get(urlfitsname, timeout=600).status_code==200:
            with fits.open(urlfitsname) as hdus:
                lcvand+=[openFits(hdus,urlfitsname)]
    return lcStack(lcvand)
 
def openEverest(epic,camp):
    import everest
    if camp in ['10','11']:
        #One of the "split" campaigns:
        st1=everest.Everest(int(epic),season=camp+'1')
        st2=everest.Everest(int(epic),season=camp+'2')
        lcev={'time':np.vstack((st1.time,st2.time)),
              'flux':np.vstack((st1.flux,st2.flux)),
              'flux_err':np.vstack((st1.fraw_err,st2.fraw_err)),
              'raw_flux':np.vstack((st1.fraw,st2.fraw)),
              'raw_flux_err':np.vstack((st1.fraw_err,st2.fraw_err)),
              'quality':np.ones(len(st1.time)+len(st2.time))}
        lcev['quality'][st1.mask]==0.0;lcev['quality'][len(st1.time)+st2.mask]==0.0
        lcev=openFits(lcev,'NA')
        #elif int(camp)>=14:
        #    lcloc='https://archive.stsci.edu/hlsps/everest/v2/c'+str(int(camp))+'/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_everest_k2_llc_'+str(epic)+'-c'+str(int(camp))+'_kepler_v2.0_lc.fits'
        #    lcev=openFits(fits.open(lcloc),lcloc)
    else:
        st1=everest.Everest(int(epic),season=int(camp))
        lcev={'time':st1.time,'flux':st1.flux,'flux_err':st1.fraw_err,
              'raw_flux':st1.fraw,'raw_flux_err':st1.fraw_err,'quality':np.ones(len(st1.time))}
        lcev['quality'][st1.mask]==0.0
        lcev=openFits(lcev,'NA')
        #logging.debug(str(len(lcev))+"-long lightcurve from everest")
    return lcev
   
def getKeplerLC(kic):
    '''
    This module uses the KIC of a planet candidate to download lightcurves
    
    Args:
        kic: EPIC (K2) or KIC (Kepler) id number

    Returns:
        lightcurve
    '''
    qcodes=[2009131105131,2009166043257,2009259160929,2009350155506,2010009091648,2010078095331,2010174085026,
            2010265121752,2010355172524,2011073133259,2011177032512,2011271113734,2012004120508,2012088054726,
            2012179063303,2012277125453,2013011073258,2013098041711,2013131215648]
    lcs=[]
    for q in qcodes:
        lcloc='http://archive.stsci.edu/pub/kepler/lightcurves/'+str(int(kic)).zfill(9)[0:4]+'/'+str(int(kic)).zfill(9)+'/kplr'+str(int(kic)).zfill(9)+'-'+str(q)+'_llc.fits'
        if requests.get(lcloc, timeout=600).status_code==200:
            with fits.open(lcloc) as hdu:
                ilc=openFits(hdu,lcloc)
                if ilc is not None:
                    lcs+=[ilc]
                hdr=hdu[1].header
    lc=lcStack(lcs)
    return lc,hdr
    
    
def CutAnomDiff(flux,thresh=4.2):
    #Uses differences between points to establish anomalies.
    #Only removes single points with differences to both neighbouring points greater than threshold above median difference (ie ~rms)
    #Fast: 0.05s for 1 million-point array.
    #Must be nan-cut first
    diffarr=np.vstack((np.diff(flux[1:]),np.diff(flux[:-1])))
    diffarr/=np.median(abs(diffarr[0,:]))
    #Adding a test for the first and last points if they are >3*thresh from median RMS wrt next two points.
    anoms=np.hstack((abs(flux[0]-np.median(flux[1:3]))<(np.median(abs(diffarr[0,:]))*thresh*5),
                     ((diffarr[0,:]*diffarr[1,:])>0)+(abs(diffarr[0,:])<thresh)+(abs(diffarr[1,:])<thresh),
                     abs(flux[-1]-np.median(flux[-3:-1]))<(np.median(abs(diffarr[0,:]))*thresh*5)))
    return anoms

def TESS_lc(tic,sector='all'):
    #Downloading TESS lc
    epoch={1:'2018206045859_0120',2:'2018234235059_0121',3:'2018263035959_0123',4:'2018292075959_0124',
           5:'2018319095959_0125',6:'2018349182459_0126',7:'2019006130736_0131',8:'2019032160000_0136',
           9:'2019058134432_0139',10:'2019085135100_0140',11:'2019112060037_0143',12:'2019140104343_0144',
           13:'2019169103026_0146'}
    lcs=[];lchdrs=[]
    if type(sector)==str and sector=='all':
        epochs=list(epoch.keys())
    else:
        epochs=[sector]
        #observed_sectors=observed(tic)
        #observed_sectors=np.array([os for os in observed_sectors if observed_sectors[os]])
        #if observed_sectors!=[-1] and len(observed_sectors)>0:
        #    observed_sectors=observed_sectors[np.in1d(observed_sectors,np.array(list(epoch.keys())))]
        #else:
        #    observed_sectors=sector
        #print(observed_sectors)
    for key in epochs:
        fitsloc="https://archive.stsci.edu/missions/tess/tid/s"+str(key).zfill(4)+"/"+str(tic).zfill(16)[:4]+"/"+str(tic).zfill(16)[4:8]+"/"+str(tic).zfill(16)[-8:-4]+"/"+str(tic).zfill(16)[-4:]+"/tess"+epoch[key].split('_')[0]+"-s"+str(key).zfill(4)+"-"+str(tic).zfill(16)+"-"+epoch[key].split('_')[1]+"-s_lc.fits"
        if requests.get(fitsloc, timeout=600).status_code==200:
            with fits.open(fitsloc) as hdus:
                lcs+=[openFits(hdus,fitsloc)]
                lchdrs+=[hdus[0].header]
                '''
                with fits.open(fitsloc, mode="readonly") as hdulist:
                time = hdulist[1].data['TIME']
                sap = hdulist[1].data['SAP_FLUX']/np.nanmedian(hdulist[1].data['SAP_FLUX'])
                pdcsap = hdulist[1].data['PDCSAP_FLUX']/np.nanmedian(hdulist[1].data['PDCSAP_FLUX'])
                bg = hdulist[1].data['SAP_BKG']/np.nanmedian(hdulist[1].data['SAP_BKG'])
                bg_err = hdulist[1].data['SAP_BKG_ERR']/np.nanmedian(hdulist[1].data['SAP_BKG'])
                if np.nansum(hdulist[1].data['PSF_CENTR2'])==0.0:
                    cent = np.sqrt((hdulist[1].data['MOM_CENTR1']-np.nanmedian(hdulist[1].data['MOM_CENTR1']))**2+
                                   (hdulist[1].data['MOM_CENTR2']-np.nanmedian(hdulist[1].data['MOM_CENTR2']))**2)
                else:
                    cent = np.sqrt((hdulist[1].data['PSF_CENTR1']-np.nanmedian(hdulist[1].data['PSF_CENTR1']))**2+
                                   (hdulist[1].data['PSF_CENTR2']-np.nanmedian(hdulist[1].data['PSF_CENTR2']))**2)

            sectlcs+=[np.column_stack((time,pdcsap,sap,bg,cent))]
            print(fitsloc)
            '''
    lc=lcStack(lcs)
    return lc,lchdrs[0]

def PeriodGaps(t,t0,dur=0.5):
    # Given the time array, the t0 of transit, and the fact that another transit is not observed, 
    #   we want to calculate a distribution of impossible periods to remove from the Period PDF post-MCMC
    # In this case, a list of impossible periods is returned, with all points within 0.5dur of those to be cut
    dist_from_t0=np.sort(abs(t0-t))
    gaps=np.where(np.diff(dist_from_t0)>(0.9*dur))[0]
    listgaps=[]
    for ng in range(len(gaps)):
        start,end=dist_from_t0[gaps[ng]],dist_from_t0[gaps[ng]+1]
        listgaps+=[np.linspace(start,end,np.ceil(2*(end-start)/dur))]
    listgaps+=[np.max(dist_from_t0)]
    return np.hstack(listgaps)

def init_model(x, y, yerr, initdepth, initt0, Rstar, rhostar, Teff, lcmask=None, logg=np.array([4.3,1.0,1.0]),initdur=None, 
               periods=None,per_index=-8/3,assume_circ=False,
               use_GP=True,constrain_LD=True,ld_mult=3,useL2=True,
               mission='TESS',FeH=0.0,LoadFromFile=False,cutDistance=0.0):
    # x - array of times
    # y - array of flux measurements
    # yerr - flux measurement errors
    # initdepth - initial depth guess
    # initt0 - initial time guess
    # Rstar - array with radius of star and error/s
    # rhostar - array with density of star and error/s
    # periods - In the case where a planet is already transiting, include the period guess as a an array with length n_pl
    # per_index - index to raise the period to. Kipping 2019 suggests -8/3 while Sandford 2019 suggests -5/3 is better
    # constrain_LD - Boolean. Whether to use 
    # ld_mult - Multiplication factor on STD of limb darkening]
    # cutDistance - cut out points further than this from transit. Default of zero does no cutting
    
    lcmask=np.tile(True,len(x)) if lcmask is None else lcmask
    
    n_pl=len(initt0)
    
    print("Teff:",Teff)
    start=None
    with pm.Model() as model:
        
        # We're gonna need a bounded normal:
        #BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=3)

        #Stellar parameters (although these aren't really fitted.)
        #Using log rho because otherwise the distribution is not normal:
        if len(rhostar)==3:
            logrho_S = pm.Normal("logrho_S", mu=np.log(rhostar[0]), sd=np.average(abs(rhostar[1:]/rhostar[0])),testval=np.log(rhostar[0]))
        else:
            logrho_S = pm.Normal("logrho_S", mu=np.log(rhostar[0]), sd=rhostar[1]/rhostar[0],testval=np.log(rhostar[0]))

        rho_S = pm.Deterministic("rho_S",tt.exp(logrho_S))
        if len(Rstar)==3:
            Rs = pm.Normal("Rs", mu=Rstar[0], sd=np.average(Rstar[1:]),testval=Rstar[0],shape=1)
        else:
            Rs = pm.Normal("Rs", mu=Rstar[0], sd=Rstar[1],testval=Rstar[0],shape=1)
        Ms = pm.Deterministic("Ms",(rho_S/1.408)*Rs**3)

        # The baseline flux
        mean = pm.Normal("mean", mu=0.0, sd=1.0,testval=0.0)
        
        # The 2nd light (not third light as companion light is not modelled) 
        # This quantity is in delta-mag
        if useL2:
            deltamag_contam = pm.Uniform("deltamag_contam", lower=-20.0, upper=20.0)
            mult = pm.Deterministic("mult",(1+tt.power(2.511,-1*deltamag_contam))) #Factor to multiply normalised lightcurve by
        else:
            mult=1.0
        

        # The time of a reference transit for each planet 
        t0 = pm.Normal("t0", mu=initt0, sd=1.0, shape=n_pl, testval=initt0)
        
        #Calculating minimum period:
        P_gap_cuts=[];pertestval=[]
        print(initt0)
        for n,nt0 in enumerate(initt0):
            #Looping over all t0s - i.e. all planets
            if periods is None or np.isnan(periods[n]) or periods[n]==0.0:
                
                dist_from_t0=np.sort(abs(nt0-x[lcmask]))
                inputdur=0.5 if initdur is None or np.isnan(initdur[n]) or initdur[n]==0.0 else initdur[n]
                P_gap_cuts+=[PeriodGaps(x[lcmask],nt0,inputdur)]
                #Estimating init P using duration:
                initvrel=(2*(1+np.sqrt(initdepth[n]))*np.sqrt(1-(0.41/(1+np.sqrt(initdepth[n])))**2))/inputdur
                initper=18226*(rhostar[0]/1.408)/(initvrel**3)
                print(initper,P_gap_cuts[n])
                if initper>P_gap_cuts[n][0]:
                    pertestval+=[np.power(initper/P_gap_cuts[n][0],per_index)]
                else:
                    pertestval+=[0.5]
            else:
                P_gap_cuts+=[0.75*periods[n]]
                pertestval+=[np.power(periods[n]/P_gap_cuts[n][0],per_index)]
        
        #Cutting points for speed of computation:
        speedmask=np.tile(False, len(x))
        for n,it0 in enumerate(initt0):
            if periods is not None and not np.isnan(periods[n]) and not periods[n]==0.0:
                #For known periodic planets, need to keep many transits, so masking in the period space:
                speedmask[(((x-it0)%periods[n])<cutDistance)|(((x-it0)%periods[n])>(periods[n]-cutDistance))]=True
            elif cutDistance>0.0:
                speedmask[abs(x-it0)<cutDistance]=True
            else:
                #No parts of the lc to cut
                speedmask=np.tile(True,len(x))
            print(np.sum(~speedmask),"points cut from lightcurve leaving",np.sum(speedmask),"to process")
        totalmask=speedmask*lcmask
        
        P_min=np.array([P_gap_cuts[n][0] for n in range(len(P_gap_cuts))]);pertestval=np.array(pertestval)
        print("Using minimum period(s) of:",P_min)
        
        #Using a normal distribution between 0.0 (inf period) and 1.0 (min period) in the index-adjusted parameter space.
        P_index = xo.distributions.UnitUniform("P_index", shape=n_pl, testval=pertestval)#("P_index", mu=0.5, sd=0.3)
        #P_index = pm.Bound("P_index", upper=1.0, lower=0.0)("P_index", mu=0.5, sd=0.33, shape=n_pl)
        period = pm.Deterministic("period", tt.power(P_index,1/per_index)*P_min)
        logp = pm.Deterministic("logp", tt.log(period))

        # The Espinoza (2018) parameterization for the joint radius ratio and
        # impact parameter distribution
        if useL2:
            #EB case as second light needed:
            RpRs, b = xo.distributions.get_joint_radius_impact(
                min_radius=0.001, max_radius=1.25,
                testval_r=np.array(initdepth)**0.5,
                testval_b=np.random.rand(n_pl)
            )
        else:
            RpRs, b = xo.distributions.get_joint_radius_impact(
                min_radius=0.001, max_radius=0.25,
                testval_r=np.array(initdepth)**0.5,
                testval_b=np.random.rand(n_pl)
            )

        r_pl = pm.Deterministic("r_pl", RpRs * Rs)
        
        if assume_circ:
            orbit = xo.orbits.KeplerianOrbit(
                r_star=Rs, rho_star=rho_S,
                period=period, t0=t0, b=b)
        else:
            # This is the eccentricity prior from Kipping (2013) / https://arxiv.org/abs/1306.4982
            BoundedBeta = pm.Bound(pm.Beta, lower=1e-5, upper=1-1e-5)
            ecc = BoundedBeta("ecc", alpha=0.867, beta=3.03, shape=n_pl,
                              testval=np.tile(0.1,n_pl))
            omega = xo.distributions.Angle("omega", shape=n_pl, testval=np.tile(0.1,n_pl))
            orbit = xo.orbits.KeplerianOrbit(
                r_star=Rs, rho_star=rho_S,
                ecc=ecc, omega=omega,
                period=period, t0=t0, b=b)
        
        vx, vy, vz = orbit.get_relative_velocity(t0)
        #vsky = 
        if n_pl>1:
            vrel=pm.Deterministic("vrel",tt.diag(tt.sqrt(vx**2 + vy**2))/Rs)
        else:
            vrel=pm.Deterministic("vrel",tt.sqrt(vx**2 + vy**2)/Rs)
        
        tdur=pm.Deterministic("tdur",(2*tt.sqrt(1-b**2))/vrel)
        
        if constrain_LD:
            n_samples=1200
            # Bounded normal distributions (bounded between 0.0 and 1.0) to constrict shape given star.
            ld_dists=getLDs(np.random.normal(Teff[0],Teff[1],n_samples),
                            np.random.normal(logg[0],logg[1],n_samples),FeH,mission=mission)
            print("contrain LDs - ",Teff[0],Teff[1],logg[0],logg[1],FeH,n_samples,
                  np.clip(np.nanmedian(ld_dists,axis=0),0,1),np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.05,1.0))
            u_star = pm.Bound(pm.Normal, lower=0.0, upper=1.0)("u_star", 
                                        mu=np.clip(np.nanmedian(ld_dists,axis=0),0,1),
                                        sd=np.clip(ld_mult*np.nanstd(ld_dists,axis=0),0.05,1.0), shape=2, testval=np.clip(np.nanmedian(ld_dists,axis=0),0,1))
        else:
            # The Kipping (2013) parameterization for quadratic limb darkening paramters
            u_star = xo.distributions.QuadLimbDark("u_star", testval=np.array([0.3, 0.2]))
        tt.printing.Print('Rs')(Rs)
        tt.printing.Print('RpRs')(RpRs)
        tt.printing.Print('u_star')(u_star)
        tt.printing.Print('r_pl')(r_pl)
        #tt.printing.Print('t0')(t0)
        print(P_min,t0,x[totalmask][:10],np.nanmedian(np.diff(x[totalmask])))
        # Compute the model light curve using starry
        light_curves = xo.LimbDarkLightCurve(u_star).get_light_curve(orbit=orbit, r=r_pl,t=x[totalmask])*1e3
        light_curve = pm.math.sum(light_curves, axis=-1)
        pm.Deterministic("light_curves", light_curves)
        
        if use_GP:
            # Transit jitter & GP parameters
            #logs2 = pm.Normal("logs2", mu=np.log(np.var(y[m])), sd=10)
            logs2 = pm.Uniform("logs2", upper=np.log(np.std(y[totalmask]))+4,lower=np.log(np.std(y[totalmask]))-4)
            
            logw0_guess = np.log(2*np.pi/10)
            cad=np.nanmedian(np.diff(x))#Limiting to <1 cadence
            lcrange=x[totalmask][-1]-x[totalmask][0]
            
            #freqs bounded from 2pi/cadence to to 2pi/(4x lc length)
            logw0 = pm.Uniform("logw0",lower=np.log((2*np.pi)/(4*lcrange)), 
                               upper=np.log((2*np.pi)/cad))

            # S_0 directly because this removes some of the degeneracies between
            # S_0 and omega_0 prior=(-0.25*lclen)*exp(logS0)
            logpower = pm.Uniform("logpower",lower=-20,upper=np.log(np.nanmedian(abs(np.diff(y[totalmask])))))
            logS0 = pm.Deterministic("logS0", logpower - 4 * logw0)
            
            #timescale = pm.Bound(pm.Exponential, upper=(4*lcrange), lower=cad)("timescale",lam=1.0)
            #logw0 = pm.Deterministic("logw0",tt.log(timescale))
            #w0 = pm.Bound(pm.Exponential, lower=(2*np.pi)/(4*lcrange), upper=(2*np.pi)/cad)("w0",lam=1.0)
            #w0 = pm.Bound(pm.Gamma, lower=(2*np.pi)/(4*lcrange), upper=(2*np.pi)/cad)("w0",alpha=1,beta=25)
            #logw0 = pm.Deterministic("logw0",tt.log(w0))
            # We'll parameterize using the maximum power (S_0 * w_0^4) instead of
            # S_0 directly because this removes some of the degeneracies between
            # S_0 and omega_0 prior=(-0.25*lclen)*exp(logS0)
            #power=pm.Bound(pm.Gamma,lower=1e-12,upper=2*np.nanmedian(abs(np.diff(y[m]))))("power",alpha=1,beta=25)
            '''
            #EVANS:lp-=priors[key][3]*np.exp(params[n])#
            min_bound=-50
            max_bound=10
            high_num_fact=(np.max([abs(min_bound),max_bound])+0.33*(max_bound-min_bound))
            np.exp(np.clip(np.log(high_num_fact+S0),np.log(high_num_fact+min_bound),np.log(high_num_fact+max_bound))-high_num_fact
            
            logpower = pm.Bound(pm.Normal,lower=min_bound,upper=max_bound)("logw0", mu=logw0_guess, sd=10)
            
            #Adding a bound on the power to keep it lower than the amplitude of the per-point rms to stop it over-fitting
            
            #power=pm.Bound(pm.Gamma,lower=1e-12,upper=2*np.nanmedian(abs(np.diff(y[m]))) )("power",alpha=1,beta=25)
            #logS0 = pm.Deterministic("logS0", tt.log(power) - 30 - 4 * logw0)
            #logpower_adj=pm.Bound(pm.Exponential,lower=0.0,upper=30+np.log(np.nanmedian(abs(np.diff(y[m]))))+0.5)("logpower_adj",lam=1.0)
            logpower = pm.Bound(pm.Normal,lower=-50,upper=np.log(np.nanmedian(abs(np.diff(y[m])))))("logpower",
                                 mu=np.log(1e-4*np.var(y[m]))+4*logw0_guess,
                                 sd=10)
            logpower_lower=np.log(1e-5*np.var(y[m]))+4*logw0_guess
            logpower_upper=np.log(np.nanmedian(abs(np.diff(y[m]))))
            print("lower:",logpower_lower,"upper:",logpower_upper)
            logpower = pm.Bound(pm.Normal,lower=logpower_lower,upper=logpower_upper)("logpower",
                                 mu=logpower_lower+2.0,
                                 sd=0.5*(logpower_upper-logpower_lower+2.0))
            '''
            
            # GP model for the light curve
            kernel = xo.gp.terms.SHOTerm(log_S0=logS0, log_w0=logw0, Q=1/np.sqrt(2))
            gp = xo.gp.GP(kernel, x[totalmask], tt.exp(logs2) + tt.zeros(np.sum(totalmask)), J=2)

            #pm.Potential("p_prior", tt.power(period)

            llk_gp = pm.Potential("transit_obs", gp.log_likelihood(y[totalmask] - light_curve/mult))
            gp_pred = pm.Deterministic("gp_pred", gp.predict())

            #chisqs = pm.Deterministic("chisqs", (y - (gp_pred + tt.sum(light_curve,axis=-1)))**2/yerr**2)
            #avchisq = pm.Deterministic("avchisq", tt.sum(chisqs))
            #llk = pm.Deterministic("llk", model.logpt)
        else:
            pm.Normal("obs", mu=light_curve, sd=yerr[totalmask], observed=y[totalmask])

        # Fit for the maximum a posteriori parameters, I've found that I can get
        # a better solution by trying different combinations of parameters in turn
        if start is None:
            start = model.test_point
        
        if not LoadFromFile:
            map_soln = xo.optimize(start=start, vars=[RpRs, b])
            map_soln = xo.optimize(start=map_soln, vars=[logs2])
            map_soln = xo.optimize(start=map_soln, vars=[P_index, t0])
            map_soln = xo.optimize(start=map_soln, vars=[logs2, logpower])
            map_soln = xo.optimize(start=map_soln, vars=[logw0])
            map_soln = xo.optimize(start=map_soln)
            
            print(model.check_test_point())
            
            return model, map_soln, totalmask, P_gap_cuts
        else:
            return model, None, totalmask, P_gap_cuts

        # This shouldn't make a huge difference, but I like to put a uniform
        # prior on the *log* of the radius ratio instead of the value. This
        # can be implemented by adding a custom "potential" (log probability).
        #pm.Potential("r_prior", -pm.math.log(r))

        
def Run(ID, initdepth, initt0, mission='TESS', stellardict=None,n_draws=1200,
        overwrite=False,LoadFromFile=False,savefileloc=None, doplots=True,do_per_gap_cuts=True, **kwargs):
    #, cutDistance=0.0):
    """#PymcSingles - Run model
    Inputs:
    #  * ID - ID of star (in TESS, Kepler or K2)
    #  * initdepth - initial detected depth (for Rp guess)
    #  * initt0 - initial detection transit time
    #  * mission - TESS or Kepler/K2
    #  * stellardict - dictionary of stellar parameters. (alternatively taken from Gaia). With:
    #         Rs, Rs_err - 
    #         rho_s, rho_s_err - 
    #         Teff, Teff_err - 
    #         logg, logg_err - 
    #  * n_draws - number of samples for the MCMC to take
    #  * overwrite - whether to overwrite saved samples
    #  * LoadFromFile - whether to load the last written sample file
    #  * savefileloc - location of savefiles. If None, creates a folder specific to the ID
    # In KWARGS:
    #  * ALL INPUTS TO INIT_MODEL
    
    Outputs:
    # model - the PyMc3 model
    # trace - the samples
    # lc - a 3-column light curve with time, flux, flux_err
    """
    
    if not LoadFromFile:
        savename=GetSavename(ID, mission, how='save', overwrite=overwrite, savefileloc=savefileloc)
    else:
        savename=GetSavename(ID, mission, how='load', overwrite=overwrite, savefileloc=savefileloc)

    if os.path.exists(savename.replace('_mcmc.pickle','.lc')) and os.path.exists(savename.replace('_mcmc.pickle','_hdr.pickle')) and not overwrite:
        print("loading from",savename.replace('_mcmc.pickle','.lc'))
        #Loading lc from file
        lc_nd = np.genfromtxt(savename.replace('_mcmc.pickle','.lc'), dtype=float, delimiter=',', names=True)
        lc={}
        for key in lc_nd.dtype.names:
            lc[key]=lc_nd[key]
        lc['mask']=lc['mask'].astype(bool)
        hdr=pickle.load(open(savename.replace('_mcmc.pickle','_hdr.pickle'),'rb'))
        
    else:
        #Opening using url search:
        if mission is 'TESS':
            lc,hdr = TESS_lc(ID)
        elif mission is 'K2':
            lc,hdr = K2_lc(ID)
        elif mission is 'Kepler':
            lc,hdr = getKeplerLC(ID)
        np.savetxt(savename.replace('_mcmc.pickle','.lc'),np.column_stack([lc[key] for key in list(lc.keys())]),header=','.join(list(lc.keys())),delimiter=',')
        pickle.dump(hdr, open(savename.replace('_mcmc.pickle','_hdr.pickle'),'wb'))

    if stellardict is None:
        Rstar, rhostar, Teff, logg, src = starpars.getStellarInfo(ID, hdr, mission, overwrite=overwrite,
                                                             fileloc=savename.replace('_mcmc.pickle','_starpars.csv'),
                                                             savedf=True)
    else:
        if type(stellardict['Rs_err'])==tuple:
            Rstar=np.array([stellardict['Rs'],stellardict['Rs_err'][0],stellardict['Rs_err'][1]])
        else:
            Rstar=np.array([stellardict['Rs'],stellardict['Rs_err'],stellardict['Rs_err']])
        if type(stellardict['rho_s_err'])==tuple:
            rhostar = np.array([stellardict['rho_s'],stellardict['rho_s_err'][0],stellardict['rho_s_err'][1]])
        else:
            rhostar = np.array([stellardict['rho_s'],stellardict['rho_s_err'],stellardict['rho_s_err']])
        if type(stellardict['Teff_err'])==tuple:
            Teff = np.array([stellardict['Teff'],stellardict['Teff_err'][0],stellardict['Teff_err'][1]])
        else:
            Teff = np.array([stellardict['Teff'],stellardict['Teff_err'],stellardict['Teff_err']])
        if type(stellardict['logg_err'])==tuple:
            logg = np.array([stellardict['logg'],stellardict['logg_err'][0],stellardict['logg_err'][1]])
        else:
            logg = np.array([stellardict['logg'],stellardict['logg_err'],stellardict['logg_err']])
    print("Initialising transit model")
    print(lc['time'],type(lc['time']),type(lc['time'][0]))
    model, soln, lcmask, P_gap_cuts = init_model(lc['time'], lc['flux'],lc['flux_err'],
                                                 initdepth, initt0, Rstar, rhostar, Teff,
                                                 lcmask=lc['mask'], logg=logg, **kwargs)
    #initdur=None,n_pl=1,periods=None,per_index=-8/3,
    #assume_circ=False,use_GP=True,constrain_LD=True,ld_mult=1.5,
    #mission='TESS',LoadFromFile=LoadFromFile,cutDistance=cutDistance)
    print("Model loaded")


    #try:
    if LoadFromFile and not overwrite:
        trace = LoadPickle(ID, mission, savename)
    else:
        trace=None
   
    if trace is None:
        #Running sampler:
        np.random.seed(int(ID))
        with model:
            trace = pm.sample(tune=int(n_draws*0.66), draws=n_draws, start=soln, chains=4,
                                  step=xo.get_dense_nuts_step(target_accept=0.9))
        SavePickle(trace, ID, mission, savename)
            
    if do_per_gap_cuts:
        #Doing Cuts for Period gaps (i.e. where photometry rules out the periods of a planet)
        #Only taking MCMC positions in the trace where either:
        #  - P<0.5dur away from a period gap in P_gap_cuts[:-1]
        #  - OR P is greater than P_gap_cuts[-1]
        tracemask=np.tile(True,len(trace['period'][:,0]))
        for n in range(len(P_gap_cuts)):
            #for each planet
            #Cutting points where P<P_gap_cuts[-1] and P is not within 0.5Tdurs of a gap:
            gap_dists=np.nanmin(abs(trace['period'][:,n][:,np.newaxis]-P_gap_cuts[n][:-1][np.newaxis,:]),axis=1)
            tracemask[(trace['period'][:,n]<P_gap_cuts[n][-1])*(gap_dists>0.5*np.nanmedian(trace['tdur'][:,n]))] = False
            
        #tracemask=np.column_stack([(np.nanmin(abs(trace['period'][:,n][:,np.newaxis]-P_gap_cuts[n][:-1][np.newaxis,:]),axis=1)<0.5*np.nanmedian(trace['tdur'][:,n]))|(trace['period'][:,n]>P_gap_cuts[n][-1]) for n in range(len(P_gap_cuts))]).any(axis=1)
        print(np.sum(~tracemask),"(",int(100*np.sum(~tracemask)/len(tracemask)),") removed due to period gap cuts")
    else:
        tracemask=None
    if doplots:
        print("plotting")
        PlotLC(lc, trace, ID, mission=mission, savename=savename.replace('mcmc.pickle','TransitFit.png'), lcmask=lcmask,tracemask=tracemask)
        PlotCorner(trace, ID, mission=mission, savename=savename.replace('mcmc.pickle','corner.png'),tracemask=tracemask)
    restable=ToLatexTable(trace, ID, mission=mission, varnames=None,order='columns',
                          savename=savename.replace('mcmc.pickle','results.txt'), overwrite=False,
                          savefileloc=None, tracemask=tracemask)
    imports()
    return {'model':model, 'trace':trace, 'light_curve':lc, 'lcmask':lcmask, 'P_gap_cuts':P_gap_cuts, 'tracemask':tracemask,'restable':restable}

def imports():
    # Getting all imported stuff
    import sys
    for module in sys.modules:
        try:
            print(module,'==',sys.modules[module].__version__)
        except:
            try:
                if  type(modules[module].version) is str:
                    print(module,'==',sys.modules[module].version)
                else:
                    print(module,'==',sys.modules[module].version())
            except:
                try:
                    print(module,'==',sys.modules[module].VERSION)
                except:
                    pass


def GetSavename(ID, mission, how='load', suffix='mcmc.pickle', overwrite=False, savefileloc=None):
    '''
    # Get unique savename (defaults to MCMC suffic) with format:
    # [savefileloc]/[T/K]IC[11-number ID]_[20YY-MM-DD]_[n]_mcmc.pickle
    #
    # INPUTS:
    # - ID
    # - mission - (TESS/K2/Kepler)
    # - how : 'load' or 'save'
    # - suffix : final part of file string. default is _mcmc.pickle
    # - overwrite : if 'save', whether to overwrite past save or not.
    # - savefileloc : file location of files to save (default: 'NamastePymc3/[T/K]ID[11-number ID]/
    #
    # OUTPUTS:
    # - filepath
    '''
    if savefileloc is None:
        savefileloc=os.path.join(NamastePymc3_path,id_dic[mission]+str(ID).zfill(11))
    if not os.path.isdir(savefileloc):
        os.mkdir(savefileloc)
    pickles=glob.glob(os.path.join(savefileloc,id_dic[mission]+str(ID).zfill(11)+"*"+suffix))
    if how is 'load' and len(pickles)>1:
        #finding most recent pickle:
        date=np.max([datetime.strptime(pick.split('_')[1],"%Y-%m-%d") for pick in pickles]).strftime("%Y-%m-%d")
        datepickles=glob.glob(os.path.join(savefileloc,id_dic[mission]+str(ID).zfill(11)+"_"+date+"_*_"+suffix))
        if len(datepickles)>1:
            nsim=np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])
        elif len(datepickles)==1:
            nsim=0
        elif len(datepickles)==0:
            print("problem - no saved mcmc files in correct format")
    elif how is 'load' and len(pickles)==1:
        date=pickles[0].split('_')[1]
        nsim=pickles[0].split('_')[2]
    else:
        #Either pickles is empty (no file to load) or we want to save a fresh file:
        #Finding unique
        date=datetime.now().strftime("%Y-%m-%d")
        datepickles=glob.glob(os.path.join(savefileloc,id_dic[mission]+str(ID).zfill(11)+"_"+date+"_*_"+suffix))
        if len(datepickles)==0:
            nsim=0
        elif overwrite:
            nsim=np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])
        else:
            #Finding next unused number with this date:
            nsim=1+np.max([int(nmdp.split('_')[2]) for nmdp in datepickles])
    
    return os.path.join(savefileloc,id_dic[mission]+str(ID).zfill(11)+"_"+date+"_"+str(int(nsim))+"_"+suffix)
                                
    
def LoadPickle(ID, mission,loadname=None,savefileloc=None):
    #Pickle file style: folder/TIC[11-number ID]_[20YY-MM-DD]_[n]_mcmc.pickle
    if loadname is None:
        loadname=GetSavename(ID, mission, how='load', suffix='mcmc.pickle', savefileloc=savefileloc)
    if os.path.exists(loadname):
        n_bytes = 2**31
        max_bytes = 2**31 - 1

        ## read
        bytes_in = bytearray(0)
        input_size = os.path.getsize(loadname)
        with open(loadname, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        trace = pickle.loads(bytes_in)
        return trace
    else:
        return None

def SavePickle(trace,ID,mission,savename=None,overwrite=False,savefileloc=None):
    if savename is None:
        savename=GetSavename(ID, mission, how='save', suffix='mcmc.pickle', overwrite=overwrite, savefileloc=savefileloc)
        
    n_bytes = 2**31
    max_bytes = 2**31 - 1

    ## write
    bytes_out = pickle.dumps(trace)
    with open(savename, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

def getLDs(Ts,logg=4.43812,FeH=0.0,mission="TESS"):
    from scipy.interpolate import CloughTocher2DInterpolator as ct2d

    if mission[0]=="T" or mission[0]=="t":
        import pandas as pd
        from astropy.io import ascii
        TessLDs=ascii.read(os.path.join(NamastePymc3_path,'data','tessLDs.txt')).to_pandas()
        TessLDs=TessLDs.rename(columns={'col1':'logg','col2':'Teff','col3':'FeH','col4':'L/HP','col5':'a',
                                           'col6':'b','col7':'mu','col8':'chi2','col9':'Mod','col10':'scope'})
        a_interp=ct2d(np.column_stack((TessLDs.Teff.values.astype(float),TessLDs.logg.values.astype(float))),TessLDs.a.values.astype(float))
        b_interp=ct2d(np.column_stack((TessLDs.Teff.values.astype(float),TessLDs.logg.values.astype(float))),TessLDs.b.values.astype(float))

        if (type(Ts)==float) or (type(Ts)==int):
            Ts=np.array([Ts])
        if type(logg) is float:
            outarr=np.column_stack((np.array([a_interp(T,logg) for T in np.clip(Ts,2300,12000)]),
                                    np.array([b_interp(T,logg) for T in np.clip(Ts,2300,12000)])))
        else:
            outarr=np.column_stack((a_interp(np.clip(Ts,2300,12000),logg),b_interp(np.clip(Ts,2300,12000),logg)))
        return outarr
    elif mission[0]=="k" or mission[0]=="K": 
        #Get Kepler Limb darkening coefficients.
        #print(label)
        types={'1':[3],'2':[4, 5],'3':[6, 7, 8],'4':[9, 10, 11, 12]}
        if how in types:
            checkint = types[how]
            #print(checkint)
        else:
            print("no key...")

        arr = np.genfromtxt(os.path.join(NamastePymc3_path,"data","KeplerLDlaws.txt"),skip_header=2)
        FeHarr=np.unique(arr[:, 2])
        FeH=find_nearest_2D(FeH,FeHarr)

        outarr=np.zeros((len(FeH),len(checkint)))
        for met in np.unique(FeH):
            #Selecting FeH manually:
            arr2=arr[arr[:,2]==met]
            for n,i in enumerate(checkint):
                ix_to_take=(FeH==met)*(Ts<50000.)*(Ts>=2000.)
                u_interp=ct2d(np.column_stack((arr2[:,0],arr2[:,1])),arr2[:,i])
                outarr[ix_to_take,n]=u_interp(np.clip(Ts[ix_to_take],3500,50000),np.clip(logg[ix_to_take],0,5))
        return outarr


def PlotCorner(trace, ID, mission='TESS', varnames=["b", "ecc", "period", "r_pl","u_star","vrel"],
               savename=None, overwrite=False,savefileloc=None,returnfig=False,tracemask=None):
    #Plotting corner of the parameters to see correlations
    import corner
    import matplotlib.pyplot as plt
    print("varnames = ",varnames)
    
    if savename is None:
        savename=GetSavename(ID, mission, how='save', suffix='_corner.png', 
                             overwrite=overwrite, savefileloc=savefileloc)
    
    if tracemask is None:
        tracemask=np.tile(True,len(trace['Rs']))
    
    samples = pm.trace_to_dataframe(trace, varnames=varnames)
    samples=samples.loc[tracemask]

    plt.figure()
    fig = corner.corner(samples);
    fig.savefig(savename,dpi=250)
    
    if returnfig:
        return fig

def vals_to_latex(vals):
    #Function to turn -1,0, and +1 sigma values into round latex strings for a table
    try:
        roundval=int(np.min([-1*np.floor(np.log10(abs(vals[1]-vals[0])))+1,-1*np.floor(np.log10(abs(vals[2]-vals[1])))+1]))
        errs=[vals[2]-vals[1],vals[1]-vals[0]]
        if np.round(errs[0],roundval-1)==np.round(errs[1],roundval-1):
            #Errors effectively the same...
            if roundval<0:
                return " $ "+str(int(np.round(vals[1],roundval)))+" \pm "+str(int(np.round(np.average(errs),roundval)))+" $ "
            else:
                return " $ "+str(np.round(vals[1],roundval))+" \pm "+str(np.round(np.average(errs),roundval))+" $ "
        else:
            if roundval<0:
                return " $ "+str(int(np.round(vals[1],roundval)))+"^{+"+str(int(np.round(errs[0],roundval)))+"}_{-"+str(int(np.round(errs[1],roundval)))+"} $ "
            else:
                return " $ "+str(np.round(vals[1],roundval))+"^{+"+str(np.round(errs[0],roundval))+"}_{-"+str(np.round(errs[1],roundval))+"} $ "
    except:
        return " - "
    
def ToLatexTable(trace, ID, mission='TESS', varnames='all',order='columns',
               savename=None, overwrite=False, savefileloc=None, tracemask=None):
    #Plotting corner of the parameters to see correlations
    print("MakingLatexTable")
    if savename is None:
        savename=GetSavename(ID, mission, how='save', suffix='_table.txt',overwrite=False, savefileloc=savefileloc)
    if tracemask is None:
        tracemask=np.tile(True,len(trace['Rs']))
    if varnames is None or varnames is 'all':
        varnames=[var for var in trace.varnames if var[-2:]!='__' and var not in ['gp_pred','light_curves']]
    
    samples = pm.trace_to_dataframe(trace, varnames=varnames)
    samples = samples.loc[tracemask]
    facts={'r_pl':109.07637,'Ms':1.0,'rho':1.0,"t0":1.0,"period":1.0,"vrel":1.0,"tdur":24}
    units={'r_pl':"$ R_\\oplus $",'Ms':"$ M_\\odot $",'rho':"$ \\rho_\\odot $",
           "t0":"BJD-2458433","period":'d',"vrel":"$R_s/d$","tdur":"hours"}
    if order=="rows":
        #Table has header as a single row and data as a single row 
        rowstring=str("ID")
        valstring=str(ID)
        for row in samples.columns:
            fact=[fact for fact in list(facts.keys()) if fact in row]
            if fact is not []:
                rowstring+=' & '+str(row)+' ['+units[fact[0]]+']'
                valstring+=' & '+vals_to_latex(np.percentile(facts[fact[0]]*samples[row],[16,50,84]))
            else:
                rowstring+=' & '+str(row)
                valstring+=' & '+vals_to_latex(np.percentile(samples[row],[16,50,84]))
        outstring=rowstring+"\n"+valstring
    else:
        #Table has header as a single column and data as a single column 
        outstring="ID & "+str(ID)
        for row in samples.columns:
            fact=[fact for fact in list(facts.keys()) if fact in row]
            if len(fact)>0:
                outstring+="\n"+row+' ['+units[fact[0]]+']'+" & "+vals_to_latex(np.percentile(facts[fact[0]]*samples[row],[16,50,84]))
            else:
                outstring+="\n"+row+" & "+vals_to_latex(np.percentile(samples[row],[16,50,84]))
    with open(savename,'w') as file_to_write:
        file_to_write.write(outstring)
    #print("appending to file,",savename,"not yet supported")
    return outstring

def PlotLC(lc, trace, ID, mission='TESS', savename=None,overwrite=False, savefileloc=None, 
           returnfig=False, lcmask=None,tracemask=None):
    
    #The tracemask is a mask used to remove samples where the period is inconsistent with the presence of photometry:
    if tracemask is None:
        tracemask=np.tile(True,len(trace['Rs']))
    
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(14,6))
    
    if lcmask is None:
        assert len(lc['time'])==len(trace['gp_pred'][0,:])
        lcmask=np.tile(True,len(lc['time']))
    else:
        assert len(lc['time'][lcmask])==len(trace['gp_pred'][0,:])
    
    #Finding if there's a single enormous gap in the lightcurve:
    x_gap=np.max(np.diff(lc['time'][lcmask]))>10
    if x_gap:
        print(" GAP IN X OF ",np.argmax(np.diff(lc['time'])))
        gs = fig.add_gridspec(4,8,wspace=0.3,hspace=0.001)
        f_all_1=fig.add_subplot(gs[:3, :3])
        f_all_2=fig.add_subplot(gs[:3, 3:6])
        f_all_resid_1=fig.add_subplot(gs[3, :3])#, sharey=f_all_2)
        f_all_resid_2=fig.add_subplot(gs[3, 3:6])#, sharey=f_all_resid_1)
    else:
        gs = fig.add_gridspec(4,4,wspace=0.3,hspace=0.001)
        f_all=fig.add_subplot(gs[:3, :3])
        f_all_resid=fig.add_subplot(gs[3, :3])
    
    # Compute the GP prediction
    gp_mod = np.median(trace["gp_pred"][tracemask,:] + trace["mean"][tracemask, None], axis=0)
    
    if 'mult' in trace.varnames:
        pred = trace["light_curves"][tracemask,:,:]/trace["mult"]
    else:
        pred = trace["light_curves"][tracemask,:,:]
    #Need to check how many planets are here:
    pred = np.percentile(pred, [16, 50, 84], axis=0)
    
    gp_pred = np.percentile(pred, [16, 50, 84], axis=0)
    
    #Plotting model with GPs:
    min_trans=abs(np.min(np.sum(pred[1,:,:],axis=1)))
    if x_gap:
        gap_pos=np.average(lc['time'][np.argmax(np.diff(lc['time'])):(1+np.argmax(np.diff(lc['time'])))])
        before_gap_lc,before_gap_gp=(lc['time']<gap_pos)&lcmask,(lc['time'][lcmask]<gap_pos)
        after_gap_lc,after_gap_gp=(lc['time']>gap_pos)&lcmask,(lc['time'][lcmask]>gap_pos)
        
        print(np.sum(before_gap_lc),len(lc['time'][before_gap_lc]),np.sum(before_gap_gp),len(gp_mod[before_gap_gp]))
        
        f_all_1.plot(lc['time'][before_gap_lc], lc['flux'][before_gap_lc]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_2.plot(lc['time'][after_gap_lc], lc['flux'][after_gap_lc]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

        f_all_1.plot(lc['time'][before_gap_lc], gp_mod[before_gap_gp]+2.5*min_trans, color="C3", label="GP fit")
        f_all_2.plot(lc['time'][after_gap_lc], gp_mod[after_gap_gp]+2.5*min_trans, color="C3", label="GP fit")

        f_all_1.plot(lc['time'][before_gap_lc], lc['flux'][before_gap_lc] - gp_mod[before_gap_gp], ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_2.plot(lc['time'][after_gap_lc], lc['flux'][after_gap_lc] - gp_mod[after_gap_gp], ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

        #Plotting residuals at the bottom:
        f_all_resid_1.plot(lc['time'][before_gap_lc], 
                         lc['flux'][before_gap_lc] - gp_mod[before_gap_gp] - np.sum(pred[1,before_gap_gp,:],axis=1), 
                         ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_resid_1.set_xlabel('Time (BJD-245700)')
        f_all_resid_1.set_xlim(lc['time'][before_gap_lc][0][0],lc['time'][before_gap_lc][-1])

        f_all_resid_2.plot(lc['time'][after_gap_lc], 
                         lc['flux'][after_gap_lc] - gp_mod[after_gap_gp] - np.sum(pred[1,after_gap_gp,:],axis=1),
                         ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_resid_2.set_xlabel('Time (BJD-245700)')
        f_all_resid_2.set_xlim(lc['time'][after_gap_lc][0],lc['time'][after_gap_lc][-1])
        #print(len(lc[:,0]),len(lc[lcmask,0]),len(gp_mod))
        f_all_resid_1.set_ylim(2*np.percentile(lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
        f_all_resid_2.set_ylim(2*np.percentile(lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
        for n_pl in range(len(pred[1,0,:])):
            f_all_1.plot(lc['time'][before_gap_lc], pred[1,before_gap_gp,n_pl], color="C1", label="model")
            art = f_all_1.fill_between(lc['time'][after_gap_lc], pred[0,before_gap_gp,n_pl], pred[2,before_gap_gp,n_pl], color="C1", alpha=0.5, zorder=1000)
            f_all_1.set_xlim(lc['time'][before_gap_lc],lc['time'][before_gap_lc][-1])

            f_all_2.plot(lc['time'][after_gap_lc], pred[1,after_gap_gp,n_pl], color="C1", label="model")
            art = f_all_2.fill_between(lc['time'][after_gap_lc], pred[0,after_gap_gp,n_pl], pred[2,after_gap_gp,n_pl], color="C1", alpha=0.5, zorder=1000)
            f_all_2.set_xlim(lc['time'][after_gap_lc][0],lc['time'][after_gap_lc][-1])
            
            f_all_1.set_ylim(np.percentile(lc['flux'][lcmask]-gp_mod,0.25),np.percentile(lc['flux'][lcmask]+2.5*min_trans,99))
            f_all_2.set_ylim(np.percentile(lc['flux'][lcmask]-gp_mod,0.25),np.percentile(lc['flux'][lcmask]+2.5*min_trans,99))
        
        f_all_1.get_xaxis().set_ticks([])
        f_all_2.get_yaxis().set_ticks([])
        f_all_2.get_xaxis().set_ticks([])
        
        f_all_resid_2.get_yaxis().set_ticks([])
        f_all_1.spines['right'].set_visible(False)
        f_all_resid_1.spines['right'].set_visible(False)
        f_all_2.spines['left'].set_visible(False)
        f_all_resid_2.spines['left'].set_visible(False)
        #
        #spines['right'].set_visible(False)
        #
        #f_all_2.set_yticks([])
        #f_all_2.set_yticklabels([])
        #f_all_1.tick_params(labelright='off')
        #f_all_2.yaxis.tick_right()
        
        f_zoom=fig.add_subplot(gs[:3, 6:])
        f_zoom_resid=fig.add_subplot(gs[3, 6:])

    else:
        #No gap in x, plotting normally:
        print(len(lc['time']),len(lc['flux']),len(lc['time'][lcmask]),len(lc['flux'][lcmask]),len(gp_mod),len(np.sum(pred[1,:,:],axis=1)))
        f_all.plot(lc['time'][lcmask], lc['flux'][lcmask]+2.5*min_trans, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all.plot(lc['time'][lcmask], gp_mod+2.5*min_trans, color="C3", label="GP fit")

        # Plot the data
        f_all.plot(lc['time'][lcmask], lc['flux'][lcmask] - gp_mod, ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)

        #Plotting residuals at the bottom:
        f_all_resid.plot(lc['time'][lcmask], lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1), ".k", label="data", zorder=-1000,alpha=0.5,markersize=0.75)
        f_all_resid.set_xlabel('Time (BJD-245700)')
        f_all_resid.set_ylim(2*np.percentile(lc['flux'][lcmask] - gp_mod - np.sum(pred[1,:,:],axis=1),[0.5,99.5]))
        f_all_resid.set_xlim(lc['time'][lcmask][0],lc['time'][lcmask][-1])
        
        for n_pl in range(len(pred[1,0,:])):
            f_all.plot(lc['time'][lcmask], pred[1,:,n_pl], color="C1", label="model")
            art = f_all.fill_between(lc['time'][lcmask], pred[0,:,n_pl], pred[2,:,n_pl], color="C1", alpha=0.5, zorder=1000)
            f_all.set_xlim(lc['time'][lcmask][0],lc['time'][lcmask][-1])
        
        f_all.set_xticks([])
        f_zoom=fig.add_subplot(gs[:3, 3])
        f_zoom_resid=fig.add_subplot(gs[3, 3])
    
    min_trans=0
    for n_pl in range(len(pred[1,0,:])):
        # Get the posterior median orbital parameters
        p = np.median(trace["period"][tracemask,n_pl])
        t0 = np.median(trace["t0"][tracemask,n_pl])
        tdur = np.nanmedian(trace['tdur'][tracemask,n_pl])
        min_trans+=abs(1.25*np.min(pred[1,:,n_pl]))
        
        #tdurs+=[(2*np.sqrt(1-np.nanmedian(trace['b'][tracemask,n_pl])**2))/np.nanmedian(trace['vrel'][tracemask,n_pl])]
        #print(min_trans,tdurs[n_pl],2*np.sqrt(1-np.nanmedian(trace['b'][tracemask,n_pl])**2),np.nanmedian(trace['vrel'][tracemask, n_pl]))
        
        phase=(lc['time'][lcmask]-t0+p*0.5)%p-p*0.5
        zoom_ind=abs(phase)<tdur
        
        f_zoom.plot(phase[zoom_ind], min_trans+lc['flux'][lcmask][zoom_ind] - gp_mod[zoom_ind], ".k", label="data", zorder=-1000,alpha=0.5)
        f_zoom.plot(phase[zoom_ind], min_trans+pred[1,zoom_ind,n_pl], color="C1", label="model")
        art = f_zoom.fill_between(phase[zoom_ind], min_trans+pred[0,zoom_ind,n_pl], min_trans+pred[2,zoom_ind,n_pl],
                                  color="C1", alpha=0.5,zorder=1000)
        
        

    f_zoom_resid.plot(lc['time'][lcmask][zoom_ind]-t0, lc['flux'][lcmask][zoom_ind] - gp_mod[zoom_ind] - np.sum(pred[1,zoom_ind,:],axis=1), ".k", label="data", zorder=-1000,alpha=0.5)
    f_zoom_resid.set_xlabel('Time - t_c')

    f_zoom_resid.set_xlim(-1*tdur,tdur)

    f_zoom.set_xticks([])
    f_zoom.set_xlim(-1*tdur,tdur)
    
                                    
    if savename is None:
        savename=GetSavename(ID, mission, how='save', suffix='_TransitFit.png', 
                             overwrite=overwrite, savefileloc=savefileloc)
    
    plt.savefig(savename,dpi=250)
    
    if returnfig:
        return fig
