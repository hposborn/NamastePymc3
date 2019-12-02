import os 
import copy
import glob
import h5py

import numpy as np
from matplotlib import pylab as plt
import pandas as pd
#import ebf
import astropy.units as units
from astropy.coordinates import SkyCoord
from dustmaps.bayestar import BayestarWebQuery
import mwdust
import pdb

from .direct import classify as classify_direct
from .grid import classify as classify_grid
from isoclassify import DATADIR

CONSTRAINTS = [
    'teff','logg','feh','lum','gmag','rmag','imag','zmag','jmag','hmag','kmag',
    'gamag','bpmag','rpmag','parallax', 'bmag','vmag', 'btmag','vtmag','numax','dnu'
]

COORDS = ['ra','dec']

def run(**kw):
    if kw['method']=='direct':
        pipe = PipelineDirect(**kw)    
    elif kw['method']=='grid':
        pipe = PipelineGrid(**kw) 
    else:
        assert False, "method {} not supported ".format(kw['method'])

    pipe.run()
    if pipe.plotmode=='show':
        plt.ion()
        plt.show()
        input('[press return to continue]:')
    elif pipe.plotmode.count('save')==1:
        pipe.savefig()

    pipe.to_csv()

def query_dustmodel_coords(ra,dec):
    reddenMap = BayestarWebQuery(version='bayestar2017')
    sightLines = SkyCoord(ra*units.deg,dec*units.deg,frame='icrs')
    reddenContainer = reddenMap(sightLines,mode='best')
    del reddenMap # To clear reddenMap from memory
    distanceSamples = np.array([0.06309573,0.07943284,0.1,0.12589255,0.15848933,0.19952627,0.25118864,0.31622776,0.3981072,0.50118726,0.6309574,0.7943282 ,1.,1.2589258,1.5848933,1.9952621,2.511887,3.1622777,3.981073,5.011873,6.3095727,7.943284,10.,12.589258,15.848933,19.952621,25.11887,31.622776,39.81073,50.11873,63.095726])*1000. # In pc, from bayestar2017 map distance samples
    
    dustModelDF = pd.DataFrame({'ra': [ra], 'dec': [dec]})

    for index in range(len(reddenContainer)):
        dustModelDF['av_'+str(round(distanceSamples[index],6))] = reddenContainer[index]
        
    return dustModelDF
    
def query_dustmodel_coords_allsky(ra,dec):
    reddenMap = mwdust.Combined15()
    sightLines = SkyCoord(ra*units.deg,dec*units.deg,frame='galactic')
    distanceSamples = np.array([0.06309573,0.07943284,0.1,0.12589255,0.15848933,0.19952627,0.25118864,0.31622776,0.3981072,0.50118726,0.6309574,0.7943282 ,1.,1.2589258,1.5848933,1.9952621,2.511887,3.1622777,3.981073,5.011873,6.3095727,7.943284,10.,12.589258,15.848933,19.952621,25.11887,31.622776,39.81073,50.11873,63.095726])*1000. # In pc, from bayestar2017 map distance samples
    reddenContainer=reddenMap(sightLines.l.value,sightLines.b.value,distanceSamples/1000.)
    del reddenMap # To clear reddenMap from memory
    
    dustModelDF = pd.DataFrame({'ra': [ra], 'dec': [dec]})
    
    for index in range(len(reddenContainer)):
        dustModelDF['av_'+str(round(distanceSamples[index],6))] = reddenContainer[index]

    return dustModelDF

class Pipeline(object):
    def __init__(self, **kw):
        assert 'csv' in kw, "must pass csv as keyword"
        assert 'outdir' in kw, "must pass outdir as keyword"
        #assert kw.has_key('csv'), "must pass csv as keyword"
        #assert kw.has_key('outdir'), "must pass outdir as keyword"

        self.plotmode = kw['plot']

        # create plot (both interactive and save modes)
        if self.plotmode=='none':
            self.plot = 0 
        else:
            self.plot = 1

        self.id_starname = kw['id_starname']
        self.outdir = kw['outdir']

        # Read in inputs
        df = pd.read_csv(kw['csv'])
        
        if (len(df.id_starname.drop_duplicates())!=len(df)):
            print('dropping duplicates')
            df=df.drop_duplicates(subset='id_starname')
            
        df.index = df.id_starname
        star = df.ix[self.id_starname]

        self.dust = star.dust

        const = {}
        for key in CONSTRAINTS:
            if key in star:
                const[key] = star[key]
                const[key+'_err'] = star[key+'_err']
            else:
                const[key] = -99
                const[key+'_err'] = 0

        for key in COORDS:
            if key in star:
                const[key] = star[key]
            else:
                const[key] = -99
        
        self.const = const
        self.const['ra'] = star['ra']
        self.const['dec'] = star['dec']
        self.const['band'] = star['band']

        self.pdffn = os.path.join(self.outdir,'output.pdf')
        self.csvfn = os.path.join(self.outdir,'output.csv')

    def addspec(self,x):
        keys = 'teff logg feh'.split()
        val = [self.const[key] for key in keys]
        err = [self.const[key+'_err'] for key in keys]
        x.addspec(val,err)

    def addlum(self,x):
        keys = 'lum'.split()
        val = [self.const[key] for key in keys]
        err = [self.const[key+'_err'] for key in keys]
        x.addlum(val,err)

    def addjhk(self,x):
        keys = 'jmag hmag kmag'.split()
        val = [self.const[key] for key in keys]
        err = [self.const[key+'_err'] for key in keys]
        x.addjhk(val,err)

    def addgriz(self,x):
        keys = 'gmag rmag imag zmag'.split()
        val = [self.const[key] for key in keys]
        err = [self.const[key+'_err'] for key in keys]
        x.addgriz(val,err)
        
    def addgaia(self,x):
        keys = 'gamag bpmag rpmag'.split()
        val = [self.const[key] for key in keys]
        err = [self.const[key+'_err'] for key in keys]
        x.addgaia(val,err)
        
    def addbvt(self,x):
        keys = 'btmag vtmag'.split()
        val = [self.const[key] for key in keys]
        err = [self.const[key+'_err'] for key in keys]
        x.addbvt(val,err)
        
    def addseismo(self,x):
        keys = 'numax dnu'.split()
        val = [self.const[key] for key in keys]
        err = [self.const[key+'_err'] for key in keys]
        x.addseismo(val,err)
    
    def addbv(self,x):
        keys = 'bmag vmag'.split()
        val = [self.const[key] for key in keys]
        err = [self.const[key+'_err'] for key in keys]
        x.addbv(val,err)

    def addplx(self,x):
        x.addplx(self.const['parallax'], self.const['parallax_err'])
    
    def addcoords(self,x):
        x.addcoords(self.const['ra'],self.const['dec'])
    
    def addmag(self,x):
        x.addmag(
            [self.const[self.const['band']]],
            [self.const[self.const['band']+'_err']]
        )

    def addcoords(self,x):
        x.addcoords(self.const['ra'], self.const['dec'])

    def print_constraints(self):
        print("id_starname {}".format(self.id_starname))
        print("dust:", self.dust)
        for key in CONSTRAINTS:
            print(key, self.const[key], self.const[key+'_err'])

        for key in COORDS:
            print(key, self.const[key])
            
    def savefig(self):
        labels = plt.get_figlabels()
        _, ext = self.plotmode.split('-')
        for label in plt.get_figlabels():
            fn = os.path.join(self.outdir,'{}.{}'.format(label,ext))
            fig = plt.figure(label)
            fig.set_tight_layout(True)
            plt.savefig(fn)
            print("created {}".format(fn))

    def to_csv(self):
        out = {}
        out['id_starname'] = self.id_starname
        out = dict(out, **self.const)
        for outcol,incol in self.outputcols.items():
            out[outcol] = getattr(self.paras, incol)
            out[outcol+'_err1'] = getattr(self.paras, incol+'ep')
            out[outcol+'_err2'] = -getattr(self.paras, incol+'em')

        out = pd.Series(out)
        
        # Re-ordering series
        block1 = []
        block2 = []
        block3 = []
        for col in list(out.index):
            if col.count('id_starname')==1:
                block1.append(col)
                continue
            if (col.count('iso_')==1) :
                block3.append(col)
                continue

            block2.append(col)

        out = out[block1 + block2 + block3]
        out.to_csv(self.csvfn)
        print("created {}".format(self.csvfn))

class PipelineDirect(Pipeline):
    outputcols = {
        'dir_dis': 'dis',
        'dir_avs': 'avs',
        'dir_rad': 'rad',
        'dir_lum': 'lum',
        'dir_teff': 'teff',
        'dir_mabs': 'mabs',
        'dir_mass': 'mass',
        'dir_rho': 'rho'
    }
    
    def run(self):
        self.print_constraints()

        #pdb.set_trace()

        fn = os.path.join(DATADIR,'bcgrid.h5')
        bcmodel = h5py.File(fn,'r', driver='core', backing_store=False)
        
        if self.dust == 'allsky':
            dustmodel = query_dustmodel_coords_allsky(
                self.const['ra'],self.const['dec']
            )
            ext = extinction('cardelli')

        if self.dust == 'green18':
            dustmodel = query_dustmodel_coords(
                self.const['ra'],self.const['dec']
            )
            ext = extinction('schlafly16')

        if self.dust == 'none':
            dustmodel = 0
            ext = extinction('cardelli')

        x = classify_direct.obsdata()
        self.addspec(x)
        #self.addlum(x)
        self.addjhk(x)
        self.addbv(x)
        self.addbvt(x)
        self.addgriz(x)
        self.addgaia(x)
        self.addplx(x)
        self.addcoords(x)
        self.addmag(x)
        self.paras = classify_direct.stparas(
            input=x, bcmodel=bcmodel, dustmodel=dustmodel, 
            band=self.const['band'], ext=ext, plot=1
        )

class PipelineGrid(Pipeline):
    outputcols = {
        'iso_age':'age',
        'iso_avs':'avs',
        'iso_dis':'dis',
        'iso_feh':'feh',
        'iso_mass':'mass',
        'iso_rad':'rad',
        'iso_lum':'lum',
        'iso_logg':'logg',
        'iso_rho': 'rho',
        'iso_teff':'teff',
    }
    def run(self):
        self.print_constraints()

#        model = ebf.read(os.path.join(DATADIR,'mesa.ebf'))
        fn = os.path.join(DATADIR,'mesa.h5')
        file = h5py.File(fn,'r+', driver='core', backing_store=False)
        model = {'age':np.array(file['age']),\
        'mass':np.array(file['mass']),\
        'feh':np.array(file['feh']),\
        'teff':np.array(file['teff']),\
        'logg':np.array(file['logg']),\
        'rad':np.array(file['rad']),\
        'lum':np.array(file['rad']),\
        'rho':np.array(file['rho']),\
        'dage':np.array(file['dage']),\
        'dmass':np.array(file['dmass']),\
        'dfeh':np.array(file['dfeh']),\
        'eep':np.array(file['eep']),\
        'bmag':np.array(file['bmag']),\
        'vmag':np.array(file['vmag']),\
        'btmag':np.array(file['btmag']),\
        'vtmag':np.array(file['vtmag']),\
        'gmag':np.array(file['gmag']),\
        'rmag':np.array(file['rmag']),\
        'imag':np.array(file['imag']),\
        'zmag':np.array(file['zmag']),\
        'jmag':np.array(file['jmag']),\
        'hmag':np.array(file['hmag']),\
        'kmag':np.array(file['kmag']),\
        'd51mag':np.array(file['d51mag']),\
        'gamag':np.array(file['gamag']),\
        'fdnu':np.array(file['fdnu']),\
        'avs':np.zeros(len(np.array(file['gamag']))),\
        'dis':np.zeros(len(np.array(file['gamag'])))}
        
        #ebf.read(os.path.join(DATADIR,'mesa.ebf'))
        # prelims to manipulate some model variables (to be automated soon ...)
        #pdb.set_trace()
        model['rho'] = np.log10(model['rho'])
        model['lum'] = model['rad']**2*(model['teff']/5777.)**4
        # next line turns off Dnu scaling relation corrections
        model['fdnu'][:]=1.
        model['avs']=np.zeros(len(model['teff']))
        model['dis']=np.zeros(len(model['teff']))

        if self.dust == 'allsky':
            dustmodel = query_dustmodel_coords_allsky(self.const['ra'],self.const['dec'])
            ext = extinction('cardelli')
        if self.dust == 'green18':
            dustmodel = query_dustmodel_coords(self.const['ra'],self.const['dec'])
            ext = extinction('schlafly16')
        if self.dust == 'none':
            dustmodel = 0
            ext = extinction('cardelli')
            
        # Instantiate model
        x = classify_grid.obsdata()
        self.addcoords(x)
        self.addspec(x)
        self.addlum(x)
        self.addjhk(x)
        self.addgriz(x)
        self.addgaia(x)
        self.addbv(x)
        self.addbvt(x)
        self.addseismo(x)
        self.addplx(x)
        self.paras = classify_grid.classify(
            input=x, model=model, dustmodel=dustmodel,ext=ext, 
            plot=self.plot, useav=0
        )

def _csv_reader(f):
    row = pd.read_csv(f,header=None,squeeze=True, index_col=0)
    return row

def scrape_csv(path):
    """
    Read in isochrones csvfiles 
    Args:
        outdir (str): where to look for isochrones.csv files
    """
    fL = glob.glob(path)
    df = []

    for i, f in enumerate(fL):
        if i%100==0:
            print(i)
        try:
            df.append(_csv_reader(f))
        except ValueError:
            print("{} failed".format(f))


    df = pd.DataFrame(df)
    df = df.reset_index()
    return df
 
# R_lambda values to convert E(B-V) given by dustmaps to extinction in
# a given passband.  The two main caveats with this are: - strictly
# speaking only cardelli is consistent with the BC tables used in the
# MIST grid, but using wrong R_lambda's for the newer Green et
# al. dustmaps is (probably) worse.  - some values were interpolated
# to passbands that aren't included in the Schlafly/Green tables.

def extinction(law):
    if (law == 'cardelli'):
        out = {
            "ab":4.1708789, 
            "av":3.1071930, 
            "abt":4.3358221, 
            "avt":3.2867038, 
            "ag":3.8281101, 
            "ar":2.7386468, 
            "ai":2.1109662, 
            "az":1.4975613, 
            "aj":0.89326176, 
            "ah":0.56273418, 
            "ak":0.35666104, 
            "aga":2.4623915
        }
        
    if (law == 'schlafly11'):
        out = {
            "ab":3.626, 
            "av":2.742, 
            "abt":4.5309214, 
            "avt":3.1026801, 
            "ag":3.303, 
            "ar":2.285, 
            "ai":1.698, 
            "az":1.263, 
            "aj":0.77510388, 
            "ah":0.50818384, 
            "ak":0.33957048, 
            "aga":1.9139634
        }

    if (law == 'schlafly16'):
        # see http://argonaut.skymaps.info/usage under "Gray Component". this is a lower limit.
        grayoffset=0.063
        out = {
            "ab":3.6060565+grayoffset, 
            "av":2.9197679+grayoffset, 
            "abt":3.7204173+grayoffset, 
            "avt":3.0353634+grayoffset, 
            "ag":3.384+grayoffset, 
            "ar":2.483+grayoffset, 
            "ai":1.838+grayoffset, 
            "az":1.414+grayoffset, 
            "aj":0.650+grayoffset, 
            "ah":0.327+grayoffset, 
            "ak":0.161+grayoffset, 
            "aga":2.2203186+grayoffset
        }
    return out

