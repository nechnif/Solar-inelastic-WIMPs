import os
import pickle, json
import numpy as np
from numpy import nan
import pandas as pd
from scipy.interpolate import RectBivariateSpline

from .dataio import *
from .solar import *
from .gaussian_kde import *
from .Selection import *


class Model(object):
    ''' Tools for scotogenic minimal DM scenarios.

        A class to initialize and work with scenarios of the scotogenic
        minimal dark matter model, as part of the IceCube solar
        inelastic WIMPs analysis.

        Parameters
        ----------
        allsets: dict
            Names and locations of all datasets used in the analysis.
        batch: str
            The batch that contains the scenario.
        configs: list
            Locations of the event selection config files.
        loc: str
            Location of the batches/scenarios.
        m: int
            Speed for KDE fine grid evaluation.
        name: str or float
            Name of the scenario (mass of the DM candidate).
        outdir: str
            Alternative output directory. For testing purposes.
        set: str
            Dataset (nominal or any systematic set).

        Attributes
        ----------
        batch: str
            The batch that contains the scenario.
        configs: list
            Locations of the event selection config files.
        flux_e: ndarray
            Neutrino energy.
        flux_numu: ndarray
            Combined elastic and inelastic muon neutrino flux.
        flux_numubar: ndarray
            Combined elastic and inelastic muon antineutrino flux.
        input_parameters: dict
            Scenario parameters.
        INT: Selection
            Hight energy selection.
        loc: str
            Location of the batches/scenarios.
        m: int
            Speed for KDE fine grid evaluation.
        mdm: float
            Mass of the DM candidate.
        modelpath: str
            Path to scenario.
        name: str or float
            Name of the scenario (mass of the DM candidate).
        observables: dict
            Scenario observables.
        OSC: Selection
            Low energy selection.
        outdir: str
            Alternative output directory. For testing purposes.
        set: str
            Dataset (nominal or any systematic set).
        setpath: str
            Path where dataset specifict scenario results are stored.
        sets: dict
            Names and locations of all datasets used in the analysis.

        Methods
        -------
        LoadPDFs
        CreateBPDF
        CreateSPDF
        FineGridEvaluation
        IntpEvalfine
        LoadResults
        SaveResults
        UpdateResults

        Notes
        -----
        Raffaela Busse, August 2022
        raffaela.busse@uni-muenster.de

    '''

    def __init__(self, set, batch, name, allsets, loc, configs, m=8, outdir='default'):

        self.set       = set
        self.sets      = allsets
        self.loc       = loc
        self.batch     = batch
        self.name      = name
        self.modelpath = loc+'batch_'+batch+'/'+name+'/'
        self.setpath   = self.modelpath+allsets[set][0]
        self.mdm       = float(self.name)
        self.configs   = configs
        self.m         = m

        ## Set the outdir if you want the results in an alternative directory:
        self.outdir = outdir

        with open(self.modelpath+self.name+'.txt', 'r') as ls:
            point = eval(ls.readline())

        ## Make input parameters accessible more easily:
        input_parameters = {}
        pars_ = {
            'lambda1Input' : 'lam1', 'lambda2Input' : 'lam2',
            'lambda3Input' : 'lam3', 'lambda4Input' : 'lam4',
            'lambda5Input' : 'lam5',
            'mEt2Input' : 'met2',
            # 'Real(YN(1,1),dp)' : 'YNr(1,1)', 'Real(YN(1,2),dp)' : 'YNr(1,2)', 'Real(YN(1,3),dp)' : 'YNr(1,3)',
            # 'Real(YN(2,1),dp)' : 'YNr(2,1)', 'Real(YN(2,2),dp)' : 'YNr(2,2)', 'Real(YN(2,3),dp)' : 'YNr(2,3)',
            # 'Real(YN(3,1),dp)' : 'YNr(3,1)', 'Real(YN(3,2),dp)' : 'YNr(3,2)', 'Real(YN(3,3),dp)' : 'YNr(3,3)',
            # 'Aimag(YN(1,1))'   : 'YNi(1,1)', 'Aimag(YN(1,2))'   : 'YNi(1,2)', 'Aimag(YN(1,3))'   : 'YNi(1,3)',
            # 'Aimag(YN(2,1))'   : 'YNi(2,1)', 'Aimag(YN(2,2))'   : 'YNi(2,2)', 'Aimag(YN(2,3))'   : 'YNi(2,3)',
            # 'Aimag(YN(3,1))'   : 'YNi(3,1)', 'Aimag(YN(3,2))'   : 'YNi(3,2)', 'Aimag(YN(3,3))'   : 'YNi(3,3)',
        }
        for key, value in pars_.items():
            input_parameters[value] = point[key]
        self.input_parameters = input_parameters

        ## Same for some important observables:
        equilibrium  = np.min([point['equi (elastic)']+point['equi (inelastic)'], 1])
        ## Determine dominant annihilation channel and ratio:
        sigv_channel = point['sigv channels'][0][1].split('-> ')[1].replace(' (null)', '')
        sigv_ratio   = point['sigv channels'][0][0]
        obs = {
            'sigpSI_el'    : point['sigpSI (elastic)'],
            'sigpSI_inel'  : point['sigpSI (inelastic)'],
            'sigv_channel' : sigv_channel,
            'sigv_ratio'   : sigv_ratio,
            'cap_el'       : point['capture rate (elastic)'],
            'cap_inel'     : point['capture rate (inelastic)'],
            'sigma v'      : point['sigma v'],
            'drate'        : point['depletion rate'],
            'Veff'         : point['Veff'],
            'equilibrium'  : equilibrium,
        }
        self.observables = obs

        ## The flux is saved in [GeV km^2 y]^-1, and convereted here
        ## to [GeV cm^2 s]^-1:
        km_yr = 1e10*3.1536*1e7    # km^2 in cm^2 times year in seconds
        e, nu_el, nubar_el, nu_inel, nubar_inel = np.loadtxt(self.modelpath+'neutrinoflux.txt', unpack=True)
        self.flux_e       = e
        self.flux_numu    = (nu_el+nu_inel)/km_yr
        self.flux_numubar = (nubar_el+nubar_inel)/km_yr

        ## Assign selections:
        INT = Selection('INT', set, configs, m)
        OSC = Selection('OSC', set, configs, m)
        self.INT = INT
        self.OSC = OSC

    def LoadPDFs(self):
        ### Load SPDFs and BPDFs.

        for selection in ['INT', 'OSC']:
            SEL = getattr(self, selection)

            if self.outdir != 'default':
                dir_B = self.outdir
                dir_S = self.outdir
            else:
                dir_B = os.readlink(self.setpath+'PDFs/'+selection+'_BPDF')
                dir_S = self.setpath+'PDFs/'

            ## Signal PDFs:
            Seval, _, _ = np.load(
                dir_S+selection+'_SPDF_KDE_evalfine.npy',
                allow_pickle=True
            )
            if selection+'_SPDF_KDE_evalfine-intp.pkl' not in os.listdir(dir_S):
                self.IntpEvalfine(selection, 'S')
            Sintp = pickle.load(
                open(dir_S+selection+'_SPDF_KDE_evalfine-intp.pkl', 'rb')
            )
            setattr(self, selection+'_SPDF_evalfine', Seval)
            setattr(self, selection+'_SPDF_evalintp', Sintp)

            ## Background PDFs:
            Beval, _, _ = np.load(
                dir_B+SEL.files['BPDF_KDE_evalfine'],
                allow_pickle=True
            )
            Bintp = pickle.load(
                open(dir_B+SEL.files['BPDF_KDE_evalintp'], 'rb')
            )
            setattr(self, selection+'_BPDF_evalfine',  Beval)
            setattr(self, selection+'_BPDF_evalintp',  Bintp)

    def LoadFarsampleFactors(self):

        if self.set != 'farsample':
            print('ERROR: Current set is "{}", not "farsample".'.format(self.set))
            return -1

        if self.outdir != 'default':
            ksfile = self.outdir+'farsample_PDFfactors.txt'
        else:
            ksfile = self.setpath+'farsample_PDFfactors.txt'

        with open(ksfile, 'r') as ksf:
            ks = eval(ksf.read())

        self.farfactors = ks

    def CreatePDF(self, selection, SB, eval=True):
        ### This function creates the PDFs. First as 2D histograms for
        ### comparison purposes, then as KDEs for the actual PDFs. The
        ### PDFs are saved as fine-grid evaluations of the KDEs, which
        ### can be used for drawing random test samples.
        ### selection = 'INT' or 'OSC'.
        ### SB = 'S' or 'B' (for signal or background).

        if self.outdir != 'default':
            dir_B = self.outdir
            dir_S = self.outdir
        else:
            dir_B = os.readlink(self.setpath+'PDFs/'+selection+'_BPDF')
            dir_S = self.setpath+'PDFs/'

        SEL = getattr(self, selection)
        print('Calculating {} {}PDF for set {}, scenario {} ...'.format(SEL.name, SB, self.set, self.name))

        if SB == 'B':
            ## Load sample:
            if self.set == 'farsample':
                sample = LoadSample(SEL.files['farsample'])
            else:
                sample = LoadSample(SEL.files['background_events'])

            ## Prepare histogram:
            if 'weight' not in sample.columns:
                sample['weight'] = np.ones(len(sample))
            W = sample['weight'].values
            bounds = SEL.bounds_b

        elif SB == 'S':
            sample = LoadSample(SEL.files['signal_events'].replace('SSSS', SEL.sets[self.set][0]))

            ## Cut sample at maximum WIMP energy:
            e_wimp = self.flux_e
            sample = sample[sample['trueE'] <= np.max(e_wimp)]

            ## Weight sample with WIMP flux:
            sample.loc[sample['PDG']== 14, 'wimp'] = sample['weight']/SunSolidAngle(sample['dist'])*WIMPweight(self, sample['trueE'], 'nu')
            sample.loc[sample['PDG']==-14, 'wimp'] = sample['weight']/SunSolidAngle(sample['dist'])*WIMPweight(self, sample['trueE'], 'nubar')

            ## Prepare histogram:
            W = sample['wimp'].values
            bounds = SEL.bounds_s

        ## Create histogram:
        X = sample['sun_psi'].values
        Y = sample['logE'].values
        xbins,  ybins  = SEL.psibins,  SEL.ebins
        hist = np.array(np.histogram2d(X, Y, bins=[xbins, ybins], weights=W, density=True), dtype=object)

        ## Create KDE with own class, from values:
        KDE = GaussianKDE(np.vstack([X, Y]), weights=W, bounds=bounds)
        KDE.set_bandwidth(*SEL.bandwidth)
        print('bandwidth:\t', SEL.bandwidth)

        ## Saving the histogram (KDE object is not saved anymore because
        ## takes a lot of space and is quick to re-create in case it is
        ## needed again):
        if SB == 'B':
            np.save(dir_B+SEL.files['BPDF_histogram'], hist)
            # pickle.dump(KDE, open(dir_B+SEL.files['BPDF_KDE'], 'wb'))
        elif SB == 'S':
            np.save(dir_S+SEL.name+'_SPDF_histogram.npy', hist)
            # pickle.dump(KDE, open(dir_S+SEL.name+'_SPDF_KDE.pkl', 'wb'))

        if eval == False:
            return KDE

        ##--- Perform finegrid evaluation: -----------------------------
        name = SB+'PDF_KDE'

        x, y      = SEL.psifine, SEL.efine
        xx, yy    = np.meshgrid(x, y)
        positions = np.vstack([xx.ravel(), yy.ravel()])

        eval = KDE.evaluate(positions, **getattr(SEL, 'kwargs_'+name))
        ## Dividing by the sum is necessary to make sure SPDF and
        ## BPDF both sum to exactly the same value (after this, 1).
        ## For the histograms this is given naturally, but since the
        ## normalization of the KDE happens inside the kernels and
        ## not for the evaluated grid (which is a simplification),
        ## it can happen that sum(SPDF) != sum(BPDF) and then shit
        ## hits all kinds of fans.
        eval = eval/eval.sum()
        # print(KDE.integrate_box(self.bounds, delta=200))

        ## The transposing happens so that the KDE has the same
        ## dimensions as the 2D histogram PDF:
        eval = np.reshape(eval, xx.shape).T

        ## Save:
        if SB=='B':
            pickle.dump((eval, x, y),
                open(dir_B+SEL.files['BPDF_KDE_evalfine'], 'wb')
            )
        else:
            pickle.dump((eval, x, y),
                open(dir_S+SEL.name+'_'+name+'_evalfine.npy', 'wb')
            )

        ## Interpolation:
        self.IntpEvalfine(selection, SB)
        print('{} {} done.'.format(SEL.id, name))

        return 0

    def FarsampleFactors(self):

        if self.set != 'farsample':
            print('ERROR: Current set is "{}", not "farsample".'.format(self.set))
            return -1

        self.LoadPDFs()
        ks = {}
        for selection in ['INT', 'OSC']:

            if self.outdir != 'default':
                dir_ = self.outdir
            else:
                dir_ = self.setpath

            # SEL  = getattr(self, selection)
            # BKDE = pickle.load(open(dir_B+SEL.files['BPDF_KDE'], 'rb'))
            # SKDE = self.CreatePDF(selection, SB='S', eval=False)
            #
            # box = [[SEL.psicut_farsample, SEL.ebins[0]], [SEL.psibins[-1], SEL.ebins[-1]]]
            # # print(box)
            # Bintgrl = BKDE.integrate_box(box=box, delta=delta)
            # Sintgrl = SKDE.integrate_box(box=box, delta=delta)
            # ks[selection+'-B'] = 1/Bintgrl
            # ks[selection+'-S'] = 1/Sintgrl
            # ks[selection+'-psicut'] = SEL.psicut_farsample

            SEL   = getattr(self, selection)
            Beval = getattr(self, selection+'_BPDF_evalfine')
            Seval = getattr(self, selection+'_SPDF_evalfine')

            indx = np.argmin(np.abs(SEL.psifine-SEL.psicut_farsample))

            Beval, Seval = Beval.T, Seval.T
            Beval = np.where(SEL.psifine<SEL.psifine[indx], 0, Beval)
            Seval = np.where(SEL.psifine<SEL.psifine[indx], 0, Seval)
            Beval, Seval = Beval.T, Seval.T

            ks[selection+'-B'] = 1/np.sum(Beval)
            ks[selection+'-S'] = 1/np.sum(Seval)
            ks[selection+'-psicut'] = SEL.psicut_farsample

        with open(dir_+'farsample_PDFfactors.txt', 'w') as ksf:
            ksf.write(str(ks)+'\n')

        return 0

    def IntpEvalfine(self, selection, SB):
        ### Interpolate fine grid evaluation.
        ### Provide selection = 'INT' or 'OSC', and SB = 'S' or 'B' for
        ### signal/ background.

        SEL = getattr(self, selection)

        if self.outdir != 'default':
            dir_B = self.outdir
            dir_S = self.outdir
        else:
            dir_B = os.readlink(self.setpath+'PDFs/'+selection+'_BPDF')
            dir_S = self.setpath+'PDFs/'

        ## Load fine grid eval:
        if SB=='B':
            eval, x, y = np.load(
                dir_B+SEL.files['BPDF_KDE_evalfine'],
                allow_pickle=True
            )
            F = RectBivariateSpline(x, y, eval)
            pickle.dump(F, open(dir_B+SEL.files['BPDF_KDE_evalintp'], 'wb'))
        else:
            eval, x, y = np.load(
                dir_S+SEL.name+'_SPDF_KDE_evalfine.npy',
                allow_pickle=True
            )
            F = RectBivariateSpline(x, y, eval)
            pickle.dump(F, open(dir_S+SEL.name+'_SPDF_KDE_evalfine-intp.pkl', 'wb'))

    def LoadResults(self):
        if 'results.txt' in os.listdir(self.modelpath):
            with open(self.modelpath+'results.txt', 'r') as rf:
                results = str(rf.readline())
            results = eval(results.split('\x00')[0])
            # results = eval(results)
        else:
            results = {}

        for key, value in self.input_parameters.items():
            results[key] = value
        for key, value in self.observables.items():
            results[key] = value

        results['batch'] = self.batch

        self.results = results

    def SaveResults(self):
        with open(self.modelpath+'results.txt', 'w') as rf:
            rf.write(str(self.results)+'\n')

    def UpdateResults(self):
        self.LoadResults()
        results = self.results

        for name, set in self.sets.items():
            setdir = self.modelpath+set[0]

            if not name in ['farsample', 'unblinded']:

                try:
                    bg = LoadSample(setdir+'TS/TS_background_00000.npy')
                    bg_median = np.median(np.sort(bg['minTS'].values*-1))
                except:
                    bg_median = np.nan

                try:
                    rdf = LoadSample(setdir+'TS/sensitivity_fit.npy')
                    mu90 = rdf['90%CL_ns'][0]
                except:
                    mu90 = np.nan

                try:
                    te, ns = np.loadtxt(setdir+'te_ns.txt')
                except:
                    te, ns = np.nan, np.nan

                try:
                    _, _, mrf_te = np.loadtxt(setdir+'MRF.txt')
                except:
                     mrf_te = np.nan

                # Creating results dictionary:
                new_results = {
                    'te'                   : te,
                    'TS0_median '   +name  : bg_median,
                    '90%CL_ns '     +name  : mu90,
                    'ns_te '        +name  : ns,
                    'MRF_te '       +name  : mrf_te,
                }


            if name in ['farsample', 'unblinded']:

                try:
                    TSmin, TSmin_ns, pvalue = np.loadtxt(setdir+'TSmin.txt', unpack=True)
                    TSmin = TSmin*-1
                except:
                    TSmin, TSmin_ns, pvalue = np.nan, np.nan, np.nan

                try:
                    rdf = LoadSample(setdir+'TS/upperlimit_fit.npy')
                    muUL = rdf['UL'][0]
                except:
                    muUL = np.nan

                # Creating results dictionary:
                new_results = {
                    'TSmin '        +name  : TSmin,
                    'TSmin_ns '     +name  : TSmin_ns,
                    'P '            +name  : pvalue,
                    'UL '           +name  : muUL,
                }


            for key, value in new_results.items():
                results[key] = value

            # results.pop('MaxlogL '   +name, None)
            # results.pop('MaxlogL_ns '+name, None)
            # results.pop('TSmin'   , None)
            # results['batch'] = self.batch
            # print(new_results)

        # results.pop('equi_el', None)
        # results.pop('equi_inel', None)

        vals = list(results.keys())
        for val in vals:
            if (isinstance(results[val], str)==False) and (np.isnan(results[val])==True):
                results.pop(val, None)

        # print(results)
        self.results = results
        self.SaveResults()
