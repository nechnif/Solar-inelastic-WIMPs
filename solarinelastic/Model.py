import os
import pickle, json
import numpy as np
from numpy import nan
import pandas as pd

from .dataio import *
from .solar import *
from .gaussian_kde import *
from .Selection import *



###--- globals ---------------------------------------------------------
int_config = 'selections.int.'    +'int_config'
osc_config = 'selections.oscnext.'+'osc_config'
defaultconfigs = [int_config, osc_config]

defaultpath  = '/data/user/rbusse/analysis/models/ana/'
sets = {
    # set             location            name
    'nominal'     : ['nominal/',         'nominal'],
    'DOMeff0.90'  : ['sys-DOMeff0.90/',  'DOMeff 0.90'],
    'DOMeff1.10'  : ['sys-DOMeff1.10/',  'DOMeff 1.10'],
    'BIceAbs0.95' : ['sys-BIceAbs0.95/', 'BIce abs 0.95'],
    'BIceAbs1.05' : ['sys-BIceAbs1.05/', 'BIce abs 1.05'],
    'BIceSct0.95' : ['sys-BIceSct0.95/', 'BIce sct 0.95'],
    'BIceSct1.05' : ['sys-BIceSct1.05/', 'BIce sct 1.05'],
    'HIceP0-1'    : ['sys-HIceP0-1/',    'HIce p0 -1'],
    'HIceP0+1'    : ['sys-HIceP0+1/',    'HIce p0 +1'],
}


class Model(object):

    def __init__(self, set, batch, name, loc=defaultpath,
    configs=defaultconfigs, m=8):

        self.set       = set
        self.loc       = loc
        self.batch     = batch
        self.batchname = 'batch_'+batch
        self.name      = name
        self.modelpath = loc+self.batchname+'/'+name+'/'
        self.setpath   = self.modelpath+sets[set][0]
        self.mdm       = float(self.name)
        self.configs   = configs
        self.m         = m

        with open(self.modelpath+self.name+'.txt', 'r') as ls:
            point = eval(ls.readline())
        self.point = point

        # try:
        #     point['equi (elastic)']   = float(point['equi (elastic)'])
        #     point['equi (inelastic)'] = float(point['equi (inelastic)'])
        #     point['Veff'] = float(point['Veff'].replace('33 ', ''))
        #     point['depletion rate'] = float(point['depletion rate'])
        #     # print(point)
        #     with open(self.modelpath+self.name+'.txt', 'w') as ls:
        #         dump = json.dumps(point)
        #         ls.write(dump+'\n')
        # except:
        #     print('Model already updated')

        ## Make input parameters accessable more easily:
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

            ## Signal PDFs:
            Seval, _, _ = np.load(
                self.setpath+'PDFs/'+selection+'_SPDF_KDE_evalfine.npy',
                allow_pickle=True
            )
            if selection+'_SPDF_KDE_evalfine-intp.pkl' not in os.listdir(self.setpath+'PDFs/'):
                self.IntpEvalfine(selection, 'S')
            Sintp = pickle.load(
                open(self.setpath+'PDFs/'+selection+'_SPDF_KDE_evalfine-intp.pkl', 'rb')
            )

            setattr(self, selection+'_SPDF_evalfine', Seval)
            setattr(self, selection+'_SPDF_evalintp', Sintp)
            # setattr(self, selection+'_cfine_psi',  fine_psi)
            # setattr(self, selection+'_cfine_logE', fine_psi)

            ## Background PDFs:
            Beval, _, _ = np.load(
                SEL.files['BPDF_KDE_evalfine'].replace('MM', str(self.m).zfill(2)),
                allow_pickle=True
            )
            Bintp = pickle.load(
                open(SEL.files['BPDF_KDE_evalintp'].replace('MM', str(self.m).zfill(2)), 'rb')
            )
            setattr(self, selection+'_BPDF_evalfine',  Beval)
            setattr(self, selection+'_BPDF_evalintp',  Bintp)

    def CreateBPDF(self, selection):
        ### Create background PDFs.
        ### Provide selection = 'INT' or 'OSC'.

        SEL = getattr(self, selection)

        sample = LoadSample(SEL.files['background_events'])

        ## Create histogram:
        X = sample['sun_psi'].values
        Y = sample['logE'].values
        if 'weight' not in sample.columns:
            sample['weight'] = np.ones(len(sample))
        W = sample['weight'].values
        xbins, ybins   = SEL.psibins,  SEL.ebins
        bounds_b = SEL.bounds_b

        hist = np.array(np.histogram2d(X, Y, bins=[xbins, ybins], weights=W, density=True), dtype=object)

        ## Create KDE with own class, from values:
        kernel = GaussianKDE(np.vstack([X, Y]), weights=W, bounds=bounds_b)
        kernel.set_bandwidth(*SEL.bw)
        # if SEL.bw:
        # SEL.bw = kernel.bandwidth
        print('background bandwidth:\t', SEL.bw)

        ## Save everything:
        np.save(SEL.files['BPDF_histogram'], hist)
        pickle.dump(kernel, open(SEL.files['BPDF_KDE'], 'wb'))

    def CreateSPDF(self, selection):
        ### Create signal PDFs.
        ### Provide selection = 'INT' or 'OSC'.

        SEL = getattr(self, selection)

        sample = LoadSample(SEL.files['signal_events'].replace('SSSS', SEL.sets[self.set][0]))
        print('Calculating {} SPDF for set {}, scenario {} ...'.format(SEL.name, self.set, self.name))

        ## Cut sample at maximum WIMP energy:
        e_wimp = self.flux_e
        sample = sample[sample['trueE'] <= np.max(e_wimp)]

        ## Weight sample with WIMP flux:
        sample.loc[sample['PDG']== 14, 'wimp'] = sample['weight']/SunSolidAngle(sample['dist'])*WIMPweight(self, sample['trueE'], 'nu')
        sample.loc[sample['PDG']==-14, 'wimp'] = sample['weight']/SunSolidAngle(sample['dist'])*WIMPweight(self, sample['trueE'], 'nubar')

        ## Create histogram:
        X = sample['sun_psi'].values
        Y = sample['logE'].values
        W = sample['wimp'].values
        xbins,  ybins  = SEL.psibins,  SEL.ebins
        bounds_s = SEL.bounds_s

        hist = np.array(np.histogram2d(X, Y, bins=[xbins, ybins], weights=W, density=True), dtype=object)

        ## Create KDE with own class, from values:
        kernel = GaussianKDE(np.vstack([X, Y]), weights=W, bounds=bounds_s)
        kernel.set_bandwidth(*SEL.bw)
        print('signal bandwidth:\t', SEL.bw)

        ## Save everything:
        np.save(self.setpath+'PDFs/'+SEL.name+'_SPDF_histogram.npy', hist)
        pickle.dump(kernel, open(self.setpath+'PDFs/'+SEL.name+'_SPDF_KDE.pkl', 'wb'))

    def FineGridEvaluation(self, selection, SB):
        ### This function creates a fine-grid evaluation of the PDF KDEs,
        ### with the purpose of drawing random toy samples from it.
        ### Provide selection = 'INT' or 'OSC'.


        SEL = getattr(self, selection)
        print('Calculating fine grid eval for {} {}PDF for set {}, scenario {} ...'.format(SEL.name, SB, self.set, self.name))

        x, y = SEL.psifine, SEL.efine
        # x = np.linspace(self.psibins[0],  self.psibins[-1],  len(self.psibins) *m-(m-1))
        # y = np.linspace(self.ebins[0], self.ebins[-1], len(self.ebins)*m-(m-1))
        xx, yy    = np.meshgrid(x, y)
        positions = np.vstack([xx.ravel(), yy.ravel()])

        name = SB+'PDF_KDE'
        if SB=='B':
            KDE = pickle.load(open(SEL.files['BPDF_KDE'], 'rb'))
        else:
            KDE = pickle.load(open(self.setpath+'PDFs/'+SEL.name+'_'+name+'.pkl', 'rb'))

        eval = KDE.evaluate(positions, **getattr(SEL, 'kwargs_'+name))
        ## Dividing by the sum is necessary to make sure SPDF and
        ## BPDF both sum to exactly the same value (after this, 1).
        ## For the histograms this is given naturally, but since the
        ## normalization of the KDE happens inside the kernels and
        ## not for the evaluated grid (which is a simplification),
        ## it can happen that sum(SPDF) != sum(BPDF) and then shit
        ## hits all kinds of fans.
        # print(eval.sum())
        eval = eval/eval.sum()
        # print(KDE.integrate_box(self.bounds, delta=200))
        ## The transposing happens so that the KDE has the same
        ## meaning as a numpy histogram2d()[0]; same reason that
        ## we are saving x, y instead of cx, cy here:
        eval = np.reshape(eval, xx.shape).T

        ## Save:
        if SB=='B':
            pickle.dump((eval, x, y),
                open(SEL.files['BPDF_KDE_evalfine'].replace('MM', str(self.m).zfill(2)), 'wb')
            )
        else:
            pickle.dump((eval, x, y),
                open(self.setpath+'PDFs/'+SEL.name+'_'+name+'_evalfine.npy', 'wb')
            )

        ## Interpolation:
        self.IntpEvalfine(selection, SB)
        print('{} {} done.'.format(SEL.ID, name))

    def IntpEvalfine(self, selection, SB):
        ### Interpolate fine grid evaluation.
        ### Provide selection = 'INT' or 'OSC'.

        SEL = getattr(self, selection)

        ## Load fine grid eval:
        if SB=='S':
            eval, x, y = np.load(
                self.setpath+'PDFs/'+SEL.name+'_SPDF_KDE_evalfine.npy',
                allow_pickle=True
            )
            F = RectBivariateSpline(x, y, eval)
            pickle.dump(F, open(self.setpath+'PDFs/'+SEL.name+'_SPDF_KDE_evalfine-intp.pkl', 'wb'))
        if SB=='B':
            eval, x, y = np.load(
                SEL.files['BPDF_KDE_evalfine'].replace('MM', str(self.m).zfill(2)),
                allow_pickle=True
            )
            F = RectBivariateSpline(x, y, eval)
            pickle.dump(F, open(SEL.files['BPDF_KDE_evalintp'].replace('MM', str(self.m).zfill(2)), 'wb'))

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

        for name, set in sets.items():
            # print(name, set)

            if 'te_ns.txt' in os.listdir(self.modelpath+set[0]):
                te, ns = np.loadtxt(self.modelpath+set[0]+'te_ns.txt')
            else:
                ns = np.nan

            if 'TS_background_00000.npy' in os.listdir(self.modelpath+set[0]+'TS/'):
                bg = LoadSample(self.modelpath+set[0]+'TS/TS_background_00000.npy')
                bg_median = np.median(np.sort(bg['minTS'].values*-1))
            else:
                bg_median = np.nan

            if 'TS_sensitivity.npy' in os.listdir(self.modelpath+set[0]+'TS/'):
                rdf = LoadSample(self.modelpath+set[0]+'TS/TS_sensitivity.npy')
                mu90 = rdf['90%CL_ns'][0]
            else:
                mu90 = np.nan

            if 'MRF.txt' in os.listdir(self.modelpath+set[0]):
                te, ns, mrf_te = np.loadtxt(self.modelpath+set[0]+'MRF.txt')
            else:
                mrf_te = np.nan

            if name=='nominal' and ('TSmin.txt' in os.listdir(self.setpath)):
                TSmin, TSmin_ns = np.loadtxt(self.setpath+'TSmin.txt', unpack=True)
            else:
                TSmin, TSmin_ns = np.nan, np.nan

            # Creating results dictionary:
            new_results = {
                'te'                   : te,
                'TS0_median '   +name  : bg_median,
                '90%CL_ns '     +name  : mu90,
                'ns_te '        +name  : ns,
                'MRF_te '       +name  : mrf_te,

                # 'TSmin'              : TSmin,
                # 'TSmin_ns'           : TSmin_ns,
            }

            for key, value in new_results.items():
                results[key] = value

            # results.pop('MaxlogL '   +name, None)
            # results.pop('MaxlogL_ns '+name, None)
            # results.pop('TSmin'   , None)
            # results['batch'] = self.batch

        # results.pop('equi_el', None)
        # results.pop('equi_inel', None)

        vals = list(results.keys())
        for val in vals:
            if (isinstance(results[val], str)==False) and (np.isnan(results[val])==True):
                results.pop(val, None)

        # print(results)
        self.results = results
        self.SaveResults()


# ## Delete KDE.pkl object (it takes a huge amount of space and is
# ## quick to re-create):
# if self.files['SPDF_KDE'].split('/')[-1] in os.listdir(model.modelpath+'PDFs/'):
#     os.remove(self.files['SPDF_KDE'].replace('$MODELPATH', model.modelpath))
