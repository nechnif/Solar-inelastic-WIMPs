import os
import time
import numpy as np
import pandas as pd
import warnings
from scipy.optimize import minimize, curve_fit

from .Model import Model
from .dataio import *
from .solar import *
from .samplemod import *


def PDF(sample, model, selection, SB):
    ### selection = 'INT' or 'OSC'.
    ### SB = 'S' or 'B'

    SEL = getattr(model, selection)
    events = sample[sample['sel']==SEL.id]
    X = events['sun_psi'].values
    Y = events['logE'].values

    name = ('SPDF_'+model.name if SB=='S' else 'BPDF')

    ## Use interpolation:
    model.LoadPDFs()
    eval = []
    F = getattr(model, SEL.name+'_'+SB+'PDF_evalintp')
    for i in range(len(X)):
        eval.append(F(X[i], Y[i])[0][0])

    ## Account for farsample PDF correction factors when applicable:
    if model.set == 'farsample':
        model.LoadFarsampleFactors()
        k = model.farfactors[selection+'-'+SB]
    else:
        k = 1.0
    eval = np.array(eval)*k

    sample.loc[sample['sel']==SEL.id, name] = eval

    return sample

def SampleFromPDF(model, selection, SB, N, seed=None):
    ### SB = S (B) for signal (background)
    ### selection = 'INT' or 'OSC'.

    model.LoadPDFs()
    SEL = getattr(model, selection)
    ## If N is 0, return empty dataframe (avoid loading a fine grid
    ## evaluation), and print a warning:
    if N == 0:
        # print('WARNING: Emptry dataframe requested.')
        return pd.DataFrame(columns=['sun_psi', 'logE', 'sel'])

    eval = getattr(model, SEL.name+'_'+SB+'PDF_evalfine')
    cx, cy = SEL.psifine, SEL.efine
    # cx   = getattr(model, SEL.name+'_cfine_psi')
    # cy   = getattr(model, SEL.name+'_cfine_logE')

    rng = np.random.default_rng(seed)

    psi, logE = [], []
    sample = rng.choice(eval.size, N, p=eval.ravel()/eval.sum())
    a, b = np.unravel_index(sample, eval.shape)
    psi.append( cx[a])
    logE.append(cy[b])

    pdfsample = pd.DataFrame(columns=['sun_psi', 'logE', 'sel'])
    pdfsample['sun_psi'], pdfsample['logE'] = np.concatenate(psi), np.concatenate(logE)
    pdfsample['sel'] = np.ones(len(pdfsample)) * SEL.id
    return pdfsample

def NumberOfSignalEvents(model, livetime, sample=None):    # Provide te in days.
    ### Calculate the number of signal events for the current set.

    day_s = 86400.        # day in seconds
    te_s  = livetime * day_s

    if not sample:
        sample = AllSignalsWeighted(model)
        ns = np.sum(sample['wimp']/SunSolidAngle(sample['dist']))*te_s
    else:
        if 'dist' not in sample.columns:
            ## This is only for test purposes!
            print('WARNING: No "dist" column found. Are you sure this is a signal sample?')
            ns = np.sum(sample['wimp'])*te_s
        else:
            ns = np.sum(sample['wimp']/SunSolidAngle(sample['dist']))*te_s

    np.savetxt(
        model.setpath+'te_ns.txt',
        np.array([[livetime], [ns]]).T, fmt='%.4e',
        header='livetime [days]\tns [per livetime]'
    )

    # print('total ns in {} days ({:.2e} years):\t{:.4e}'.format(livetime, livetime/365, ns))
    return livetime, ns

def Likelihood(ns, sample, model):
    N = len(sample)
    L = np.prod(
        ns/N*sample['SPDF_'+model.name].values + (1-ns/N)*sample['BPDF'].values
    )
    return L

def LogLikelihood(ns, sample, model):
    N = len(sample)
    Lterms = np.log(
        ns/N*sample['SPDF_'+model.name].values + (1-ns/N)*sample['BPDF'].values
    )
    L = -np.sum(Lterms)
    return L

def TestStatistics(ns, sample, model):
    N = len(sample)
    TS_terms = -2 * np.log( ns/N * sample['S/B'] + (1-ns/N))
    # TS = np.sum(TS_terms)
    ## Sorting is important to make the sum numerically stable:
    TS = np.sum(np.sort(TS_terms)[::-1])
    return TS

def TSdist(model, tests, mu, kind='data', livetime=None, N=None, ratio=None, mode='save', div=1, physical=False, seed=None):
    ### If ratio, provide as e.g. [1, 0]. Sum of ratio[0] and ratio[1]
    ### needs to be 1, and N is mandatory.
    ### If livetime, N is calculated automatically.

    if (kind == 'unblinded') or (kind == 'samedata'):
        ## For the unblinded/ test-unblinded set, N is set
        ## automatically; livetime and ratio are ignored.
        print('WARNING: Using unblinded dataset!' if kind=='unblinded' else None)
        livetime, ratio = None, None
        N_int, N_osc = np.rint(model.INT.total), np.rint(model.OSC.total)
        N = int(N_int+N_osc)
        ratio_int, ratio_osc = model.INT.frac, model.OSC.frac
    elif livetime and not (N and ratio):
        N_int, N_osc = np.rint(livetime*model.INT.eventsperday), np.rint(livetime*model.OSC.eventsperday)
        N = int(N_int+N_osc)
        ratio_int, ratio_osc = model.INT.frac, model.OSC.frac
    elif not livetime and (N and ratio):
        N_int, N_osc = N*ratio[0], N*ratio[1]
        ratio_int, ratio_osc = ratio[0], ratio[1]
    else:
        raise ValueError('Please provide <livetime> OR <N and ratio>.')

    ## We pre-load the experimental set to save time in the loop
    ## (only relevant if kind in ['data', 'samedata']):
    model.INT.LoadEvents('exp')
    model.OSC.LoadEvents('exp')

    ## Load the nominal set:
    model_nom = Model(
        'nominal', model.batch, model.name,
        allsets=model.sets,
        loc=model.loc,
        configs=model.configs,
        m=model.m
    )

    tests = int(tests/div)
    mu_inj, TS, ns, TS_corr, ns_corr = [], [], [], [], []

    starttime = time.time()

    ## Pick random numbers:
    rng = np.random.default_rng(seed=seed)
    seeds = np.array(rng.random(tests)*1e5).astype(int)
    # print(seeds)

    ## Run tests:
    print('i\tN\tmu\tmu_inj\tns\tminTS')
    for i in range(tests):
        ## Determine number of signals to be injected:
        if mu == 0:
            ns_int, ns_osc = 0, 0
            mu_inj.append(0)
        else:
            ns_int = rng.poisson(mu*ratio_int)
            ns_osc = rng.poisson(mu*ratio_osc)
            mu_inj.append(ns_int+ns_osc)
        nb_int = int(N_int) - ns_int
        nb_osc = int(N_osc) - ns_osc

        t0 = time.time()
        ## Injecting signals from the given sample:
        signal_int = SampleFromPDF(model, 'INT', 'S', ns_int, seed=seeds[i])
        signal_osc = SampleFromPDF(model, 'OSC', 'S', ns_osc, seed=seeds[i])
        t1 = time.time()

        if kind == 'PDF':
            ## This is for test purposes only. Samples the background
            ## from the background PDFs instead of using data.
            background_int = SampleFromPDF(model_nom, 'INT', 'B', nb_int, seed=seeds[i])
            background_osc = SampleFromPDF(model_nom, 'OSC', 'B', nb_osc, seed=seeds[i])
            background = pd.concat([background_int, background_osc], ignore_index=True)
        elif (kind == 'unblinded') and (model.set == 'unblinded'):
            ## This is the mode for the unblinded set. Uses the same
            ## UNSCRAMBLED background for each test, and only variates
            ## the signal portion.
            model.INT.LoadEvents('unblinded')
            model.OSC.LoadEvents('unblinded')
            if mu==0:
                ## Use *all* events when no signal is inserted (this
                ## ensures a correct and reproducible TS value):
                int_ = model.INT.df_unblinded
                osc_ = model.OSC.df_unblinded
            else:
                ## When signal is injected, subtract the according number
                ## of background events:
                int_ = model.INT.df_unblinded.sample(n=nb_int)
                osc_ = model.OSC.df_unblinded.sample(n=nb_osc)
            background = pd.concat([int_, osc_])
        elif (kind == 'samedata') and (isinstance(seed, int)):
            ## This is to test the distribution of mu90 (similar to the
            ## above but with the same SCRAMBLDED background):
            background = CreateCombined(nb_int, nb_osc, model.INT, model.OSC, seed=seed, farsample=model.set)
        elif kind == 'data':
            ## This is the normal mode to create background TS and sensitivities:
            background = CreateCombined(nb_int, nb_osc, model.INT, model.OSC, seed=seeds[i], farsample=model.set)
        else:
            raise ValueError((
                'No seed provided for kind=="samedata".' if kind=='samedata' else
                'Either kind="{}" unknown'
                'or not allowed in combination with set="{}".').format(kind, model.set))

        # print(background.sort_values('sun_psi'))
        # bint = background[background['sel']==1]
        # bosc = background[background['sel']==0]
        # print(bint.sort_values('sun_psi'))
        # print(bosc.sort_values('sun_psi'))
        # # print(np.rad2deg(np.min(bint['sun_psi'])))
        # # print(np.rad2deg(np.min(bosc['sun_psi'])))

        sample_ = pd.concat([background, signal_int, signal_osc], ignore_index=True).sample(frac=1, random_state=seeds[i])
        sample_ = sample_.drop([
            'azi', 'zen', 'sun_zen', 'sun_azi',
        ], axis=1)

        t2 = time.time()

        ## Interpreting events always with the nominal PDFs, except when
        ## using the far-sample:
        if model.set=='farsample':
            PDF(sample_, model,     'INT', 'B')
            PDF(sample_, model,     'INT', 'S')
            PDF(sample_, model,     'OSC', 'B')
            PDF(sample_, model,     'OSC', 'S')
        else:
            PDF(sample_, model_nom, 'INT', 'B')
            PDF(sample_, model_nom, 'INT', 'S')
            PDF(sample_, model_nom, 'OSC', 'B')
            PDF(sample_, model_nom, 'OSC', 'S')

        sample_['S/B'] = sample_['SPDF_'+model.name].values/sample_['BPDF'].values
        # sample_ = sample_.sort_values('S/B')
        t3 = time.time()

        # ## For testing purposes:
        # SaveSample(sample_, '/data/user/rbusse/analysis/verifications/220921_minimizer-farsample/testsample-'+str(i).zfill(2)+'.npy')

        ## Determine lower minimizer bound (to avoid weird behaviour
        ## in the log function of TS):
        warnings.filterwarnings('error', category=RuntimeWarning)
        testns = np.linspace(-300, 300, 61)
        testTS = []
        for t_ in testns:
            try:
                testTS.append(TestStatistics(t_, sample_, model))
            except:
                testTS.append(np.nan)
        testTS = np.array(testTS)
        warnings.filterwarnings('default', category=RuntimeWarning)
        lowbound = testns[np.where(np.isfinite(testTS))][0]
        # print(lowbound)
        t4 = time.time()

        ## Minimize test statistic:
        if physical == True:
            popt = minimize(TestStatistics, mu_inj[i], args=(sample_, model), bounds=((0, N),))
        else:
            popt = minimize(TestStatistics, mu_inj[i], args=(sample_, model), bounds=((lowbound, N),))
        mu_     = popt.x[0]
        ns.append(mu_)
        ts_     = TestStatistics((0 if mu_<0 else mu_), sample_, model)
        TS.append(ts_)
        print('{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.6f}'.format(i, N, mu, mu_inj[i], mu_, ts_))
        t5 = time.time()

        # print(t1-t0)
        # print(t2-t1)
        # print(t3-t2)
        # print(t4-t3)
        # print(t5-t4)
        # print(t5-t0)

    endtime = time.time()
    print('This calculation took {:.2f} s ({:.2f} h).'.format(endtime-starttime, (endtime-starttime)/3600))

    cols = ['N', 'mu', 'mu_inj', 'ns', 'minTS']
    results = np.array([np.array(c) for c in [
        N*np.ones(tests),
        mu*np.ones(tests),
        mu_inj,
        ns,
        TS,
    ]])
    resultsdf = pd.DataFrame(data=results.T, columns=cols)

    if (model.set=='unblinded') or (mode=='return'):
        return resultsdf

    if model.outdir == 'default':
        outfile = model.setpath+'/TS/TS_'+('background_' if mu==0 else '')+'{:05d}'.format(mu)+'.npy'
    else:
        outfile = model.outdir+'TS_'+('background_' if mu==0 else '')+'{:05d}'.format(mu)+'.npy'

    if div != 1:
        digit = rng.choice(range(100000))
        SaveSample(resultsdf, outfile.replace('.npy', '_'+str(digit).zfill(6)+'.npy'))
    else:
        SaveSample(resultsdf, outfile)

    return 0

def Sensitivity(model, livetime, numtests=500, points=None, append=False, seed=None):

    if model.set == 'farsample':
        print('Sensitivity calculation not available for farsample set.')
        return -1

    if model.outdir == 'default':
        outdir = model.setpath+'TS/'
    else:
        outdir = model.outdir

    bgfile  = outdir+'TS_background_00000.npy'
    outfile = outdir+'sensitivity.npy'

    ## Look for sensitivity file to append to:
    if (append==True) and (outfile.split('/')[-1] not in os.listdir(outdir)):
        print('Could not find sensitivity file to append to.')
        return -1

    ## Points for fit:
    if isinstance(points, (list, np.ndarray)):
        testpoints = points
    else:
        testpoints = [10., 20., 30., 55., 85.]

    kind = ('unblinded' if model.set=='unblinded' else 'data')
    mus, fracs, dfs = [], [], []
    for mu in testpoints:
        df    = TSdist(model, tests=numtests, mu=mu, livetime=livetime, kind=kind, mode='return', physical=False, seed=seed)
        tsmin = df['minTS'].values*-1
        dfs.append(df)

    bigdf = pd.concat(dfs)

    ## Save the full calculation for mu90 distribution plots:
    if append == True:
        ## Use this if you want to improve your sensitivity with
        ## successive calculations:
        olddf = LoadSample(outfile)
        newdf = pd.concat([olddf, bigdf]).sort_values('mu')
        print(newdf)
        SaveSample(newdf, outfile)
    else:
        ## Use this if you want to start fresh (default):
        SaveSample(bigdf, outfile)

    return 0

def SensitivityFit(model, testunblinding=False, seed=None):

    if model.outdir == 'default':
        outdir = model.setpath+'TS/'
    else:
        outdir = model.outdir

    bgfile = outdir+'TS_background_00000.npy'
    infile = outdir+'sensitivity.npy'
    outfile = outdir+('upperlimit_fit.npy' if model.set=='unblinded' else 'sensitivity_fit.npy')

    ## Look for sensitivity file to append to:
    if infile.split('/')[-1] not in os.listdir(outdir):
        print('Could not find sensitivity file.')
        return -1

    ## Load background TS:
    bg = LoadSample(bgfile)
    bg['minTS'] = np.abs(bg['minTS'])
    bg_TS = np.median(np.sort(bg['minTS'].values))

    ## Determine background TS value (which is the median background TS
    ## except for the unblinded sample or test unblinding):
    if model.set == 'unblinded':
        df0 = TSdist(model, tests=1, mu=0, kind='unblinded', mode='return')
        bg_TS, bg_ns = np.abs(df0['minTS'][0]), df0['ns'][0]
    elif testunblinding==True:
        df0 = TSdist(model, tests=1, mu=0, kind='samedata', mode='return', seed=seed)
        bg_TS, bg_ns, = np.abs(df0['minTS'][0]), df0['ns'][0]
    else:
        bg_TS, bg_ns = np.median(np.sort(bg['minTS'].values)), 0.0

    bg_ratio  = len(bg[bg['minTS']>bg_TS])/len(bg)
    rdfbg = pd.DataFrame(data={'mu':[0], 'frac>bg_TS':[bg_ratio]})

    ## Load sensitivity file and add background TS:
    sen = LoadSample(infile)
    mus, fracs, dfs = [], [], []
    for mu in np.unique(sen['mu'].values):
        mus.append(mu)
        df = sen.loc[sen['mu']==mu]
        tsmin = np.abs(df['minTS'].values)
        fracs.append(len(tsmin[tsmin>bg_TS])/len(tsmin))
        dfs.append(df)
    rdf = pd.DataFrame(data={'mu': mus, 'frac>bg_TS': fracs})

    rdf = pd.concat([rdf, rdfbg]).sort_values('mu')

    ## Fit a saturation function to the data points:
    def fitf(x, a, b):
        return 1-(1-b)*np.exp(-x/a)

    x = np.linspace(0, 200, 400)
    popt, pcov = curve_fit(fitf, rdf['mu'], rdf['frac>bg_TS'])
    F = interp1d(fitf(x, *popt), x, kind='linear', fill_value='extrapolate')
    ns90 = float(F(0.9))
    rdf['90%CL_ns'] = np.ones(len(rdf['mu']))*ns90

    if model.set=='unblinded':
        rdf = rdf.rename(columns={'90%CL_ns': 'UL'})

    print(rdf)
    if testunblinding==True:
        return 0
    else:
        SaveSample(rdf, outfile)

    return 0

def TSmin(model, livetime, test=True, seed=None):

    ## This only makes sense with the nominal set:
    if model.set not in ['nominal', 'farsample']:
        print('Please use nominal or farsample set!')
        return -1

    if test == True:
        n_int = int(np.rint(livetime*model.INT.eventsperday))
        n_osc = int(np.rint(livetime*model.OSC.eventsperday))
        if seed:
            sample = CreateCombined(n_int, n_osc, INT=model.INT, OSC=model.OSC, seed=seed)    # good seeds: 1007
        else:
            sample = CreateCombined(n_int, n_osc, INT=model.INT, OSC=model.OSC)
    else:
        dataset = ('farsample' if model.set=='farsample' else 'unblinded')
        model.INT.LoadEvents(dataset)
        model.OSC.LoadEvents(dataset)
        int_ = getattr(model.INT, 'df_'+dataset)
        osc_ = getattr(model.OSC, 'df_'+dataset)
        sample = pd.concat([int_, osc_])
    N = len(sample)
    # print(sample)

    model.LoadPDFs()
    PDF(sample, model, 'INT', 'B')
    PDF(sample, model, 'INT', 'S')
    PDF(sample, model, 'OSC', 'B')
    PDF(sample, model, 'OSC', 'S')
    sample['S/B'] = sample['SPDF_'+model.name].values/sample['BPDF'].values
    # sample = sample.sort_values('S/B')
    # sample = sample.sample(frac=1)
    # print(sample)

    ## The value of ns that maximizes the LogLikelihood also maximizes the
    ## test statistics. The test statistics is way easier to maximize (or
    ## minimize in this case) and not so prone to be stuck in local minima,
    ## that's why we're using the test statistics here.

    ## Determine lower minimizer bound (to avoid weird behaviour
    ## in the log function of TS):
    warnings.filterwarnings('error', category=RuntimeWarning)
    testns = np.linspace(-300, 300, 61)
    testTS = []
    for t_ in testns:
        try:
            testTS.append(TestStatistics(t_, sample, model))
        except:
            testTS.append(np.nan)
    testTS = np.array(testTS)
    warnings.filterwarnings('default', category=RuntimeWarning)
    lowbound = testns[np.where(np.isfinite(testTS))][0]

    popt = minimize(TestStatistics, 0, args=(sample, model), bounds=((lowbound, N),))
    TSmin_ns = popt.x[0]
    TSmin    = np.abs(TestStatistics((0 if TSmin_ns<0 else TSmin_ns), sample, model))

    if model.outdir == 'default':
        dir_  = model.setpath
        dirbg = model.setpath+'TS/'
    else:
        dir_  = model.outdir
        dirbg = model.outdir

    if (model.set == 'farsample'):
        TS   = LoadSample(dirbg+'TS_background_00000.npy')
        TSbg = np.sort(np.abs(TS['minTS'].values))

        ## P-value:
        # indx = np.argmin(np.abs(TSbg-TSmin))
        # numx = len(TSbg)-(indx+1)
        # pvalue = 1-numx/len(TSbg)
        ## P-value (same as above, but in one line):
        pvalue = 1 - len(TSbg[np.where(TSbg>TSmin)])/len(TSbg)
    else:
        pvalue = np.nan

    np.savetxt(
        dir_+'TSmin.txt',
        np.array([[TSmin], [TSmin_ns], [pvalue]]).T, fmt='%.4e',
        header='TSmin\tTSmin_ns\tp-value'
    )

    print('{} {} nsmin, TSmin, P: {:.4f}\t{:.4f}\t{:.6f}'.format(model.batch, model.name, TSmin_ns, TSmin, pvalue))
    # model.LoadResults()
    # model.results['TSmin']    = TSmin
    # model.results['TSmin_ns'] = TSmin_ns
    # model.SaveResults()
    # model.UpdateResults()
    return 0

def MRF(model, livetime):
    ### Calculate the model rejection factor.

    model.UpdateResults()
    if '90%CL_ns '+model.set in model.results.keys():
        mu90 = model.results['90%CL_ns '+model.set]
    else:
        print('{} {} sensitivity not found'.format(model.batch, model.name))
        return -1

    ## Load number of signal events:
    if 'ns_te '+model.set in model.results.keys():
        te = model.results['te']
        ns = model.results['ns_te '+model.set]
        if te != livetime:
            te, ns = NumberOfSignalEvents(model, livetime)
    else:
        te, ns = NumberOfSignalEvents(model, livetime)
    # te, ns = NumberOfSignalEvents(model, livetime)

    ## The MRF is the rescaling factor that is needed for the
    ## theoretical flux to yield ns90 events:
    mrf_te = mu90/ns

    np.savetxt(
        model.setpath+'MRF.txt',
        np.array([[te], [ns], [mrf_te]]).T, fmt='%.4e',
        header='livetime [days]\tns [per livetime]\tMRF'
    )

    # ## Updating results dictionary:
    # results = {
    #     'MRF_te '+model.set : mrf_te,
    # }
    #
    # for key, value in results.items():
    #     model.results[key] = value
    # model.SaveResults()
    # model.UpdateResults()

    print('{} {} {:15s}\tns_te, MRF: {:.4e}\t{:4e}'.format(model.batch, model.name, model.set, ns, mrf_te))
    return 0
