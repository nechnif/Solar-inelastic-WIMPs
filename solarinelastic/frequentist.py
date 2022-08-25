import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit

from .Model import Model
from .dataio import *
from .solar import *
from .samplemod import *


def PDF(sample, model, selection, SB):
    ### selection = 'INT' or 'OSC'.
    ### SB = 'S' or 'B'

    SEL = getattr(model, selection)
    events = sample[sample['sel']==SEL.ID]
    X = events['sun_psi'].values
    Y = events['logE'].values

    name = ('SPDF_'+model.name if SB=='S' else 'BPDF')

    ## Use interpolation:
    model.LoadPDFs()
    eval = []
    F = getattr(model, SEL.name+'_'+SB+'PDF_evalintp')
    for i in range(len(X)):
        eval.append(F(X[i], Y[i])[0][0])

    sample.loc[sample['sel']==SEL.ID, name] = eval

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

    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    psi, logE = [], []
    sample = rng.choice(eval.size, N, p=eval.ravel()/eval.sum())
    a, b = np.unravel_index(sample, eval.shape)
    psi.append( cx[a])
    logE.append(cy[b])

    pdfsample = pd.DataFrame(columns=['sun_psi', 'logE', 'sel'])
    pdfsample['sun_psi'], pdfsample['logE'] = np.concatenate(psi), np.concatenate(logE)
    pdfsample['sel'] = np.ones(len(pdfsample)) * SEL.ID
    return pdfsample

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
    TS_terms = -2 * np.log( ns/N * sample['SPDF_'+model.name].values/sample['BPDF'].values + (1-ns/N))
    return np.sum(TS_terms)

def TSdist(outfile, model, tests, mu, livetime=None, N=None, ratio=None, kind='data', mode='normal', modefile='', seed=None, div=1, physical=False, correction=False):
    ### If ratio, provide as e.g. [1, 0]. Sum of ratio[0] and ratio[1]
    ### needs to be 1, and N is mandatory.
    ### If livetime, N is calculated automatically.

    if livetime and (N or ratio):
        raise ValueError('Provide either livetime, or N and ratio, not both.')
    if not livetime and not (N and ratio):
        raise ValueError('Values for both N and ratio are required')

    # INT = Selection('INT', model.set)
    # OSC = Selection('OSC', model.set)
    ## We pre-load the experimental set to save time in the loop:
    model.INT.LoadEvents('exp')
    model.OSC.LoadEvents('exp')

    ## Load the nominal set:
    model_nom = Model(
        'nominal', model.batch, model.name,
        loc=model.loc,
        configs=model.configs,
        m=model.m
    )

    tests = int(tests/div)
    rng = np.random.default_rng()
    mu_inj, TS, ns, TS_corr, ns_corr = [], [], [], [], []

    if livetime:
        # N = int(np.rint(livetime*INT.eventsperday)+np.rint(livetime*OSC.eventsperday))
        N_int, N_osc = np.rint(livetime*model.INT.eventsperday), np.rint(livetime*model.OSC.eventsperday)
        N = int(N_int+N_osc)
        ratio_int, ratio_osc = model.INT.frac, model.OSC.frac
    else:
        N_int, N_osc = N*ratio[0], N*ratio[1]
        ratio_int, ratio_osc = ratio[0], ratio[1]

    starttime = time.time()

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
        # print(ns_int, ns_osc)

        t0 = time.time()
        ## Injecting signals from the given sample:
        signal_int = SampleFromPDF(model, 'INT', 'S', ns_int)
        signal_osc = SampleFromPDF(model, 'OSC', 'S', ns_osc)
        t1 = time.time()

        if kind == 'PDF':
            background_int = SampleFromPDF(model_nom, 'INT', 'B', nb_int)
            background_osc = SampleFromPDF(model_nom, 'OSC', 'B', nb_osc)
            background = pd.concat([background_int, background_osc], ignore_index=True)
        elif kind == 'data':
            background = CreateCombined(nb_int, nb_osc, model.INT, model.OSC, seed=seed)
        else:
            raise ValueError('kind ' + str(kind) + ' unknown.')

        sample_ = pd.concat([background, signal_int, signal_osc], ignore_index=True).sample(frac=1)
        sample_ = sample_.drop([
            'azi', 'zen', 'sun_zen', 'sun_azi',
        ], axis=1)

        t2 = time.time()

        ## Interpreting signals always with the nominal PDFs:
        PDF(sample_, model_nom, 'INT', 'B')
        PDF(sample_, model_nom, 'INT', 'S')
        PDF(sample_, model_nom, 'OSC', 'B')
        PDF(sample_, model_nom, 'OSC', 'S')
        t3 = time.time()

        # check = InspectPDFs(sample_, model)
        ## Minimize test statistic:
        if physical == True:
            popt = minimize(TestStatistics, mu_inj[i], args=(sample_, model), bounds=((0, N),))
        else:
            popt = minimize(TestStatistics, mu_inj[i], args=(sample_, model), bounds=((-100, N),))
        mu_     = popt.x[0]
        ns.append(mu_)
        ts_     = TestStatistics((0 if mu_<0 else mu_), sample_, model)
        TS.append(ts_)
        print('{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}'.format(i, N, mu, mu_inj[i], mu_, ts_))
        t4 = time.time()

        # print(t1-t0)
        # print(t2-t1)
        # print(t3-t2)
        # print(t4-t3)
        # print(t4-t0)

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

    if mode == 'return':
        return resultsdf

    if div != 1:
        digit = rng.choice(range(100000))
        SaveSample(resultsdf, outfile+'_{:05d}'.format(mu)+'_'+str(digit).zfill(6)+'.npy')
    else:
        SaveSample(resultsdf, outfile+'_{:05d}'.format(mu)+'.npy')

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

    # model.LoadResults()
    # model.results['te'] = livetime
    # model.results['ns_te '+model.set] = ns
    # model.SaveResults()

    # print('total ns in {} days ({:.2e} years):\t{:.2e}'.format(te, te/365, ns))
    return livetime, ns

def MaxLikelihood(model, livetime):

    ## This only makes sense with the nominal set:
    if model.set != 'nominal':
        print('Please use nominal set!')
        return -1

    n_int = int(np.rint(livetime*model.INT.eventsperday))
    n_osc = int(np.rint(livetime*model.OSC.eventsperday))
    sample = CreateCombined(n_int, n_osc, INT=model.INT, OSC=model.OSC, seed=1010)    # good seeds: 1007
    N = len(sample)

    model.LoadPDFs()
    PDF(sample, model, 'INT', 'B')
    PDF(sample, model, 'INT', 'S')
    PDF(sample, model, 'OSC', 'B')
    PDF(sample, model, 'OSC', 'S')

    ## The value of ns that maximizes the LogLikelihood also maximizes the
    ## test statistics. The test statistics is way easier to maximize (or
    ## minimize in this case) and not so prone to be stuck in local minima,
    ## that's why we're using the test statistics here.
    popt = minimize(TestStatistics, 0, args=(sample, model), bounds=((-100, N),))
    # popt = basinhopping(TestStatistics, 0, minimizer_kwargs={'args': (sample, model)}, niter=3)
    TSmin_ns = popt.x[0]
    TSmin    = TestStatistics((0 if TSmin_ns<0 else TSmin_ns), sample, model)
    # popt = basinhopping(LogLikelihood, 0, minimizer_kwargs={'args': (sample, model)}, seed=1111)
    # nsmin = popt.x[0]
    # Lmin  = LogLikelihood(nsmin, sample, model)
    print('{} {} nsmin, logLmin: {:.4f}, {:4f}'.format(model.batch, model.name, TSmin_ns, TSmin))

    np.savetxt(
        model.setpath+'TSmin.txt',
        np.array([[TSmin], [TSmin_ns]]).T, fmt='%.4e',
        header='TSmin\tTSmin_ns'
    )

    # model.LoadResults()
    # model.results['TSmin']    = TSmin
    # model.results['TSmin_ns'] = TSmin_ns
    # model.SaveResults()
    # model.UpdateResults()
    return 0

def Sensitivity(model, livetime, numtests=500, points=None, append=False):

    ## Look for sensitivity file to append to:
    if (append==True) and ('TS_sensitivity.npy' not in os.listdir(model.setpath+'TS/')):
        print('Could not find sensitivity file to append to.')
        return -1

    ## Load background TS and background median:
    try:
        bg = LoadSample(model.setpath+'TS/TS_background_00000.npy')
        bg['minTS'] = bg['minTS']*-1
        bg_median = np.median(np.sort(bg['minTS'].values))
        bg_ratio  = len(bg[bg['minTS']>bg_median])/len(bg)
        rdfbg = pd.DataFrame(data={'mu':[0], 'frac>bg_median':[bg_ratio]})
        # print('{:.2f} {:.2f}'.format(bg_median, bg_ratio))
    except:
        bg_median, bg_ratio, rdfbg = 0.0, None, None

    ## Points for fit:
    if isinstance(points, (list, np.ndarray)):
        testpoints = points
    else:
        testpoints = [20., 30., 55., 85.]

    mus, fracs = [], []
    for mu in testpoints:
        df    = TSdist('', model, tests=numtests, mu=mu, livetime=livetime, mode='return', physical=False)
        tsmin = df['minTS'].values*-1
        mus.append(mu)
        fracs.append(len(tsmin[tsmin>bg_median])/len(tsmin))

    if append == True:
        ## Use this if you want to improve your sensitivity with
        ## successive calculations:
        rdfold = LoadSample(model.setpath+'TS/TS_sensitivity.npy')
        rdfnew = pd.DataFrame(data={'mu': mus, 'frac>bg_median': fracs})
        rdf = pd.concat([rdfold, rdfnew]).sort_values(['mu'])
    else:
        ## Use this if you want to start fresh (default):
        rdf  = pd.DataFrame(data={'mu': mus, 'frac>bg_median': fracs})

    ## Include backround-only TS (if there is one):
    if (0 not in rdf['mu'].values) and bg_ratio:
        rdf = pd.concat([rdf, rdfbg]).sort_values('mu')
    elif bg_ratio:
        rdf.loc[rdf['mu']==0, 'frac>bg_median'] = bg_ratio
    else:
        pass

    ## Fit a saturation function to the data points:
    def fitf(x, a, b):
        return 1-(1-b)*np.exp(-x/a)

    x = np.linspace(0, 200, 400)
    popt, pcov = curve_fit(fitf, rdf['mu'], rdf['frac>bg_median'])
    F = interp1d(fitf(x, *popt), x, kind='linear')
    ns90 = float(F(0.9))
    rdf['90%CL_ns'] = np.ones(len(rdf['mu']))*ns90

    ## The old way. Don't use, it's not accurate.
    # F    = interp1d(rdf['frac>bg_median'], rdf['mu'], kind='linear', fill_value='extrapolate')
    # ns90 = float(F(0.9))

    SaveSample(rdf, model.setpath+'TS/TS_sensitivity.npy')

    # ## Creating results dictionary:
    # setname = model.set.replace(' ', '')
    # results = {
    #     'TS0_median_'+setname   : bg_median,
    #     '90%CL_ns_'+setname     : ns90,
    #     'te'                    : livetime,
    # }
    # model.LoadResults()
    # for key, value in results.items():
    #     model.results[key] = value
    # model.SaveResults()
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
            print('hello1')
            te, ns = NumberOfSignalEvents(model, livetime)
    else:
        print('hello2')
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

    print('{} {} ns_te, MRF: {:.4e}\t{:4e}'.format(model.batch, model.name, ns, mrf_te))
    return 0
