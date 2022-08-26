import numpy as np

#--- bins --------------------------------------------------------------
psicut  = np.deg2rad(10.0)
logEcut = [2.0, 6.0]

m = 8
psibins = np.linspace(0,          psicut,     20)
ebins   = np.linspace(logEcut[0], logEcut[1], 20)
psifine = np.linspace(psibins[0], psibins[-1], len(psibins)*m-(m-1))
efine   = np.linspace(ebins[0],   ebins[-1],   len(ebins)  *m-(m-1))

bounds_b = [[-np.inf,         np.min(ebins)],
            [np.max(psibins), np.inf       ]]
bounds_s = [[np.min(psibins), np.min(ebins)],
            [np.inf,          np.inf       ]]

bandwidth = [0.0020, 0.020]
tag = '({:.4f}-{:.4f})'.format(bandwidth[0], bandwidth[1])

kwargs_BPDF_KDE = {
    'bound_method'  : 'reflect',
    'speed'         : 32,
}
kwargs_SPDF_KDE = {
    'bound_method'  : 'reflect',
    'speed'         : 4,
}

#--- constants ---------------------------------------------------------
# Identifier:
id = 1.0

total           = 6592
livetime        = 3304
eventsperday    = 1.9950
eventsper9years = 6589.3533
## Fraction of total analysis sample (obtained by
## int_events/(int_events+oscevents) from the exp. datasets):
frac            = 0.047229

#--- file locations ----------------------------------------------------
name = 'INT'
selectionpath = '/data/user/rbusse/analysis/selections/int/'

sets = {
    # systematic set      name                # of oversamples
    'nominal'     : ['nominal',               25000],
    'DOMeff0.90'  : ['21047_dom_090',         75000],
    'DOMeff1.10'  : ['21047_dom_110',         70000],
    'BIceAbs0.95' : ['21006_abs95',           30000],
    'BIceAbs1.05' : ['pass2_abs105_ds21005',  30000],
    'BIceSct0.95' : ['21004_scat95',          30000],
    'BIceSct1.05' : ['pass2_scat105_ds21003', 30000],
    'HIceP0-1'    : ['21047_holep0_-1.0',     50000],     # -1, 0(?) | -1, -0.05
    'HIceP0+1'    : ['21047_holep0_+1.0',     50000],     # +1, 0(?) | +0.3, -0.05
}

files = {
    'sun'               : '/data/user/rbusse/analysis/sun/sun_positions/',
    'exp'               : selectionpath+'data/exp/INT_IC86_11-19_exp_vrbusse_scrambled-01.npy',
    'background_events' : selectionpath+'data/exp/oversample/INT_IC86_11-19_exp_vrbusse_scrambled-01_10deg_ov150.npy',
    'BPDF_histogram'    : selectionpath+'backgroundPDF/INT_BPDF_histogram'+tag+'.npy',
    'BPDF_KDE'          : selectionpath+'backgroundPDF/INT_BPDF_KDE'+tag+'.pkl',
    'BPDF_KDE_evalfine' : selectionpath+'backgroundPDF/INT_BPDF_KDE_evalfine'+tag+'.npy',
    'BPDF_KDE_evalintp' : selectionpath+'backgroundPDF/INT_BPDF_KDE_evalfine'+tag+'-intp.pkl',
    'sim'               : selectionpath+'data/sim/IC79_IC86_MC_SSSS_vrbusse.npy',
    'signal_events'     : selectionpath+'data/sim/oversample/IC79_IC86_MC_SSSS_vrbusse_ov.npy',
    'Aeff'              : selectionpath+'Aeff/'+str(id)+'_Aeff.npy',
}

#--- functions ---------------------------------------------------------
def CutDiffuse(df):
    ### The following cuts are applied by the Diffuse group and should
    ### be applied here too (already included in the files provided
    ### by hmniederhausen and tglauch).
    cosZenith_bins = np.linspace(-1., np.cos(85./180.*np.pi), 34)
    logEnergy_bins = np.linspace(0., 8., 61)

    print(len(df))
    # Some simple extra quality cuts:
    df = df[df['zenith_exists'] == 1]
    df = df[df['zenith_fit_status'] == 0]
    df = df[df['zen'] < np.arccos(np.min(cosZenith_bins))]
    df = df[df['zen'] > np.arccos(np.max(cosZenith_bins))]
    df = df[df['dec'] < -np.arcsin(np.min(cosZenith_bins))]
    df = df[df['dec'] > -np.arcsin(np.max(cosZenith_bins))]
    df = df[df['energy_exists'] == 1]
    df = df[df['energy_fit_status'] == 0]
    df = df[df['energy'] > 10**(np.min(logEnergy_bins))]
    df = df[df['energy'] < 10**(np.max(logEnergy_bins))]

    # Make energy cut and apply angular error floor
    mask = (df['logE'] >= 2.) & (df['logE'] < 8.0)
    df = df[mask]
    df['angErr'][np.degrees(df['angErr'])<0.1] = np.radians(0.1)

    df = df.drop([
        'zenith_exists', 'zenith_fit_status',
        'energy_exists', 'energy_fit_status', 'energy'
    ], axis=1)
    print(len(df), '\n')

    return df

def CutSystematics(df):
    ### The files created by tglauch located in
    ### /data/user/tglauch/diffuse_ps_final_level_processing/v005p02/mc
    ### are missing the final energy cut and the angular error floor
    ### correction. That's whats done here.

    print(len(df))

    mask = (df['TUM_dnn_energy_hive'] >= 2.) & (df['TUM_dnn_energy_hive'] < 8.0)
    df = df[mask]

    df.loc[np.degrees(df['TUM_angErr'])<0.1, 'TUM_angErr'] = np.radians(0.1)

    print(len(df), '\n')

    return df
