import numpy as np

#--- bins --------------------------------------------------------------
psicut = np.deg2rad(180.0)
psicut_farsample = np.deg2rad(89.70)
logEcut = [np.log10(5), np.log10(300)]

m = 8
psibins = np.linspace(0,          psicut,     20)
ebins   = np.linspace(logEcut[0], logEcut[1], 20)
psifine = np.linspace(psibins[0], psibins[-1], len(psibins)*m-(m-1))
efine   = np.linspace(ebins[0],   ebins[-1],   len(ebins)  *m-(m-1))

bounds_b = [[-np.inf,          np.min(ebins)],
            [ np.inf,          np.inf       ]]
bounds_s = [[np.min(psibins), -np.inf       ],
            [ np.inf,          np.inf       ]]

bandwidth = [0.042, 0.020]
tag = '({:.4f}-{:.4f})'.format(bandwidth[0], bandwidth[1])

kwargs_BPDF_KDE = {
    'bound_method'  : 'reflect',
    'speed'         : 4,
}
kwargs_SPDF_KDE = {
    'bound_method'  : 'reflect',
    'speed'         : 4,
}

#--- constants ---------------------------------------------------------
# Identifier:
id = 0.0

total           = 135063
livetime        = 3357
eventsperday    = 40.2332
eventsper9years = 132930.6381
## Fraction of total analysis sample (obtained by int_events/(int_events+oscevents)
## from the background_events datasets):
frac            = 0.952771

#--- file locations ----------------------------------------------------
name = 'OSC'
selectionpath = '/data/user/rbusse/analysis/selections/oscnext/'

sets = {
    # systematic set   name    # of oversamples
    'nominal'      : ['0000',   150000],
    'DOMeff0.90'   : ['0001',   150000],
    'DOMeff1.10'   : ['0004',   150000],
    'BIceAbs0.95'  : ['0517',   500000],
    'BIceAbs1.05'  : ['0516',   500000],
    'BIceSct0.95'  : ['0519',   500000],
    'BIceSct1.05'  : ['0518',   500000],
    'HIceP0-1'     : ['0300',   350000],     # -1, 0(?) | -1, -0.05
    'HIceP0+1'     : ['0303',   350000],     # +1, 0(?) | +0.3, -0.05
    'OSC-MCv02.05' : ['1122',   150000],
}

files = {
    'sun'               : selectionpath+'sun/sundir_01.npy',
    'exp'               : selectionpath+'data/exp/oscNext_pisa_data_11-20_vrbusse_scrambled-01.npy',
    'background_events' : selectionpath+'data/exp/oversample/oscNext_pisa_data_11-20_vrbusse_scrambled-01_ov8.npy',
    'BPDF_histogram'    : 'OSC_BPDF_histogram'+tag+'.npy',
    'BPDF_KDE'          : 'OSC_BPDF_KDE'+tag+'.pkl',
    'BPDF_KDE_evalfine' : 'OSC_BPDF_KDE_evalfine'+tag+'.npy',
    'BPDF_KDE_evalintp' : 'OSC_BPDF_KDE_evalfine'+tag+'-intp.pkl',
    'sim'               : selectionpath+'data/sim/oscnext_pisa_genie_SSSS_vrbusse.npy',
    'signal_events'     : selectionpath+'data/sim/oversample/oscnext_pisa_genie_SSSS_vrbusse_ov.npy',
    'Aeff'              : selectionpath+'Aeff/'+str(id)+'_Aeff.npy',

    'farsample'         : selectionpath+'data/exp/oscNext_pisa_data_11-20_vrbusse_UNSCRAMBLED_farsample.npy',
    'farsample_ov'      : selectionpath+'data/exp/oversample/oscNext_pisa_data_11-20_vrbusse_UNSCRAMBLED_farsample_ov.npy',
    'unblinded'         : selectionpath+'data/exp/oscNext_pisa_data_11-20_vrbusse_UNSCRAMBLED.npy',
}

#--- functions ---------------------------------------------------------
def CutOsc(df):
    ### The following cuts are applied by the OscNext group and should
    ### be applied here too. Reference:
    ### https://github.com/icecube/wg-oscillations-fridge/blob/master/analysis/oscNext_std_osc/settings/pipeline/stages/oscNext_sample.cfg
    # original:
    # cuts = [
    #     (l7_muon_classifier_prob_nu > 0.4) &
    #     (L4_NoiseClassifier_ProbNu > 0.95) &
    #     (reco_z > -500.) & (reco_z < -200.) &
    #     (reco_rho < 300.) &
    #     (L5_SANTA_DirectPulsesHitMultiplicity.n_hit_doms > 2.5) &
    #     ( L7_CoincidentMuon_Variables.n_top15 < 2.5 ) &
    #     ( L7_CoincidentMuon_Variables.n_outer < 7.5 ) &
    #     ( L7_reconstructed_time < 14500. ) &
    #     (reco_energy >= 5.) &
    #     (reco_energy <= 300.) &
    #     (reco_coszen <= 0.3)
    # ]

    df = df[df['prob_nu_l7_as'] >     0.4  ]
    df = df[df['prob_nu_l4']    >     0.95 ]
    df = df[df['reco_z']        >  -500.   ]
    df = df[df['reco_z']        <  -200.   ]
    df = df[df['reco_rho']      <   300.   ]
    df = df[df['n_hit']         >     2.5  ]
    df = df[df['n_top15']       <     2.5  ]
    df = df[df['n_outer']       <     7.5  ]
    df = df[df['recotime']      < 14500.   ]
    df = df[df['E']            >=     5.0  ]
    df = df[df['E']            <=   300.   ]
    df = df[df['coszen']       <=     0.3  ]

    # Distinguish set from INT sample:
    df = df[df['pass_muon'] == 0.0]

    print(len(df), '\n')
    return df
