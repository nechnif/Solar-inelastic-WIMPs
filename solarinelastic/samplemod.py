import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .Selection import *
from .dataio import *
from .solar import *


def CreateCombined(n_int, n_osc, INT, OSC, seed=None, farsample=False):

    if not hasattr(INT, 'df_exp'):
        INT.LoadEvents('exp')
    if not hasattr(OSC, 'df_exp'):
        OSC.LoadEvents('exp')
    dfint = INT.df_exp
    dfosc = OSC.df_exp

    dfint['azi'] = ScrambleAzi(len(dfint), seed)
    dfint['sun_psi'] = SpaceAngle(dfint['sun_zen'], dfint['sun_azi'], dfint['zen'], dfint['azi'])
    dfint = dfint[dfint['sun_psi']<=INT.psicut]
    dfint = dfint[dfint['logE']<=INT.logEcut[1]]
    dfint = dfint[dfint['logE']>=INT.logEcut[0]]
    if farsample=='farsample':
        dfint = dfint[dfint['sun_psi']>=INT.psicut_farsample]

    if len(dfint) < n_int:
        dfint = pd.concat([
            dfint.sample(frac=1),
            dfint.sample(n=n_int-len(dfint))
        ])
    else:
        if seed:
            dfint = dfint.sample(n=n_int, random_state=seed)
        else:
            dfint = dfint.sample(n=n_int)
    dfint = dfint.sort_values('logE')

    dfosc['azi'] = ScrambleAzi(len(dfosc), seed)
    dfosc['sun_psi'] = SpaceAngle(dfosc['sun_zen'], dfosc['sun_azi'], dfosc['zen'], dfosc['azi'])
    dfosc = dfosc[dfosc['sun_psi']<=OSC.psicut]
    dfosc = dfosc[dfosc['logE']<=OSC.logEcut[1]]
    dfosc = dfosc[dfosc['logE']>=OSC.logEcut[0]].sample(n=n_osc, random_state=seed)
    if farsample=='farsample':
        dfosc = dfosc[dfosc['sun_psi']>=OSC.psicut_farsample]

    df = pd.concat([dfint, dfosc], ignore_index=True)

    ## To save some memory:
    drops = [
        'ra', 'dec', 'pass_lowup', 'pass_muon', 'angErr', 'recotime',
        'run', 'event', 'subevent', 'mjd'
    ]
    df = df.drop([d for d in drops if d in df.columns], axis=1)
    return df

def ScrambleAzi(length, seed=None):
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    newazi = rng.random(length)*2*np.pi
    # print(newazi)
    return newazi

def WIMPweight(model, E, kind):
    ### Returning interpolated WIMP flux weights.

    ## Flux is loaded in [GeV cm^2 s]^-1:
    wimpe, wimpnu, wimpnubar = model.flux_e, model.flux_numu, model.flux_numubar

    Fnu    = interp1d(wimpe, wimpnu,    kind='linear', bounds_error=False, fill_value=0)
    Fnubar = interp1d(wimpe, wimpnubar, kind='linear', bounds_error=False, fill_value=0)
    nu_weight    = Fnu(E)
    nubar_weight = Fnubar(E)

    if kind=='nu':
        return nu_weight
    elif kind=='nubar':
        return nubar_weight
    else:
        raise ValueError(
            '"kind" needs to be either "nu" or "nubar"'
        )

def AllSignalsWeighted(model, sample='signal_events'):

    # from ..classes.Selection import Selection
    # INT = Selection('INT', model.set)
    # OSC = Selection('OSC', model.set)

    model.INT.LoadEvents(sample)
    model.OSC.LoadEvents(sample)
    intsignals = getattr(model.INT, 'df_'+sample)
    oscsignals = getattr(model.OSC, 'df_'+sample)

    intsignals.loc[intsignals['PDG']== 14, 'wimp'] = intsignals['weight']*WIMPweight(model, intsignals['trueE'], 'nu',  )
    intsignals.loc[intsignals['PDG']==-14, 'wimp'] = intsignals['weight']*WIMPweight(model, intsignals['trueE'], 'nubar')
    oscsignals.loc[oscsignals['PDG']== 14, 'wimp'] = oscsignals['weight']*WIMPweight(model, oscsignals['trueE'], 'nu',  )
    oscsignals.loc[oscsignals['PDG']==-14, 'wimp'] = oscsignals['weight']*WIMPweight(model, oscsignals['trueE'], 'nubar')

    intsignals['sel'] = np.ones(len(intsignals))
    oscsignals['sel'] = np.zeros(len(oscsignals))

    signals = pd.concat([intsignals, oscsignals], ignore_index=True)
    return signals

def CrossCheck(model, intsel, oscsel, te):

    fourpi = 4.*np.pi
    yr_s   = 3.1536*1e7    # year in seconds
    sun_r, sun_d = 6.96e10, 1.496e13
    solarsolid = 2*np.pi*(1-np.cos(np.arctan(sun_r/sun_d)))
    print('Solar solid angle: {:.2e}'.format(solarsolid))

    ## Load effective areas:
    intsel.LoadAeff()
    intaeffnu, zbins, ebins = intsel.Aeffnu
    intaeffnubar, _, _      = intsel.Aeffnubar
    oscsel.LoadAeff()
    oscaeffnu,    _, _      = oscsel.Aeffnu
    oscaeffnubar, _, _      = oscsel.Aeffnubar
    cebins = ebins[:-1]+np.diff(ebins)/2.

    intaeffnu_e    = np.array([np.sum(intaeffnu.T[z]   *np.diff(zbins)) for z in range(len(intaeffnu.T))])   /np.sum(np.diff(zbins))
    intaeffnubar_e = np.array([np.sum(intaeffnubar.T[z]*np.diff(zbins)) for z in range(len(intaeffnubar.T))])/np.sum(np.diff(zbins))
    oscaeffnu_e    = np.array([np.sum(oscaeffnu.T[z]   *np.diff(zbins)) for z in range(len(oscaeffnu.T))])   /np.sum(np.diff(zbins))
    oscaeffnubar_e = np.array([np.sum(oscaeffnubar.T[z]*np.diff(zbins)) for z in range(len(oscaeffnubar.T))])/np.sum(np.diff(zbins))
    aeffnu_e     = (intaeffnu_e    + oscaeffnu_e)
    aeffnubar_e  = (intaeffnubar_e + oscaeffnubar_e)

    # ## Should be exactly the same:
    # # print('{:.2e} {:.2e}'.format(np.sum(aeff_e), np.sum(aeff_z)))
    # # print('{:.2e} {:.2e}'.format(np.sum(aeff_e*np.diff(ebins))   *np.sum(np.diff(zbins)), np.sum(aeff_z*np.diff(zbins))   *np.sum(np.diff(ebins))))
    # # print('{:.2e} {:.2e}'.format(np.sum(intaeff_e*np.diff(ebins))*np.sum(np.diff(zbins)), np.sum(intaeff_z*np.diff(zbins))*np.sum(np.diff(ebins))))
    # # print('{:.2e} {:.2e}'.format(np.sum(oscaeff_e*np.diff(ebins))*np.sum(np.diff(zbins)), np.sum(oscaeff_z*np.diff(zbins))*np.sum(np.diff(ebins))))

    fluxnu    = WIMPweight(model, cebins, 'nu')
    fluxnubar = WIMPweight(model, cebins, 'nubar')
    ## Sum:
    dnudE    = fluxnu    * aeffnu_e   *np.diff(ebins) * yr_s
    dnubardE = fluxnubar * aeffnubar_e*np.diff(ebins) * yr_s
    N1_sum = np.sum(dnudE)+np.sum(dnubardE)
    ## Integral:
    dnudE    = fluxnu    * aeffnu_e    * yr_s
    dnubardE = fluxnubar * aeffnubar_e * yr_s
    N1_int = (np.trapz(dnudE, cebins)+np.trapz(dnubardE, cebins))

    ## N2: sum of sim weighted with flux
    intsel.LoadEvents('sim')
    oscsel.LoadEvents('sim')
    simint, simosc = intsel.df_sim, oscsel.df_sim
    ## The 1/fourpi is necessary because the wimp flux is a point source
    ## flux with units 1/[GeV cm**2 s], and the weight has units
    ## [GeV cm**2 sr]:
    simint.loc[simint['PDG']== 14, 'wimp'] = simint['weight']/fourpi*WIMPweight(model, simint['trueE'], 'nu',  )
    simint.loc[simint['PDG']==-14, 'wimp'] = simint['weight']/fourpi*WIMPweight(model, simint['trueE'], 'nubar')
    simosc.loc[simosc['PDG']== 14, 'wimp'] = simosc['weight']/fourpi*WIMPweight(model, simosc['trueE'], 'nu',  )
    simosc.loc[simosc['PDG']==-14, 'wimp'] = simosc['weight']/fourpi*WIMPweight(model, simosc['trueE'], 'nubar')
    N2 = (np.sum(simint['wimp'].values)+np.sum(simosc['wimp'].values)) * yr_s

    ## N3: sum of actual signal events:
    signals = AllSignalsWeighted(model, intsel, oscsel)
    N3 = np.sum(signals['wimp'].values/SunSolidAngle(signals['dist'].values)) * yr_s

    print('{:.4e}\t{:.4e}'.format(N1_sum, N1_int))
    print('{:.4e}\t{:.4e}'.format(N2, N2*solarsolid))
    print('{:.4e}\t{:.4e}'.format(N3, N3*solarsolid))
