import importlib
import numpy as np
import pandas as pd

from .dataio import *
from .solar import *
from .samplemod import *


sunpath = '/data/user/rbusse/analysis/sun/sun_positions/'


class Selection(object):

    def __init__(self, name, set, configs, m):
        if name=='INT':
            config = importlib.import_module(configs[0])
        elif name=='OSC':
            config = importlib.import_module(configs[1])
        else:
            print('Name {} not recognized! Aborting.'.format(name))
            return -1

        self.config = config
        self.ID = config.id
        self.name = config.name
        self.sets = config.sets
        self.set = set
        self.selectionpath = config.selectionpath

        ## Bins:
        self.psibins = config.psibins
        self.ebins   = config.ebins
        self.psifine = config.psifine
        self.efine   = config.efine
        # ## Bin centers:
        # self.cpsibins  = self.psibins[:-1] +self.binwidth_psi/2.
        # self.cebins = self.ebins[:-1]+self.binwidth_logE/2.

        self.bounds_b = config.bounds_b
        self.bounds_s = config.bounds_s

        self.bw = config.bandwidth
        self.kwargs_BPDF_KDE = config.kwargs_BPDF_KDE
        self.kwargs_SPDF_KDE = config.kwargs_SPDF_KDE

        self.total           = config.total
        self.livetime        = config.livetime
        self.eventsperday    = config.eventsperday
        self.eventsper9years = config.eventsper9years
        self.frac            = config.frac
        self.psicut          = config.psicut
        self.logEcut         = config.logEcut

        self.files = config.files

    def DetermineLivetime(self, te):
        import astropy.coordinates as coord
        import astropy.units as u
        from astropy.time import Time

        # Determine livetime in days:
        self.LoadEvents('background_events')
        bg = self.df_background_events.sort_values('mjd', axis=0)
        # exp = LoadSample(self.files['background_events']).sort_values('mjd', axis=0)
        times = Time(bg['mjd'].values, format='mjd')
        delta = times[-1]-times[0]
        livetime = delta.to_datetime().days

        # Determine total number of events:
        if 'weight' not in bg.columns:
            bg['weight'] = np.ones(len(bg))
        total = len(bg)*bg['weight'][0]

        evp9y = total/livetime*te

        print(
            'events:\t\t\t{}\n'
            'livetime:\t\t{}\n'
            'events per day:\t\t{:.4f}\n'
            'events in {:.2f} years:\t{:.4f}'.format(
            total, livetime, total/livetime, te/365, evp9y)
        )

    def OversampleBackground(self, rep):
        insample = LoadSample(self.files['exp'])
        if self.ID == 0.0:
            drops = [
                'pass_lowup', 'pass_muon',
                'run', 'event', 'subevent',
                'recotime',
            ]
            insample = insample.drop([d for d in drops if d in insample.columns], axis=1)
        elif self.ID == 1.0:
            drops = [
                'ra', 'dec',
                'run', 'event', 'subevent',
                'angErr',
            ]
            insample = insample.drop([d for d in drops if d in insample.columns], axis=1)

        dfs = []
        for r in range(rep):
            print('cycle ', r, ' ...')
            df_ = insample.copy()
            df_['azi'] = ScrambleAzi(len(df_))
            df_['sun_psi'] = SpaceAngle(
                df_['sun_zen'], df_['sun_azi'],
                df_['zen'], df_['azi']
            )
            df_ = df_[df_['sun_psi'] <= self.psibins[-1]]
            df_ = df_[df_['logE']    >= self.ebins[0]]
            df_ = df_[df_['logE']    <= self.ebins[-1]]
            dfs.append(df_)

        outsample = pd.concat(dfs, ignore_index=True)
        outsample['weight'] = np.ones(len(outsample))/float(rep)
        print(outsample)
        # print(np.rad2deg(min(df_['zen'])), np.rad2deg(max(df_['zen'])))
        SaveSample(outsample, self.files['background_events'])

    def OversampleSignal(self, div=1):
        insample = LoadSample(self.files['sim'].replace('SSSS', self.sets[self.set][0]))
        rep = self.sets[self.set][1]
        print('oversampling MC set {} with {} reps ...'.format(self.set, rep))

        if div != 1:
            if int(rep/div) != rep/div:
                print('{} can not be divided by {}'.format(rep, div))
            else:
                rep = int(rep/div)

        ## Load sun data:
        sunfiles = []
        ## This takes too much memory:
        # for file in os.listdir(sunpath):
        #     if 'sundir_' in file:
        #         sunfiles.append(LoadSample(sunpath+file))
        # sunlocs  = pd.concat(sunfiles)
        ## Use this instead:
        sunlocs = LoadSample(sunpath+'sundir_2011_01.npy').sample(n=len(insample), replace=True)
        ## convert distance earth-sun to [cm]:
        sunlocs['dist'] = sunlocs['dist']*1e5

        dfs = []
        for r in range(rep):
            print('cycle ', r, ' ...')

            df_  = insample.copy(deep=False)
            # df_  = insample.copy()
            sun_ = sunlocs.sample(n=len(df_)).reset_index(drop=True)
            df_ = df_.reset_index(drop=True).assign(**sun_)

            df_['true_sun_psi'] = SpaceAngle(
                df_['sun_zen'], df_['sun_azi'],
                df_['trueZen'], df_['trueAzi']
            )
            df_['sun_psi'] = SpaceAngle(
                df_['sun_zen'], df_['sun_azi'],
                df_['zen'], df_['azi']
            )

            df_ = FilterCutAngle(df_)
            # ## Weight with solar solid angle (afterwards weights will be
            # ## in units [GeV cm**2]):
            # ## NOT DOING THIS so that weights remain in units [GeV cm**2 sr]
            # solarsolid = SunSolidAngle(df_['dist'])
            # df_['weight'] = df_['weight']/solarsolid

            df_ = df_[df_['sun_psi'] <= self.psibins[-1]]
            df_ = df_[df_['logE']    >= self.ebins[0]]
            df_ = df_[df_['logE']    <= self.ebins[-1]]
            # print(df_)
            dfs.append(df_)

        outsample = pd.concat(dfs, ignore_index=True)
        outsample['weight'] = outsample['weight']/float(rep*div)
        print(outsample)

        if div != 1:
            rnd = np.random.default_rng()
            digit = rnd.choice(range(100000))
            SaveSample(outsample, self.files['signal_events'].replace('SSSS', set).replace('.npy', '_{:06d}.npy'.format(digit)))
        else:
            SaveSample(outsample, self.files['signal_events'].replace('SSSS', set))

    def LoadEvents(self, samplename):
        df = LoadSample(self.files[samplename].replace('SSSS', self.sets[self.set][0]))
        setattr(self, 'df_'+samplename, df)

    def Aeff(self):
        ### Calculates the effective area of the selection.
        ### Unit: cm^2

        # ## From the whole simulation set:
        # self.LoadEvents('sim')
        # df = self.df_sim

        ## From the signal-events-only simulation set:
        self.LoadEvents('signal_events')
        df = self.df_signal_events
        df['weight'] = df['weight']/SunSolidAngle(df['dist'])*4*np.pi

        dfnu    = df[df['PDG'] ==  14]
        dfnubar = df[df['PDG'] == -14]

        ## Eight bins per decade:
        ebins=np.logspace(0, 9, 73)
        ## This only works with cos(theta) bins:
        zbins = np.linspace(-1, 1, 19)
        dOmega = 2*np.pi*np.diff(zbins)

        aeff_hist_nu    = list(np.histogram2d(np.cos(dfnu['trueZen']),    dfnu['trueE'],    bins=[zbins, ebins], weights=   dfnu['weight']))
        aeff_hist_nubar = list(np.histogram2d(np.cos(dfnubar['trueZen']), dfnubar['trueE'], bins=[zbins, ebins], weights=dfnubar['weight']))

        for aeff_hist, name in zip([aeff_hist_nu, aeff_hist_nubar], ['nu', 'nubar']):
            aeff = aeff_hist[0]

            for z in range(len(aeff)):
                aeff[z] /= np.diff(ebins)

            aeff = aeff.T
            for e in range(len(aeff)):
                aeff[e] /= dOmega
            aeff = aeff.T

            ## These should be exactly the same:
            # aeff_e = np.array([np.sum(aeff.T[z]*np.diff(zbins)) for z in range(len(aeff.T))])
            # aeff_z = np.array([np.sum(aeff[e]  *np.diff(ebins)) for e in range(len(aeff))])
            # print('{:.2e} {:.2e}'.format(np.sum(aeff_e*np.diff(ebins)), np.sum(aeff_z*np.diff(zbins))))

            aeff_hist[0] = aeff
            np.save(self.selectionpath+'Aeff/'+str(self.ID)+'_'+name+'_Aeff.npy', tuple(aeff_hist))

    def LoadAeff(self):
        ### Unit: cm^2

        for name in ['nu', 'nubar']:
            Aeff = np.load(self.files['Aeff'].replace('Aeff.npy', name+'_Aeff.npy'), allow_pickle=True)
            setattr(self, 'Aeff'+name, Aeff)
