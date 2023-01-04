import sys
import numpy as np
import pandas as pd

from .dataio import *
from .solar import *
from .samplemod import *


class Selection(object):
    ''' Event selection tools.

        A class to initialize and work with different event selections
        as part of the IceCube solar inelastic WIMPs analysis.

        Parameters
        ----------
        name: str
            Name of selection, 'INT' or 'OSC'.
        set: str
            Dataset (nominal or any systematic set).
        configs: list
            Locations of the event selection config files.
        m: int
            Speed for KDE fine grid evaluation.

        Attributes
        ----------
        config: module
            Module file that contains all selection specific attributes.

        Methods
        -------
        DetermineLivetime
        OversampleBackground
        OversampleSignal
        LoadEvents
        Aeff
        LoadAeff

        Notes
        -----
        Raffaela Busse, August 2022
        raffaela.busse@uni-muenster.de

    '''

    def __init__(self, name, set, configs, m):
        if name=='INT':
            sys.path.append('/'.join(configs[0].split('/')[:-1]))
            import int_config as config
        elif name=='OSC':
            sys.path.append('/'.join(configs[1].split('/')[:-1]))
            import osc_config as config
        else:
            print('Name {} not recognized!'.format(name))
            return -1

        self.set = set
        self.config = config

        attrs = [
            ## Identification of event selection:
            'id', 'name', 'selectionpath',
            ## events and livetime:
            'eventsper9years', 'eventsperday', 'frac', 'livetime', 'total',
            ## All sets (nominal, systematics, ect.) in this selection:
            'sets',
            ## Cut angles:
            'logEcut', 'psicut', 'psicut_farsample',
            ## PDF and KDE specific attributes:
            'ebins', 'efine', 'psibins', 'psifine',
            'bandwidth', 'bounds_b', 'bounds_s', 'm', 'tag',
            'kwargs_BPDF_KDE', 'kwargs_SPDF_KDE',
            ## File locations:
            'files',
        ]

        for attr in attrs:
            setattr(self, attr, getattr(self.config, attr))

    def DetermineLivetime(self, te):
        ## Determine livetime in days.

        import astropy.coordinates as coord
        import astropy.units as u
        from astropy.time import Time

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

    def OversampleBackground(self, rep, outfile='default'):
        ## Background oversampling is needed to smooth out scrambling
        ## artefacts and to avoid empty bins in the background PDFs.

        insample = LoadSample(self.files['exp'])
        if self.id == 0.0:
            drops = [
                'pass_lowup', 'pass_muon',
                'run', 'event', 'subevent',
                'recotime',
            ]
            insample = insample.drop([d for d in drops if d in insample.columns], axis=1)
        elif self.id == 1.0:
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

        if outfile=='default':
            outfile = self.files['background_ov']
        else:
            outfile = outfile
        SaveSample(outsample, outfile)

    def OversampleSignal(self, div=1):
        ## Signal oversampling is needed for sufficient statistics.

        insample = LoadSample(self.files['sim'].replace('SSSS', self.sets[self.set][0]))
        rep = self.sets[self.set][1]
        print('oversampling MC set {} with {} reps ...'.format(self.set, rep))

        if div != 1:
            if int(rep/div) != rep/div:
                print('{} can not be divided by {}'.format(rep, div))
            else:
                rep = int(rep/div)

        ## Load sun data (which was pre-calculated to save time and and memory):
        sunlocs = LoadSample(self.files['sun']).sample(n=len(insample), replace=True)
        ## Convert distance earth-sun to [cm]:
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
            SaveSample(outsample, self.files['signal_events'].replace('SSSS', self.sets[self.set][0]).replace('.npy', '_{:06d}.npy'.format(digit)))
        else:
            SaveSample(outsample, self.files['signal_events'].replace('SSSS', self.sets[self.set][0]))

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
            np.save(self.selectionpath+'Aeff/'+str(self.id)+'_'+name+'_Aeff.npy', tuple(aeff_hist))

    def LoadAeff(self):
        ### Unit: cm^2

        for name in ['nu', 'nubar']:
            Aeff = np.load(self.files['Aeff'].replace('Aeff.npy', name+'_Aeff.npy'), allow_pickle=True)
            setattr(self, 'Aeff'+name, Aeff)
