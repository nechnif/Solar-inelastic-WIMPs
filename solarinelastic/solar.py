import numpy as np
import pandas as pd

from .dataio import *

#--- Solar stuff -------------------------------------------------------
sun_radius = 6.9e10              # [cm]
mean_dist  = 1.496e13            # [cm]


def DeltaPhi(phi1, phi2):
    ### The angle in a 1D circle for two positions on the circumference.

    delta = np.abs(np.abs(phi1-phi2) - (2*np.pi if np.abs(phi1-phi2)>np.pi else 0))
    return delta

def SunCoordinates(mjd, i3=True):
    ### There are two equivalent ways to calculate the sun coordinates from
    ### a given MJD. One uses the icecube environment, the other the
    ### astropy package. Both versions are given here. The return array is
    ### of the form np.array( [zenith], [azimuth] )

    if i3 == True:
        from icecube import astro
        sundir = astro.sun_dir(mjd)
        return sundir

    else:
        ## The astropy package is not important globally because it relies
        ## on some files that might be missing on some systems (e.g. the
        ## cluster) and cause everything to crash upon import.
        import astropy.coordinates as coord
        import astropy.units as u
        from astropy.time import Time
        t = Time(mjd, format='mjd')
        # print(t.mjd)    # modified julian date
        # print(t.iso)    # human readable time
        Sun = coord.get_sun(t)
        icl = coord.EarthLocation(lat=-89.9944*u.deg, lon=-62.6081*u.deg, height=883.9*u.m)
        astro_altaz = Sun.transform_to(coord.AltAz(location=icl, obstime=t))
        astroi3_zen = 90*u.deg - astro_altaz.alt
        astroi3_az  = (90*u.deg - astro_altaz.az - icl.lon).wrap_at(360*u.deg)

        sundir = np.array(astroi3_zen, astroi3_az)
        return sundir

def SunSolidAngle(sun_distance):
    ### The sun solid angle depends on the seasonal variation of the
    ### distance sun-earth.
    ### Please provide sun_distance in [cm].

    return 2*np.pi * (1-np.cos(sun_radius/sun_distance))

def CutAngle(sun_distance):
    ### Calculates the angle in a right triangle where the hypotenuse
    ### is unknown. Used as a cut angle for neutrinos with true directions
    ### from the sun.
    ### Please provide sun_distance in [cm].

    return np.arctan(sun_radius/sun_distance)

def SpaceAngle(zen1, azi1, zen2, azi2):
    ### Orthodrome (smallest distance between two points on a sphere,
    ### see Vincenty's formulae).
    ### zen1/azi1: sun directions
    ### zen2/azi2: object directions

    # psi = np.arccos(
    #       np.cos(azi2)*np.sin(zen2)*np.cos(azi1)*np.sin(zen1)
    #     + np.sin(azi2)*np.sin(zen2)*np.sin(azi1)*np.sin(zen1)
    #     + np.cos(zen2)*np.cos(zen1)
    # )

    ## Faster:
    psi = np.arccos(
          np.sin(zen2)*np.sin(zen1)*np.cos(azi2-azi1)
        + np.cos(zen2)*np.cos(zen1)
    )
    return psi

def FilterCutAngle(sample):
    ### For signal samples: Cuts away all neutrinos that are not
    ### coming from within the disk of the sun.

    psi_cut = CutAngle(sample['dist'])
    # print(psi_cut)
    return sample[sample['true_sun_psi'] <= psi_cut]

def CreateSunFile(len, iso_range, outname):
    ### Pre-creates a file with random MJDs and corresponding coordinates
    ### of the sun. This is done to save time later in the oversampling
    ### process of the signal samples.
    ### iso_range to be given as e.g. [2014-01-01, 2015-01-01]

    ## The astropy package is not important globally because it relies
    ## on some files that might be missing on some systems (e.g. the
    ## cluster) and cause everything to crash upon import.
    ## Same with icecube, which needs a load-in of the environment first.
    import astropy.coordinates as coord
    import astropy.units as u
    from astropy.time import Time
    from icecube import astro

    print(iso_range)
    mjd_range = [
        float(Time(iso_range[0], format='iso').mjd),
        float(Time(iso_range[1], format='iso').mjd)
    ]
    print(mjd_range)

    ## Create uniformly sampled MJDs between two dates, e.g. spring and autumn equinoxes (03/20/14 - 09/23/14):
    mjds = np.random.uniform(mjd_range[0], mjd_range[1], len)

    ## Calculate locations of the Sun (may take a while):
    sundir = np.array(astro.sun_dir(mjds)).T
    sundir = np.insert(sundir, 0, mjds, axis=1)
    sundir_df = pd.DataFrame(sundir)
    sundir_df.columns = ['mjd', 'sun_zen', 'sun_azi']
    ## Calculate distance earth-sun (may also take a while):
    dist = coord.get_sun(Time(sundir_df['mjd'], format='mjd')).distance.km
    sundir_df['dist'] = dist
    # print(sundir_df)
    np.save(outname, sundir_df.to_records(index=False))
