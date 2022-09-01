import numpy as np
import pandas as pd


#--- Solar stuff -------------------------------------------------------
sun_radius = 6.9e10              # [cm]
mean_dist  = 1.496e13            # [cm]


def DeltaPhi(phi1, phi2):
    ### The angle in a 1D circle for two positions on the circumference.
    delta = np.abs(np.abs(phi1-phi2) - (2*np.pi if np.abs(phi1-phi2)>np.pi else 0))
    return delta

def SunCoordinates(mjd, i3=True):
    ## There are two equivalent ways to calculate the sun coordinates from
    ## a given MJD. One requires the icecube environment, the other the
    ## astropy package. Both versions are given here. The return array is
    ## of the form np.array( [zenith], [azimuth] )

    if i3 == True:
        from icecube import astro
        sundir = astro.sun_dir(mjd)
        return sundir

    else:
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
    # Faster:
    psi = np.arccos(
          np.sin(zen2)*np.sin(zen1)*np.cos(azi2-azi1)
        + np.cos(zen2)*np.cos(zen1)
    )
    return psi

def FilterCutAngle(sample):
    psi_cut = CutAngle(sample['dist'])
    # print(psi_cut)
    return sample[sample['true_sun_psi'] <= psi_cut]
