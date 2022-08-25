#!/usr/bin/env python3
from typing import Dict, List, Tuple
import numpy as np
from scipy.special import erf


def bandwidth_scott(dataset: np.ndarray, weights: np.ndarray,
dim: int) -> float or np.ndarray:
    ''' Automatic bandwidth determination using the Silverman Rule.

        Calculation of the kernel bandwidth(s) for each dimension of the
        input data. For 2D data it is assumend that the bandwidths are
        not correlated.

        Parameters
        ----------
        dataset: ndarray
            1D or 2D array of the form (# of dimensions, # of points).
        weights: ndarray
            1D array with weights corresponding to each point in the
            dataset.
        dim: int
            Dimension of dataset.

        Returns
        -------
        b: float or ndarray
            The kernel bandwidth(s) of the input data.

        Notes
        -----
        D.W. Scott: Multivariate Density Estimation: Theory, Practice,
        and Visualization, John Wiley & Sons, New York, Chicester, 1992.

    '''

    # N = np.size(dataset.T, axis=0)
    N_eff = np.sum(weights)**2/np.sum(weights**2)

    std = np.std(dataset, axis=1)
    b = N_eff**(-1/(dim+4))*std

    return b

def bandwidth_silverman(dataset: np.ndarray, weights: np.ndarray,
dim: int) -> float or np.ndarray:
    ''' Automatic bandwidth determination using the Silverman Rule.

        Calculation of the kernel bandwidth(s) for each dimension of the
        input data. For 2D data it is assumend that the bandwidths are
        not correlated.

        Parameters
        ----------
        dataset: ndarray
            1D or 2D array of the form (# of dimensions, # of points).
        weights: ndarray
            1D array with weights corresponding to each point in the
            dataset.
        dim: int
            Dimension of dataset.

        Returns
        -------
        b: float or ndarray
            The kernel bandwidth(s) of the input data.

        Notes
        -----
        B.W. Silverman: Density estimation for statistics and data
        analysis, volume 26. CRC press, 1986.

    '''

    # N = np.size(dataset.T, axis=0)
    N_eff = np.sum(weights)**2/np.sum(weights**2)

    std = np.std(dataset, axis=1)
    b = (N_eff*(dim+2)/4)**(-1/(dim+4))*std

    return b


def kernel_gaussian(position: np.ndarray, values: float or np.ndarray,
weights: float or np.ndarray, bandwidth: float or np.ndarray, dim: int,
hasbounds: bool, bounds: np.ndarray, bound_method: str,
threshold: float, speed: float) -> np.ndarray:
    ''' Gaussian kernel creation.

        Creates a gaussian kernel for a single data point and returns
        the values of the kernel at the given positions.

        Parameters
        ----------
        positions: ndarray
            The position of the kernel, in the form array([m, n]).
            E.g. in 1D: array([0]), in 2D: array([0, 1]).
        values: float or ndarray
            Datapoints in the form array([[x-values], [y-values]]).
            Equivalent to the expectation values of the individual
            kernels.
        weights: float or ndarray
            The weights to values. Lengths must be equal to number of
            values/ value-pairs.
        bandwidth: float or ndarray
            Bandwdith of dataset. If 2D: array([bx, by]).
        dim: int
            Dimension of value (of dataset that value originates from).
        hasbounds: bool
            Indicates whether the dataset has boundaries.
        bounds: ndarray
            Boundaries of the dataset.
        bound_method: str
            Method to handle the boundaries.
        threshold: float
            Threshold of relevant kernels out of total kernels for which
            boundary correction is applied. This prevents over-
            estimation of borders where few kernels are present.
        speed:
            Take into account only kernels that are within a reasonable
            range around the position to be evaluated. That range
            calculates to binwidth*speed. So speed should be > 1.
            Calculation speed-up is significant for values < 10.
            Careful though, this can cause the evaluation to be Zero
            if the data is sparse.

        Returns
        -------
        Kernels: ndarray
            Gaussian kernels for all input values evaluated at the given
            position. Length of retun array is equal to number of
            values/ value-pairs.

    '''

    if speed != 0:
        a = np.where(
            (values[0]>=position[0]-bandwidth[0]*speed)&
            (values[0]<=position[0]+bandwidth[0]*speed)
        )
        values_ = values.T[a].T
        weights_ = weights[a]
        b = np.where(
            (values_[1]>=position[1]-bandwidth[1]*speed)&
            (values_[1]<=position[1]+bandwidth[1]*speed)
        )
        values_ = values_.T[b].T
        weights_ = weights_[b]
    else:
        values_, weights_  = values, weights

    w = np.sum(weights_)/np.sum(weights)

    norm = 1/np.sqrt((2*np.pi)**dim)

    if not hasbounds:
        kernel = (
            norm
          * 1/(np.prod(bandwidth) if dim==2 else bandwidth[0])
          * weights_
          * np.exp(-1/2. * (
              ((position[0]-values_[0])**2/bandwidth[0]**2)
           + (((position[1]-values_[1])**2/bandwidth[1]**2) if dim==2 else 0)
            ))
        )
    elif hasbounds and bound_method == 'truncate':
        if w < threshold:
            norm_trunc = 1
        else:
            norm_trunc_x = 1/(
                1/2 * (1 + erf((bounds[1][0]-values_[0])/bandwidth[0]/np.sqrt(2)))
              - 1/2 * (1 + erf((bounds[0][0]-values_[0])/bandwidth[0]/np.sqrt(2)))
            )
            norm_trunc_y = 1/(
               (1/2 * (1 + erf((bounds[1][1]-values_[1])/bandwidth[1]/np.sqrt(2)))
              - 1/2 * (1 + erf((bounds[0][1]-values_[1])/bandwidth[1]/np.sqrt(2))))
            )
            norm_trunc = norm_trunc_x * (norm_trunc_y if dim==2 else 1)

        kernel = (
            norm
          * norm_trunc
          * 1/(np.prod(bandwidth) if dim==2 else bandwidth[0])
          * weights_
          * np.exp(-1/2. * (
              ((position[0]-values_[0])**2/bandwidth[0]**2)
           + (((position[1]-values_[1])**2/bandwidth[1]**2) if dim==2 else 0)
            ))
        )


    elif hasbounds and bound_method == 'reflect':
        posss = []
        if w < threshold:
            posss.append(position)
        else:
            t = 4
            posss.append(position)
            if (position[0] <= bounds[0][0]+bandwidth[0]*t):
                posss.append(np.array([2*bounds[0][0]-position[0],
                                       position[1]]))
            if (position[0] >= bounds[1][0]-bandwidth[0]*t):
                posss.append(np.array([2*bounds[1][0]-position[0],
                                       position[1]]))
            if (position[1] <= bounds[0][1]+bandwidth[1]*t):
                posss.append(np.array([position[0],
                                       2*bounds[0][1]-position[1]]))
            if (position[1] >= bounds[1][1]-bandwidth[1]*t):
                posss.append(np.array([position[0],
                                       2*bounds[1][1]-position[1]]))

        kernels = []
        for pos in posss:
            # print(pos)
            kernels.append(
                norm
              * 1/(np.prod(bandwidth) if dim==2 else bandwidth[0])
              * weights_
              * np.exp(-1/2. * (
                   ((pos[0]-values_[0])**2/bandwidth[0]**2)
                + (((pos[1]-values_[1])**2/bandwidth[1]**2) if dim==2 else 0)))
            )
            kernel = np.sum(kernels)

    # print(kernel)
    return kernel


class GaussianKDE(object):
    ''' Kernel density estimation with gaussian kernels for bounded data

        A class for kernel density estimation (KDE) of a 1D or 2D
        dataset, as a smooth alternative to histograms. This class is
        similar to the scipy.stats.gaussian_kde, but features additional
        methods that allow for a better estimation of bounded data.

        Currently only gaussian kernels are implemented. Automatic
        bandwidth calculation is done by means of the Silverman Rule.
        Reflection is used for boundary correction.

        Parameters
        ----------
        dataset : array-like
            1D or 2D array of the form (# of dimensions, # of points).
        bw_method: str
            Currently, only "silverman" is supported.
        bounds: array-like
            A pair or pair of pairs in the format [lower, higher],
            according to the dimension of the dataset.

        Attributes
        ----------
        dim: int
            Dimension of dataset.
        values: array-like
            The dataset that GaussianKDE was initialized with.
        weights: array-like
            The weights to values. Lengths must be equal to number of
            values/ value-pairs.
        bandwdith: float or np.ndarray
            Kernel-bandwidth. Float in case of 1D, array with shape (2,)
            in case of 2D.
        bounds: ndarray
            Boundaries of the dataset.
        hasbounds: bool
            Set True when bounds are given as parameters, else False.

        Methods
        -------
        set_bandwidth
        evaluate
        integrate_box

        Notes
        -----
        Raffaela Busse, November 2021
        raffaela.busse@uni-muenster.de

    '''

    def __init__(self, dataset: np.ndarray,
    weights: int or np.ndarray=0, bw_method: str='silverman',
    bounds: list or np.ndarray=None):

        dataset = np.array(dataset)
        # Determine dimension:
        dim = dataset.ndim
        if dim == 1:
            # Expand dataset to 2D and fill second dimension with nans:
            dataset = np.array([dataset, np.full_like(dataset, np.nan)])
        elif dim == 2:
            pass
        else:
            raise ValueError(
                'Only 1D and 2D datasets are supported'
            )

        # Normalize weights:
        if isinstance(weights, int):
            weights = np.ones(len(dataset.T))
        weightnorm = len(weights)/np.sum(weights)
        weights = weights * weightnorm

        ## Sort the dataset by first column:
        sort = np.vstack([dataset, weights]).T
        sort = sort[sort[:, 0].argsort()].T
        dataset = np.vstack([sort[0], sort[1]])
        weights = sort[2]

        # Calculate bandwidth:
        if bw_method == 'scott':
            bandwidth = bandwidth_scott(dataset, weights, dim)
        elif bw_method == 'silverman':
            bandwidth = bandwidth_silverman(dataset, weights, dim)
        else:
            raise ValueError(
                'No such bandwith method available. Try "scott" or'
                'silverman.'
            )

        if bounds:
            self.hasbounds = True
            bounds = np.array(bounds, dtype=float)
            if dim == 1:
                bounds = np.pad(
                    bounds, (0, 2), 'constant', constant_values=np.nan
                ).reshape(2, 2).T
            else:
                pass
        else:
            self.hasbounds = False

        # Assign core attributes:
        self.dim       = dim
        self.values    = dataset
        self.weights   = weights
        self.bandwidth = bandwidth
        self.bounds    = bounds

    def set_bandwidth(self, bx: float, by: float=np.nan):
        ''' Set the bandwidth manually.

            Parameters
            ----------
            bx, by: float
                Bandwidth for first and (optional) second dimension.

        '''

        if self.dim == 2 and np.isnan(by):
            by = bx
        self.bandwidth = np.array([bx, by])

    def evaluate(self, positions: int or float or list or np.ndarray
    , bound_method: str='truncate', threshold: float=0.0,
    speed: float=0) -> np.ndarray:
        ''' Evaluation of KDE object at given positions

            Creates the kernels and evaluates them at the given
            positions. If bounds are provided, the kernels are
            reflected at the boundaries.

            Parameters
            ----------
            positions: scalar or array-like
                Can either be a single point, or a whole range/ grid of
                points. Needs to match the dimension of the KDE. E.g. in
                1D: positions = 0
                    positions = [0, 1, 2, 3]
                2D:
                    positions = [0, 0]
                    positions = [[x-positions], [y-positions]]
            bound_method: str
                Method to handle the boundaries.
            threshold: float
                Threshold of relevant kernels out of total kernels for which
                boundary correction is applied. This prevents over-
                estimation of borders where few kernels are present.
            speed:
                Take into account only kernels that are within a reasonable
                range around the position to be evaluated. That range
                calculates to binwidth*speed. So speed should be > 1.
                Calculation speed-up is significant for values < 10.
                Careful though, this can cause the evaluation to be Zero
                if the data is sparse.

            Returns
            -------
            eval: np.ndarray
                Sum over all kernels at the requested positions.

            Raises
            ------
            ValueError
                If the dimension of the input positions does not match
                the dimension of the initialized KDE object.

        '''

        # Bring positions into a neat shape:
        if (self.dim==1 and (isinstance(positions, int)
        or isinstance(positions, float))):
            positions = np.array([float(positions)])
            positions = np.array(
                [positions, np.full_like(positions, np.nan)]
            )
        elif self.dim==1 and np.array(positions).ndim==1:
            positions = np.array(
                [positions, np.full_like(positions, np.nan)]
            )
        elif self.dim==2 and np.shape(positions)==(2,):
            positions = np.reshape(positions, (2,1))
        elif (self.dim==2 and np.array(positions).ndim==2 and
        len(positions[0])==len(positions[1])):
            positions = np.array(positions)
        else:
            raise ValueError(
                'Dimension of input does not match dimension of KDE'
            )

        eval = []
        for pos in positions.T:
            # Check if position is out of bounds:
            if self.hasbounds:
                if not (  pos[0]>=self.bounds[0][0] and
                          pos[0]<=self.bounds[1][0]):
                    eval.append(0.)
                    continue
                elif self.dim==2 and not (
                          pos[1]>=self.bounds[0][1] and
                          pos[1]<=self.bounds[1][1]):
                    eval.append(0.)
                    continue
                else:
                    pass

            kernels = np.sum(kernel_gaussian(pos, self.values,
                self.weights, self.bandwidth, self.dim,
                hasbounds=self.hasbounds, bounds=self.bounds,
                bound_method=bound_method, threshold=threshold,
                speed=speed))
            eval.append(kernels)

        eval = np.array(eval)
        eval = 1/len(self.values.T)*eval

        return eval

    def integrate_box(self, box: list or np.ndarray,
    delta: int=50, bound_method: str='truncate') -> float:
        ''' Evaluation of KDE object at given positions

            Creates the kernels and evaluates them at the given
            positions. If bounds are provided, the kernels are
            reflected at the boundaries.

            Parameters
            ----------
            box: array-like
                A pair or pair of pairs in the format [lower, higher],
                according to the dimension of the dataset, defining the
                desired boundaries of the integral.
            delta: int
                "Precision" (dx) of the integration.

            Returns
            -------
            intgrl: float
                Sum over all KDE values evaluated on the grid of given
                positions, normalized by the bin area (determined by
                delta).

        '''

        # A finer grid will give better precision, but will also take
        # a lot of time:

        if self.dim == 1:
            bins = np.linspace(box[0], box[1], delta)
            binarea = bins[1]-bins[0]
            eval = self.evaluate(bins, bound_method)
        elif self.dim == 2:
            xbins = np.linspace(box[0][0], box[1][0], delta)
            ybins = np.linspace(box[0][1], box[1][1], delta)
            binarea = (xbins[1]-xbins[0])*(ybins[1]-ybins[0])
            xx, yy = np.meshgrid(xbins, ybins)
            positions = np.vstack([xx.ravel(), yy.ravel()])
            eval = self.evaluate(positions, bound_method)

        intgrl = np.sum(eval)*binarea

        return intgrl
