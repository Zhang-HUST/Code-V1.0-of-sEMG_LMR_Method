import numpy as np
from scipy.signal import detrend
from sklearn.preprocessing import MinMaxScaler
from pywt import WaveletPacket, threshold


# =====================================================================
# Auxiliary functions
# =====================================================================
def Energy(x):
    """! Computes the energy of a signal. The energy is essentially the
    magnitude of the signal (inner product of x with itself).

    @param x Input signal as numpy array (1D)

    @return The energy of the input signal x
    """
    return np.dot(x, x)


def EuclideanNorm(x):
    """! Computes the Euclidean norm (p-norm with p=2) of the input
    1D vector (signal) x.

    @param x The input signal (numpy float 1D ndarray)

    @return The norm of the input signal as a float scaler
    """
    return np.linalg.norm(x)


def mad(x):
    """! Estimates the Median Absolute Deviation (MAD). MAD is defined to be
    the median of the absolute difference between the input X and median(X).

    @param x The input signal (1D ndarray)
    @return The median absolute deviation of the input signal

    @note More details on the MAD can be found on the Wikipedia page:
    please see https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    return 1.482579 * np.median(np.abs(x - np.median(x)))


def meanad(x):
    """! Estimates the Mean Absolute Deviation (MeanAD). MeanAD is defined to
    be the mean of the absolute difference between the input X and mean(X).

    @param x The input signal (1D ndarray)
    @return The mean absolute deviation of the input signal
    """
    return 1.482579 * np.mean(np.abs(x - np.mean(x)))


def grad_g_fun(x, thr=1):
    return (x >= thr) * 1 + (x <= -thr) * 1 + (np.abs(x) <= thr) * 0


def NearestEvenInteger(n):
    """! Returns the nearest even integer to number n.

    @param n Input number for which one requires the nearest even integer

    @return The even nearest integer to the input number
    """
    if n % 2 == 0:
        res = n
    else:
        res = n-1
    return res


def DyadicLength(x):
    """! Returns the length and the dyadic length of the input 1D array x.

    @param x The input signal (float 1D ndarray)

    @return Returns the length m and the least power of 2 greater than m

    @note This function has been taken from the pyYAWT package
    (https://pyyawt.readthedocs.io/_modules/pyyawt/denoising.html).
    """
    m = x.shape[0]
    j = np.ceil(np.log(m) / np.log(2.)).astype('i')
    return m, j


def SoftHardThresholding(x, thr=1, method='s'):
    """! Performs either a soft or hard thresholding on the input signal x.

    @param x The 1D input signal
    @param thr The threshold value (float, default=1)
    @param method A string that indicates if either the soft or the hard
    thresholding is being used (default=soft, s for soft, h for hard)

    @return Returns the thresholded signal
    """
    if method.lower() == 'h':
        res = x * (np.abs(x) > thr)
    elif method.lower() == 's':
        res = ((x >= thr) * (x - thr) + (x <= -thr) * (x + thr)
               + (np.abs(x) <= thr) * 0)
    else:
        print("Thresholding method not found! Choose s (soft) or h (hard)")
        res = None
    return res


# =====================================================================
# Main wavelets denoising class
# =====================================================================
class WaveletPacketDenoising:
    """! Denoising class """
    def __init__(self,
                 normalize=False,
                 wavelet='db5',
                 level=3,
                 recon_mode='symmetric',
                 thr_mode='garrote',
                 method='universal',
                 energy_perc=0.9):
        """! Constructor of WaveletDenoising class.
        @param normalize Enables the normalization of the input signal into
        [0, 1] (bool)

        @param wavelet Wavelet's type, e.g. 'db1', 'haar' (str)

        @param level   Decomposition level (n), the default value is 1.

        @param recon_mode Reconstruction signal extension mode. This can be
        one of the following: 'smooth', 'symmetric', 'antisymmetric', 'zero',
        'constant', 'periodic', 'reflect' (str).

        @param thr_mode Type of thresholding ('soft' or 'hard') (str)

        @param method Type of threshold determination method. This can be one
        of the following:
        - 'universal' - The threshold is the sqrt(2*length(signal))*mad
        - 'sqtwolog' - The threshold is the sqrt(2*length(signal))
        - 'stein' - Stein's unbiased risk estimator
        - 'heurstein' - Heuristic of rigsure
        - 'energy' - Computes the energy of the coefficients and retains a
        predefined percentage of it.

        @param energy_perc Energy level retained in the coefficients when one
        uses the energy thresholding method.

        @return Nothing

        @note For more details on modes, see:
        https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#ref-modes
        """
        self.wavelet = wavelet
        self.level = level
        self.method = method
        self.thr_mode = thr_mode
        self.recon_mode = recon_mode
        self.energy_perc = energy_perc
        self.normalize = normalize

        # Check if level is None and set it to 1
        if level is None:
            self.nlevel = 1
        else:
            self.nlevel = level
        self.normalized_data = None
        self.wp = None

    def fit(self, signal):
        """! This method executes the denoising algorithm by invoking all the
        necessary methods.
            i. Preprocessing
            ii. Multilevel Wavelet Decomposition
            iii. Denoise the coefficients

        @param signal A noisy input signal

        @return A denoised version of the input signal

        """
        tmp_signal = signal.copy()
        tmp_signal = self.Preprocess(tmp_signal)
        self.wp = self.WavPacketTransform(tmp_signal)
        denoised_signal = self.Denoise()
        return denoised_signal

    # ********************************************************************
    # Preprocessing methods
    def Preprocess(self, signal):
        """! This method removes all the trends from the input signal, such as
        DC currents. Furthermore, it can normalize the input signal into the
        interval [0, 1].

        @param signal The input signal (1D ndarray)
        @param normalize A flag that determines if the input signal will be
                         normalized into [0, 1] or not (bool)

        @return A detrended signal (and normalized in case the normalization
        flag is set to True)

        """
        # Remove all the unnecessary trends (DC, etc)
        xhat = detrend(signal)

        # Normalize the data (bring them into [0, 1]) and keep the scaler for
        # future use or inversing the normalization
        if self.normalize:
            self.scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
            xhat = self.scaler.fit_transform(xhat.reshape(-1, 1))[:, 0]
            self.normalized_data = xhat.copy()
        return xhat

    # ********************************************************************
    # Multilevel wavelet decomposition
    def WavPacketTransform(self, signal):
        """! Performs a Wavelet multilevel decomposition on the input signal.
        This method first will estimate the power of two nearest to the length
        of the signal. Then it will check the values of level (n), and in case
        level is set to zero it will compute the optimal level using the
        function dwt_max_level. Finally, it will perform the decomposition
        on the signal[:size], where size is a power of two closest to the
        length of the input signal.

        @param signal The input signal (1D ndarray)

        @return The wavelet coefficients as a list. First appears the
        approximation coefficient cA and then the detail coefficients in the
        order cD_n, cD_n-1, ..., cD2, cD_1

        """
        # Find the nearest even integer to input signal's length
        size = NearestEvenInteger(signal.shape[0])
        wp = WaveletPacket(data=signal[:size], wavelet=self.wavelet, mode=self.recon_mode, maxlevel=self.nlevel)
        return wp

    # ********************************************************************
    # Main denoising method
    def Denoise(self):
        """! Denoises the input signal based on its wavelet coefficients. This
        method first computes the SD of the detail coefficients, then
        determines the appropriate threshold, applies the threshold on the
        coefficients, and then proceed in the signal's denoising. Finally,
        if the normalization flag is True, it renormalizes the input signal
        back to its original space.

        @return The denoised signal
        """
        # Estimate the SD of the wavelet coefficients
        wp = self.wp
        for node in wp.get_level(wp.maxlevel, 'freq'):
            if node.path.endswith('a'):  # 低频节点
                continue
            else:  # 高频节点
                coeff = node.data
                sigma = 1.4825 * np.median(np.abs(coeff))
                thr = self.DetermineThreshold(coeff / sigma, self.energy_perc) * sigma
                processed_coeff = threshold(coeff, value=thr, mode=self.thr_mode)
                node.data = processed_coeff
        # 小波包重构
        denoised_signal = wp.reconstruct(update=False)
        # Inverse normalization in case the input signal was normalized
        if self.normalize:
            denoised_signal = self.scaler.inverse_transform(
                    denoised_signal.reshape(-1, 1))[:, 0]
        return denoised_signal

    # ********************************************************************
    # Thresholding methods
    def DetermineThreshold(self, signal, energy_perc=0.9):
        """! Determines the value of the threshold. It offers five different
        methods:
        - 'universal' - The threshold is the sqrt(2*length(signal))*mad
        - 'sqtwolog' - The threshold is the sqrt(2*length(signal))
        - 'stein' - Stein's unbiased risk estimator
        - 'heurstein' - Heuristic implementation of rigsure
        - 'energy' - Computes the energy of the coefficients and retains a
        predefined percentage of it.
        The method is defined in the constructor (see self.method).

        @param signal The input signal (1D ndarray)
        @param energy_perc The percentage of energy to be retained in the case
        one uses the energy method to determine the threshold (float)

        @return The value of the threshold (float)

        @note In case the method provided by the user does not exist, this
        method will fall back to the 'universal' method.
        """
        thr = 0.0
        if self.method == 'universal':
            thr = self.UniversalThreshold(signal)
        elif self.method == 'sqtwolog':
            thr = self.UniversalThreshold(signal, sigma=False)
        elif self.method == 'stein':
            thr = self.SteinThreshold(signal)
        elif self.method == 'heurstein':
            thr = self.HeurSteinThreshold(signal)
        elif self.method == 'energy':
            thr = self.EnergyThreshold(signal, perc=energy_perc)
        else:
            print("No such method detected!")
            print("Set back to default (universal thresholding)!")
            thr = self.UniversalThreshold(signal)
        return thr

    def UniversalThreshold(self, signal, sigma=True):
        """! Universal threshold
        @param signal Input signal (1D ndarray of floats)
        @param sigma If true multiplies the term sqrt(2xlog(m)) with the MAD
        value of the input signal (m is the input signal's length)

        @return A float scaler representing the threshold value
        """
        m = signal.shape[0]
        if sigma:
            sd = mad(signal)
            # sigma = meanad(signal)
        else:
            sd = 1.0
        # thr = sd * np.sqrt((2*np.log(m)) / m)
        thr = sd * np.sqrt(2 * np.log(m))
        return thr

    def SteinThreshold(self, signal):
        """! An implementation of Stein's unbiased rist estimator based on
        PyYAWT package.

        @param signal The input signal

        @return The value of the threshold for the input signal
        """
        m = signal.shape[0]
        sorted_signal = np.sort(np.abs(signal))**2
        c = np.linspace(m-1, 0, m)
        s = np.cumsum(sorted_signal) + c * sorted_signal
        risk = (m - (2.0 * np.arange(m)) + s) / m
        ibest = np.argmin(risk)
        thr = np.sqrt(sorted_signal[ibest])
        return thr

    def HeurSteinThreshold(self, signal):
        """! A heuristic implementation of Stein's unbiased rist estimator
        based on PyYAWT package.

        @param signal The input signal

        @return The value of the threshold for the input signal
        """
        m, j = DyadicLength(signal)
        magic = np.sqrt(2 * np.log(m))
        eta = (np.linalg.norm(signal)**2 - m) / m
        critical = j**(1.5)/np.sqrt(m)
        if eta < critical:
            thr = magic
        else:
            thr = np.min((self.SteinThreshold(signal), magic))
        return thr

    def SquareRootLogThreshold(self, signal):
        """! It computes a threshold for the input signal. The threshold value
        is given by the squared-root of 2 x log(m), where m is the length of
        the input signal.

        Square-root of 2log threshold
        @param signal Input signal (1D ndarray of floats)

        @return A float scaler representing the threshold value
        """
        m = len(signal)
        thr = np.sqrt(2.0 * np.log(m))
        return thr

    def EnergyThreshold(self, signal, perc=0.1):
        """! Energy-based threshold method. It estimates a threshold value for
        the input signal based on the signal's energy.

        @param signal Input signal (1D ndarray)
        @param perc   Energy retained percentage (float)

        @return A float scaler representing the threshold value
        """
        tmp_signal = np.sort(np.abs(signal))[::-1]
        energy_thr = perc * Energy(tmp_signal)
        energy_tmp = 0
        for sig in tmp_signal:
            energy_tmp += sig**2
            if energy_tmp >= energy_thr:
                thr = sig
                break
        return thr