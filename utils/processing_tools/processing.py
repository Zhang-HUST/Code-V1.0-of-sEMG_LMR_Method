import math
import numpy as np
import librosa
from utils.processing_tools.wavelet_denoising import WaveletDenoising
from utils.processing_tools.wavelet_packet_denoising import WaveletPacketDenoising
from utils.processing_tools.iceemdan_pe_denoising import ICEEMDANPEDenoising
from scipy.signal import butter, lfilter, iirfilter
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

"""Signal resampling"""


def signal2d_resampling(rawdata, raw_fs, tar_fs):
    channel = rawdata.shape[1]
    x = []
    for i in range(channel):
        temp = librosa.resample(rawdata[:, i], orig_sr=raw_fs, target_sr=tar_fs, res_type='kaiser_best',
                                fix=True, scale=False)
        x.append(temp)
    x = np.array(x).transpose()
    return x


"""emg filters: notch filter, band pass filter, low pass filter"""


class EMGFilteringTools:
    def __init__(self, fs):
        self.fs = fs

    def Implement_Notch_Filter(self, data, imf_band, imf_freq, order, filter_type='butter'):
        # Required input defintions are as follows;
        # time:   Time between samples
        # band:   The bandwidth around the centerline freqency that you wish to filter
        # freq:   The centerline frequency to be filtered
        # ripple: The maximum passband ripple that is allowed in db
        # order:  The filter order.  For FIR notch filters this is best set to 2 or 3,
        #         IIR filters are best suited for high values of order.  This algorithm
        #         is hard coded to FIR filters
        # filter_type: 'butter', 'bessel', 'cheby1', 'cheby2', 'ellip'
        # data:         the data to be filtered
        fs = self.fs

        nyq = fs / 2.0
        freq, band = imf_freq, imf_band
        low = freq - band / 2.0
        high = freq + band / 2.0
        low = low / nyq
        high = high / nyq
        b, a = iirfilter(order, [low, high], btype='bandstop', analog=False, ftype=filter_type)
        filtered_data = lfilter(b, a, data)

        return filtered_data

    def butter_bandpass(self, lowcut, highcut, order):
        fs = self.fs
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')

        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, order):
        b, a = self.butter_bandpass(lowcut, highcut, order)
        y = lfilter(b, a, data)

        return y

    def butter_lowpass(self, cutoff, order):
        fs = self.fs
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        return b, a

    def butter_lowpass_filter(self, data, cutoff, order):
        b, a = self.butter_lowpass(cutoff, order)
        y = lfilter(b, a, data)

        return y


"""emg denoising"""


class Signal2dDenoise:
    def __init__(self, data, denoise_method):
        self.data = data
        self.denoise_method = denoise_method
        assert self.denoise_method in ['rawdata', 'WD-GT', 'WPD-GT', 'EMD-PE-GT', 'EMD-PE-SVD',
                                       'EEMD-PE-GT', 'EEMD-PE-SVD', 'ICEEMDAN-PE-GT', 'ICEEMDAN-PE-SVD']
        print('emg denoising method: %s' % self.denoise_method)
        if self.denoise_method == 'WD-GT':
            self.wd = WaveletDenoising(normalize=False, wavelet='db5', level=3, thr_mode='garrote',
                                       selected_level=None, method="universal", energy_perc=0.90)
        elif self.denoise_method == 'WPD-GT':
            self.wpd = WaveletPacketDenoising(normalize=False, wavelet='db5', level=3, thr_mode='garrote',
                                              method="universal", energy_perc=0.90)
        elif self.denoise_method == 'EMD-PE-GT':
            self.iceemdan_pe_denoising = ICEEMDANPEDenoising(decomposition_method='emd',
                                                             denoise_method='garrote_threshold')
        elif self.denoise_method == 'EMD-PE-SVD':
            self.iceemdan_pe_denoising = ICEEMDANPEDenoising(decomposition_method='emd',
                                                             denoise_method='svd', svd_threshold_type='mutation_value')
        elif self.denoise_method == 'EEMD-PE-GT':
            self.iceemdan_pe_denoising = ICEEMDANPEDenoising(decomposition_method='eemd',
                                                             denoise_method='garrote_threshold')
        elif self.denoise_method == 'EEMD-PE-SVD':
            self.iceemdan_pe_denoising = ICEEMDANPEDenoising(decomposition_method='eemd',
                                                             denoise_method='svd', svd_threshold_type='mutation_value')
        elif self.denoise_method == 'ICEEMDAN-PE-GT':
            self.iceemdan_pe_denoising = ICEEMDANPEDenoising(decomposition_method='iceemdan',
                                                             denoise_method='garrote_threshold')
        else:
            self.iceemdan_pe_denoising = ICEEMDANPEDenoising(decomposition_method='iceemdan',
                                                             denoise_method='svd', svd_threshold_type='mutation_value')
        self.denoised_data = None

    def forward(self):
        print('Signal Length: %d' % len(self.data))
        if self.denoise_method == 'rawdata':
            self.denoised_data = self.data
        elif self.denoise_method == 'WD-GT':
            temp = []
            for i in range(self.data.shape[1]):
                print('Prosessing Signal Channel: %d' % (i + 1))
                temp.append(self.wd.fit(self.data[:, i]))
            self.denoised_data = np.array(temp).T
        elif self.denoise_method == 'WPD-GT':
            temp = []
            for i in range(self.data.shape[1]):
                print('Prosessing Signal Channel: %d' % (i + 1))
                temp.append(self.wpd.fit(self.data[:, i]))
            self.denoised_data = np.array(temp).T
        else:
            temp = []
            for i in range(self.data.shape[1]):
                print('Prosessing Signal Channel: %d' % (i + 1))
                denoised_signal = self.iceemdan_pe_denoising.fit(self.data[:, i])
                temp.append(denoised_signal)
            self.denoised_data = np.array(temp).T

        return self.denoised_data


"""Multi-modal multi-channel data normalization methods, 
which support the methods: 'min-max', 'max-abs', 'positive_negative_one; Normalization level: 'matrix', 'rows'"""


def data_nomalize(data, normalize_method, normalize_level):
    if normalize_level == 'matrix':
        if normalize_method == 'min-max':
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data)
            normalized_data = (data - np.min(scaler.data_min_)) / (np.max(scaler.data_max_) - np.min(scaler.data_min_))
        elif normalize_method == 'positive_negative_one':
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            normalized_data = ((data - np.min(scaler.data_min_)) / (
                    np.max(scaler.data_max_) - np.min(scaler.data_min_))) * 2 - 1
        elif normalize_method == 'max-abs':
            scaler = MaxAbsScaler()
            scaler.fit(data)
            normalized_data = (data / np.maximum(np.abs(np.max(scaler.data_max_)),
                                                 np.abs(np.min(scaler.data_min_)))) * scaler.scale_
        else:
            raise ValueError('Unsupported normalize_method!')
    elif normalize_level == 'rows':
        if normalize_method == 'min-max':
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        elif normalize_method == 'positive_negative_one':
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        elif normalize_method == 'max-abs':
            scaler = MaxAbsScaler()
            scaler.fit(data)
            normalized_data = scaler.transform(data)
        else:
            raise ValueError('Unsupported normalize_method!')
    else:
        raise ValueError('Unsupported normalize_level!')

    return normalized_data


"""Sample set segmentation based on sliding overlapping window sampling"""


def overlapping_windowing_movement_classification(emg_data_act, movement, window, step):
    length = math.floor((np.array(emg_data_act).shape[0] - window) / step)
    emg_sample, movement_label = [], []
    for j in range(length):
        sub_emg_sample = emg_data_act[step * j:(window + step * j), :]
        emg_sample.append(sub_emg_sample)
        movement_label.append(movement)

    return np.array(emg_sample), np.array(movement_label)


"""Overlapping window segmentation"""


def movement_classification_sample_segmentation(movement, emg_data_act, window, step):
    print('       Overlapping window segmentation...')
    emg_sample, movement_label = overlapping_windowing_movement_classification(emg_data_act, movement, window, step)
    print('       emg_sample.shape: ', emg_sample.shape, ', movement_label.shape: ', movement_label.shape)

    return emg_sample, movement_label


def get_emg_act_signal(movement, emg_raw_data, status_label):
    if movement in ['WAK', 'UPS', 'DNS']:
        emg_data_act = emg_raw_data
    else:
        indices_a = np.where(status_label == 'A')
        emg_data_act = emg_raw_data[indices_a[0], :]

    return emg_data_act


"""EMG Filtering"""


def emg_filtering(input_data, fs):
    # FIR notch filter of order 3
    imf_band, imf_freq, order_1 = 2, 50, 3
    # 15-450 Hz 7th order Butterworth bandpass filtering
    lowcut, highcut, order_2 = 15, 450, 7

    cows, rows = input_data.shape[0], input_data.shape[1]
    # 1. De-dc component
    emg_de_direct_data = np.zeros((cows, rows))
    for i in range(rows):
        emg_de_direct_data[:, i] = input_data[:, i] - np.mean(input_data[:, i])
    emg_filtering = EMGFilteringTools(fs)
    # 2. notch filter
    emg_inf_data = np.zeros((cows, rows))
    for i in range(rows):
        emg_inf_data[:, i] = emg_filtering.Implement_Notch_Filter(emg_de_direct_data[:, i], imf_band, imf_freq, order_1)
    # 3. Butterworth bandpass filtering
    emg_bpf_data = np.zeros((cows, rows))
    for i in range(rows):
        emg_bpf_data[:, i] = emg_filtering.butter_bandpass_filter(emg_inf_data[:, i], lowcut, highcut, order_2)

    return emg_bpf_data

