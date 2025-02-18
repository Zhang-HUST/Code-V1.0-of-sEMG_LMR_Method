from utils.common_utils import get_feature_list, is_string_in_list
from utils.processing_tools.processing import data_nomalize
from utils.feature_extraction_tools.feature_extraction_utils import *
from utils.common_params import tar_fs


def emg_feature_extraction(emg_sample, emg_channels, emg_feature_type, fea_normalize_method, fea_normalize_level):
    """
    Supported emg features
        1.1. 15 time domain features['VAR', 'RMS', 'IEMG', 'MAV', 'LOG', 'WL', 'AAC', 'DASDV',
                                    'ZC', 'WAMP', 'MYOP', 'SSC', 'SSI', 'KF', 'TM3']
        1.2. 9 frequency domain features['FR', 'MNP', 'TOP', 'MNF', 'MDF', 'PKF', 'SM1', 'SM2', 'SM3']
        1.3. 1 time-frequency domain feature['WENT']
        1.4. 3 entropy features['AE', 'SE’, 'FE']
    """

    # 1. Gets the name of the feature list for all channels, shape: num*len(emg_channels)*len(feature_type)
    all_emg_fea_names = get_feature_list(emg_channels, emg_feature_type, concatenation=False)

    # 2. emg feature extraction, shape: num*[len(emg_channels)*len(feature_type)]
    all_emg_feas = []
    for i in range(emg_sample.shape[0]):
        temp1 = []
        for j in range(emg_sample.shape[2]):
            sub_emg_data = emg_sample[i, :, j]
            sub_emg_feas = emg_feature_extraction_alone(sub_emg_data, emg_feature_type)
            temp1.extend(np.array(sub_emg_feas))
        all_emg_feas.append(temp1)
    all_emg_feas = np.array(all_emg_feas)

    # 3. Feature set normalization
    emg_feas_normalize = data_nomalize(all_emg_feas, fea_normalize_method, fea_normalize_level)

    # 4. Sort the feature set as: num*len(channels) *len(feature_type)
    all_emg_feas_pre = np.reshape(emg_feas_normalize,
                                  (emg_feas_normalize.shape[0], len(emg_channels), len(emg_feature_type)))
    print('       emg_feas.shape: ', all_emg_feas_pre.shape)
    return all_emg_feas_pre, all_emg_fea_names


def emg_feature_extraction_alone(x, feature_type):
    emg_feas = []
    th = np.mean(x) + 3 * np.std(x)

    # Time domain features
    if is_string_in_list(feature_type, 'VAR'):
        fea_var = np.var(x)
        emg_feas.append(fea_var)
    if is_string_in_list(feature_type, 'RMS'):
        fea_rms = np.sqrt(np.mean(x ** 2))
        emg_feas.append(fea_rms)
    if is_string_in_list(feature_type, 'IEMG'):
        fea_iemg = np.sum(abs(x))
        emg_feas.append(fea_iemg)
    if is_string_in_list(feature_type, 'MAV'):
        fea_mav = np.sum(np.absolute(x)) / len(x)
        emg_feas.append(fea_mav)
    if is_string_in_list(feature_type, 'LOG'):
        fea_log = np.exp(np.sum(np.log10(np.absolute(x))) / len(x))
        emg_feas.append(fea_log)
    if is_string_in_list(feature_type, 'WL'):
        fea_wl = np.sum(abs(np.diff(x)))
        emg_feas.append(fea_wl)
    if is_string_in_list(feature_type, 'AAC'):
        fea_aac = np.sum(abs(np.diff(x))) / len(x)
        emg_feas.append(fea_aac)
    if is_string_in_list(feature_type, 'DASDV'):
        fea_dasdv = math.sqrt((1 / (len(x) - 1)) * np.sum((np.diff(x)) ** 2))
        emg_feas.append(fea_dasdv)
    if is_string_in_list(feature_type, 'ZC'):
        fea_zc = get_emg_feature_zc(x, th)
        emg_feas.append(fea_zc)
    if is_string_in_list(feature_type, 'WAMP'):
        fea_wamp = get_emg_feature_wamp(x, th)
        emg_feas.append(fea_wamp)
    if is_string_in_list(feature_type, 'MYOP'):
        fea_myop = get_emg_feature_myop(x, th)
        emg_feas.append(fea_myop)
    if is_string_in_list(feature_type, 'SSC'):
        fea_ssc = get_emg_feature_ssc(x, threshold=0.000001)
        emg_feas.append(fea_ssc)
    if is_string_in_list(feature_type, 'SSI'):
        fea_ssi = get_emg_feature_ssi(x)
        emg_feas.append(fea_ssi)
    if is_string_in_list(feature_type, 'KF'):
        fea_kf = get_emg_feature_kf(x)
        emg_feas.append(fea_kf)
    if is_string_in_list(feature_type, 'TM3'):
        fea_tm3 = get_emg_feature_tm3(x)
        emg_feas.append(fea_tm3)

    # Frequency domain features
    frequency, power = get_signal_spectrum(x, fs=tar_fs)
    if is_string_in_list(feature_type, 'FR'):
        fea_fr = get_emg_feature_fr(frequency, power)  # Frequency ratio
        emg_feas.append(fea_fr)
    if is_string_in_list(feature_type, 'MNP'):
        fea_mnp = np.sum(power) / len(power)  # Mean power
        emg_feas.append(fea_mnp)
    if is_string_in_list(feature_type, 'TOP'):
        fea_top = np.sum(power)  # Total power
        emg_feas.append(fea_top)
    if is_string_in_list(feature_type, 'MNF'):
        fea_mnf = get_emg_feature_mnf(frequency, power)  # Mean frequency
        emg_feas.append(fea_mnf)
    if is_string_in_list(feature_type, 'MDF'):
        fea_mdf = get_emg_feature_mdf(frequency, power)  # Median frequency
        emg_feas.append(fea_mdf)
    if is_string_in_list(feature_type, 'PKF'):
        fea_pkf = frequency[power.argmax()]  # Peak frequency
        emg_feas.append(fea_pkf)
    if is_string_in_list(feature_type, 'SM1'):
        fea_sm1 = get_emg_feature_sm1(x, fs_global)  # Spectral Moment 1
        emg_feas.append(fea_sm1)
    if is_string_in_list(feature_type, 'SM2'):
        fea_sm2 = get_emg_feature_sm2(x, fs_global)  # Spectral Moment 2
        emg_feas.append(fea_sm2)
    if is_string_in_list(feature_type, 'SM3'):
        fea_sm3 = get_emg_feature_sm3(x, fs_global)  # Spectral Moment 3
        emg_feas.append(fea_sm3)

    # Time-frequency domain features
    if is_string_in_list(feature_type, 'WENT'):
        fea_went = get_emg_feature_went(x)  # Wavelet energy
        emg_feas.append(fea_went)

    # Entropy features
    if is_string_in_list(feature_type, 'AE'):
        fea_ae = get_emg_feature_AE(x, m=3, r=0.15)  # Approximate entropy
        emg_feas.append(fea_ae)
    if is_string_in_list(feature_type, 'SE'):
        fea_se = get_emg_feature_SE(x, m=3, r=0.15)  # Sample entropy
        emg_feas.append(fea_se)
    if is_string_in_list(feature_type, 'FE'):
        fea_fe = get_emg_feature_FE(x, m=3, r=0.15, n=2)  # Fuzzy entropy
        emg_feas.append(fea_fe)

    return emg_feas
