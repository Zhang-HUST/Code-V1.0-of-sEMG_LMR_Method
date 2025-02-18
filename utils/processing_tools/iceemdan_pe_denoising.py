from PyEMD import CEEMDAN
from PyEMD import EMD, EEMD
from pyentrp.entropy import permutation_entropy
from pywt import threshold
import numpy as np

thread_number = 12  # Number of threads used for EMD class decomposition


def mad(x):
    return 1.482579 * np.median(np.abs(x - np.median(x)))


def svd_denoise_signal(noisy_signal, svd_threshold_type='mutation_value'):
    N = len(noisy_signal)
    indices = np.arange(N // 2)[:, None] + np.arange(N // 2 + 1)
    A = noisy_signal[indices]
    U, S_values, V = np.linalg.svd(A)
    S = np.zeros((U.shape[0], V.shape[0]))
    np.fill_diagonal(S, S_values)

    # SVD mutation point
    if svd_threshold_type == 'mutation_value':
        diff_S_left = np.abs(np.diff(S_values[0:-1]))
        diff_S_right = np.abs(np.diff(S_values[1:]))
        window_means = (diff_S_left + diff_S_right) / 2
        diff_S = np.abs(np.diff(window_means))
        max_diff_S_index = np.argmax(diff_S)
        n_svd_thr = max_diff_S_index + 1
        print('...mutation_value_index of svd: %d ...' % n_svd_thr)
    elif svd_threshold_type == 'mean_value':
        mean_value = np.mean(S_values)
        larger_than_mean = S_values[S_values > mean_value]
        last_larger_index = np.where(S_values == larger_than_mean[-1])[0][-1]
        n_svd_thr = last_larger_index + 1
        print('... mean_value_index of svd: %d ...' % n_svd_thr)
    else:
        raise ValueError('unsupported svd_threshold_type')

    # Signal reconstruction
    X = np.zeros_like(A)
    for i in range(n_svd_thr):
        X += np.outer(U[:, i], np.outer(S[i, i], V[i, :]))

    # Calculate the sum of the anti-diagonal elements
    anti_diagonal_means = []
    num_anti_diagonals = min(X.shape)
    for i in range(num_anti_diagonals):
        temp_matrix = X[:i + 1, :i + 1]
        anti_diagonal_mean = np.trace(np.fliplr(temp_matrix)) / len(temp_matrix)
        anti_diagonal_means.append(anti_diagonal_mean)
    for i in range(num_anti_diagonals):
        temp_matrix = X[i:, i + 1:]
        anti_diagonal_mean = np.trace(np.fliplr(temp_matrix)) / len(temp_matrix)
        anti_diagonal_means.append(anti_diagonal_mean)

    denoised_signal = np.array(anti_diagonal_means)

    return denoised_signal


class ICEEMDANPEDenoising:
    def __init__(self, decomposition_method='emd', denoise_method='svd', svd_threshold_type='mutation_value'):
        assert decomposition_method in ['emd', 'eemd', 'iceemdan']
        self.decomposition_method = decomposition_method
        self.denoise_method = denoise_method
        # iceemdan
        self.trails, self.epsilon = 100, 0.005
        # pe
        self.pe_length, self.pe_order, self.pe_tao = 2048, 6, 1
        assert self.denoise_method in ['garrote_threshold', 'svd']
        if self.denoise_method == 'svd':
            self.svd_threshold_type = svd_threshold_type

    def fit(self, signal):
        print('...1. %s...' % self.decomposition_method)
        IMFs, residue, nIMFs = self.get_CEEMD_residue(signal)
        print('...number of total IMFs: %d ...' % nIMFs)
        print('...2. calculate_pe...')
        K, pe_values = self.calculate_pe(IMFs)
        print('...number of noisy IMFs: %d ...' % K)
        print('...pe_values:', pe_values, '...')
        print('...3. denoise by %s ...' % self.denoise_method)
        denoised_signal = self.denoise(IMFs, K)
        return denoised_signal

    def get_CEEMD_residue(self, signal):
        if self.decomposition_method == 'emd':
            ceemd = EMD()
            ceemd.emd(signal)
        elif self.decomposition_method == 'eemd':
            ceemd = EEMD(trials=self.trails, noise_width=0.05, parallel=True, processes=thread_number)
            ceemd.eemd(signal)
        else:
            ceemd = CEEMDAN(trials=self.trails, epsilon=self.epsilon, parallel=True, processes=thread_number)
            ceemd.extrema_detection = "parabol"
            ceemd.ceemdan(signal)

        IMFs, residue = ceemd.get_imfs_and_residue()
        IMFs = IMFs.T
        nIMFs = IMFs.shape[1]
        return IMFs, residue, nIMFs

    def calculate_pe(self, signal):
        m = 1
        prev_pe = None
        pe_values = []

        while True:
            # Reconstruction of the first m-order IMFs
            if m == 1:
                reconstructed_imf = np.squeeze(signal[:, 0])
            else:
                reconstructed_imf = np.sum(signal[:, 0:m], axis=1)

            # The permutation entropy of the reconstructed sequence is calculated
            pe = permutation_entropy(reconstructed_imf[:min(len(signal), self.pe_length)], order=self.pe_order, delay=self.pe_tao, normalize=True)
            if prev_pe is None or pe > prev_pe:
                pe_values.append(pe)
                prev_pe = pe
                m += 1
            else:
                pe_values.append(pe)
                break

        return m - 1, pe_values

    def denoise(self, IMFs, K):
        if self.denoise_method == 'garrote_threshold':
            noise_imfs = IMFs[:, 0:K]
            signal_imfs = IMFs[:, K:]
            processd_noise_imfs = np.zeros((noise_imfs.shape[0], noise_imfs.shape[1]))
            for i in range(noise_imfs.shape[1]):
                noise = np.squeeze(noise_imfs[:, i])
                # Improvement of the generic threshold
                sigma = 1.4825 * np.median(np.abs(noise))
                thr = sigma * np.sqrt(2 * np.log(noise.shape[0])) / np.log(i + 2)
                processed_noise = threshold(noise, value=thr, mode='garrote')
                processd_noise_imfs[:, i] = processed_noise
            reconstructed_signal = np.sum(np.hstack((processd_noise_imfs, signal_imfs)), axis=1)
        else:
            N = len(IMFs)
            if N % 2 == 0:
                noise_imfs = IMFs[:, 0:K]
                signal_imfs = IMFs[:, K:]
            else:
                noise_imfs = IMFs[0:N - 1, 0:K]
                signal_imfs = IMFs[0:N - 1, K:]

            processd_noise_imfs = np.zeros((noise_imfs.shape[0], noise_imfs.shape[1]))
            for i in range(noise_imfs.shape[1]):
                noise = np.squeeze(noise_imfs[:, i])
                processd_noise_imfs[:, i] = svd_denoise_signal(noise, svd_threshold_type=self.svd_threshold_type)
            reconstructed_signal = np.sum(np.hstack((processd_noise_imfs, signal_imfs)), axis=1)

        return reconstructed_signal
