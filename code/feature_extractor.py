import numpy as np
import pandas as pd
import pyeeg
import scipy.stats as stats
import scipy.signal as signal


# time_feature 계산
def time_feature_extractor(df):
    mean = np.mean(df, axis=1)
    std = np.std(df, axis=1)
    skewness = stats.skew(df, axis=1)
    kurotosis = stats.kurtosis(df, axis=1)

    hjorth_mob = df.apply(lambda x: pyeeg.hjorth(list(x))[0], axis=1)
    hjorth_com = df.apply(lambda x: pyeeg.hjorth(list(x))[1], axis=1)
    hurst = df.apply(lambda x: pyeeg.hurst(x), axis=1)
    dfa = df.apply(lambda x: pyeeg.dfa(x), axis=1)
    hfd = df.apply(lambda x: pyeeg.hfd(list(x), 8), axis=1)
    pfd = df.apply(lambda x: pyeeg.pfd(x), axis=1)

    time_feature_df = pd.DataFrame({"mean": mean, "std": std, "skewness": skewness, "kurotosis": kurotosis, "hjorth_mob": hjorth_mob,
                                    "hjorth_com": hjorth_com, "hurst": hurst, "dfa": dfa, "hfd": hfd, "pfd": pfd})
    return time_feature_df


# frequency_feature 계산
def frequency_feature_extractor(df):
    # sampling frequency 정의
    sampling_frequency = 128
    sampling_period = 1/128

    frequency_feature_list = []
    for channel in df.index:
        row = df.loc[channel, :]
        pxx_den = signal.welch(row, fs=sampling_frequency, nperseg=128)[1]
        pxx_spec = signal.welch(row, fs=sampling_frequency, nperseg=128, scaling="spectrum")[1]

        max_psd = np.sqrt(np.max(pxx_den))
        max_freq = np.argmax(pxx_den)
        rms = np.sqrt(np.max(pxx_spec))
        frequency_feature_list.append([max_psd, max_freq, rms])

    frequency_feature_df = pd.DataFrame(frequency_feature_list, columns=["MAX_PSD", "MAX_freq", "RMS"])
    return frequency_feature_df


# time-frequency-feature 계산 (13-42Hz 이용)
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def time_frequency_feature_extractor(df):
    # sampling frequency 정의
    sampling_frequency = 128
    sampling_period = 1/128

    # band-pass frequency 정의
    low_frequency = 13
    high_frequency = 42

    time_frequency_list = []
    beta_gamma_max_freq_list = []
    for channel in df.index:
        row = df.loc[channel, :].to_numpy()

        # band-pass
        bandpass_signal = butter_bandpass_filter(row, low_frequency, high_frequency, sampling_frequency, order=15)
        f, t, zxx = signal.stft(bandpass_signal, fs=sampling_frequency, nperseg=128, noverlap=0)

        # time_frequency_extract
        time_frequency_list.append(pd.DataFrame(np.abs(zxx)))
        beta_gamma_max_freq_list.append(np.argmax(signal.welch(bandpass_signal, fs=sampling_frequency, nperseg=128)[1]))

    return beta_gamma_max_freq_list, time_frequency_list
