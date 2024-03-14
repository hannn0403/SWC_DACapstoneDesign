import numpy as np
import pandas as pd
from scipy import stats
import pyeeg
import pretty_midi as pm

# 피아노 오른손자리 옥타브 주파수
piano_frequency = np.array([261.63, 277.18, 293.67, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25])

# major 코드별 스케일
C_major_scale = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4']
C_sharp_major_scale = ["Db4", 'Eb4', 'F4', 'Gb4', 'Ab4', 'Bb4', "C5"]
D_major_scale = ['D4', 'E4', 'F#4', 'G4', 'A4', 'B4', 'C#5']
D_sharp_major_scale = ['Eb4', 'F4', 'G4', 'Ab4', 'Bb4', 'C5', 'D5']
E_major_scale = ['E4', 'F#4', 'G#4', 'A4', 'B4', 'C#5', "B#5"]
F_major_scale = ['F4', 'G4', 'A4', 'A#4', 'C5', 'D5', "E5"]
F_sharp_major_scale = ['F#4', 'G#4', 'A#4', 'B4', 'C#5', 'D#5', 'F5']
G_major_scale = ['G4', 'A4', 'B4', 'C5', 'D5', 'E5', "F#5"]
G_sharp_major_scale = ['Ab4', 'Bb4', 'C5', 'Db5', 'Eb5', 'F5', 'G5']
A_major_scale = ['A4', 'B4', 'C#5', 'D5', 'E5', 'F#5', "G#5"]
A_sharp_major_scale = ['Bb4', 'C5', 'D5', 'Eb5', 'F5', 'G5', 'A5']
B_major_scale = ['B4', 'C#5', 'D#5', 'E5', 'F#5', 'G#5', "A#5"]
major_scale_list = [C_major_scale, C_sharp_major_scale, D_major_scale, D_sharp_major_scale, E_major_scale, F_major_scale, F_sharp_major_scale,
                    G_major_scale, G_sharp_major_scale, A_major_scale, A_sharp_major_scale, B_major_scale]
major_scale_names = ["C major", "C sharp major", "D major", "D sharp major", "E major", "F major", "F sharp major",
                     "G major", "G sharp major", "A major", "A sharp major", "B major"]

# minor 코드별 스케일
C_minor_scale = ['C4', 'D4', 'Eb4', 'F4', 'G4', 'Ab4', 'Bb4']
C_sharp_minor_scale = ['C#4', 'D#4', 'E4', 'F#4', 'G#4', 'A4', 'B4']
D_minor_scale = ['D4', 'E4', 'F4', 'G4', 'A4', 'Bb4', 'C5']
D_sharp_minor_scale = ['D#4', 'F4', 'F#4', 'G#4', 'A#4', 'B4', 'C#5']
E_minor_scale = ['E4', 'F#4', 'G4', 'A4', 'B4', 'C5', 'D5']
F_minor_scale = ['F4', 'G4', 'Ab4', 'Bb4', 'C5', 'Db5', 'Eb5']
F_sharp_minor_scale = ['F#4', 'G#4', 'A4', 'B4', 'C#5', 'D5', 'E5']
G_minor_scale = ['G4', 'A4', 'Bb4', 'C5', 'D5', 'Eb5', 'F5']
G_sharp_minor_scale = ['G#4', 'A#4', 'B4', 'C#5', 'D#5', 'E5', 'F#5']
A_minor_scale = ['A4', 'B4', 'C5', 'D5', 'E5', 'F5', 'G5']
A_sharp_minor_scale = ['A#4', 'C5', 'C#5', 'D#5', 'F5', 'F#5', 'G#5']
B_minor_scale = ['B4', 'C#5', 'D5', 'E5', 'F#5', 'G5', 'A5', 'B5']
minor_scale_list = [C_minor_scale, C_sharp_minor_scale, D_minor_scale, D_sharp_minor_scale, E_minor_scale, F_minor_scale, F_sharp_minor_scale,
                    G_minor_scale, G_sharp_minor_scale, A_minor_scale, A_sharp_minor_scale, B_minor_scale]
minor_scale_names = ["C minor", "C sharp minor", "D minor", "D sharp minor", "E minor", "F minor", "F sharp minor",
                     "G minor", "G sharp minor", "A minor", "A sharp minor", "B minor"]

major_or_minor = [major_scale_list, minor_scale_list]
major_or_minor_names = [major_scale_names, minor_scale_names]


def scaling_index(frequency):
    scale_frequency = piano_frequency[0] + (piano_frequency[-1] - piano_frequency[0]) * (frequency - 13) / (42 - 13)
    idx = np.abs(piano_frequency - scale_frequency).argmin()
    return idx


def octave_control(note_name, number):
    octave = str(int(note_name[-1]) + number)
    note_name = note_name[:-1] + octave
    return note_name


def chord_and_melody(valence_label, stft_list, max_psf_list):
    # 12개의 channel에서 측정한 stft와 max_psf의 평균 계산
    max_stft_signal_mean = np.array([])
    max_psf_mean = 0

    for stft, mpsf in zip(stft_list, max_psf_list):
        max_stft_signal = [stft.loc[:, column].argmax() for column in stft.columns]
        if max_stft_signal_mean.shape == (0,):
            max_stft_signal_mean = np.array(max_stft_signal)
        else:
            max_stft_signal_mean += np.array(max_stft_signal)
        max_psf_mean += mpsf

    max_stft_signal_mean = max_stft_signal_mean / 12
    max_psf_mean = max_psf_mean / 12

    # Chord 결정
    scale_idx = scaling_index(max_psf_mean)
    if valence_label.index[0] == 1:
        key = 0
        base_octave = 1
    elif valence_label.index[0] == 0:
        key = 1
        base_octave = 0

    return key, scale_idx, base_octave, max_stft_signal_mean


def tempo(arousal_label):
    # Tempo 조절
    index = arousal_label.index
    value = arousal_label.values
    if index[0] == 0:
        interval = 0.4 + 0.05 * value[0]
        interpol_interval = interval / 2
    elif index[0] == 1:
        if value[0] == 12:
            interval = 0.4
            interpol_interval = 0.2
        else:
            interval = 0.4 + 0.05 * value[1]
            interpol_interval = interval / 2

    return interval, interpol_interval


def volume_and_octave(eeg_signal):
    # 볼륨 조절
    high_kurtosis = pd.read_csv("../music_control_feature/high_kurtosis.csv", header=None)
    high_kurtosis_estimator = stats.gaussian_kde(high_kurtosis.mean(axis=0), bw_method='silverman')
    low_kurtosis = pd.read_csv("../music_control_feature/low_kurtosis.csv", header=None)
    low_kurtosis_estimator = stats.gaussian_kde(low_kurtosis.mean(axis=0), bw_method='silverman')

    start_index = 0
    sliding_window = 1

    kurtosis_list = pd.DataFrame([])
    while start_index + 10 <= 60:
        temp = eeg_signal.iloc[:, 128 * start_index: 128 * (start_index + 10)]
        kurtosis_list = pd.concat([kurtosis_list, pd.Series(stats.kurtosis(temp, axis=1))], axis=1)
        start_index += sliding_window
    kurtosis_list = np.mean(kurtosis_list, axis=0)

    volume_list = []
    for i in range(len(kurtosis_list)):
        K1 = high_kurtosis_estimator(kurtosis_list.iloc[i])
        K2 = low_kurtosis_estimator(kurtosis_list.iloc[i])
        volume_list.append(0.5 if K1 > K2 else -0.5)

    # Key 조절
    high_dfa = pd.read_csv("../music_control_feature/high_dfa.csv", header=None)
    high_dfa_estimator = stats.gaussian_kde(high_dfa.mean(axis=0), bw_method='silverman')
    low_dfa = pd.read_csv("../music_control_feature/low_dfa.csv", header=None)
    low_dfa_estimator = stats.gaussian_kde(low_dfa.mean(axis=0), bw_method='silverman')

    start_index = 0
    sliding_window = 30

    dfa_list = pd.DataFrame([])
    while start_index + sliding_window <= 60:
        temp = eeg_signal.iloc[:, 128 * start_index: 128 * (start_index + sliding_window)]
        dfa_list = pd.concat([dfa_list, temp.apply(lambda x: pyeeg.dfa(x), axis=1)], axis=1)
        start_index += sliding_window
    dfa_list = np.mean(dfa_list, axis=0)

    key_list = []
    for i in range(len(dfa_list)):
        K1 = high_dfa_estimator(dfa_list.iloc[i])
        K2 = low_dfa_estimator(dfa_list.iloc[i])
        key_list.append(2 if K1 > K2 else -2)

    return volume_list, key_list


def left_note_record(scale, pitch_idx, base_octave):
    first_note_name = octave_control(scale[pitch_idx], base_octave - 2)
    first_note_number = pm.note_name_to_number(first_note_name)

    # 두번째 음부터 한 옥타브가 올라가는 경우
    if pitch_idx + 2 > 6:
        second_note_name = octave_control(scale[pitch_idx - 5], base_octave - 1)
        second_note_number = pm.note_name_to_number(second_note_name)
        third_note_name = octave_control(scale[pitch_idx - 3], base_octave - 1)
        third_note_number = pm.note_name_to_number(third_note_name)

    # 세번째 음부터 한 옥타브가 올라가는 경우
    elif (pitch_idx + 2 <= 6) and (pitch_idx + 4 > 6):
        second_note_name = octave_control(scale[pitch_idx + 2], base_octave - 2)
        second_note_number = pm.note_name_to_number(second_note_name)
        third_note_name = octave_control(scale[pitch_idx - 3], base_octave - 1)
        third_note_number = pm.note_name_to_number(third_note_name)

    # 옥타브가 올라가지 않는 경우
    else:
        second_note_name = octave_control(scale[pitch_idx + 2], base_octave - 2)
        second_note_number = pm.note_name_to_number(second_note_name)
        third_note_name = octave_control(scale[pitch_idx + 4], base_octave - 2)
        third_note_number = pm.note_name_to_number(third_note_name)

    return first_note_number, second_note_number, third_note_number


def convert_signal_to_music(inst, eeg_signal, arousal_label, valence_label, stft_list, max_psf_list):
    # 악기 결정
    pretty_midi_music = pm.PrettyMIDI()
    instrument_program = pm.instrument_name_to_program(inst)
    instrument = pm.Instrument(program=instrument_program)

    # Chord and Key melody 결정
    scale_name_list = []
    key, scale_idx, base_octave, max_stft_signal_mean = chord_and_melody(valence_label, stft_list, max_psf_list)
    scale = major_or_minor[key][scale_idx]
    scale_name = major_or_minor_names[key][scale_idx]
    scale_name_list.append(scale_name)

    # Tempo 결정
    start_time = 0.0
    interval, interpol_interval = tempo(arousal_label)

    # Volume / Key 조절
    right_base_volume, left_base_volume = 80, 60
    volume_list, key_list = volume_and_octave(eeg_signal)
    volume_list = [0] * 10 + volume_list

    # note 및 간격 list 초기화
    right_note = []
    right_volume = []
    right_start_time = []
    right_end_time = []

    left_note = []
    left_volume = []
    left_start_time = []
    left_end_time = []

    # 음악 만들기 -> 오른손과 왼손에서 입력할 note, 시작 시간, 끝 시간을 List up
    pre_pitch_idx = scaling_index(max_stft_signal_mean[0]) % 7
    for idx in range(0, len(max_stft_signal_mean)):

        # volume 및 key 조절 설정
        right_base_volume = right_base_volume + volume_list[idx]
        left_base_volume = left_base_volume + volume_list[idx]

        if idx % 30 == 0 and (0 < idx // 30 < 2):
            scale_idx = (scale_idx + key_list[idx // 30]) % 12
            scale = major_or_minor[key][scale_idx]
            scale_name = major_or_minor_names[key][scale_idx]
            scale_name_list.append(scale_name)

        # stft값을 통해서 해당하는 가장 가까운 음을 찾는다.
        pitch_idx = scaling_index(max_stft_signal_mean[idx]) % 7

        # 오른손 - Interpolate Note를 추가하는 경우
        while np.abs(pre_pitch_idx - pitch_idx) > 1:
            if pre_pitch_idx > pitch_idx:
                pre_pitch_idx -= 1
            elif pre_pitch_idx < pitch_idx:
                pre_pitch_idx += 1
            note_name = octave_control(scale[pre_pitch_idx], base_octave)
            note_number = pm.note_name_to_number(note_name)

            # 오른손 list에 추가
            right_note.append(note_number)
            right_volume.append(right_base_volume)
            right_start_time.append(start_time)
            right_end_time.append(start_time + interpol_interval)

            # 왼손 list에 추가
            if start_time // 1 != (start_time + interpol_interval) // 1:
                left_note.append(left_note_record(scale, pitch_idx, base_octave))
                left_volume.append(left_base_volume)
                left_start_time.append(start_time)
                left_end_time.append(start_time + interpol_interval)

            start_time += interpol_interval

        # 오른손 일반적인 음을 추가하는 경우
        note_name = octave_control(scale[pitch_idx], base_octave)
        note_number = pm.note_name_to_number(note_name)

        # 오른손 List 들에 추가
        right_note.append(note_number)
        right_volume.append(right_base_volume)
        right_start_time.append(start_time)
        right_end_time.append(start_time + interval)

        # 왼손 일반적인 음을 추가하는 경우
        if start_time // 1 != (start_time + interval) // 1:
            left_note.append(left_note_record(scale, pitch_idx, base_octave))
            left_volume.append(left_base_volume)
            left_start_time.append(start_time)
            left_end_time.append(start_time + interval)

        start_time += interval
        pre_pitch_idx = pitch_idx

    # List up이 완료되면, 하나씩 객체에 입력
    # 오른손
    for i in range(len(right_note)):
        note = pm.Note(velocity=int(right_volume[i]), pitch=right_note[i], start=right_start_time[i],
                       end=right_end_time[i])
        instrument.notes.append(note)

        # 왼손 -> 첫음 채워 넣기
        # 3개의 음을 동시에 치기 위해서 각각의 음들을 first, second, third list에 저장하고 동일한 인덱스의 위치에 저장한다.
        note = pm.Note(velocity=int(left_volume[0]), pitch=left_note[0][0], start=0, end=left_start_time[1])
        instrument.notes.append(note)
        note = pm.Note(velocity=int(left_volume[0]), pitch=left_note[0][1], start=0, end=left_start_time[1])
        instrument.notes.append(note)
        note = pm.Note(velocity=int(left_volume[0]), pitch=left_note[0][2], start=0, end=left_start_time[1])
        instrument.notes.append(note)

    for i in range(len(left_note)):

        # 마지막 인덱스인 경우
        if i == len(left_note) - 1:
            # 3개의 음을 동시에 치기 위해서 각각의 음들을 first, second, third list에 저장하고 동일한 인덱스의 위치에 저장한다.
            note = pm.Note(velocity=int(left_volume[i]), pitch=left_note[i][0], start=left_start_time[i],
                           end=right_end_time[-1])
            instrument.notes.append(note)
            note = pm.Note(velocity=int(left_volume[i]), pitch=left_note[i][1], start=left_start_time[i],
                           end=right_end_time[-1])
            instrument.notes.append(note)
            note = pm.Note(velocity=int(left_volume[i]), pitch=left_note[i][2], start=left_start_time[i],
                           end=right_end_time[-1])
            instrument.notes.append(note)

        # 마지막 인덱스가 아닌 경우
        else:
            # 3개의 음을 동시에 치기 위해서 각각의 음들을 first, second, third list에 저장하고 동일한 인덱스의 위치에 저장한다.
            note = pm.Note(velocity=int(left_volume[i]), pitch=left_note[i][0], start=left_start_time[i],
                           end=left_start_time[i + 1])
            instrument.notes.append(note)
            note = pm.Note(velocity=int(left_volume[i]), pitch=left_note[i][1], start=left_start_time[i],
                           end=left_start_time[i + 1])
            instrument.notes.append(note)
            note = pm.Note(velocity=int(left_volume[i]), pitch=left_note[i][2], start=left_start_time[i],
                           end=left_start_time[i + 1])
            instrument.notes.append(note)

    pretty_midi_music.instruments.append(instrument)
    return scale_name_list, pretty_midi_music
