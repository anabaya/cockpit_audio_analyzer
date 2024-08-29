import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import random
import streamlit as st
import streamlit.components.v1 as components
from pydub import AudioSegment
from pydub.generators import Sine
import numpy as np
import random
import json
import io
from scipy.io.wavfile import write
from db.database import get_dataset_by_id, insert_audio, insert_audio_alarm
import os

VOLUMES = np.linspace(0, 0.5, 10).tolist()


# Utils
@st.cache_data
def pick_value_in_range(range):
    return int(random.uniform(range[0], range[1]))


@st.cache_data
def pick_initial_value(range, key):
    return random.randint(range[0], range[1])


@st.cache_data
def pick_initial_values(range, key):
    return [random.randint(range[0], range[1]), random.randint(range[0], range[1])]


# Générer la base du temps
@st.cache_data
def generate_time_base(duration, sample_rate):
    return np.linspace(0, duration, int(sample_rate * duration), endpoint=False)


# Générer une onde sinusoïdale de base
@st.cache_data
def generate_sinus_wave(
    fundamental_freq_range,
    amplitude_range,
    t,
    phase_shift=0,
):
    fundamental_freq = pick_value_in_range(fundamental_freq_range)
    amplitude = pick_value_in_range(amplitude_range)
    return (
        VOLUMES[amplitude]
        * normalize(np.sin(2 * np.pi * fundamental_freq * t + phase_shift)),
        fundamental_freq,
        amplitude,
    )


# Générer une onde en dent de scie
@st.cache_data
def generate_sawtooth_wave(fundamental_freq_range, amplitude_range, t, phase_shift=0):
    fundamental_freq = pick_value_in_range(fundamental_freq_range)
    amplitude = pick_value_in_range(amplitude_range)
    sawtooth_wave = normalize(2 * (t * fundamental_freq % 1 + phase_shift) - 1)
    return VOLUMES[amplitude] * sawtooth_wave, fundamental_freq, amplitude


# Création des caractéristiques des harmoniques
@st.cache_data
def create_harmonics(nbr_harmonics_range, start_intensity):
    nbr_harmonics = int(pick_value_in_range(nbr_harmonics_range))
    start_intensity = pick_value_in_range(start_intensity)
    st.write(
        f"There are {nbr_harmonics} harmonics in the audio below with a starting intensity of {start_intensity}"
    )
    harmonics = []
    for index_harmonique in range(nbr_harmonics):
        harmonics.append((index_harmonique + 1, start_intensity / 2))
        start_intensity = start_intensity / 2
    return harmonics


# Ajouter des harmoniques
@st.cache_data
def add_harmonics(waveform, fundamental_freq, harmonics, t):
    for multiple, amplitude in harmonics:
        waveform += amplitude * np.sin(2 * np.pi * fundamental_freq * multiple * t)
    return waveform


# Ajouter un bruit blanc
@st.cache_data
def add_white_noise(duration, sample_rate, noise_level_range):
    noise_level = pick_value_in_range(noise_level_range)
    return VOLUMES[noise_level] * normalize(
        np.random.randn(duration * sample_rate)
    ), noise_level


# Ajouter une onde en dent de scie
@st.cache_data
def add_sawtooth_wave(fundamental_freq_range, t, amplitude_range):
    fundamental_freq = pick_value_in_range(fundamental_freq_range)
    amplitude = pick_value_in_range(amplitude_range)
    sawtooth_wave = 2 * (t * fundamental_freq % 1) - 1
    st.write(
        f"The sawtooth wave below has a fundamental frequency of : {fundamental_freq} Hz and an amplitude of {amplitude}"
    )
    return amplitude / 100 * sawtooth_wave


# Normaliser le signal
def normalize(waveform):
    if waveform is None:
        return None
    return waveform / np.max(np.abs(waveform))


# Fonction pour créer un filtre passe-bas
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


# Appliquer un filtre passe-bas
def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    return lfilter(b, a, data)


# Fonction pour générer le bruit des pales/hélices
# def generate_propeller_noise(duration, sample_rate, base_frequency, harmonics):
#     t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
#     signal = np.zeros_like(t)
#     for i, amplitude in enumerate(harmonics):
#         harmonic_freq = base_frequency * (i + 1)
#         signal += amplitude * np.sin(2 * np.pi * harmonic_freq * t)
#     return signal


# Fonction pour générer le ronronnement mécanique du moteur
# def generate_engine_hum(
#     duration, sample_rate, base_frequency, harmonics, noise_amplitude
# ):
#     t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
#     signal = np.zeros_like(t)
#     for i, amplitude in enumerate(harmonics):
#         harmonic_freq = base_frequency * (i + 1)
#         signal += amplitude * np.sin(2 * np.pi * harmonic_freq * t)
#     noise = noise_amplitude * np.random.normal(size=t.shape)
#     signal += noise
#     return signal


# Fonction pour combiner les sons
def combine_sounds(sounds):
    if len(sounds) == 0:
        return None
    combined_signal = np.sum(sounds, axis=0)
    # combined_signal /= np.max(np.abs(combined_signal))
    return combined_signal


# Fonction pour appliquer un filtre passe-bas
# def apply_lowpass_filter(data, cutoff_freq, sample_rate, order=5):
#     nyquist = 0.5 * sample_rate
#     normal_cutoff = cutoff_freq / nyquist
#     b, a = butter(order, normal_cutoff, btype="low", analog=False)
#     filtered_data = lfilter(b, a, data)
#     return filtered_data


# def generate_bip(frequency, duration, volume):
#     bip = Sine(frequency).to_audio_segment(duration=duration).apply_gain(volume)
#     return bip


# def add_silence(sequence, sequence_duration, duration_range, sample_rate):
#     duration = random.randint(min(duration_range), max(duration_range))
#     return sequence + AudioSegment.silent(duration=duration, frame_rate=sample_rate)


# def add_bip(sequence, frequency, duration, volume):
#     bip = generate_bip(frequency, duration, volume)
#     return sequence + bip


def create_alarm_audio(
    audio_duration,
    audio_sample_rate,
    nbr_alarm_range,
    alarm_duration,
    alarm_frequency,
    alarm_volume,
):
    nbr_alarm = pick_value_in_range(nbr_alarm_range)
    t = generate_time_base(audio_duration, audio_sample_rate)
    nbr_points = len(t)
    silent_points_left = nbr_points - nbr_alarm * alarm_duration * audio_sample_rate
    audio = np.zeros(nbr_points)
    starting_block = 0
    for idx_alarm in range(nbr_alarm):
        start_idx_alarm = starting_block + random.randint(0, silent_points_left)
        end_idx_alarm = start_idx_alarm + alarm_duration * audio_sample_rate
        if end_idx_alarm > nbr_points:
            end_idx_alarm = nbr_points
            return audio
        audio[start_idx_alarm:end_idx_alarm] += (
            normalize(
                np.sin(2 * np.pi * alarm_frequency * t[start_idx_alarm:end_idx_alarm])
            )
            * VOLUMES[alarm_volume]
        )

        starting_block = end_idx_alarm
    return audio


# def create_audio_sequence_with_superposition(
#     audio_duration,
#     audio_sample_rate,
#     nbr_alarm_range,
#     alarm_duration,
#     alarm_frequency,
#     alarm_volume,
# ):
#     full_duration = 0
#     sequence = AudioSegment.empty()
#     sequence = add_silence(sequence, full_duration, [0, 10000], audio_sample_rate)

#     nbr_alarm = pick_value_in_range(nbr_alarm_range)
#     for i in range(nbr_alarm):
#         sequence = add_bip(
#             sequence, alarm_frequency, alarm_duration * 1000, alarm_volume
#         )
#         sequence = add_silence(sequence, full_duration, [0, 10000], audio_sample_rate)

#     if full_duration < audio_duration * 1000:
#         duration_final_silence = 2 * (audio_duration * 1000 - full_duration)
#         sequence = add_silence(
#             sequence,
#             full_duration,
#             [duration_final_silence, duration_final_silence + 1],
#             audio_sample_rate,
#         )
#     sequence = sequence[: audio_duration * 1000]
#     return np.array(sequence.get_array_of_samples())


def turn_to_bytes(audios, sample_rate):
    for idx_audio in range(len(audios)):
        if audios[idx_audio] is not None:
            output = io.BytesIO()
            write(output, sample_rate, audios[idx_audio].astype(np.float32))
            audios[idx_audio] = output.getvalue()
    return audios


def turn_to_bytes_single_audio(audio, sample_rate):
    output = io.BytesIO()
    write(output, sample_rate, audio.astype(np.float32))
    return output.getvalue()


def create_audios_dataset(
    dataset_id,
    dataset_characteristics,
    noise_characteristics,
    sinus_characteristics,
    sawtooth_characteristics,
    alarms_characteristics,
):
    num_training_audios = int(
        dataset_characteristics[3] * dataset_characteristics[4] / 100
    )
    num_validation_audios = dataset_characteristics[3] - num_training_audios
    training_audios_bool = [1] * num_training_audios + [0] * num_validation_audios
    random.shuffle(training_audios_bool)

    for idx_audio in range(dataset_characteristics[3]):
        audios = create_audio(
            dataset_id,
            idx_audio,
            training_audios_bool[idx_audio],
            dataset_characteristics,
            noise_characteristics,
            sinus_characteristics,
            sawtooth_characteristics,
            alarms_characteristics,
        )

        insert_audio(
            dataset_id,
            training_audios_bool[idx_audio],
            turn_to_bytes(audios, dataset_characteristics[1]),
        )


def create_audio(
    dataset_id,
    idx_audio,
    training_audio,
    dataset_characteristics,
    noise_characteristics,
    sinus_characteristics,
    sawtooth_characteristics,
    alarms_characteristics,
):
    t = generate_time_base(dataset_characteristics[2], dataset_characteristics[1])
    noise = np.zeros(len(t), dtype=np.float64)
    engine_sound_sinus = np.zeros(len(t), dtype=np.float64)
    engine_sound_sawtooth = np.zeros(len(t), dtype=np.float64)
    alarms = np.zeros(len(t), dtype=np.float64)

    if noise_characteristics is not None:
        noise, _ = add_white_noise(
            dataset_characteristics[2],
            dataset_characteristics[1],
            [noise_characteristics[0], noise_characteristics[1]],
        )
        noise = normalize(noise)
    else:
        noise = None
    if sinus_characteristics is not None and sawtooth_characteristics is not None:
        engine_sound_sinus = []
        for idx_sinus in range(len(sinus_characteristics)):
            new_sinus_wave, _, _ = generate_sinus_wave(
                [
                    sinus_characteristics[idx_sinus][0],
                    sinus_characteristics[idx_sinus][1],
                ],
                [
                    sinus_characteristics[idx_sinus][2],
                    sinus_characteristics[idx_sinus][3],
                ],
                t,
                idx_sinus * np.pi / 2,
            )
            engine_sound_sinus.append(new_sinus_wave)
        engine_sound_sinus = combine_sounds(engine_sound_sinus)
        if dataset_characteristics[12]:
            engine_sound_sinus = lowpass_filter(
                engine_sound_sinus,
                dataset_characteristics[13],
                dataset_characteristics[1],
                1,
            )
        if dataset_characteristics[14]:
            engine_sound_sinus = normalize(engine_sound_sinus)

        engine_sound_sawtooth = []
        for idx_sawtooth in range(len(sawtooth_characteristics)):
            new_sawtooth_wave, _, _ = generate_sawtooth_wave(
                [
                    sawtooth_characteristics[idx_sawtooth][0],
                    sawtooth_characteristics[idx_sawtooth][1],
                ],
                [
                    sawtooth_characteristics[idx_sawtooth][2],
                    sawtooth_characteristics[idx_sawtooth][3],
                ],
                t,
            )
            engine_sound_sawtooth.append(new_sawtooth_wave)
        engine_sound_sawtooth = combine_sounds(engine_sound_sawtooth)
        if dataset_characteristics[15]:
            engine_sound_sawtooth = lowpass_filter(
                engine_sound_sawtooth,
                dataset_characteristics[16],
                dataset_characteristics[1],
                1,
            )
        if dataset_characteristics[17]:
            engine_sound_sawtooth = normalize(engine_sound_sawtooth)
    else:
        engine_sound_sawtooth = None
        engine_sound_sinus = None

    if alarms_characteristics is not None:
        for idx_alarm in range(len(alarms_characteristics)):
            new_alarm = normalize(
                create_alarm_audio(
                    dataset_characteristics[2],
                    dataset_characteristics[1],
                    [
                        alarms_characteristics[idx_alarm][0],
                        alarms_characteristics[idx_alarm][1],
                    ],
                    alarms_characteristics[idx_alarm][2],
                    alarms_characteristics[idx_alarm][3],
                    alarms_characteristics[idx_alarm][4],
                )
            )
            insert_audio_alarm(
                dataset_id,
                idx_alarm + 1,
                idx_audio + 1,
                training_audio,
                turn_to_bytes_single_audio(new_alarm, dataset_characteristics[1]),
            )
            alarms += new_alarm
        alarms = alarms
    else:
        alarms = None
    waveform = np.zeros(len(t), dtype=int)
    engine_sound = combine_sounds(
        clean_audios([engine_sound_sinus, engine_sound_sawtooth])
    )
    if dataset_characteristics[9]:
        engine_sound = lowpass_filter(
            engine_sound, dataset_characteristics[10], dataset_characteristics[1], 1
        )
    if dataset_characteristics[11]:
        engine_sound = normalize(engine_sound)

    waveform = combine_sounds(clean_audios([noise, engine_sound, alarms]))
    return [
        finite(waveform),
        finite(noise),
        finite(engine_sound),
        finite(alarms),
        None,
    ]


def finite(array):
    if array is None:
        return None
    if not np.isfinite(array).all():
        array = np.nan_to_num(
            array,
            nan=0.0,
            posinf=np.finfo(np.float64).max,
            neginf=np.finfo(np.float64).min,
        )
    return array


def clean_audios(sources_audios):
    cleaned_sources_audios = []
    for source_audio in sources_audios:
        if source_audio is not None:
            cleaned_sources_audios.append(source_audio)
    return cleaned_sources_audios
