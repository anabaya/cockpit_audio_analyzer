from locale import normalize
import streamlit as st
import streamlit.components.v1 as components
import audio_functions
import soundfile as sf
import io
import numpy as np
from db.database import (
    insert_dataset,
    insert_sinusoidal_wave,
    insert_sawtooth_wave,
    insert_noise,
    insert_alarm,
)
import random
import soundfile as sf
import os
import re


def initialize_dataset_creator():
    st.markdown("### Dataset Characteristics")
    container = st.container(border=True)
    cols = container.columns(3)
    dataset_characteristics = get_dataset_characteristics(cols[:2])
    (
        audio_components,
        waveform,
        alarms_waveform,
        noise_characteristics,
        sinus_characteristics,
        sawtooth_characteristics,
        alarms_characteristics,
        audios,
    ) = display_audio_components(
        duration=dataset_characteristics[2],
        sample_rate=dataset_characteristics[1],
        col=cols[2],
    )
    dataset_characteristics.extend(audio_components)
    build_dataset(
        waveform,
        alarms_waveform,
        dataset_characteristics,
        noise_characteristics,
        sinus_characteristics,
        sawtooth_characteristics,
        alarms_characteristics,
        audios,
    )


def get_dataset_characteristics(cols):
    col1, col2 = cols
    with col1:
        name = col1.text_input(
            "Dataset Name", key="name_dataset", value="new_dataset_1"
        )
        sample_rate = col1.number_input(
            "Select the sample rate : ",
            value=8000,
            min_value=8000,
            max_value=48000,
            step=1,
            key="sample_rate",
        )
        duration = col1.number_input(
            "Select the duration of the audio in seconds : ",
            value=10,
            step=1,
            key="duration",
        )
    with col2:
        nbr_of_audios = col2.number_input(
            "Pick the number of audios in the dataset",
            step=1,
            key="nbr_audios",
            min_value=5,
            value=10,
        )
        percentage_train_audios = col2.slider(
            "Select the percentage of training audios you want : ",
            value=80,
            min_value=1,
            max_value=99,
            step=1,
            key="percentage_train_audios",
        )
    dataset_characteristics = [
        name,
        int(sample_rate),
        int(duration),
        nbr_of_audios,
        percentage_train_audios,
    ]
    return dataset_characteristics


def display_audio(audio_object, sample_rate, container=None):
    buffer = io.BytesIO()
    sf.write(buffer, audio_object, sample_rate, format="WAV")
    buffer.seek(0)
    if container == None:
        st.audio(buffer, format="audio/wav")
    else:
        container.audio(buffer, format="audio/wav")


def display_audio_components(duration, sample_rate, col=st):
    container = col.container()
    add_background_noise = container.toggle("Add Background Noise")
    add_engine_sound = container.toggle("Add Engine Sound")
    add_alarms = container.toggle("Add Alarms")
    add_voice = container.toggle("Add Voice")

    nbr_sinus = 0
    nbr_sawtooth = 0
    nbr_alarms_type = 0
    superposition_alarms = False
    nbr_sources = 0
    lowpass_filter_selected = False
    cutoff = None
    normalize_selected = False
    lowpass_filter_selected_sinus = False
    cutoff_sinus = None
    normalize_selected_sinus = False
    lowpass_filter_selected_sawtooth = False
    cutoff_sawtooth = None
    normalize_selected_sawtooth = False
    alarms_waveform = None

    audios = [None for _ in range(5)]

    if add_background_noise:
        noise, noise_level = show_building_background_noise(duration, sample_rate)
        audios[1] = noise
        waveform = audio_functions.combine_sounds([noise])
        nbr_sources += 1
    if add_engine_sound:
        (
            lowpass_filter_selected,
            cutoff,
            normalize_selected,
            lowpass_filter_selected_sinus,
            cutoff_sinus,
            normalize_selected_sinus,
            lowpass_filter_selected_sawtooth,
            cutoff_sawtooth,
            normalize_selected_sawtooth,
            nbr_sinus,
            sinus_waveform,
            sinus_characteristics,
            nbr_sawtooth,
            sawtooth_waveform,
            sawtooth_characteristics,
        ) = show_building_engine_sound(duration, sample_rate)
        audios[2] = audio_functions.combine_sounds([sinus_waveform, sawtooth_waveform])
        if locals().get("waveform", None) is None:
            waveform = audio_functions.combine_sounds(
                [sinus_waveform, sawtooth_waveform]
            )
        else:
            waveform = audio_functions.combine_sounds(
                [waveform, sinus_waveform, sawtooth_waveform]
            )
        nbr_sources += 1
    if add_alarms:
        (
            nbr_alarms_type,
            superposition_alarms,
            alarms_waveform,
            alarms_characteristics,
        ) = show_building_alarms(duration, sample_rate)
        audios[3] = alarms_waveform
        if locals().get("waveform", None) is None:
            waveform = audio_functions.combine_sounds([alarms_waveform])
        else:
            waveform = audio_functions.combine_sounds([waveform, alarms_waveform])
        nbr_sources += 1
    if add_voice:
        show_building_voice()
        nbr_sources += 1

    if locals().get("waveform", None) is None:
        t = audio_functions.generate_time_base(duration, sample_rate)
        waveform = np.zeros(len(t), dtype=int)
    # else:
    #     waveform = audio_functions.normalize(waveform)
    audios[0] = waveform
    return (
        [
            add_background_noise,
            add_engine_sound,
            nbr_sinus,
            nbr_sawtooth,
            lowpass_filter_selected,
            cutoff,
            normalize_selected,
            lowpass_filter_selected_sinus,
            cutoff_sinus,
            normalize_selected_sinus,
            lowpass_filter_selected_sawtooth,
            cutoff_sawtooth,
            normalize_selected_sawtooth,
            add_alarms,
            nbr_alarms_type,
            superposition_alarms,
            add_voice,
            nbr_sources,
        ],
        waveform,
        alarms_waveform,
        locals().get("noise_level", None),
        locals().get("sinus_characteristics", None),
        locals().get("sawtooth_characteristics", None),
        locals().get("alarms_characteristics", None),
        audios,
    )


def show_building_background_noise(duration, sample_rate):
    st.markdown("### Building Background Noise")
    container = st.container(border=True)
    noise_level_range = container.select_slider(
        "Pick the noise level :",
        options=range(0, 10),
        value=audio_functions.pick_initial_values([0, 9], "noise_level_range"),
        key="noise_level_range",
    )
    noise, noise_level = audio_functions.add_white_noise(
        duration, sample_rate, noise_level_range
    )
    container.write(f"The noise level of the audio below is : {noise_level}")
    display_audio(noise, sample_rate, container)
    return noise, [noise_level_range[0], noise_level_range[1]]


def show_dashboard_sinus_wave(duration, sample_rate, idx_sinus, phase_shift):
    container = st.container(border=True)
    t = audio_functions.generate_time_base(duration, sample_rate)
    fundamental_freq_range = container.select_slider(
        "Pick a range for the fundamental frequency of the sinus wave : ",
        options=range(0, 500),
        value=audio_functions.pick_initial_values(
            [0, 499], f"fundamental_freq_range_sinus{idx_sinus}"
        ),
        key=f"fundamental_freq_range_sinus{idx_sinus}",
    )
    amplitude_range = container.select_slider(
        "Pick a range for the amplitude of the sinus wave : ",
        options=range(0, 10),
        value=audio_functions.pick_initial_values(
            [0, 9], f"amplitude_range_sinus_{idx_sinus}"
        ),
        key=f"amplitude_range_sinus_{idx_sinus}",
    )
    sinus_wave, fundamental_freq, amplitude = audio_functions.generate_sinus_wave(
        fundamental_freq_range, amplitude_range, t, phase_shift
    )
    container.write(
        f"The audio right below has a fundamental frequency of {fundamental_freq} Hz and an amplitude of {amplitude}"
    )

    # nbr_harmonics_range = container.select_slider("Select a range for the number of harmonics : ", options=range(0,11), value=audio_functions.pick_initial_values([0,10], f"nbr_harmonics_range_sinus{idx_sinus}"), key=f"nbr_harmonics_range_sinus{idx_sinus}")
    # start_intensity_harmonic = container.select_slider("Select a starting intensity for the harmonics : ", options=range(0,3), value=audio_functions.pick_initial_values([0,2], f"start_intensity_harmonic_range_sinus{idx_sinus}"), key=f"start_intensity_harmonic_range_sinus{idx_sinus}")
    # harmonics = audio_functions.create_harmonics(nbr_harmonics_range, start_intensity_harmonic)
    # sinus_wave = audio_functions.add_harmonics(sinus_wave, fundamental_freq, harmonics, t)
    display_audio(sinus_wave, sample_rate, container)
    # return sinus_wave, [fundamental_freq_range[0], fundamental_freq_range[1], amplitude_range[0], amplitude_range[1], nbr_harmonics_range[0], nbr_harmonics_range[1], start_intensity_harmonic[0], start_intensity_harmonic[1]]
    return sinus_wave, [
        fundamental_freq_range[0],
        fundamental_freq_range[1],
        amplitude_range[0],
        amplitude_range[1],
        0,
        0,
        0,
        0,
    ]


def show_sinus_waves(duration, sample_rate):
    container = st.container(border=True)
    container.markdown("##### Sinus Waves")
    nbr_sinus_waves = container.select_slider(
        "Choose the number of sinus waves you want : ",
        options=range(1, 6),
        value=audio_functions.pick_initial_value([1, 5], "nbr_sinus_waves"),
        key="nbr_sinus_waves",
    )
    tabs = container.tabs([f"Sinus Wave {i+1}" for i in range(nbr_sinus_waves)])
    sinus_waves = [
        audio_functions.generate_time_base(duration, sample_rate)
        for _ in range(nbr_sinus_waves)
    ]
    characteristics = []
    for idx_sinus, tab in enumerate(tabs):
        with tab:
            sinus_waves[idx_sinus], sinus_characteristics = show_dashboard_sinus_wave(
                duration, sample_rate, idx_sinus, idx_sinus * np.pi / 2
            )
            characteristics.append(sinus_characteristics)
        waveform = audio_functions.normalize(
            audio_functions.combine_sounds(sinus_waves)
        )
    waveform = audio_functions.combine_sounds(sinus_waves)
    col1, col2, col3 = container.columns(3)
    with col1:
        lowpass_filter_selected = col1.checkbox(
            "Lowpass filter", key="lowpass_filter_sinus"
        )
    with col2:
        if lowpass_filter_selected:
            cutoff = col2.select_slider(
                "Cutoff frequency : ",
                options=range(0, 1000),
                value=audio_functions.pick_initial_value(
                    [0, 1000], "cutoff_frequency_sinus"
                ),
                key="cutoff_frequency_sinus",
            )
            waveform = audio_functions.lowpass_filter(waveform, cutoff, sample_rate, 1)
        else:
            cutoff = None
    with col3:
        normalize_selected = col3.checkbox("Normalize", key="normalize_sinus")
        if normalize_selected:
            waveform = audio_functions.normalize(waveform)
    container.markdown("##### The sum of all the sinus waves is : ")
    display_audio(waveform, sample_rate, container)
    return (
        nbr_sinus_waves,
        lowpass_filter_selected,
        cutoff,
        normalize_selected,
        waveform,
        characteristics,
    )


def show_dashboard_sawtooth_wave(duration, sample_rate, idx_sawtooth, phase_shift=0):
    container = st.container(border=True)
    t = audio_functions.generate_time_base(duration, sample_rate)
    fundamental_freq_range = container.select_slider(
        "Pick a range for the fundamental frequency of the sawtooth wave : ",
        options=range(0, 500),
        value=audio_functions.pick_initial_values(
            [0, 499], f"fundamental_freq_range_sawtooth{idx_sawtooth}"
        ),
        key=f"fundamental_freq_range_sawtooth{idx_sawtooth}",
    )
    amplitude_range = container.select_slider(
        "Pick a range for the amplitude of the sawtooth wave : ",
        options=range(0, 10),
        value=audio_functions.pick_initial_values(
            [0, 9], f"amplitude_range_sawtooth_{idx_sawtooth}"
        ),
        key=f"amplitude_range_sawtooth_{idx_sawtooth}",
    )
    sawtooth_wave, fundamental_freq, amplitude = audio_functions.generate_sawtooth_wave(
        fundamental_freq_range, amplitude_range, t, phase_shift
    )
    container.write(
        f"The audio right below has a fundamental frequency of {fundamental_freq} Hz and an amplitude of {amplitude}"
    )

    # TODO : ici les harmoniques ajoutés correspondent à des harmoniques sinus à voir pour faire des harmoniques dent de scie
    # il serait aussi chouette de visualisez les dents de scie générer avec la fonction mano et de comparer avec celle de pydub generators
    # nbr_harmonics_range = container.select_slider("Select a range for the number of harmonics : ", options=range(0,11), value=audio_functions.pick_initial_values([0,10], f"nbr_harmonics_range_sawtooth{idx_sawtooth}"), key=f"nbr_harmonics_range_sawtooth{idx_sawtooth}")
    # start_intensity_harmonic = container.select_slider("Select a starting intensity for the harmonics : ", options=range(0,3), value=audio_functions.pick_initial_values([0,2], f"start_intensity_harmonic_range_sawtooth{idx_sawtooth}"), key=f"start_intensity_harmonic_range_sawtooth{idx_sawtooth}")
    # harmonics = audio_functions.create_harmonics(nbr_harmonics_range, start_intensity_harmonic)
    # sawtooth_wave = audio_functions.add_harmonics(sawtooth_wave, fundamental_freq, harmonics, t)
    display_audio(sawtooth_wave, sample_rate, container)
    return sawtooth_wave, [
        fundamental_freq_range[0],
        fundamental_freq_range[1],
        amplitude_range[0],
        amplitude_range[1],
        0,
        0,
        0,
        0,
    ]


def show_sawtooth_waves(duration, sample_rate):
    container = st.container(border=True)
    container.markdown("##### Sawtooth Waves")
    nbr_sawtooth_waves = container.select_slider(
        "Choose the number of sawtooth waves you want : ",
        options=range(1, 6),
        value=audio_functions.pick_initial_value([1, 5], "nbr_sawtooth_waves"),
        key="nbr_sawtooth_waves",
    )
    tabs = container.tabs([f"Sawtooth {i+1}" for i in range(nbr_sawtooth_waves)])
    sawtooth_waves = [
        audio_functions.generate_time_base(duration, sample_rate)
        for _ in range(nbr_sawtooth_waves)
    ]
    characteristics = []
    for idx_sawtooth, tab in enumerate(tabs):
        with tab:
            sawtooth_waves[idx_sawtooth], sawtooth_characteristics = (
                show_dashboard_sawtooth_wave(duration, sample_rate, idx_sawtooth)
            )
            characteristics.append(sawtooth_characteristics)
    waveform = audio_functions.combine_sounds(sawtooth_waves)
    col1, col2, col3 = container.columns(3)
    with col1:
        lowpass_filter_selected = col1.checkbox(
            "Lowpass filter", key="lowpass_filter_sawtooth"
        )
    with col2:
        if lowpass_filter_selected:
            cutoff = col2.select_slider(
                "Cutoff frequency : ",
                options=range(0, 1000),
                value=audio_functions.pick_initial_value(
                    [0, 1000], "cutoff_frequency_sawtooth"
                ),
                key="cutoff_frequency_sawtooth",
            )
            waveform = audio_functions.lowpass_filter(waveform, cutoff, sample_rate, 1)
        else:
            cutoff = None
    with col3:
        normalize_selected = col3.checkbox("Normalize", key="normalize_sawtooth")
        if normalize_selected:
            waveform = audio_functions.normalize(waveform)
    container.markdown("##### The sum of all the sawtooth waves is : ")
    display_audio(waveform, sample_rate, container)
    return (
        nbr_sawtooth_waves,
        lowpass_filter_selected,
        cutoff,
        normalize_selected,
        waveform,
        characteristics,
    )


def show_building_engine_sound(duration, sample_rate):
    st.markdown("### Building Engine Sound")
    left_col, right_col = st.columns(2)
    with left_col:
        (
            nbr_sinus,
            lowpass_filter_selected_sinus,
            cutoff_sinus,
            normalize_selected_sinus,
            sinus_waveform,
            sinus_characteristics,
        ) = show_sinus_waves(duration, sample_rate)
    with right_col:
        (
            nbr_sawtooth,
            lowpass_filter_selected_sawtooth,
            cutoff_sawtooth,
            normalize_selected_sawtooth,
            sawtooth_waveform,
            sawtooth_characteristics,
        ) = show_sawtooth_waves(duration, sample_rate)
    combined_waveforms = audio_functions.combine_sounds(
        [sinus_waveform, sawtooth_waveform]
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        lowpass_filter_selected = col1.checkbox("Lowpass filter", key="lowpass_filter")
    with col2:
        cutoff = None
        if lowpass_filter_selected:
            cutoff = col2.select_slider(
                "Cutoff frequency : ",
                options=range(0, 1000),
                value=audio_functions.pick_initial_value([0, 1000], "cutoff_frequency"),
                key="cutoff_frequency",
            )
            combined_waveforms = audio_functions.lowpass_filter(
                combined_waveforms, cutoff, sample_rate, 1
            )
    with col3:
        normalize_selected = col3.checkbox("Normalize", key="normalize")
        if normalize_selected:
            combined_waveforms = audio_functions.normalize(combined_waveforms)
    display_audio(combined_waveforms, sample_rate)
    return (
        lowpass_filter_selected,
        cutoff,
        normalize_selected,
        lowpass_filter_selected_sinus,
        cutoff_sinus,
        normalize_selected_sinus,
        lowpass_filter_selected_sawtooth,
        cutoff_sawtooth,
        normalize_selected_sawtooth,
        nbr_sinus,
        sinus_waveform,
        sinus_characteristics,
        nbr_sawtooth,
        sawtooth_waveform,
        sawtooth_characteristics,
    )


def show_building_alarms(duration, sample_rate):
    st.markdown("### Building Alarms")
    container = st.container(border=True)
    nbr_alarms_type = container.select_slider(
        "Choose the number of different alarms you want : ",
        options=range(0, 11),
        value=3,
    )
    superposition = container.toggle("Enable alarm superposition", value=True)
    tabs = container.tabs([f"Alarm {i+1}" for i in range(nbr_alarms_type)])

    alarms = [
        audio_functions.generate_time_base(duration, sample_rate)
        for _ in range(nbr_alarms_type)
    ]
    characteristics = []
    for idx_alarm, tab in enumerate(tabs):
        with tab:
            alarms[idx_alarm], alarm_characteristics = show_dashboard_alarm(
                duration, sample_rate, idx_alarm, superposition
            )
            characteristics.append(alarm_characteristics)
    waveform = audio_functions.combine_sounds(alarms)
    container.markdown("##### The sum of all the alarms : ")
    display_audio(waveform, sample_rate, container)
    return nbr_alarms_type, superposition, waveform, characteristics


def show_dashboard_alarm(duration, sample_rate, idx_alarm, superposition):
    container_alarm_characteristics = st.container(border=True)
    col1, col2 = container_alarm_characteristics.columns(2)
    with col1:
        nbr_alarm_range = col1.select_slider(
            "Pick the number of this alarm type you want : ",
            options=range(0, 11),
            value=[1, 3],
            key=f"nbr_alarm_range_{idx_alarm}",
        )
        alarm_duration = col1.select_slider(
            "Pick the duration of this alarm type : ",
            options=range(0, 5),
            value=1,
            key=f"duration_alarm_{idx_alarm}",
        )

    with col2:
        alarm_frequency = col2.select_slider(
            "Pick the frequency you want for this alarm type : ",
            options=range(0, 2000),
            value=1000,
            key=f"frequency_alarm_{idx_alarm}",
        )
        alarm_volume = col2.select_slider(
            "Pick the volume you want for this alarm type :",
            options=range(0, 10),
            value=5,
            key=f"volume_alarm_{idx_alarm}",
        )

    audio = audio_functions.create_alarm_audio(
        duration,
        sample_rate,
        nbr_alarm_range,
        alarm_duration,
        alarm_frequency,
        alarm_volume,
    )

    display_audio(audio, sample_rate, container_alarm_characteristics)
    return audio, [
        nbr_alarm_range[0],
        nbr_alarm_range[1],
        alarm_duration,
        alarm_frequency,
        alarm_volume,
    ]


def show_created_sound(waveform, sample_rate, form=st):
    form.markdown("##### An Example of Created Audio")
    display_audio(waveform, sample_rate, form)


def show_building_voice():
    st.markdown("### Building Voice")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col3:
        st.image("static/work-in-progress.png")


def build_dataset(
    waveform,
    alarms_waveform,
    dataset_characteristics,
    noise_characteristics,
    sinus_characteristics,
    sawtooth_characteristics,
    alarms_characteristics,
    audios,
):
    st.markdown("### Building Dataset")

    dataset_form = st.container(border=True)
    show_created_sound(waveform, dataset_characteristics[1], dataset_form)
    cols = dataset_form.columns(9)
    with cols[0]:
        submit_button = cols[0].button("Create Dataset", use_container_width=True)
    if submit_button:
        msg = st.toast("Dataset is being created")
        dataset_id = insert_dataset(dataset_characteristics)

        if dataset_id == None:
            msg.toast("Dataset name already exists, please choose another name")

        else:
            if dataset_characteristics[5]:
                insert_noise(dataset_id, noise_characteristics)
            if dataset_characteristics[6]:
                for index_sinus in range(dataset_characteristics[7]):
                    insert_sinusoidal_wave(
                        dataset_id, sinus_characteristics[index_sinus]
                    )
                for index_sawtooth in range(dataset_characteristics[8]):
                    insert_sawtooth_wave(
                        dataset_id, sawtooth_characteristics[index_sawtooth]
                    )
            if dataset_characteristics[18]:
                for index_alarm in range(dataset_characteristics[19]):
                    insert_alarm(dataset_id, alarms_characteristics[index_alarm])
            audio_functions.create_audios_dataset(
                dataset_id,
                dataset_characteristics,
                noise_characteristics,
                sinus_characteristics,
                sawtooth_characteristics,
                alarms_characteristics,
            )
            msg.toast("Dataset created")
    with cols[8]:
        one_audio_button = cols[8].button("One Audio", use_container_width=True)
        if one_audio_button:
            output_path = os.path.join(os.getcwd(), "inference_audios")
            version_audio = compute_version_new_audio(output_path)
            sf.write(
                os.path.join(output_path, f"audio_{version_audio}_SS1.wav"),
                waveform,
                dataset_characteristics[1],
                format="wav",
            )
            if alarms_waveform is not None:
                sf.write(
                    os.path.join(output_path, f"audio_{version_audio}_SS2.wav"),
                    alarms_waveform,
                    dataset_characteristics[1],
                    format="wav",
                )


def compute_version_new_audio(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    files = os.listdir(output_path)
    wav_files = [f for f in files if f.endswith(".wav")]
    numbers = []
    for file in wav_files:
        match = re.search(r"audio_(\d+)_", file)
        if match:
            number = int(match.group(1))
            numbers.append(number)
    if numbers:
        max_number = max(numbers)
        version = max_number + 1
    else:
        version = 1
    return version


st.set_page_config(page_title="Dataset Builder", layout="wide")
st.markdown("# Dataset Builder")

initialize_dataset_creator()
