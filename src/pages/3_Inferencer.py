# import array
# from db.models import Training
import librosa
import re
import tensorflow as tf
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from db.database import (
    get_dataset_by_id,
    delete_pretrained_model_by_id,
    get_all_pretrained_models,
    get_pretrained_models_by_module,
    get_pretrained_model_by_id,
)
import os
import soundfile as sf
import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from skimage.measure import label, regionprops

# from scipy.signal import butter, filtfilt
from inference import extract_number, inference_SS1, inference_SS2
import io
from scipy.io.wavfile import write


def tensorboard_iframe(logdir, port=6006):
    import os

    os.system(f"tensorboard --logdir={logdir} --port={port} &")
    tensorboard_url = f"http://localhost:{port}"
    st.markdown(
        f'<iframe src="{tensorboard_url}" width="100%" height="800px"></iframe>',
        unsafe_allow_html=True,
    )


def plot_tensorboard_data(steps, values, tag):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, label=tag)
    plt.xlabel("Steps")
    plt.ylabel(tag)
    plt.title(f"{tag} over time")
    plt.legend()
    plt.grid(True)
    return plt


def get_tensorboard_data(logdir, tag):
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    if tag not in event_acc.Tags()["scalars"]:
        raise ValueError(f"Tag '{tag}' not found in TensorBoard logdir: {logdir}")
    scalars = event_acc.Scalars(tag)
    steps = [x.step for x in scalars]
    values = [x.value for x in scalars]
    return steps, values


def show_choose_pretrained_models():
    st.markdown("#### Choose the Pretrained Models")
    container = st.container(border=True)
    col1, col2 = container.columns(2)
    sr = 44100
    with col1:
        container_col1 = col1.container()
        pretrained_models_SS1 = get_pretrained_models_by_module("Source Separation 1")
        selected_model_SS1 = container_col1.selectbox(
            "Select the pretrained model for Source Separation 1",
            [f"{model.id} - {model.name}" for model in pretrained_models_SS1],
        )
        nbr_sources_SS1 = None
        if (
            selected_model_SS1 is not None
            and get_dataset_by_id(
                extract_number(
                    get_pretrained_model_by_id(
                        extract_number(selected_model_SS1)
                    ).dataset_name
                )
            )
            is not None
        ):
            nbr_sources_SS1 = get_pretrained_model_by_id(
                extract_number(selected_model_SS1)
            ).nbr_sources
            sr = get_dataset_by_id(
                extract_number(
                    get_pretrained_model_by_id(
                        extract_number(selected_model_SS1)
                    ).dataset_name
                )
            ).sample_rate
    with col2:
        container_col2 = col2.container()
        pretrained_models_SS2 = get_pretrained_models_by_module("Source Separation 2")
        selected_model_SS2 = container_col2.selectbox(
            "Select the pretrained model for Source Separation 2",
            [f"{model.id} - {model.name}" for model in pretrained_models_SS2],
        )
        nbr_sources_SS2 = None
        if (
            selected_model_SS2 is not None
            and get_dataset_by_id(
                extract_number(
                    get_pretrained_model_by_id(
                        extract_number(selected_model_SS1)
                    ).dataset_name
                )
            )
            is not None
        ):
            nbr_sources_SS2 = get_pretrained_model_by_id(
                extract_number(selected_model_SS2)
            ).nbr_sources
            sr = get_dataset_by_id(
                extract_number(
                    get_pretrained_model_by_id(
                        extract_number(selected_model_SS1)
                    ).dataset_name
                )
            ).sample_rate
    loss_curves = container.toggle("Show Details")
    if loss_curves:
        show_loss_curves(container, selected_model_SS1, selected_model_SS2)
    return sr, selected_model_SS1, nbr_sources_SS1, selected_model_SS2, nbr_sources_SS2


def show_loss_curves(container, model_SS1, model_SS2):
    container.markdown("##### Curves")
    tag = container.selectbox("Select the tag", ["val_loss", "epoch"])
    col1, col2 = container.columns(2)

    with col1:
        if model_SS1 is not None:
            selected_training_SS1 = get_pretrained_model_by_id(
                extract_number(model_SS1)
            )

            steps, values = get_tensorboard_data(
                selected_training_SS1.tensorboard_logdir, tag
            )
            plt = plot_tensorboard_data(steps, values, tag)
            col1.pyplot(plt)
    with col2:
        if model_SS2 is not None:
            selected_training_SS2 = get_pretrained_model_by_id(
                extract_number(model_SS2)
            )

            steps, values = get_tensorboard_data(
                selected_training_SS2.tensorboard_logdir, tag
            )
            plt = plot_tensorboard_data(steps, values, tag)
            col2.pyplot(plt)


def show_available_pretrained_models():
    st.markdown(f"#### Available Pretrained Models")
    container = st.container(border=True)
    training_names = []
    trainings = pd.DataFrame(
        {
            "id_training": pd.Series(dtype="int64"),
            "name": pd.Series(dtype="string"),
            "module": pd.Series(dtype="string"),
            "nbr_epochs": pd.Series(dtype="int64"),
            "batch_size": pd.Series(dtype="int64"),
            "learning_rate": pd.Series(dtype="float64"),
            "dataset_name": pd.Series(dtype="string"),
            "model_name": pd.Series(dtype="string"),
        }
    )
    for training in get_all_pretrained_models():
        data = {
            "id_training": training.id,
            "name": training.name,
            "module": training.module,
            "nbr_epochs": training.nbr_epochs,
            "batch_size": training.batch_size,
            "learning_rate": training.learning_rate,
            "dataset_name": training.dataset_name,
            "model_name": training.model_name,
        }
        trainings.loc[len(trainings)] = data
        training_names.append(f"{training.id} - {training.name}")
    trainings.set_index("id_training", inplace=True)
    container.dataframe(trainings, use_container_width=True)
    show_training_delete(container, training_names)


def extract_bip_frequency(melspectrogram, processed_melspectrogram, sr):
    gray_melspectrogram = np.mean(melspectrogram, axis=2)
    indices = np.unravel_index(
        np.argmax(gray_melspectrogram, axis=None), gray_melspectrogram.shape
    )

    mel_frequencies = librosa.core.mel_frequencies(
        fmin=0, fmax=sr, n_mels=gray_melspectrogram.shape[0]
    )[::-1]

    mean_pitch = mel_frequencies[indices[0]]

    return mean_pitch


def extract_bip_frequency_binary(binary_melspectrogram, sr):
    contours, _ = cv2.findContours(
        binary_melspectrogram, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    y_positions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        y_center = y + h // 2
        y_positions.append(y_center)

    mean_y_position = np.mean(y_positions)

    mel_frequencies = librosa.core.mel_frequencies(
        fmin=0, fmax=sr, n_mels=binary_melspectrogram.shape[0]
    )[::-1]

    mean_pitch = mel_frequencies[int(mean_y_position)]

    return mean_pitch


def count_nbr_bips(binary_melspectrogram, sr):
    label_image = label(binary_melspectrogram)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots()
    cax = ax.matshow(label_image, cmap=cmap)
    ax.axis("off")
    properties = regionprops(label_image)
    nbr_bips = len(properties)
    return fig, nbr_bips, properties, label_image


def compute_length_bip(properties):
    length_bips = []
    for prop in properties:
        _, x_min, _, x_max = prop.bbox
        length_bips.append(convert_index_to_ms(x_max - x_min))
    mean_length_bip = np.mean(length_bips)

    return int(mean_length_bip)


def compute_time_between_bips(properties):
    if len(properties) <= 1:
        return None

    distance_between_bips = []
    for index_bip in range(1, len(properties)):
        _, x_min, _, _ = properties[index_bip].bbox
        _, _, _, x_max_prev = properties[index_bip - 1].bbox
        distance_between_bips.append(convert_index_to_ms(x_min - x_max_prev))
    mean_distance_between_bips = np.mean(distance_between_bips)

    return mean_distance_between_bips


def convert_index_to_ms(value_index):
    length_audio = 15000
    x_length_melspectrogram = 646
    value_ms = length_audio * value_index / x_length_melspectrogram
    return value_ms


def compute_binary_melspectrogram(melspectrogram, criteria, k):
    melspectrograms = [melspectrogram]
    original_shape = melspectrograms[0].shape
    pixel_melspectrogram = melspectrograms[0].reshape((-1, 1))
    pixel_melspectrogram = np.float32(pixel_melspectrogram)
    retval, labels, centers = cv2.kmeans(
        pixel_melspectrogram,
        k,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((original_shape))
    return segmented_image


def show_alarm_data(
    container, mean_pitch, mean_length_bip, mean_distance_between_bips, nbr_bips
):
    container.markdown("##### Alarm Data")
    data = {
        "Propriété": [
            "Fréquence",
            "Longueur moyenne du bip en ms",
            # "Temps moyen entre les bips en ms",
            "Nombre de bips",
        ],
        "Valeur": [
            mean_pitch,
            mean_length_bip,
            # mean_distance_between_bips,
            nbr_bips,
        ],
    }
    df = pd.DataFrame(data)
    container.table(df)


def show_image_segmentation(audio_alarms, sr):
    container = st.container(border=True)
    container.markdown("##### Image Segmentation")
    separated_audios_melspectrograms, array_separated_audios_melspectrograms = (
        compute_melspectrograms(audio_alarms, sr)
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.5)
    k = 2

    if len(audio_alarms) == 1:
        cols = container.columns(3)
        compute_image_segmentation(
            cols[1],
            k,
            criteria,
            separated_audios_melspectrograms[0],
            array_separated_audios_melspectrograms[0],
            sr,
        )

    else:
        cols = container.columns(len(audio_alarms))
        for idx_col, col in enumerate(cols):
            with col:
                pitches, magnitudes = librosa.core.piptrack(
                    y=audio_alarms[idx_col], sr=sr
                )
                index = magnitudes.argmax()
                row, column = np.unravel_index(index, magnitudes.shape)
                fundamental_frequency = pitches[row, column]
                compute_image_segmentation(
                    col,
                    k,
                    criteria,
                    separated_audios_melspectrograms[idx_col],
                    array_separated_audios_melspectrograms[idx_col],
                    sr,
                    fundamental_frequency,
                )


def compute_image_segmentation(
    container,
    k,
    criteria,
    melspectrogram,
    array_melspectrogram,
    sr,
    fundamental_frequency,
):
    container.pyplot(melspectrogram)
    # binary_melspectrogram = compute_binary_melspectrogram(
    #     cv2.cvtColor(array_melspectrogram, cv2.COLOR_BGR2GRAY),
    #     criteria,
    #     k,
    # )
    gray_melspectrogram = cv2.cvtColor(array_melspectrogram, cv2.COLOR_RGB2GRAY)
    melspectrogram_dB = librosa.power_to_db(gray_melspectrogram, ref=np.max)
    value_range = np.max(melspectrogram_dB) - np.min(melspectrogram_dB)
    threshold = np.max(melspectrogram_dB) - (value_range * 2 / 100)
    binary_melspectrogram = melspectrogram_dB > threshold
    buf = io.BytesIO()
    fig, ax = plt.subplots()
    ax.imshow(binary_melspectrogram, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")  # Supprimer les axes
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    container.image(buf)
    fig, nbr_bips, properties, label_image = count_nbr_bips(binary_melspectrogram, sr)
    container.pyplot(fig)
    length_bip = compute_length_bip(properties)
    time_between_bips = compute_time_between_bips(properties)
    show_alarm_data(
        container, fundamental_frequency, length_bip, time_between_bips, nbr_bips
    )


def apply_morphological_operations(
    binary_image, kernel_size=(10, 10), apply_dilation=True
):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    if apply_dilation:
        morphed_image = cv2.dilate(eroded_image, kernel, iterations=1)
    else:
        morphed_image = eroded_image
    return morphed_image


# def show_inferencing(model_SS1, nbr_sources_SS1, model_SS2, nbr_sources_SS2, sr):
#     st.markdown("#### Inference")
#     form = st.form(key="inference", border=True)
#     audio_file_SS1 = form.file_uploader(
#         "Upload your audio file for Module SS1", type=["wav"], key="audio_file_SS1"
#     )
#     audio_file_SS2 = form.file_uploader(
#         "Upload your audio file for Module SS2", type=["wav"], key="audio_file_SS2"
#     )
#     submit_button = form.form_submit_button("Start Inference")
#     if submit_button:
#         st.markdown("#### Results for each module")
#         msg = st.toast("Inference is running ...")

#         if model_SS1 is None or model_SS2 is None:
#             msg.toast(
#                 "Both model_SS1 and model_SS2 must be loaded before starting inference."
#             )
#         else:
#             model_SS1 = get_pretrained_model_by_id(extract_number(model_SS1))
#             audio, separated_audios = inference(audio_file_SS1, model_SS1, sr)

#             separated_audios = show_inference_results_SS1(
#                 audio,
#                 audio_file_SS1,
#                 separated_audios,
#                 "Source_Separation_1",
#                 nbr_sources_SS1,
#                 sr,
#             )
#             model_SS2 = get_pretrained_model_by_id(extract_number(model_SS2))
#             # audio, separated_audios = inference(audio_file_SS2, model_SS2, sr)
#             audio, separated_audios = inference(separated_audios[1], model_SS2, sr)

#             separated_audios = show_inference_results_SS2(
#                 audio,
#                 audio_file_SS2,
#                 separated_audios,
#                 "Source_Separation_2",
#                 nbr_sources_SS2,
#                 sr,
#             )
#             show_image_segmentation(separated_audios, sr)


def show_inferencing(model_SS1, nbr_sources_SS1, model_SS2, nbr_sources_SS2, sr):
    st.markdown("#### Inference")
    form = st.form(key="inference", border=True)
    audio_file_SS1 = form.file_uploader(
        "Upload your audio file for Module SS1", type=["wav"], key="audio_file_SS1"
    )
    submit_button = form.form_submit_button("Start Inference")
    if submit_button:
        st.markdown("#### Results for each module")
        msg = st.toast("Inference is running ...")

        if model_SS1 is None or model_SS2 is None:
            msg.toast(
                "Both model_SS1 and model_SS2 must be loaded before starting inference."
            )
        else:
            model_SS1 = get_pretrained_model_by_id(extract_number(model_SS1))
            audio, separated_audios = inference_SS1(audio_file_SS1, model_SS1, sr)

            separated_audios = show_inference_results_SS1(
                audio,
                audio_file_SS1,
                separated_audios,
                "Source_Separation_1",
                nbr_sources_SS1,
                sr,
            )
            model_SS2 = get_pretrained_model_by_id(extract_number(model_SS2))
            audio, separated_audios = inference_SS2(
                separated_audios[1].T, model_SS2, sr
            )

            separated_audios = show_inference_results_SS2(
                audio,
                audio_file_SS1,
                separated_audios,
                "Source_Separation_2",
                nbr_sources_SS2,
                sr,
            )
            show_image_segmentation(separated_audios, sr)


def extract_number_inference_audio(string):
    match = re.search(r"\d+", string)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"Aucun nombre trouvé dans la chaîne : {string}")


def display_melspectrogram(audio, sr):
    if not isinstance(audio, np.ndarray):
        # audio_to_display = audio.cpu().numpy()
        audio_to_display = np.array(audio)
    else:
        # audio_to_display = prepare_audio(audio, sr)
        audio_to_display = audio
    S = librosa.feature.melspectrogram(y=audio_to_display, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, sr=sr, ax=ax, x_axis="time", y_axis="mel")
    ax.set(title="Mel-frequency spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()

    return fig


def compute_melspectrograms(audios, sr):
    melspectrograms = []
    array_melspectrograms = []
    for audio in audios:
        # if not isinstance(audio, np.ndarray):
        #     audio_to_display = audio.cpu().numpy()
        # else:
        #     audio_to_display = audio
        audio_to_display = prepare_audio_for_melspectrogram(audio)
        S = librosa.feature.melspectrogram(y=audio_to_display, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Création du melspectrogramme pour affichage
        fig, ax = plt.subplots()
        img = librosa.display.specshow(S_dB, sr=sr, ax=ax)
        plt.axis("off")  # Remove axis
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
        plt.tight_layout(pad=0)
        melspectrograms.append(fig)
        plt.close(fig)

        fig, ax = plt.subplots()
        img = librosa.display.specshow(S_dB, sr=sr, ax=ax, cmap="gray")
        plt.axis("off")  # Remove axis
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
        plt.tight_layout(pad=0)
        # Conversion de la figure en tableau numpy
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
            int(height), int(width), 3
        )
        array_melspectrograms.append(image_array)

        # melspectrograms.append(fig)
        plt.close(fig)

    return melspectrograms, array_melspectrograms


def prepare_audio_for_melspectrogram(audio):
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    if len(audio.shape) > 1:
        audio = audio.flatten()
    return audio


def show_inference_results_SS1(
    audio, audio_file, separated_audios, module, nbr_sources, sr
):
    container = st.container(border=True)
    container.markdown(f"##### {module}")
    cols = container.columns(3)

    audio = prepare_audio_for_melspectrogram(audio)
    cols[1].pyplot(display_melspectrogram(audio, sr))
    container.audio(audio_file)
    if nbr_sources == 1:
        cols = container.columns(3)
        cols[1].pyplot(display_melspectrogram(separated_audios[0][0].T, sr))
        container.audio(separated_audios[0][0].T, sample_rate=sr, format="audio/wav")
    else:
        cols = container.columns(nbr_sources)
        for idx_col, col in enumerate(cols):
            with col:
                col.pyplot(display_melspectrogram(separated_audios[0][idx_col].T, sr))
                col.audio(
                    separated_audios[0][idx_col].T, sample_rate=sr, format="audio/wav"
                )

    output_path = os.path.join(os.getcwd(), "inference_audios", module)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for idx_audio, waveform in enumerate(separated_audios[0]):
        version = list(
            filter(
                lambda string: f"audio_{extract_number_inference_audio(audio_file.name)}_source_{idx_audio + 1}"
                in string,
                os.listdir(output_path),
            )
        )
        version = len(version)
        sf.write(
            os.path.join(
                output_path,
                f"audio_{extract_number_inference_audio(audio_file.name)}_source_{idx_audio + 1}_{version+1}.wav",
            ),
            waveform,
            sr,
            format="wav",
        )
    return separated_audios[0]


def show_inference_results_SS2(
    audio_input, audio_file, separated_audios, module, nbr_sources, sr
):
    audios = []
    container = st.container(border=True)
    container.markdown(f"##### {module}")
    cols = container.columns(3)
    audio = prepare_audio_for_melspectrogram(audio_input)
    cols[1].pyplot(display_melspectrogram(audio, sr))
    audio_input_np = audio_input.numpy()
    buffer = io.BytesIO()
    write(buffer, sr, audio_input_np)
    buffer.seek(0)
    container.audio(buffer, format="audio/wav")
    if nbr_sources == 1:
        cols = container.columns(3)
        cols[1].pyplot(display_melspectrogram(separated_audios[0][0].T, sr))
        container.audio(separated_audios[0][0].T, sample_rate=sr, format="audio/wav")
        audios.append(separated_audios[0][0].T)
    else:
        cols = container.columns(nbr_sources)
        for idx_col, col in enumerate(cols):
            with col:
                col.pyplot(display_melspectrogram(separated_audios[0][idx_col].T, sr))
                col.audio(
                    separated_audios[0][idx_col].T, sample_rate=sr, format="audio/wav"
                )
            audios.append(separated_audios[0][idx_col].T)

    output_path = os.path.join(os.getcwd(), "inference_audios", module)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for idx_audio, waveform in enumerate(separated_audios[0]):
        version = list(
            filter(
                lambda string: f"audio_{extract_number_inference_audio(audio_file.name)}_source_{idx_audio + 1}"
                in string,
                os.listdir(output_path),
            )
        )
        version = len(version)
        sf.write(
            os.path.join(
                output_path,
                f"audio_{extract_number_inference_audio(audio_file.name)}_source_{idx_audio + 1}_{version+1}.wav",
            ),
            waveform,
            sr,
            format="wav",
        )
    return audios


def prepare_audio(audio_input, sr):
    if isinstance(audio_input, str) or hasattr(audio_input, "read"):
        audio, _ = librosa.load(audio_input, sr=sr)
        audio = torch.tensor(audio, dtype=torch.float32)
    else:
        audio = (
            torch.tensor(audio_input, dtype=torch.float32)
            if not torch.is_tensor(audio_input)
            else audio_input
        )
    return audio


def show_training_delete(container, training_names):
    selected_trainings = container.multiselect(
        "",
        [training_name for training_name in training_names],
    )
    selected_datasets_ids = [
        extract_number(training_name) for training_name in selected_trainings
    ]
    if container.button("Delete Training"):
        delete_pretrained_model_by_id(selected_datasets_ids)
        st.rerun()


st.set_page_config(page_title="Inferencer", layout="wide")
st.markdown("# Inferencer")

show_available_pretrained_models()
sr, model_SS1, nbr_sources_SS1, model_SS2, nbr_sources_SS2 = (
    show_choose_pretrained_models()
)
show_inferencing(model_SS1, nbr_sources_SS1, model_SS2, nbr_sources_SS2, sr)
