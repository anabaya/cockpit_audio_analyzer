import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import time
from pydub import AudioSegment
from pydub.playback import play
from scipy.io import wavfile
import pandas as pd
import torch
import threading


from db.database import (
    insert_training,
    get_all_datasets,
    get_dataset_by_id,
    get_nbr_datasets,
    get_audios_SS1,
    get_audios_SS2,
    get_all_alarms,
    delete_datasets_by_id,
    save_audios_to_wav_SS1,
    save_audios_to_wav_SS2,
    get_pretrained_models_by_module,
)
from training import start_training, prepare_data, get_session_state
import re
import os


def initialize_models_trainer(session_state):
    tab_names = ["Source Separation 1", "Source Separation 2", "Alarm Segmentation"]
    models = st.tabs(tab_names)
    for idx_model, model in enumerate(models):
        with model:
            if idx_model != 2:
                if idx_model == 0:
                    model.markdown("#### Available Datasets")
                    container = model.container(border=True)
                    datasets_names, datasets_nbr_sources = show_available_datasets_SS1(
                        container
                    )
                    show_dataset_delete(container, datasets_names)
                    show_training_parameters_SS1(
                        session_state,
                        model,
                        idx_model,
                        tab_names[idx_model],
                        datasets_names,
                        datasets_nbr_sources,
                    )
                elif idx_model == 1:
                    datasets_names, datasets_nbr_sources = show_available_datasets_SS2(
                        model
                    )
                    show_training_parameters_SS2(
                        session_state,
                        model,
                        idx_model,
                        tab_names[idx_model],
                        datasets_names,
                        datasets_nbr_sources,
                    )

            else:
                col1, col2, col3, col4, col5 = model.columns(5)
                with col3:
                    col3.image("static/work-in-progress.png")


def extract_data_from_object(obj):
    if hasattr(obj, "__dict__"):
        return vars(obj)


def highlight_datasets(table):
    # colors = ['#ADD8E6', '#FFFACD', '#FFDAB9', '#BDFCC9', '#E6E6FA']
    colors = ["#FFEEEE", "#FFFFFF"]
    return [f"background-color: {colors[table.dataset_id%len(colors)]}"] * len(table)


def show_available_datasets_SS2(tab):
    tab.markdown("#### Available Datasets")
    alarm_datasets = pd.DataFrame(
        {
            "id_alarm": pd.Series(dtype="int64"),
            "dataset_id": pd.Series(dtype="int64"),
            "dataset_name": pd.Series(dtype="string"),
            "min_nbr_alarm": pd.Series(dtype="int64"),
            "max_nbr_alarm": pd.Series(dtype="int64"),
            "alarm_duration": pd.Series(dtype="int64"),
            "alarm_frequency": pd.Series(dtype="int64"),
            "alarm_volume": pd.Series(dtype="int64"),
        }
    )
    datasets_names = []
    for alarm_dataset in get_all_alarms():
        data = {
            "id_alarm": alarm_dataset.id,
            "dataset_id": alarm_dataset.dataset_id,
            "dataset_name": alarm_dataset.dataset.name,
            "min_nbr_alarm": alarm_dataset.min_nbr_alarm,
            "max_nbr_alarm": alarm_dataset.max_nbr_alarm,
            "alarm_duration": alarm_dataset.alarm_duration,
            "alarm_frequency": alarm_dataset.alarm_frequency,
            "alarm_volume": alarm_dataset.alarm_volume,
        }

        alarm_datasets.loc[len(alarm_datasets)] = data
        if (
            f"{alarm_dataset.dataset_id} - {get_dataset_by_id(alarm_dataset.dataset_id).name}"
            not in datasets_names
        ):
            datasets_names.append(
                f"{alarm_dataset.dataset_id} - {get_dataset_by_id(alarm_dataset.dataset_id).name}"
            )
    alarm_datasets.set_index("id_alarm", inplace=True)

    grouped = alarm_datasets.groupby("dataset_id").size().reset_index(name="counts")
    datasets_nbr_alarms = grouped.set_index("dataset_id").to_dict()["counts"]

    tab.dataframe(
        alarm_datasets.style.apply(highlight_datasets, axis=1), use_container_width=True
    )

    return datasets_names, datasets_nbr_alarms


def training_SS1(
    session_state,
    module_name,
    training_name,
    nbr_epochs,
    batch_size,
    learning_rate,
    dataset,
    model,
    sample_rate,
    nbr_sources,
):
    print(nbr_sources)
    (
        sample_rate,
        training_audios_mixtures,
        training_audios_sources,
        validation_audios_mixtures,
        validation_audios_sources,
    ) = get_audios_SS1(dataset[0])

    train_loader, val_loader = prepare_data(
        sample_rate,
        training_audios_mixtures,
        training_audios_sources,
        validation_audios_mixtures,
        validation_audios_sources,
        batch_size,
    )
    start_training(
        session_state,
        training_name,
        train_loader,
        val_loader,
        nbr_epochs,
        model,
        batch_size,
        nbr_sources,
        learning_rate,
    )


def training_SS2(
    session_state,
    module_name,
    training_name,
    nbr_epochs,
    batch_size,
    learning_rate,
    dataset,
    model,
    sample_rate,
    nbr_sources,
    id_selected_SS1_training,
):
    (
        sample_rate,
        training_audios_mixtures,
        training_audios_sources,
        validation_audios_mixtures,
        validation_audios_sources,
    ) = get_audios_SS2(dataset[0], id_selected_SS1_training)

    train_loader, val_loader = prepare_data(
        sample_rate,
        training_audios_mixtures,
        training_audios_sources,
        validation_audios_mixtures,
        validation_audios_sources,
        batch_size,
    )
    start_training(
        session_state,
        training_name,
        train_loader,
        val_loader,
        nbr_epochs,
        model,
        batch_size,
        nbr_sources,
        learning_rate,
    )


def show_available_datasets_SS1(container):
    datasets = pd.DataFrame(
        {
            "id_dataset": pd.Series(dtype="int64"),
            "name": pd.Series(dtype="string"),
            "duration (s)": pd.Series(dtype="int64"),
            "nbr_of_audios": pd.Series(dtype="int64"),
            "sample_rate (Hz)": pd.Series(dtype="int64"),
            "percentage_training_audios": pd.Series(dtype="int64"),
            "nbr_sources": pd.Series(dtype="int64"),
            "add_background_noise": pd.Series(dtype="bool"),
            "add_engine_sound": pd.Series(dtype="bool"),
            "add_alarms": pd.Series(dtype="bool"),
            "add_voices": pd.Series(dtype="bool"),
        }
    )
    datasets_names = []
    datasets_nbr_sources = []
    for dataset in get_all_datasets():
        data = {
            "id_dataset": dataset.id,
            "name": dataset.name,
            "duration (s)": dataset.duration,
            "nbr_of_audios": dataset.nbr_of_audios,
            "sample_rate (Hz)": dataset.sample_rate,
            "percentage_training_audios": dataset.percentage_training_audios,
            "nbr_sources": dataset.add_background_noise
            + dataset.add_engine_sound
            + dataset.add_alarms
            + dataset.add_voices,
            "add_background_noise": dataset.add_background_noise,
            "add_engine_sound": dataset.add_engine_sound,
            "add_alarms": dataset.add_alarms,
            "add_voices": dataset.add_voices,
        }
        datasets.loc[len(datasets)] = data
        datasets_names.append(f"{dataset.id} - {dataset.name}")
        datasets_nbr_sources.append(
            dataset.add_background_noise
            + dataset.add_engine_sound
            + dataset.add_alarms
            + dataset.add_voices
        )

    datasets.set_index("id_dataset", inplace=True)
    container.dataframe(datasets, use_container_width=True)
    return datasets_names, datasets_nbr_sources


def extract_number(expression):
    match = re.search(r"(\d+) - .+", expression)
    if match:
        return int(match.group(1))
    else:
        return None


def show_dataset_delete(tab, datasets_names):
    selected_datasets = tab.multiselect(
        "",
        [dataset_name for dataset_name in datasets_names],
    )
    selected_datasets_ids = [
        extract_number(dataset_name) for dataset_name in selected_datasets
    ]
    if tab.button("Delete Dataset"):
        delete_datasets_by_id(selected_datasets_ids)
        st.rerun()


def show_training_parameters_SS2(
    session_state, tab, idx_model, tab_name, dataset_names, datasets_nbr_sources
):
    tab.markdown("#### Training Parameters")
    trainer_form = tab.form(f"trainer_form_model_{idx_model}")
    col1, col2 = trainer_form.columns(2)
    with col1:
        training_name = col1.text_input(
            "Training name", value=f"training_1", key=f"training_name_{idx_model}"
        )
        nbr_epochs = col1.number_input(
            "Number of epochs", value=10, key=f"nbr_epochs_ model_{idx_model}"
        )
        batch_size = col1.number_input(
            "Batch size", value=32, key=f"batch_size_ model_{idx_model}"
        )
        learning_rate = col1.number_input(
            "Learning rate (10^X)",
            value=-6,
            key=f"learning_rate_ model_{idx_model}",
            min_value=-6,
            max_value=1,
            step=1,
        )

    with col2:
        dataset = col2.selectbox(
            "Dataset",
            [dataset_name for dataset_name in dataset_names],
            key=f"dataset_selector_model_{idx_model}",
        )
        trained_models_SS1 = get_pretrained_models_by_module("Source Separation 1")
        training_SS1 = col2.selectbox(
            "Select the trained model for Source Separation 1",
            [
                f"{model.id} - {model.name} - {model.nbr_epochs}_epochs - {model.dataset_name}"
                for model in trained_models_SS1
            ],
        )
        id_training_SS1 = training_SS1[0]

        model = col2.selectbox(
            "Model",
            # ["TasNet", "ConvTasNet", "TwoStep", "DPRNN-2", "DPRNN-16"],
            ["ConvTasNet", "DPRNNTasNet", "LSTMTasNet", "DPTNet", "SuDORMRFNet"],
            key=f"model_selector_model_{idx_model}",
        )

    submit_button = trainer_form.form_submit_button("Start Training")
    if submit_button:
        # save_audios_to_wav_SS2(extract_number(dataset))
        msg = st.toast("Training is running")
        tensorboard_path = os.path.join("tb_logs", model)
        os.makedirs(tensorboard_path, exist_ok=True)

        saved_model_path = os.path.join("models", model)
        os.makedirs(saved_model_path, exist_ok=True)

        training_id = insert_training(
            [
                tab_name,
                training_name,
                nbr_epochs,
                batch_size,
                learning_rate,
                dataset,
                model,
                datasets_nbr_sources[extract_number(dataset)],
                os.path.join(tensorboard_path, training_name),
                os.path.join(saved_model_path, f"{training_name}.ckpt"),
            ]
        )
        if training_id == None:
            msg.toast("Training name already exists, please choose another name")
        else:
            training_SS2(
                session_state,
                tab_name,
                training_name,
                nbr_epochs,
                batch_size,
                learning_rate,
                dataset,
                model,
                get_dataset_by_id(extract_number(dataset)).sample_rate,
                datasets_nbr_sources[extract_number(dataset)],
                id_training_SS1,
            )


def show_training_parameters_SS1(
    session_state, tab, idx_model, tab_name, dataset_names, datasets_nbr_sources
):
    tab.markdown("#### Training Parameters")
    trainer_form = tab.form(f"trainer_form_model_{idx_model}")
    col1, col2 = trainer_form.columns(2)
    with col1:
        training_name = col1.text_input(
            "Training name", value=f"training_1", key=f"training_name_{idx_model}"
        )
        nbr_epochs = col1.number_input(
            "Number of epochs", value=10, key=f"nbr_epochs_ model_{idx_model}"
        )
        batch_size = col1.number_input(
            "Batch size", value=32, key=f"batch_size_ model_{idx_model}"
        )

    with col2:
        learning_rate = col2.number_input(
            "Learning rate (10^X)",
            value=-6,
            key=f"learning_rate_ model_{idx_model}",
            min_value=-6,
            max_value=1,
            step=1,
        )
        dataset = col2.selectbox(
            "Dataset",
            [dataset_name for dataset_name in dataset_names],
            key=f"dataset_selector_model_{idx_model}",
        )
        model = col2.selectbox(
            "Model",
            # ["TasNet", "ConvTasNet", "TwoStep", "DPRNN-2", "DPRNN-16"],
            ["ConvTasNet", "DPRNNTasNet", "LSTMTasNet", "DPTNet", "SuDORMRFNet"],
            key=f"model_selector_model_{idx_model}",
        )
    submit_button = trainer_form.form_submit_button("Start Training")

    if submit_button:
        # save_audios_to_wav_SS1(extract_number(dataset))
        msg = st.toast("Training is running")
        tensorboard_path = os.path.join("tb_logs", model)
        os.makedirs(tensorboard_path, exist_ok=True)

        saved_model_path = os.path.join("models", model)
        os.makedirs(saved_model_path, exist_ok=True)

        training_id = insert_training(
            [
                tab_name,
                training_name,
                nbr_epochs,
                batch_size,
                learning_rate,
                dataset,
                model,
                datasets_nbr_sources[dataset_names.index(dataset)],
                os.path.join(tensorboard_path, training_name),
                os.path.join(saved_model_path, f"{training_name}.ckpt"),
            ]
        )
        if training_id == None:
            msg.toast("Training name already exists, please choose another name")
        else:
            training_SS1(
                session_state,
                tab_name,
                training_name,
                nbr_epochs,
                batch_size,
                learning_rate,
                dataset,
                model,
                get_dataset_by_id(extract_number(dataset)).sample_rate,
                datasets_nbr_sources[dataset_names.index(dataset)],
            )


st.set_page_config(page_title="Trainer", layout="wide")
st.markdown("# Trainer")

st.markdown(
    "### Diagram of the Processing Chain for Separation/Identification of Audio Sources"
)
st.image("static/schema_chaine_traitement_v1.png")
session_state = get_session_state(training_status="Not Started")

initialize_models_trainer(session_state)
