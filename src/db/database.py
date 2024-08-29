from pkgutil import get_data
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import (
    Base,
    Dataset,
    SinusoidalWave,
    SawtoothWave,
    Noise,
    Alarm,
    Audio,
    Training,
    Audio_Alarm,
)

# from pydub import AudioSegment
import librosa
import io
import streamlit as st
import os
import shutil
import soundfile as sf
from inference import inference_training_SS2

# Configuration de la base de données
engine = create_engine(
    "sqlite:///db/datasets.db", connect_args={"check_same_thread": False}
)
Base.metadata.create_all(engine)

# Création d'une session
Session = sessionmaker(bind=engine)
session = Session()

CURRENT_FOLDER = os.getcwd()
DATABASE_PATH_SS1 = os.path.join(CURRENT_FOLDER, "database", "SS1")
DATABASE_PATH_SS2 = os.path.join(CURRENT_FOLDER, "database", "SS2")

TRAIN_MIXTURES_AUDIO_PATH_SS1 = os.path.join(DATABASE_PATH_SS1, "train", "mixtures")
TRAIN_SOURCES_AUDIO_PATH_SS1 = os.path.join(DATABASE_PATH_SS1, "train", "sources")
VAL_MIXTURES_AUDIO_PATH_SS1 = os.path.join(DATABASE_PATH_SS1, "val", "mixtures")
VAL_SOURCES_AUDIO_PATH_SS1 = os.path.join(DATABASE_PATH_SS1, "val", "sources")


TRAIN_MIXTURES_AUDIO_PATH_SS2 = os.path.join(DATABASE_PATH_SS2, "train", "mixtures")
TRAIN_SOURCES_AUDIO_PATH_SS2 = os.path.join(DATABASE_PATH_SS2, "train", "sources")
VAL_MIXTURES_AUDIO_PATH_SS2 = os.path.join(DATABASE_PATH_SS2, "val", "mixtures")
VAL_SOURCES_AUDIO_PATH_SS2 = os.path.join(DATABASE_PATH_SS2, "val", "sources")


def insert_dataset(dataset_characteristics):
    try:
        new_dataset = Dataset(
            name=dataset_characteristics[0],
            sample_rate=dataset_characteristics[1],
            duration=dataset_characteristics[2],
            nbr_of_audios=dataset_characteristics[3],
            percentage_training_audios=dataset_characteristics[4],
            add_background_noise=dataset_characteristics[5],
            add_engine_sound=dataset_characteristics[6],
            nbr_sinus=dataset_characteristics[7],
            nbr_sawtooth=dataset_characteristics[8],
            lowpass_filter_engine=dataset_characteristics[9],
            cutoff_frequency_engine=dataset_characteristics[10],
            normalize_engine=dataset_characteristics[11],
            lowpass_filter_sinus=dataset_characteristics[12],
            cutoff_frequency_sinus=dataset_characteristics[13],
            normalize_sinus=dataset_characteristics[14],
            lowpass_filter_sawtooth=dataset_characteristics[15],
            cutoff_frequency_sawtooth=dataset_characteristics[16],
            normalize_sawtooth=dataset_characteristics[17],
            add_alarms=dataset_characteristics[18],
            nbr_alarm_types=dataset_characteristics[19],
            superposition_alarms=dataset_characteristics[20],
            add_voices=dataset_characteristics[21],
            nbr_sources=dataset_characteristics[22],
        )
        session.add(new_dataset)
        session.commit()
        print("Dataset ajouté avec succès.")
        return new_dataset.id
    except Exception as e:
        session.rollback()  # Annule les changements dans la session
        print(f"Erreur lors de l'ajout du dataset : {str(e)}")
        return None


def get_all_datasets():
    return session.query(Dataset).all()


def get_nbr_datasets():
    return session.query(Dataset).count()


def get_dataset_by_id(dataset_id):
    return session.query(Dataset).filter(Dataset.id == dataset_id).first()


def delete_datasets_by_id(dataset_ids):
    for dataset_id in dataset_ids:
        session.query(Dataset).filter(Dataset.id == dataset_id).delete()
        delete_associated_data(dataset_id)
    session.commit()


def delete_associated_data(dataset_id):
    session.query(Audio).filter(Audio.dataset_id == dataset_id).delete()
    session.query(Noise).filter(Noise.dataset_id == dataset_id).delete()
    session.query(SinusoidalWave).filter(
        SinusoidalWave.dataset_id == dataset_id
    ).delete()
    session.query(SawtoothWave).filter(SawtoothWave.dataset_id == dataset_id).delete()
    session.query(Alarm).filter(Alarm.dataset_id == dataset_id).delete()
    session.query(Audio_Alarm).filter(Audio_Alarm.dataset_id == dataset_id).delete()
    session.commit()


def insert_noise(dataset_id, noise_level):
    new_noise = Noise(
        dataset_id=dataset_id,
        min_noise_level=noise_level[0],
        max_noise_level=noise_level[1],
    )
    session.add(new_noise)
    session.commit()
    return new_noise.id


def insert_sinusoidal_wave(dataset_id, sinus_characteristics):
    # st.write(sinus_characteristics)
    new_sinusoid = SinusoidalWave(
        dataset_id=dataset_id,
        min_fundamental_frequency=sinus_characteristics[0],
        max_fundamental_frequency=sinus_characteristics[1],
        min_amplitude=sinus_characteristics[2],
        max_amplitude=sinus_characteristics[3],
        min_nbr_harmonics=sinus_characteristics[4],
        max_nbr_harmonics=sinus_characteristics[5],
        min_starting_intensity_harmonics=sinus_characteristics[6],
        max_starting_intensity_harmonics=sinus_characteristics[7],
    )
    session.add(new_sinusoid)
    session.commit()
    return new_sinusoid.id


def insert_sawtooth_wave(dataset_id, sawtooth_characteristics):
    new_sawtooth = SawtoothWave(
        dataset_id=dataset_id,
        min_fundamental_frequency=sawtooth_characteristics[0],
        max_fundamental_frequency=sawtooth_characteristics[1],
        min_amplitude=sawtooth_characteristics[2],
        max_amplitude=sawtooth_characteristics[3],
        min_nbr_harmonics=sawtooth_characteristics[4],
        max_nbr_harmonics=sawtooth_characteristics[5],
        min_starting_intensity_harmonics=sawtooth_characteristics[6],
        max_starting_intensity_harmonics=sawtooth_characteristics[7],
    )
    session.add(new_sawtooth)
    session.commit()
    return new_sawtooth.id


def insert_alarm(dataset_id, alarm_characteristics):
    new_alarm = Alarm(
        dataset_id=dataset_id,
        min_nbr_alarm=alarm_characteristics[0],
        max_nbr_alarm=alarm_characteristics[1],
        alarm_duration=alarm_characteristics[2],
        alarm_frequency=alarm_characteristics[3],
        alarm_volume=alarm_characteristics[4],
    )
    session.add(new_alarm)
    session.commit()
    return new_alarm.id


def get_all_alarms():
    return session.query(Alarm).all()


def insert_audio(dataset_id, training_audio, audios):
    new_audio = Audio(
        dataset_id=dataset_id,
        training_audio=training_audio,
        audio=audios[0],
        noise=audios[1],
        engine_sound=audios[2],
        alarms=audios[3],
        voices=audios[4],
    )
    session.add(new_audio)
    session.commit()
    return new_audio.id


def clean_sources(sources_audios):
    cleaned_sources_audios = []
    for sources_audio in sources_audios:
        cleaned_sources_audio = [
            source for source in sources_audio if source is not None
        ]
        cleaned_sources_audios.append(cleaned_sources_audio)
    return cleaned_sources_audios


def decode_audio(audio_data, sr):
    if audio_data is not None:
        audio_np, _ = librosa.load(io.BytesIO(audio_data), sr=sr)
        return audio_np
    return None


@st.cache_data
def get_audios_SS1(id_dataset):
    print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    sr = get_dataset_by_id(id_dataset).sample_rate
    print("wwwwwwwwwwwwwwwwwwwww")
    training_audios = (
        session.query(Audio)
        .filter(Audio.dataset_id == id_dataset)
        .filter(Audio.training_audio == True)
        .all()
    )
    validation_audios = (
        session.query(Audio)
        .filter(Audio.dataset_id == id_dataset)
        .filter(Audio.training_audio == False)
        .all()
    )
    training_audios_mixtures = [
        decode_audio(audio.audio, sr) for audio in training_audios
    ]
    validation_audios_mixtures = [
        decode_audio(audio.audio, sr) for audio in validation_audios
    ]
    training_audios_sources = [
        [
            decode_audio(audio.noise, sr),
            decode_audio(audio.engine_sound, sr),
            decode_audio(audio.alarms, sr),
            decode_audio(audio.voices, sr),
        ]
        for audio in training_audios
    ]
    validation_audios_sources = [
        [
            decode_audio(audio.noise, sr),
            decode_audio(audio.engine_sound, sr),
            decode_audio(audio.alarms, sr),
            decode_audio(audio.voices, sr),
        ]
        for audio in validation_audios
    ]

    return (
        sr,
        training_audios_mixtures,
        clean_sources(training_audios_sources),
        validation_audios_mixtures,
        clean_sources(validation_audios_sources),
    )


def save_audios_to_wav_SS1(id_dataset):
    os.makedirs(TRAIN_MIXTURES_AUDIO_PATH_SS1, exist_ok=True)
    os.makedirs(TRAIN_SOURCES_AUDIO_PATH_SS1, exist_ok=True)
    os.makedirs(VAL_MIXTURES_AUDIO_PATH_SS1, exist_ok=True)
    os.makedirs(VAL_SOURCES_AUDIO_PATH_SS1, exist_ok=True)

    sr = get_dataset_by_id(id_dataset).sample_rate

    training_audios = (
        session.query(Audio)
        .filter(Audio.dataset_id == id_dataset)
        .filter(Audio.training_audio == True)
        .all()
    )
    for i, audio in enumerate(training_audios):
        decoded_audio = decode_audio(audio.audio, sr)
        output_file = os.path.join(TRAIN_MIXTURES_AUDIO_PATH_SS1, f"mixture_{i+1}.wav")
        sf.write(output_file, decoded_audio, sr)

        training_sources = [
            decode_audio(audio.noise, sr),
            decode_audio(audio.engine_sound, sr),
            decode_audio(audio.alarms, sr),
            decode_audio(audio.voices, sr),
        ]
        save_audio_sources(training_sources, TRAIN_SOURCES_AUDIO_PATH_SS1, i, sr)

    validation_audios = (
        session.query(Audio)
        .filter(Audio.dataset_id == id_dataset)
        .filter(Audio.training_audio == False)
        .all()
    )

    for i, audio in enumerate(validation_audios):
        decoded_audio = decode_audio(audio.audio, sr)
        output_file = os.path.join(VAL_MIXTURES_AUDIO_PATH_SS1, f"mixture_{i+1}.wav")
        sf.write(output_file, decoded_audio, sr)

        validation_sources = [
            decode_audio(audio.noise, sr),
            decode_audio(audio.engine_sound, sr),
            decode_audio(audio.alarms, sr),
            decode_audio(audio.voices, sr),
        ]
        save_audio_sources(validation_sources, VAL_SOURCES_AUDIO_PATH_SS1, i, sr)


def save_audios_to_wav_SS2(id_dataset):
    os.makedirs(TRAIN_MIXTURES_AUDIO_PATH_SS2, exist_ok=True)
    os.makedirs(TRAIN_SOURCES_AUDIO_PATH_SS2, exist_ok=True)
    os.makedirs(VAL_MIXTURES_AUDIO_PATH_SS2, exist_ok=True)
    os.makedirs(VAL_SOURCES_AUDIO_PATH_SS2, exist_ok=True)

    sr = get_dataset_by_id(id_dataset).sample_rate

    training_audios = (
        session.query(Audio)
        .filter(Audio.dataset_id == id_dataset)
        .filter(Audio.training_audio == True)
        .all()
    )
    for i, audio in enumerate(training_audios):
        decoded_audio = decode_audio(audio.alarms, sr)
        output_file = os.path.join(TRAIN_MIXTURES_AUDIO_PATH_SS2, f"mixture_{i+1}.wav")
        sf.write(output_file, decoded_audio, sr)

    validation_audios = (
        session.query(Audio)
        .filter(Audio.dataset_id == id_dataset)
        .filter(Audio.training_audio == False)
        .all()
    )

    for i, audio in enumerate(validation_audios):
        decoded_audio = decode_audio(audio.alarms, sr)
        output_file = os.path.join(VAL_MIXTURES_AUDIO_PATH_SS2, f"mixture_{i+1}.wav")
        sf.write(output_file, decoded_audio, sr)

    training_audios_sources = (
        session.query(Audio_Alarm)
        .filter(Audio_Alarm.dataset_id == id_dataset)
        .filter(Audio_Alarm.training_audio == True)
        .all()
    )
    training_audios_sources_dict = {}
    for audio_alarm in training_audios_sources:
        audio_id = audio_alarm.audio_id
        if audio_id not in training_audios_sources_dict:
            training_audios_sources_dict[audio_id] = []
        training_audios_sources_dict[audio_id].append(
            decode_audio(audio_alarm.audio_alarm, sr)
        )
    training_audios_sources = list(training_audios_sources_dict.values())
    save_audios_sources(training_audios_sources, TRAIN_SOURCES_AUDIO_PATH_SS2, sr)

    validation_audios_sources = (
        session.query(Audio_Alarm)
        .filter(Audio_Alarm.dataset_id == id_dataset)
        .filter(Audio_Alarm.training_audio == False)
        .all()
    )
    validation_audios_sources_dict = {}
    for audio_alarm in validation_audios_sources:
        audio_id = audio_alarm.audio_id
        if audio_id not in validation_audios_sources_dict:
            validation_audios_sources_dict[audio_id] = []
        validation_audios_sources_dict[audio_id].append(
            decode_audio(audio_alarm.audio_alarm, sr)
        )
    validation_audios_sources = list(validation_audios_sources_dict.values())
    save_audios_sources(validation_audios_sources, VAL_SOURCES_AUDIO_PATH_SS2, sr)


def save_audios_sources(audios_sources, base_path, sr):
    for audio_index, sources_audio in enumerate(audios_sources):
        for source_index, source in enumerate(sources_audio):
            source_folder = os.path.join(base_path, f"source_{source_index+1}")
            os.makedirs(source_folder, exist_ok=True)
            output_file = os.path.join(source_folder, f"mixture_{audio_index+1}.wav")
            sf.write(output_file, source, sr)


def save_audio_sources(audio_sources, base_path, audio_index, sr):
    counter_source = 1
    for i, source in enumerate(audio_sources):
        if source is not None:
            source_folder = os.path.join(base_path, f"source_{counter_source}")
            counter_source += 1
            os.makedirs(source_folder, exist_ok=True)
            output_file = os.path.join(source_folder, f"mixture_{audio_index+1}.wav")
            sf.write(output_file, source, sr)


# def get_sources_SS2_by_audio_id(audio_id, sr):
#     sources = session.query(Audio_Alarm).filter(Audio_Alarm.audio_id == audio_id).all()
#     return [decode_audio(source.audio_alarm, sr) for source in sources]


# @st.cache_data
# def get_audios_SS2(id_dataset, id_selected_SS1_training):
#     sr = get_dataset_by_id(id_dataset).sample_rate
#     training_audios = (
#         session.query(Audio)
#         .filter(Audio.dataset_id == id_dataset)
#         .filter(Audio.training_audio == True)
#         .all()
#     )
#     validation_audios = (
#         session.query(Audio)
#         .filter(Audio.dataset_id == id_dataset)
#         .filter(Audio.training_audio == False)
#         .all()
#     )
#     training_audios_mixtures = [
#         decode_audio(audio.alarms, sr) for audio in training_audios
#     ]
#     validation_audios_mixtures = [
#         decode_audio(audio.alarms, sr) for audio in validation_audios
#     ]
#     training_audios_sources_raw = (
#         session.query(Audio_Alarm)
#         .filter(Audio_Alarm.dataset_id == id_dataset)
#         .filter(Audio_Alarm.training_audio == True)
#         .all()
#     )
#     validation_audios_sources_raw = (
#         session.query(Audio_Alarm)
#         .filter(Audio_Alarm.dataset_id == id_dataset)
#         .filter(Audio_Alarm.training_audio == False)
#         .all()
#     )

#     training_audios_sources_dict = {}
#     for audio_alarm in training_audios_sources_raw:
#         audio_id = audio_alarm.audio_id
#         if audio_id not in training_audios_sources_dict:
#             training_audios_sources_dict[audio_id] = []
#         training_audios_sources_dict[audio_id].append(
#             decode_audio(audio_alarm.audio_alarm, sr)
#         )

#     training_audios_sources = list(training_audios_sources_dict.values())

#     validation_audios_sources_dict = {}
#     for audio_alarm in validation_audios_sources_raw:
#         audio_id = audio_alarm.audio_id
#         if audio_id not in validation_audios_sources_dict:
#             validation_audios_sources_dict[audio_id] = []
#         validation_audios_sources_dict[audio_id].append(
#             decode_audio(audio_alarm.audio_alarm, sr)
#         )

#     validation_audios_sources = list(validation_audios_sources_dict.values())

#     return (
#         sr,
#         training_audios_mixtures,
#         clean_sources(training_audios_sources),
#         validation_audios_mixtures,
#         clean_sources(validation_audios_sources),
#     )


@st.cache_data
def get_audios_SS2(id_dataset, id_selected_SS1_training):
    model = get_pretrained_model_by_id(id_selected_SS1_training)
    sr = get_dataset_by_id(id_dataset).sample_rate
    training_audios = (
        session.query(Audio)
        .filter(Audio.dataset_id == id_dataset)
        .filter(Audio.training_audio == True)
        .all()
    )
    validation_audios = (
        session.query(Audio)
        .filter(Audio.dataset_id == id_dataset)
        .filter(Audio.training_audio == False)
        .all()
    )
    training_audios_mixtures = [
        decode_audio(audio.audio, sr) for audio in training_audios
    ]
    for index_mixture, mixture in enumerate(training_audios_mixtures):
        training_audios_mixtures[index_mixture] = inference_training_SS2(model, mixture)

    validation_audios_mixtures = [
        decode_audio(audio.audio, sr) for audio in validation_audios
    ]
    for index_mixture, mixture in enumerate(validation_audios_mixtures):
        validation_audios_mixtures[index_mixture] = inference_training_SS2(
            model, mixture
        )
    training_audios_sources_raw = (
        session.query(Audio_Alarm)
        .filter(Audio_Alarm.dataset_id == id_dataset)
        .filter(Audio_Alarm.training_audio == True)
        .all()
    )
    validation_audios_sources_raw = (
        session.query(Audio_Alarm)
        .filter(Audio_Alarm.dataset_id == id_dataset)
        .filter(Audio_Alarm.training_audio == False)
        .all()
    )

    training_audios_sources_dict = {}
    for audio_alarm in training_audios_sources_raw:
        audio_id = audio_alarm.audio_id
        if audio_id not in training_audios_sources_dict:
            training_audios_sources_dict[audio_id] = []
        training_audios_sources_dict[audio_id].append(
            decode_audio(audio_alarm.audio_alarm, sr)
        )

    training_audios_sources = list(training_audios_sources_dict.values())

    validation_audios_sources_dict = {}
    for audio_alarm in validation_audios_sources_raw:
        audio_id = audio_alarm.audio_id
        if audio_id not in validation_audios_sources_dict:
            validation_audios_sources_dict[audio_id] = []
        validation_audios_sources_dict[audio_id].append(
            decode_audio(audio_alarm.audio_alarm, sr)
        )

    validation_audios_sources = list(validation_audios_sources_dict.values())

    return (
        sr,
        training_audios_mixtures,
        clean_sources(training_audios_sources),
        validation_audios_mixtures,
        clean_sources(validation_audios_sources),
    )


def insert_audio_alarm(dataset_id, alarm_id, audio_id, training_audio, audio_alarm):
    new_audio_alarm = Audio_Alarm(
        dataset_id=dataset_id,
        alarm_id=alarm_id,
        audio_id=audio_id,
        training_audio=training_audio,
        audio_alarm=audio_alarm,
    )
    session.add(new_audio_alarm)
    session.commit()
    return new_audio_alarm.id


def insert_training(training_characteristics):
    try:
        new_training = Training(
            module=training_characteristics[0],
            name=training_characteristics[1],
            nbr_epochs=training_characteristics[2],
            batch_size=training_characteristics[3],
            learning_rate=training_characteristics[4],
            dataset_name=training_characteristics[5],
            model_name=training_characteristics[6],
            nbr_sources=training_characteristics[7],
            tensorboard_logdir=training_characteristics[8],
            saved_model_path=training_characteristics[9],
        )
        session.add(new_training)
        session.commit()
        print(f"Training ajouté avec succès")
        return new_training.id
    except Exception as e:
        session.rollback()  # Annule les changements dans la session
        print(f"Erreur lors de l'ajout du Training : {str(e)}")
        return None


def get_all_pretrained_models():
    return session.query(Training).all()


def get_pretrained_models_by_module(module):
    return session.query(Training).filter(Training.module == module).all()


def get_pretrained_model_by_id(id):
    return session.query(Training).filter(Training.id == id).first()


def delete_pretrained_model_by_id(ids):
    for training_id in ids:
        training = get_pretrained_model_by_id(training_id)
        remove_folder(training.tensorboard_logdir)
        remove_file(training.saved_model_path)
        session.query(Training).filter(Training.id == training_id).delete()
    session.commit()


def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"{file_path} a été supprimé avec succès.")
    except FileNotFoundError:
        print(f"Le fichier {file_path} n'existe pas.")
    except PermissionError:
        print(f"Permission refusée pour supprimer le fichier {file_path}.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")


def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Le dossier {folder_path} et son contenu ont été supprimés avec succès.")
    except FileNotFoundError:
        print(f"Le dossier {folder_path} n'existe pas.")
    except OSError as e:
        print(f"Erreur lors de la suppression du dossier {folder_path} : {e}")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
