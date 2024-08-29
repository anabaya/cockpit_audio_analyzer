from re import A

from pyparsing import null_debug_action
from sklearn.calibration import column_or_1d
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    LargeBinary,
    ForeignKey,
    Boolean,
    Float,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    sample_rate = Column(Integer, nullable=False)
    duration = Column(Integer, nullable=False)
    nbr_of_audios = Column(Integer, nullable=False)
    percentage_training_audios = Column(Integer, nullable=False)

    add_background_noise = Column(Boolean, nullable=False)
    add_engine_sound = Column(Boolean, nullable=False)
    nbr_sinus = Column(Integer, nullable=False)
    nbr_sawtooth = Column(Integer, nullable=False)
    add_alarms = Column(Boolean, nullable=False)
    nbr_alarm_types = Column(Integer, nullable=False)
    superposition_alarms = Column(Boolean, nullable=False)
    add_voices = Column(Boolean, nullable=False)
    nbr_sources = Column(Integer, nullable=False)
    lowpass_filter_engine = Column(Boolean, nullable=False)
    cutoff_frequency_engine = Column(Integer, nullable=True)
    normalize_engine = Column(Boolean, nullable=False)
    lowpass_filter_sinus = Column(Boolean, nullable=False)
    cutoff_frequency_sinus = Column(Integer, nullable=True)
    normalize_sinus = Column(Boolean, nullable=False)
    lowpass_filter_sawtooth = Column(Boolean, nullable=False)
    cutoff_frequency_sawtooth = Column(Integer, nullable=True)
    normalize_sawtooth = Column(Boolean, nullable=False)

    # audios = relationship('Audio', back_populates='dataset', cascade="all, delete-orphan", passive_deletes=True)
    noises = relationship("Noise", back_populates="dataset")
    sinusoidal_waves = relationship("SinusoidalWave", back_populates="dataset")
    sawtooth_waves = relationship("SawtoothWave", back_populates="dataset")
    alarms = relationship("Alarm", back_populates="dataset")
    audios = relationship("Audio", back_populates="dataset")
    audio_alarms = relationship("Audio_Alarm", back_populates="dataset")
    # training = relationship('Training', back_populates='dataset')


class Noise(Base):
    __tablename__ = "noises"
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    min_noise_level = Column(Integer)
    max_noise_level = Column(Integer)
    dataset = relationship("Dataset", back_populates="noises")


class SinusoidalWave(Base):
    __tablename__ = "sinusoidal_waves"
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    min_fundamental_frequency = Column(Integer, nullable=False)
    max_fundamental_frequency = Column(Integer, nullable=False)
    min_amplitude = Column(Integer, nullable=False)
    max_amplitude = Column(Integer, nullable=False)
    min_nbr_harmonics = Column(Integer, nullable=False)
    max_nbr_harmonics = Column(Integer, nullable=False)
    min_starting_intensity_harmonics = Column(Integer, nullable=False)
    max_starting_intensity_harmonics = Column(Integer, nullable=False)
    dataset = relationship("Dataset", back_populates="sinusoidal_waves")


class SawtoothWave(Base):
    __tablename__ = "sawtooth_waves"
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    min_fundamental_frequency = Column(Integer, nullable=False)
    max_fundamental_frequency = Column(Integer, nullable=False)
    min_amplitude = Column(Integer, nullable=False)
    max_amplitude = Column(Integer, nullable=False)
    min_nbr_harmonics = Column(Integer, nullable=False)
    max_nbr_harmonics = Column(Integer, nullable=False)
    min_starting_intensity_harmonics = Column(Integer, nullable=False)
    max_starting_intensity_harmonics = Column(Integer, nullable=False)
    dataset = relationship("Dataset", back_populates="sawtooth_waves")


class Alarm(Base):
    __tablename__ = "alarms"
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    min_nbr_alarm = Column(Integer)
    max_nbr_alarm = Column(Integer)
    alarm_duration = Column(Integer)
    alarm_frequency = Column(Integer)
    alarm_volume = Column(Integer)
    dataset = relationship("Dataset", back_populates="alarms")
    audio_alarms = relationship("Audio_Alarm", back_populates="alarm")


class Audio(Base):
    __tablename__ = "audios"
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    training_audio = Column(Boolean, nullable=False)
    audio = Column(LargeBinary, nullable=False)
    noise = Column(LargeBinary, nullable=True)
    engine_sound = Column(LargeBinary, nullable=True)
    alarms = Column(LargeBinary, nullable=True)
    voices = Column(LargeBinary, nullable=True)
    dataset = relationship("Dataset", back_populates="audios")
    audio_alarms = relationship("Audio_Alarm", back_populates="audio")


class Audio_Alarm(Base):
    __tablename__ = "audio_alarms"
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    alarm_id = Column(Integer, ForeignKey("alarms.id"), nullable=False)
    audio_id = Column(Integer, ForeignKey("audios.id"), nullable=False)
    training_audio = Column(Integer, nullable=False)
    audio_alarm = Column(LargeBinary, nullable=False)
    dataset = relationship("Dataset", back_populates="audio_alarms")
    alarm = relationship("Alarm", back_populates="audio_alarms")
    audio = relationship("Audio", back_populates="audio_alarms")


class Training(Base):
    __tablename__ = "training"
    id = Column(Integer, primary_key=True, autoincrement=True)
    module = Column(String, nullable=False)
    name = Column(String, nullable=False, unique=True)
    nbr_epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    dataset_name = Column(String(100), nullable=False)
    model_name = Column(String, nullable=False)
    nbr_sources = Column(Integer, nullable=False)
    tensorboard_logdir = Column(String, nullable=False)
    saved_model_path = Column(String, nullable=False)
    # dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    # dataset = relationship('Dataset', back_populates='training')
