import torch
import librosa
import numpy as np
from asteroid.models import ConvTasNet, DPRNNTasNet, LSTMTasNet, DPTNet, SuDORMRFNet
from asteroid.losses import (
    PITLossWrapper,
    pairwise_neg_sisdr,
    pairwise_neg_snr,
    pairwise_mse,
)
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
import json
import streamlit as st
import threading
from db.database import session
import shutil

model_classes = {
    "ConvTasNet": ConvTasNet,
    "DPRNNTasNet": DPRNNTasNet,
    "LSTMTasNet": LSTMTasNet,
    "DPTNet": DPTNet,
    "SuDORMRFNet": SuDORMRFNet,
}


class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def get_session_state(**kwargs):
    session_state = st.session_state.get("session_state", None)
    if session_state is None:
        session_state = SessionState(**kwargs)
        st.session_state["session_state"] = session_state
    return session_state


def set_json_infos(path, data):
    with open(path, "w") as destination_json_file:
        json.dump(data, destination_json_file, indent=4)


def compute_model_index(path, name):
    try:
        existing_folders = [f.name for f in os.scandir(path) if f.is_dir()]
    except FileNotFoundError:
        existing_folders = []
    matching_folders = [
        folder for folder in existing_folders if folder.startswith(name)
    ]
    idx = len(matching_folders) + 1
    return idx


class MyDataset(Dataset):
    def __init__(self, mixtures, sources, sr=8000):
        self.sr = sr
        self.mixtures = mixtures
        self.sources = sources

    def __len__(self):
        return len(self.mixtures)

    def __getitem__(self, idx):
        mixture = self.mixtures[idx]
        if mixture is not None:
            mixture_np = np.array(mixture, dtype=np.float32)

            sources = []
            for source in self.sources[idx]:
                source_np = np.array(source, dtype=np.float32)
                sources.append(source_np)
            sources = np.stack(sources, axis=0)

            mixture_tensor = torch.tensor(mixture_np, dtype=torch.float32)
            sources_tensor = torch.tensor(sources, dtype=torch.float32)

        return mixture_tensor, sources_tensor


# @st.cache_data
def prepare_data(
    sample_rate,
    training_audios_mixtures,
    training_audios_sources,
    validation_audios_mixtures,
    validation_audios_sources,
    batch_size,
):
    train_set = MyDataset(
        mixtures=training_audios_mixtures,
        sources=training_audios_sources,
        # sr=sample_rate,
    )
    val_set = MyDataset(
        mixtures=validation_audios_mixtures,
        sources=validation_audios_sources,
        # sr=sample_rate,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=11,
        # persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=11,
        # persistent_workers=True,
    )

    return train_loader, val_loader


def prepare_model(model_name, nbr_sources):
    if model_name in model_classes:
        model = model_classes[model_name](nbr_sources)
    else:
        raise ValueError(f"Model {model_name} not found")
    return model


def fit_trainer(trainer, system, model_name, model, training_name):
    trainer.fit(system)
    save_model(model_name, model, training_name)


# def prepare_model(model_name, nbr_sources):
#     model = None
#     if model_name == "ConvTasNet":
#         # model = ConvTasNet(
#         #     n_src=nbr_sources,
#         #     n_repeats=3,
#         #     n_blocks=8,
#         #     n_filters=512,
#         #     kernel_size=3,
#         #     stride=16,
#         # )
#         model = ConvTasNet(n_src=nbr_sources)
#     elif model_name == "DPRNNTasNet":
#         # model = DPRNNTasNet(
#         #     n_src=nbr_sources,
#         #     n_repeats=3,
#         #     n_blocks=8,
#         #     n_filters=512,
#         #     kernel_size=3,
#         #     stride=16,
#         # )
#         model = DPRNNTasNet(n_src=nbr_sources)
#     elif model_name == "LSTMTasNet":
#         # model = LSTMTasNet(
#         #     n_src=nbr_sources,
#         #     n_repeats=3,
#         #     n_blocks=8,
#         #     n_filters=512,
#         #     kernel_size=3,
#         #     stride=16,
#         # )
#         print(f"nbr_sources : {nbr_sources}")
#         model = LSTMTasNet()
#     elif model_name == "DPTNet":
#         # model = DPTNet(
#         #     n_src=nbr_sources,
#         #     n_repeats=3,
#         #     n_blocks=8,
#         #     n_filters=512,
#         #     kernel_size=3,
#         #     stride=16,
#         # )
#         model = DPTNet(n_src=nbr_sources)
#     elif model_name == "SuDORMRFNet":
#         # model = SuDORMRFNet(
#         #     n_src=nbr_sources,
#         #     n_repeats=1,
#         #     n_blocks=2,
#         #     # n_filters=512,
#         #     # kernel_size=3,
#         #     # stride=16,
#         # )
#         model = SuDORMRFNet(n_src=nbr_sources)
#     else:
#         st.error(f"Unknown model name: {model_name}")
#         return
#     return model


def start_training(
    session_state,
    training_name,
    train_loader,
    val_loader,
    nbr_epoch,
    model_name,
    batch_size,
    nbr_sources,
    learning_rate,
):
    session_state.training_status = "Training started..."
    model = prepare_model(model_name, nbr_sources)

    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    optimizer = torch.optim.Adam(model.parameters(), lr=10**learning_rate)
    system = System(
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader,
    )

    logger = TensorBoardLogger(
        "tb_logs",
        name=model_name,
        version=training_name,
    )

    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        max_epochs=nbr_epoch,
        # callbacks=[checkpoint_callback],
        logger=logger,
        accelerator=device,
        devices=1,
        precision=16,
        accumulate_grad_batches=1,
    )

    session_state.training_status = "Training in progress"
    thread = threading.Thread(
        target=fit_trainer, args=(trainer, system, model_name, model, training_name)
    )
    thread.start()
    session_state.training_status = "Training finished!"


def save_model(model_name, model, training_name):
    path_model = os.path.join("models", model_name)
    os.makedirs(path_model, exist_ok=True)
    model_version = len(os.listdir(path_model))
    torch.save(model.state_dict(), os.path.join(path_model, f"{training_name}.ckpt"))


def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        st.error(f"NaN detected in {name}")
