import torch
from asteroid.models import ConvTasNet, DPRNNTasNet, LSTMTasNet, DPTNet, SuDORMRFNet
import re
import librosa
import os
import soundfile as sf

model_classes = {
    "ConvTasNet": ConvTasNet,
    "DPRNNTasNet": DPRNNTasNet,
    "LSTMTasNet": LSTMTasNet,
    "DPTNet": DPTNet,
    "SuDORMRFNet": SuDORMRFNet,
}


def extract_number(expression):
    match = re.search(r"(\d+) - .+", expression)
    if match:
        return int(match.group(1))
    else:
        return None


# def prepare_model(model, nbr_sources):
#     selected_training = get_pretrained_model_by_id(extract_number(model))
#     model_type = selected_training.model_name
#     try:
#         model_class = model_classes[model_type]
#     except KeyError:
#         raise ValueError(f"Model {model_type} not found")
#     if model_type == "SuDORMRFNet":
#         model = model_class(
#             n_src=nbr_sources,
#             n_repeats=1,
#             n_blocks=2,
#         )
#     else:
#         model = model_class(
#             n_src=nbr_sources,
#             n_repeats=3,
#             n_blocks=8,
#             n_filters=512,
#             kernel_size=3,
#             stride=16,
#         )
#     model.load_state_dict(torch.load(selected_training.saved_model_path), strict=False)
#     model.eval()
#     return model


def prepare_model(model):
    model_type = model.model_name
    try:
        model_class = model_classes[model_type]
    except KeyError:
        raise ValueError(f"Model {model_type} not found")
    setup_model = model_class(n_src=model.nbr_sources)
    setup_model.load_state_dict(torch.load(model.saved_model_path), strict=False)
    setup_model.eval()
    return setup_model


def inference_SS1(audio_input, model, sr):
    model = prepare_model(model)
    audio, _ = librosa.load(
        os.path.join(os.getcwd(), "inference_audios", audio_input.name), sr=8000
    )
    audio = torch.tensor(audio[None, None, :])
    with torch.no_grad():
        separated_audios = model(audio)

    # test = separated_audios
    # test = test.cpu().numpy()
    # os.makedirs(os.path.join(os.getcwd(), "output"), exist_ok=True)
    # version = len(os.listdir(os.path.join(os.getcwd(), "output"))) + 1
    # for i, source in enumerate(test[0]):
    #     sf.write(
    #         os.path.join(
    #             "output",
    #             f"audio_{version+i}.wav",
    #         ),
    #         source.T,
    #         8000,
    #     )
    return audio, separated_audios.cpu().numpy()


def inference_SS2(audio_input, model, sr):
    model = prepare_model(model)
    audio = torch.tensor(audio_input[None, None, :])
    with torch.no_grad():
        separated_audios = model(audio)
    return audio, separated_audios.cpu().numpy()


# TODO : methode pour s'adapter uax audios qu on a et aux sources
def inference_training_SS2(model_training_SS1, audio):
    model = prepare_model(model_training_SS1)
    audio = torch.tensor(audio[None, None, :])
    with torch.no_grad():
        separated_audios = model(audio)
    separated_audios = separated_audios.cpu().numpy()
    return separated_audios[0][1].T
