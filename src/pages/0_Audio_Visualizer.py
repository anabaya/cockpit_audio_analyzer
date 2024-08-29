import streamlit as st
import wave
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from matplotlib.widgets import Cursor
import streamlit.components.v1 as components
import base64
import pydub
import mpld3
from simplification.cutil import simplify_coords
from mpld3 import plugins


@st.cache_data
def display_fourier_spectrum(audio_array, frequency_rate, nb_samples, epsilon):
    y_fft = fft(audio_array)
    freq = np.linspace(0, frequency_rate / 2, nb_samples // 2)
    magnitude = np.abs(y_fft[:nb_samples // 2])
    coords = np.column_stack((freq, magnitude))
    simplified_coords = simplify_coords(coords, epsilon)

    figure = plt.figure(figsize=(12, 5))
    st.write(f"The chart right below is made out of: {len(simplified_coords[:, 0])} points")
    plt.plot(simplified_coords[:, 0], simplified_coords[:, 1])
    plt.title('Magnitude de la FFT du Signal')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude')

    figure_html = mpld3.fig_to_html(figure)
    components.html(figure_html, height=600)


@st.cache_data
def save_spectrogram(audio_array, x_lim, frequency_rate):
    plt.figure(figsize=(15, 5))
    plt.specgram(audio_array, Fs=frequency_rate, vmin=-50, vmax=50)
    # plt.title('Frequency')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (s)')
    plt.yticks([])
    plt.xticks([])
    plt.xlim(0, x_lim)
    plt.savefig("static/spectrogram.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    # plt.colorbar()
    # st.pyplot(plt)


@st.cache_data
def save_amplitude_graph(audio_array, x_lim, x_array):
    plt.figure(figsize=(15, 5))
    plt.plot(x_array, audio_array)
    # plt.title('Amplitude')
    # plt.ylabel('Signal Value')
    # plt.xlabel('Time (s)')
    plt.yticks([])
    plt.xticks([])
    plt.xlim(0, x_lim)
    plt.savefig("static/amplitude_graph.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    # st.pyplot(plt)

def display_audio_and_infos(selected_audio, sample_frequency, n_samples):
    st.write(selected_audio.name)
    st.write("sample frequency : ", sample_frequency)
    st.write("number of samples : ", n_samples)
    # st.audio(selected_audio.name, format=selected_audio.type)


def load_audio(selected_audio):
    wave_obj = wave.open(f"static/{selected_audio.name}", 'rb')
    sample_frequency = wave_obj.getframerate()
    n_samples = wave_obj.getnframes()
    t_audio = n_samples/sample_frequency
    signal_wave = wave_obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)
    if  wave_obj.getnchannels()!=1:
        signal_array = signal_array[0::2]
    return signal_array, sample_frequency, n_samples

def prepare_audio_for_display(selected_audio):
    audio_bytes = selected_audio.getvalue()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    audio_url = f"data:audio/wav;base64,{audio_base64}"
    return audio_url



st.set_page_config(page_title="Audio_Visualizer", layout="wide")
st.markdown("# Audio Visualizer")

audio = st.file_uploader("Upload your audio file", type=["wav"])

if audio:
    if audio.name.find(".wav") != -1:
        temp = pydub.AudioSegment.from_wav(audio)
        temp.export(f"static/{audio.name}", format="wav")


    signal_array, sample_frequency, n_samples = load_audio(audio)
    display_audio_and_infos(audio, sample_frequency, n_samples)

    t_audio = n_samples/sample_frequency
    times = np.linspace(0, n_samples/sample_frequency, num = n_samples)

    audio_url = prepare_audio_for_display(audio)
    save_amplitude_graph(signal_array, t_audio, times)
    save_spectrogram(signal_array, t_audio, sample_frequency)

    components.html(
        """
            <!DOCTYPE html>
            <html>
            <head>
            <title>Title of the document</title>
            <style>
                #container {
                    width: 100%;
                }

                .chart {
                    position: relative; /* Important to work with absolute */
                    box-sizing: border-box;
                    height: 300px;
                    border: 1px solid;
                    border-color: black;
                    margin: 20px;
                    background-size: 100% 100%;
                    background-repeat: no-repeat;
                    background-position: center;
                    //background-attachment: fixed;
                }

                .bar {
                    position: absolute;
                    border: 1px solid;
                    border-color: white;
                    top: 1px;
                    right: 0px;
                    width: 100;
                    height: 296px;
                    background-color: white;
                    opacity: 1;
                }

                #first {
                    background-image: url("/app/static/amplitude_graph.png");
                }

                #second {
                    background-image: url("/app/static/spectrogram.png");

                }

            </style>
            </head>

            <body>
            <div id="container">
            <figure>
                <center>
                    <audio controls src="[[audio_url]]"
                        id="audio" style="width: 100%"></audio>
                </center>
            </figure>
            <div id="first" class="chart">
                <div class="bar"></div>
            </div>
            <div id="second" class="chart">
                <div class="bar"></div>
            </div>

            </div>

            <script>
                const audio = document.getElementById('audio');

                const updateBars = () => {
                    const progress = audio.currentTime / audio.duration * 100;
                    document.querySelectorAll('.bar').forEach(bar => {
                        bar.style.width = 100-progress + '%';
                    });
                };

                audio.addEventListener('timeupdate', updateBars);
            </script>
            </body>

            </html>
        """.replace("[[audio_url]]", f"/app/static/{audio.name}"),
        height = 900
    )

    st.write("## Fourier Spectrum")
    value = st.slider("Select a value for Epsilon", 0, 100, 5)
    display_fourier_spectrum(signal_array, sample_frequency, n_samples, value)
