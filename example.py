import os
import re
import json
import numpy as np
import torch
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser
import scipy.signal
import resampy
import num2words
import matplotlib.pyplot as plt

def plot_data(data, figsize=(9, 3.6)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower',
                       interpolation='none', cmap='inferno')
    plt.show()

# Setup Pronunciation Dictionary
pronunciation_dict_path = 'merged.dict.txt'
pronunciation_dict = {}
if os.path.exists(pronunciation_dict_path):
    with open(pronunciation_dict_path, "r") as f:
        for line in f:
            word, pronunciation = line.strip().split(" ", 1)
            pronunciation_dict[word.upper()] = pronunciation

def ARPA(text, punctuation=r"!?,.;", EOS_Token=True):
    out = ''
    for word_ in text.split(" "):
        word = word_
        end_chars = ''
        while any(c in punctuation for c in word) and len(word) > 1:
            if word[-1] in punctuation:
                end_chars = word[-1] + end_chars
                word = word[:-1]
            else:
                break
        word_arpa = pronunciation_dict.get(word.upper(), word)
        word = f"{{{word_arpa}}}" if word_arpa != word else word
        out = f"{out} {word}{end_chars}".strip()
    return f"{out};" if EOS_Token and out[-1] != ';' else out

def load_hifigan(model_path, config_path):
    with open(config_path, 'r') as f:
        hifigan_config = AttrDict(json.load(f))
    hifigan = Generator(hifigan_config).to(torch.device("cuda"))
    state_dict = torch.load(model_path, map_location="cuda")
    hifigan.load_state_dict(state_dict["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="normal")
    return hifigan, hifigan_config, denoiser

def load_tacotron2(model_path):
    hparams = create_hparams()
    hparams.ignore_layers = ["embedding.weight"]
    model = Tacotron2(hparams).cuda()
    state_dict = torch.load(model_path, map_location="cuda")['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    return model, hparams

def end_to_end_infer(text, pron_dict, model, hifigan, hifigan_sr, denoiser, hifigan_config, hifigan_sr_config, superres_strength=1.0):
    for i in [x for x in text.split("\n") if x.strip()]:
        if pron_dict:
            i = ARPA(i)
        sequence = np.array(text_to_sequence(i, ['basic_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

            y_g_hat = hifigan(mel_outputs_postnet.float())
            audio = y_g_hat.squeeze() * MAX_WAV_VALUE
            audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0].cpu().numpy().reshape(-1)

            normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
            audio_denoised *= normalize

            wave = resampy.resample(
                audio_denoised,
                hifigan_config.sampling_rate,
                hifigan_sr_config.sampling_rate,
                filter="sinc_window",
                window=scipy.signal.windows.hann,
                num_zeros=8,
            )
            wave_out = wave.astype(np.int16)

            wave = wave / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(torch.device("cuda"))
            new_mel = mel_spectrogram(
                wave.unsqueeze(0),
                hifigan_sr_config.n_fft,
                hifigan_sr_config.num_mels,
                hifigan_sr_config.sampling_rate,
                hifigan_sr_config.hop_size,
                hifigan_sr_config.win_size,
                hifigan_sr_config.fmin,
                hifigan_sr_config.fmax,
            )
            
            y_g_hat2 = hifigan_sr(new_mel)
            audio2 = y_g_hat2.squeeze() * MAX_WAV_VALUE
            audio2_denoised = denoiser(audio2.view(1, -1), strength=35)[:, 0].cpu().numpy().reshape(-1)

            b = scipy.signal.firwin(101, cutoff=10500, fs=hifigan_sr_config.sampling_rate, pass_zero=False)
            y = scipy.signal.lfilter(b, [1.0], audio2_denoised) * superres_strength
            y_out = y.astype(np.int16)

            y_padded = np.zeros(wave_out.shape)
            y_padded[:y_out.shape[0]] = y_out

            sr_mix = wave_out + y_padded
            sr_mix /= normalize

model_path_tacotron2 = "MLPTTS"
model_path_hifigan = "hifimodel_config_v1"
config_path_hifigan = "hifi-gan/config_v1.json"
model_path_hifigan_sr = "hifimodel_config_32k"
config_path_hifigan_sr = "hifi-gan/config_32k.json"

# Load models
model, hparams = load_tacotron2(model_path_tacotron2)
hifigan, hifigan_config, denoiser = load_hifigan(model_path_hifigan, config_path_hifigan)
hifigan_sr, hifigan_sr_config, denoiser_sr = load_hifigan(model_path_hifigan_sr, config_path_hifigan_sr)

pronunciation_dictionary = False
superres_strength = 1.0

print("Introduce tu texto:")
while True:
    try:
        line = input("Texto: ").strip()
        if not line:
            continue
        line = ' '.join([num2words.num2words(i, lang='es') if i.isdigit() else i for i in line.split()])
        end_to_end_infer(line, pronunciation_dictionary, model, hifigan, hifigan_sr, denoiser, hifigan_config, hifigan_sr_config, superres_strength)
    except KeyboardInterrupt:
        print("Ejecución detenida.")
        break
