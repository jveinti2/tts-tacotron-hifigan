import os
import re
import json
import numpy as np
import torch
import scipy.signal
import resampy
from hparams import create_hparams
from model import Tacotron2
from audio_processing import griffin_lim
from text import text_to_sequence
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser

# Cargar el diccionario de pronunciación
with open('merged.dict.txt', "r") as file:
    thisdict = {line.split(" ", 1)[0]: line.split(" ", 1)[1].strip() for line in file}

def ARPA(text, punctuation=r"!?,.;", EOS_Token=True):
    out = ''
    for word_ in text.split(" "):
        word = word_
        end_chars = ''
        while any(elem in word for elem in punctuation) and len(word) > 1:
            if word[-1] in punctuation:
                end_chars = word[-1] + end_chars
                word = word[:-1]
            else:
                break
        try:
            word_arpa = thisdict[word.upper()]
            word = "{" + str(word_arpa) + "}"
        except KeyError:
            pass
        out = (out + " " + word + end_chars).strip()
    if EOS_Token and out[-1] != ";":
        out += ";"
    return out

# Cargar modelos
hparams = create_hparams()
hparams.ignore_layers = ["embedding.weight"]
hparams.sampling_rate = 22050  # Asegúrate de que este parámetro es igual
hparams.max_decoder_steps = 3000  # Debe coincidir
hparams.gate_threshold = 0.25  # Debe coincidir
model = Tacotron2(hparams).cuda()
state_dict = torch.load('es_co_fe', map_location=torch.device("cuda"))['state_dict']
model.load_state_dict(state_dict)
model.eval()

def load_hifigan(model_path, config_path):
    hifigan_pretrained_model = f'hifimodel_{config_path}'
    conf = os.path.join(model_path, f'{config_path}.json')
    with open(conf) as f:
        json_config = json.load(f)
    h = AttrDict(json_config)
    torch.cuda.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device("cuda"))
    state_dict_g = torch.load(hifigan_pretrained_model, map_location=torch.device("cuda"))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="normal")
    return hifigan, h, denoiser

# Cargar HiFi-GAN
hifigan, h, denoiser = load_hifigan('hifi-gan', 'config_v1')
hifigan_sr, h2, denoiser_sr = load_hifigan('hifi-gan', 'config_32k')

# Función de síntesis
def end_to_end_infer(text, pronounciation_dictionary):
    audio_segments = []
    for line in [x for x in text.split("\n") if len(x.strip())]:
        if pronounciation_dictionary:
            line = ARPA(line)
        else:
            if line[-1] != ";":
                line += ";"
        
        with torch.no_grad():
            # Convertir texto a secuencia
            sequence = np.array(text_to_sequence(line, ['basic_cleaners']))[None, :]
            sequence = torch.from_numpy(sequence).cuda().long()

            # Generar salida del modelo Tacotron2
            mel_outputs, mel_outputs_postnet, _, _ = model.inference(sequence)

            # Generar audio con HiFi-GAN
            y_g_hat = hifigan(mel_outputs_postnet.float())
            audio = y_g_hat.squeeze() * MAX_WAV_VALUE
            audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0].cpu().numpy().reshape(-1)

            # Resample para superresolución
            wave = resampy.resample(
                audio_denoised,
                h.sampling_rate,
                h2.sampling_rate,
                filter="sinc_window",
                window=scipy.signal.windows.hann,
                num_zeros=8,
            )
            wave_out = wave.astype(np.int16)

            # HiFi-GAN super-resolución
            wave = wave / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(torch.device("cuda"))
            new_mel = mel_spectrogram(
                wave.unsqueeze(0),
                h2.n_fft,
                h2.num_mels,
                h2.sampling_rate,
                h2.hop_size,
                h2.win_size,
                h2.fmin,
                h2.fmax,
            )
            y_g_hat2 = hifigan_sr(new_mel)
            audio2 = y_g_hat2.squeeze() * MAX_WAV_VALUE
            audio2_denoised = denoiser(audio2.view(1, -1), strength=35)[:, 0].cpu().numpy().reshape(-1)

            # Mezcla y normalización
            audio2_denoised = audio2_denoised.cpu().numpy().reshape(-1)
            b = scipy.signal.firwin(101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False)
            y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
            y_out = y.astype(np.int16)

            # Combinar audio original y procesado
            sr_mix = wave_out + np.pad(y_out, (0, max(0, wave_out.size - y_out.size)), 'constant')
            sr_mix = sr_mix / np.max(np.abs(sr_mix))
            audio_segments.append(sr_mix)

    return audio_segments

# Ejemplo de uso
text = "Hola, ¿cómo estás?"
end_to_end_infer(text, pronounciation_dictionary=True)
