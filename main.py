import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'tacotron2'))
sys.path.append(os.path.join(os.getcwd(), 'hifi-gan'))


import re
import json
import numpy as np
import torch
from num2words import num2words
from hparams import create_hparams
from model import Tacotron2
from text import text_to_sequence
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser
import scipy.signal
import resampy

class TTSSystem:
    def __init__(self, superres_strength=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.superres_strength = superres_strength
        self.load_pronunciation_dict()
        self.initialize_tacotron()
        self.initialize_hifigan()
    
    def load_pronunciation_dict(self):
        try:
            with open('merged.dict.txt', "r", encoding='utf-8') as file:
                self.pronunciation_dict = {}
                for line in reversed(file.readlines()):
                    if " " in line:
                        key, value = line.split(" ", 1)
                        self.pronunciation_dict[key] = value.strip()
        except FileNotFoundError:
            print("Warning: Pronunciation dictionary not found.")
            self.pronunciation_dict = {}

    def initialize_tacotron(self):
        hparams = create_hparams()
        hparams.sampling_rate = 22050
        hparams.max_decoder_steps = 3000
        hparams.gate_threshold = 0.25
        
        self.model = Tacotron2(hparams).to(self.device)
        state_dict = torch.load('es_co_fe', map_location=self.device)['state_dict']
        
        # Check for MMI model
        if any(True for x in state_dict.keys() if "mi." in x):
            raise Exception("MMI models are not supported")
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.hparams = hparams

    def initialize_hifigan(self):
        def load_hifigan_model(config_path):
            conf_file = os.path.join("hifi-gan", f'{config_path}.json')
            with open(conf_file) as f:
                config = AttrDict(json.load(f))
            
            generator = Generator(config).to(self.device)
            checkpoint = torch.load(f'hifimodel_{config_path}', 
                                  map_location=self.device)
            generator.load_state_dict(checkpoint["generator"])
            generator.eval()
            generator.remove_weight_norm()
            
            return generator, config, Denoiser(generator)

        self.hifigan, self.h, self.denoiser = load_hifigan_model('config_v1')
        self.hifigan_sr, self.h2, self.denoiser_sr = load_hifigan_model('config_32k')

    def convert_numbers_to_words(self, text, lang='es'):
        """Convert numbers in text to words."""
        words = text.split()
        converted = []
        for word in words:
            if word.replace('.', '').isdigit():
                converted.append(num2words(float(word), lang=lang))
            else:
                converted.append(word)
        return ' '.join(converted)

    def process_text(self, text, use_pronunciation=False):
        # Clean and normalize text
        text = re.sub('(\d+(\.\d+)?)', r' \1 ', text)
        text = text.strip()
        
        # Convert numbers to words
        text = self.convert_numbers_to_words(text)
        
        if use_pronunciation:
            words = text.split()
            processed = []
            for word in words:
                try:
                    arpa = self.pronunciation_dict.get(word.upper())
                    if arpa:
                        processed.append(f"{{{arpa}}}")
                    else:
                        processed.append(word)
                except KeyError:
                    processed.append(word)
            text = ' '.join(processed)
        
        if not text[-1] in "!?.,;":
            text += ";"
            
        return text

    def synthesize(self, text, use_pronunciation=False):
        """Generate audio from text."""
        processed_text = self.process_text(text, use_pronunciation)
        
        with torch.no_grad():
            sequence = torch.LongTensor(
                text_to_sequence(processed_text, ['basic_cleaners'])
            ).unsqueeze(0).to(self.device)
            
            mel_outputs, mel_outputs_postnet, _, _ = self.model.inference(sequence)
            
            # Generate initial audio
            y_g_hat = self.hifigan(mel_outputs_postnet.float())
            audio = y_g_hat.squeeze() * MAX_WAV_VALUE
            audio_denoised = self.denoiser(audio.view(1, -1), strength=35)[:, 0]
            
            # Normalize before super-resolution
            audio_denoised = audio_denoised.cpu().numpy().reshape(-1)
            normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
            audio_denoised = audio_denoised * normalize
            
            # Resample for super-resolution
            wave = resampy.resample(
                audio_denoised,
                self.h.sampling_rate,
                self.h2.sampling_rate,
                filter="sinc_window",
                window=scipy.signal.windows.hann,
                num_zeros=8
            )
            wave_out = wave.astype(np.int16)
            
            # Super-resolution
            wave = wave / MAX_WAV_VALUE
            wave = torch.FloatTensor(wave).to(self.device)
            new_mel = mel_spectrogram(
                wave.unsqueeze(0),
                self.h2.n_fft,
                self.h2.num_mels,
                self.h2.sampling_rate,
                self.h2.hop_size,
                self.h2.win_size,
                self.h2.fmin,
                self.h2.fmax
            )
            
            y_g_hat2 = self.hifigan_sr(new_mel)
            audio2 = y_g_hat2.squeeze() * MAX_WAV_VALUE
            audio2_denoised = self.denoiser_sr(audio2.view(1, -1), strength=35)[:, 0]
            
            # High-pass filter and mixing
            audio2_denoised = audio2_denoised.cpu().numpy().reshape(-1)
            b = scipy.signal.firwin(101, cutoff=10500, fs=self.h2.sampling_rate, pass_zero=False)
            y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
            
            # Apply super-resolution strength
            y *= self.superres_strength
            
            # Final mixing
            y_out = y.astype(np.int16)
            y_padded = np.zeros(wave_out.shape)
            y_padded[:y_out.shape[0]] = y_out
            sr_mix = wave_out + y_padded
            sr_mix = sr_mix / normalize
            
            return sr_mix.astype(np.int16), self.h2.sampling_rate

if __name__ == "__main__":
    tts = TTSSystem(superres_strength=1.0)
    
    while True:
        try:
            print("-" * 50)
            text = input("Enter text (Ctrl+C to exit): ")
            if not text.strip():
                continue
                
            audio, sr = tts.synthesize(text)
            
            from scipy.io.wavfile import write
            output_path = "output_audio.wav"
            write(output_path, sr, audio)
            print(f"Generated: {output_path}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            continue