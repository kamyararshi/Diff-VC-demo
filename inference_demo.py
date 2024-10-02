import argparse
import json
import os
import numpy as np
from tqdm import tqdm
import soundfile as sf
import torch
use_gpu = torch.cuda.is_available()

import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)

import params
from model import DiffVC

import sys
sys.path.append('hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path


class GenDiffVC(): 
    def __init__(self, vc_path, hfg_path, enc_path, DEVICE, tgt_path, output_path="./output_demo"):

        self.device = DEVICE
        self._load_models(vc_path, hfg_path, enc_path)
        self._set_prompt(tgt=tgt_path)

        
    def _load_models(self, vc_path='checkpts/vc/vc_libritts_wodyn.pt', hfg_path='checkpts/vocoder/', enc_path='checkpts/spk_encoder/pretrained.pt'):
        #  load voice conversion 
        generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads, 
                        params.layers, params.kernel, params.dropout, params.window_size, 
                        params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim, 
                        params.beta_min, params.beta_max)
                    
        generator = generator.to(self.device)
        generator.load_state_dict(torch.load(vc_path))
        generator.eval()


        # loading HiFi-GAN vocoder
         # HiFi-GAN path

        with open(hfg_path + 'config.json') as f:
            h = AttrDict(json.load(f))

        hifigan_universal = HiFiGAN(h).to(self.device)
        hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])

        _ = hifigan_universal.eval()
        hifigan_universal.remove_weight_norm()


        # loading speaker encoder
        enc_model_fpath = Path(enc_path) # speaker encoder path
        spk_encoder.load_model(enc_model_fpath, device=self.device)

        self.generator = generator
        self.spk_encoder = spk_encoder
        self.hifigan_universal = hifigan_universal

    def get_mel(self, wav, sr=None):
        if isinstance(wav, str) and os.path.exists(wav):
            wav, _ = load(wav, sr=22050)
        elif isinstance(wav, np.ndarray) and sr!=None:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=22050)
        else:
            raise ValueError(type(audio))
        
        wav = wav[:(wav.shape[0] // 256)*256]
        wav = np.pad(wav, 384, mode='reflect')
        stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
        stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
        mel_spectrogram = np.matmul(mel_basis, stftm)
        log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
        return log_mel_spectrogram

    def get_embed(self, wav_path):
        wav_preprocessed = self.spk_encoder.preprocess_wav(wav_path)
        embed = self.spk_encoder.embed_utterance(wav_preprocessed)
        return embed

    def noise_median_smoothing(self, x, w=5):
        y = np.copy(x)
        x = np.pad(x, w, "edge")
        for i in range(y.shape[0]):
            med = np.median(x[i:i+2*w+1])
            y[i] = min(x[i+w+1], med)
        return y

    def mel_spectral_subtraction(self, mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5):
        mel_len = mel_source.shape[-1]
        energy_min = 100000.0
        i_min = 0
        for i in range(mel_len - silence_window):
            energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i+silence_window]))
            if energy_cur < energy_min:
                i_min = i
                energy_min = energy_cur
        estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min+silence_window]), axis=-1)
        if smoothing_window is not None:
            estimated_noise_energy = self.noise_median_smoothing(estimated_noise_energy, smoothing_window)
        mel_denoised = np.copy(mel_synth)
        for i in range(mel_len):
            signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
            estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
            mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
        return mel_denoised


    @torch.no_grad()
    def _set_prompt(self, tgt, sr=None):
        self.mel_target = (torch.from_numpy(self.get_mel(tgt, sr)).float().unsqueeze(0)).to(self.device)
        self.mel_target_lengths = (torch.LongTensor([self.mel_target.shape[-1]])).to(self.device)
        self.embed_target = (torch.from_numpy(self.get_embed(tgt)).float().unsqueeze(0)).to(self.device)
        

    @torch.no_grad()
    def infer(self, src, n_timesteps=30, sr=16000): 
        
        # source_basename = os.path.basename(src_path).split('.wav')[0]
        # target_basename = os.path.basename(tgt_path).split('.wav')[0]
        # output_basename = f'{source_basename}_to_{target_basename}'
        # output_wav = os.path.join(self.output_path, output_basename+'.wav')
        
        mel_source = (torch.from_numpy(self.get_mel(src)).float().unsqueeze(0)).to(self.device)
        mel_source_lengths = (torch.LongTensor([mel_source.shape[-1]])).to(self.device)
        # if self.use_gpu:
        #     mel_source = mel_source.cuda()
        
        # if self.use_gpu:
        #     mel_source_lengths = mel_source_lengths.cuda()
        
        # mel_target = torch.from_numpy(self.get_mel(tgt_path)).float().unsqueeze(0)
        # if self.use_gpu:
        #     mel_target = mel_target.cuda()
        # mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
        # if self.use_gpu:
        #     mel_target_lengths = mel_target_lengths.cuda()

        # embed_target = torch.from_numpy(self.get_embed(tgt_path)).float().unsqueeze(0)
        # if self.use_gpu:
        #     embed_target = embed_target.cuda()
            
            
        # performing voice conversion
        mel_encoded, mel_ = self.generator.forward(mel_source, mel_source_lengths, self.mel_target, self.mel_target_lengths, self.embed_target, 
                                            n_timesteps=n_timesteps, mode='ml')
        mel_synth_np = mel_.cpu().detach().squeeze().numpy()
        mel_source_np = mel_.cpu().detach().squeeze().numpy()
        mel = (torch.from_numpy(self.mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)).to(self.device)
        
        audio = self.hifigan_universal.forward(mel).cpu().squeeze().clamp(-1, 1)

        return audio

