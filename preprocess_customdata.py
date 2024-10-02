import os
import numpy as np
from tqdm import tqdm
from scipy.io.wavfile import write

import torch
import torchaudio
import torchaudio.transforms as T
use_gpu = torch.cuda.is_available()

import librosa
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
mel_basis = librosa_mel_fn(22050, 1024, 80, 0, 8000)

import sys

sys.path.append('speaker_encoder/')
from encoder import inference as spk_encoder
from pathlib import Path


def get_mel(wav_path: str) -> np.array:
    wav, _ = load(wav_path, sr=22050)
    wav = wav[:(wav.shape[0] // 256)*256]
    wav = np.pad(wav, 384, mode='reflect')
    stft = librosa.core.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
    mel_spectrogram = np.matmul(mel_basis, stftm)
    log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
    return log_mel_spectrogram

def get_embed(wav_path: str) -> np.array:
    wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
    embed = spk_encoder.embed_utterance(wav_preprocessed)
    return embed

def resample_audio(file_path, target_sample_rate=22050):
    # Load the audio file
    waveform, original_sample_rate = torchaudio.load(file_path)
    
    # Resample the audio
    resampler = T.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
    resampled_waveform = resampler(waveform)
    
    # Save the resampled audio back to the original file path
    torchaudio.save(file_path, resampled_waveform, target_sample_rate)

if __name__=="__main__":
    for id in ['trump']:
        DATASET_PATH = f"./dataset/wavs/{id}"
        MELPATH = f"./dataset/mels/{id}"
        EMBEDSPATH = f"./dataset/embeds/{id}"
        filenames = os.listdir(DATASET_PATH)

        # Load embedding model
        spk_encoder.load_model("./checkpts/spk_encoder/pretrained.pt", device="cuda")

        val_errs=[]
        for file in tqdm(filenames):
            try:
                file_path = os.path.join(DATASET_PATH, file)
                mel_path = os.path.join(MELPATH, os.path.splitext(file)[0]+'_mel.npy')
                embed_path = os.path.join(EMBEDSPATH, os.path.splitext(file)[0]+'_embed.npy')
                
                # xtract and save mel
                log_mel_spectrogram = get_mel(file_path)
                np.save(mel_path, log_mel_spectrogram)

                # xtract and save embeds
                embed = get_embed(file_path)
                np.save(embed_path, embed)
            except ValueError:
                val_errs.append(file_path)

        print(id, " is done!")
        print("None") if len(val_errs)==0 else print(val_errs)
        