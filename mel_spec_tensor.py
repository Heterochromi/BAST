from torchaudio._torchcodec import load_with_torchcodec
import torchaudio.transforms as T
import torch
import numpy as np
def generate_mel_spectrogram_torch_tensor(input_audio_path, output_image_path, return_tensor=False):
    # 1. Load the audio file
    waveform, sample_rate = load_with_torchcodec(input_audio_path)

# 2. Define the Mel-spectrogram transform
    mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=256,
    n_mels=64
    )

# 3. Apply the transform
    mel_spectrogram_left = mel_spectrogram_transform(waveform[0])
    mel_spectrogram_right = mel_spectrogram_transform(waveform[1])


# 4. Convert to decibels
    amp_to_db = T.AmplitudeToDB()
    mel_spectrogram_db_left = amp_to_db(mel_spectrogram_left) 
    mel_spectrogram_db_right = amp_to_db(mel_spectrogram_right) 

    stereo_mel_spec = torch.stack([mel_spectrogram_db_left, mel_spectrogram_db_right], dim=0)
    # stereo_mel_spec = stereo_mel_spec.numpy()

    if return_tensor:
        return stereo_mel_spec
    else:
        torch.save(stereo_mel_spec, f"{output_image_path}.pt")  # ← Use torch.save instead
    



if __name__ == "__main__":      
    input = "output/dataset/sounds/footsteps/carpet_01_az0_el-90.wav"
    generate_mel_spectrogram_torch_tensor(input, "test")
    