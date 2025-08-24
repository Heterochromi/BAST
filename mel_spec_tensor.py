from torchaudio._torchcodec import load_with_torchcodec
import torchaudio.transforms as T
import torch
import numpy as np
def generate_mel_spectrogram_torch_tensor(input_audio_path, output_path, return_tensor=False):
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
        torch.save(stereo_mel_spec, output_path)
    



def crop_tensor_to_100ms(tensor_path, target_frames=18):
    """
    Crop an existing .pt tensor file to first 100ms (18 frames with current settings)
    """
    # Load the tensor
    tensor = torch.load(tensor_path)

    # Crop to target frames
    if tensor.shape[2] > target_frames:
        cropped = tensor[:, :, :target_frames]
        print(f"Cropped {tensor_path}: {tensor.shape} → {cropped.shape}")

        # Save back
        torch.save(cropped, tensor_path)
        return cropped
    else:
        print(f"{tensor_path} already ≤ {target_frames} frames: {tensor.shape}")
        return tensor

def batch_crop_tensors(tensor_directory, target_frames=18):
    """
    Crop all .pt files in a directory to first 100ms
    """
    import glob
    import os

    pt_files = glob.glob(os.path.join(tensor_directory, "*.pt"))
    print(f"Found {len(pt_files)} .pt files")

    for pt_file in pt_files:
        try:
            crop_tensor_to_100ms(pt_file, target_frames)
        except Exception as e:
            print(f"Error processing {pt_file}: {e}")

if __name__ == "__main__":
    # Test single file
    # input = "output/dataset/sounds/footsteps/carpet_01_az0_el-90.wav"
    # generate_mel_spectrogram_torch_tensor(input, "test.pt")

    # To crop existing tensors, uncomment and modify:
    # crop_tensor_to_100ms("path/to/your/tensor.pt")
    batch_crop_tensors("output_tensors")
    