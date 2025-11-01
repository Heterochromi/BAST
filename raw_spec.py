import torchaudio
import torchaudio.transforms as T
import torch
import numpy as np


def generate_raw_spectrogram_torch_tensor(
    input_audio_path, output_path=None, return_tensor=False, max_duration=0.1
):
    """
    Generate raw/complex spectrogram using STFT.

    Args:
        input_audio_path: Path to input audio file
        output_path: Path to save the spectrogram tensor (optional)
        return_tensor: If True, return the tensor instead of saving
        max_duration: Maximum duration in seconds to process (default: 30s)

    Returns:
        Complex spectrogram tensor of shape [2, freq_bins, time_frames] if return_tensor=True
    """
    # 1. Load the audio file using standard torchaudio.load (more stable)
    waveform, sample_rate = torchaudio.load(input_audio_path)

    # Print audio info
    duration = waveform.shape[1] / sample_rate
    print(
        f"Audio info: {sample_rate} Hz, {duration:.2f} seconds, {waveform.shape[1]} samples"
    )

    # Ensure stereo (2 channels)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)  # Duplicate mono to stereo
    elif waveform.shape[0] > 2:
        waveform = waveform[:2, :]  # Take only first 2 channels

    max_samples = int(max_duration * sample_rate)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    # 2. Define the STFT transform for raw/complex spectrogram
    stft_transform = T.Spectrogram(
        n_fft=256,
        hop_length=256,
        win_length=256,
        window_fn=torch.hann_window,
        power=None,  # None returns complex spectrogram, 2 would return power spectrogram
        normalized=False,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )

    # 3. Apply the transform to both channels
    complex_spectrogram_left = stft_transform(waveform[0])
    complex_spectrogram_right = stft_transform(waveform[1])

    # 4. Stack stereo channels
    stereo_complex_spec = torch.stack(
        [complex_spectrogram_left, complex_spectrogram_right], dim=0
    )

    if return_tensor:
        return stereo_complex_spec
    else:
        if output_path:
            torch.save(stereo_complex_spec, output_path)
        return stereo_complex_spec


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys

    # Test with an audio file
    # Update this path to your actual audio file
    if len(sys.argv) > 1:
        test_audio_path = sys.argv[1]
    else:
        test_audio_path = "dataset_parallel_100ms/sample_0025.wav"

    print("Generating raw/complex spectrogram...")
    try:
        # Process only first 10 seconds to save memory
        complex_spec = generate_raw_spectrogram_torch_tensor(
            test_audio_path, return_tensor=True, max_duration=0.1
        )
    except Exception as e:
        print(f"Error: {e}")
        print("\nTips to fix memory issues:")
        print("1. Reduce max_duration (currently 10 seconds)")
        print("2. Reduce n_fft (try 512 instead of 1024)")
        print("3. Increase hop_length (try 512 instead of 256)")
        sys.exit(1)

    print(f"Spectrogram shape: {complex_spec.shape}")
    print(f"Spectrogram dtype: {complex_spec.dtype}")
    print(f"Is complex: {torch.is_complex(complex_spec)}")
    print(
        f"a single data point example {complex_spec[1, 1, 1], complex_spec[1, 1, 1].real, complex_spec[1, 1, 1].imag}"
    )

    # Visualize complex spectrograms with magnitude and phase encoded together
    # Using HSV color space: Hue = Phase, Saturation = 1, Value = Magnitude
    from matplotlib.colors import hsv_to_rgb

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    def complex_to_rgb(complex_spec_channel):
        """Convert complex spectrogram to RGB image encoding both magnitude and phase."""
        # Extract magnitude and phase
        magnitude = torch.abs(complex_spec_channel).numpy()
        phase = torch.angle(complex_spec_channel).numpy()

        # Normalize magnitude for better visualization (using log scale)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        magnitude_normalized = (magnitude_db - magnitude_db.min()) / (
            magnitude_db.max() - magnitude_db.min() + 1e-10
        )

        # Phase is in radians [-π, π], normalize to [0, 1] for hue
        phase_normalized = (phase + np.pi) / (2 * np.pi)

        # Create HSV image: H=phase, S=1 (full saturation), V=magnitude
        h, w = magnitude.shape
        hsv = np.zeros((h, w, 3))
        hsv[:, :, 0] = phase_normalized  # Hue = Phase
        hsv[:, :, 1] = 1.0  # Saturation = full
        hsv[:, :, 2] = magnitude_normalized  # Value = Magnitude

        # Convert to RGB
        rgb = hsv_to_rgb(hsv)
        return rgb

    # Convert both channels to RGB
    rgb_left = complex_to_rgb(complex_spec[0])
    rgb_right = complex_to_rgb(complex_spec[1])

    # Plot left channel
    axes[0].imshow(rgb_left, aspect="auto", origin="lower", interpolation="nearest")
    axes[0].set_title(
        "Left Channel - Complex Spectrogram (Color=Phase, Brightness=Magnitude)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].set_xlabel("Time Frames", fontsize=12)
    axes[0].set_ylabel("Frequency Bins", fontsize=12)

    # Plot right channel
    axes[1].imshow(rgb_right, aspect="auto", origin="lower", interpolation="nearest")
    axes[1].set_title(
        "Right Channel - Complex Spectrogram (Color=Phase, Brightness=Magnitude)",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_xlabel("Time Frames", fontsize=12)
    axes[1].set_ylabel("Frequency Bins", fontsize=12)

    # Add a phase color wheel legend
    fig.text(
        0.5,
        0.1,
        "Color indicates Phase: Red(0°) → Yellow(60°) → Green(120°) → Cyan(180°) → Blue(240°) → Magenta(300°) → Red(360°)",
        ha="center",
        fontsize=10,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig("raw_spectrogram_visualization.png", dpi=150, bbox_inches="tight")
    print("Visualization saved as 'raw_spectrogram_visualization.png'")

    # Optionally save the tensor
    # torch.save(complex_spec, 'raw_spectrogram.pt')
    # print("Tensor saved as 'raw_spectrogram.pt'")
