import torch
import matplotlib.pyplot as plt
import numpy as np
from mel_spec_tensor import generate_mel_spectrogram_torch_tensor
import os
import seaborn as sns


def visualize_tensor_structure(tensor_path=None, audio_path=None):
    """
    Visualize the structure and content of a mel spectrogram tensor

    Args:
        tensor_path: Path to existing .pt tensor file
        audio_path: Path to audio file to generate tensor from
    """
    # Load or generate tensor
    if tensor_path and os.path.exists(tensor_path):
        tensor = torch.load(tensor_path)
        print(f"Loaded tensor from: {tensor_path}")
    elif audio_path and os.path.exists(audio_path):
        tensor = generate_mel_spectrogram_torch_tensor(
            audio_path, None, return_tensor=True
        )
        print(f"Generated tensor from: {audio_path}")
    else:
        print("Error: Please provide either a valid tensor_path or audio_path")
        return

    # Print tensor information
    print(f"\nTensor Shape: {tensor.shape}")
    print(f"Tensor Type: {tensor.dtype}")
    print(f"Min value: {tensor.min().item():.2f}")
    print(f"Max value: {tensor.max().item():.2f}")
    print(f"Mean value: {tensor.mean().item():.2f}")

    # Explain dimensions
    channels, freq_bins, time_frames = tensor.shape
    print(f"\nDimension Breakdown:")
    print(f"  Channels (stereo): {channels}")
    print(f"  Frequency bins (mel): {freq_bins}")
    print(f"  Time frames (40ms): {time_frames}")

    # Calculate time per frame
    sample_rate = 44100  # Assume standard sample rate
    hop_length = 256
    time_per_frame_ms = (hop_length / sample_rate) * 1000
    total_time_ms = time_frames * time_per_frame_ms

    print(f"\nTime Analysis:")
    print(f"  Time per frame: {time_per_frame_ms:.2f} ms")
    print(f"  Total time covered: {total_time_ms:.2f} ms")
    print(f"  Target duration: 40 ms")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "40ms Audio Tensor Visualization: [2, 64, 3] = [Stereo, Frequency, Time]",
        fontsize=16,
    )

    # Left channel spectrogram
    ax1 = axes[0, 0]
    im1 = ax1.imshow(tensor[0].numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax1.set_title("Left Channel Spectrogram\n(64 freq bins × 3 time frames)")
    ax1.set_xlabel("Time Frames (≈13.3ms each)")
    ax1.set_ylabel("Mel Frequency Bins")
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["0-13ms", "13-27ms", "27-40ms"])
    plt.colorbar(im1, ax=ax1, label="Amplitude (dB)")

    # Right channel spectrogram
    ax2 = axes[0, 1]
    im2 = ax2.imshow(tensor[1].numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax2.set_title("Right Channel Spectrogram\n(64 freq bins × 3 time frames)")
    ax2.set_xlabel("Time Frames (≈13.3ms each)")
    ax2.set_ylabel("Mel Frequency Bins")
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["0-13ms", "13-27ms", "27-40ms"])
    plt.colorbar(im2, ax=ax2, label="Amplitude (dB)")

    # Difference between channels
    ax3 = axes[1, 0]
    channel_diff = tensor[0] - tensor[1]
    im3 = ax3.imshow(channel_diff.numpy(), aspect="auto", origin="lower", cmap="RdBu")
    ax3.set_title("Stereo Difference (L - R)\n(Shows spatial audio information)")
    ax3.set_xlabel("Time Frames")
    ax3.set_ylabel("Mel Frequency Bins")
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(["Frame 1", "Frame 2", "Frame 3"])
    plt.colorbar(im3, ax=ax3, label="Difference (dB)")

    # Time-averaged frequency response
    ax4 = axes[1, 1]
    left_avg = tensor[0].mean(dim=1)  # Average across time
    right_avg = tensor[1].mean(dim=1)
    freq_bins = np.arange(64)
    ax4.plot(freq_bins, left_avg.numpy(), label="Left Channel", linewidth=2)
    ax4.plot(freq_bins, right_avg.numpy(), label="Right Channel", linewidth=2)
    ax4.set_title("Average Frequency Response\n(Averaged across 3 time frames)")
    ax4.set_xlabel("Mel Frequency Bin (0=low, 63=high)")
    ax4.set_ylabel("Average Amplitude (dB)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, tensor


def print_tensor_values_sample(tensor, max_frames_to_show=3):
    """
    Print actual tensor values for inspection
    """
    print(f"\n=== TENSOR VALUES SAMPLE ===")
    print(f"Showing first 5 frequency bins, all {tensor.shape[2]} time frames")
    print(
        f"Remember: This is NOT RGB! It's [stereo_channels, frequency_bins, time_frames]"
    )

    for channel in range(tensor.shape[0]):
        channel_name = "Left" if channel == 0 else "Right"
        print(f"\n{channel_name} Channel (first 5 frequency bins):")
        print("Freq\\Time  ", end="")
        for t in range(min(tensor.shape[2], max_frames_to_show)):
            print(f"Frame{t + 1:2d}    ", end="")
        print()

        for freq in range(min(5, tensor.shape[1])):
            print(f"Bin {freq:2d}:    ", end="")
            for t in range(min(tensor.shape[2], max_frames_to_show)):
                value = tensor[channel, freq, t].item()
                print(f"{value:8.2f}  ", end="")
            print()


def compare_tensor_to_image():
    """
    Educational comparison between audio tensor and image tensor
    """
    print("\n" + "=" * 60)
    print("TENSOR SHAPE COMPARISON")
    print("=" * 60)
    print("IMAGE TENSOR (RGB):")
    print("  Shape: [3, 224, 224]")
    print("  Dim 0 (3):   RGB color channels")
    print("  Dim 1 (224): Image height (pixels)")
    print("  Dim 2 (224): Image width (pixels)")
    print("  Values:      0-255 (color intensities)")
    print()
    print("YOUR AUDIO TENSOR (Mel Spectrogram):")
    print("  Shape: [2, 64, 3]")
    print("  Dim 0 (2):   Stereo audio channels (L/R)")
    print("  Dim 1 (64):  Mel frequency bins (like pitch buckets)")
    print("  Dim 2 (3):   Time frames (slices of your 40ms audio)")
    print("  Values:      dB values (negative, log-scale)")
    print()
    print("KEY DIFFERENCE:")
    print("  • Images: spatial dimensions (height × width)")
    print("  • Audio: frequency × time dimensions")
    print("=" * 60)


if __name__ == "__main__":
    # Educational comparison first
    compare_tensor_to_image()

    # Find and visualize a sample file
    sample_audio = "dataset_parallel_100ms/sample_0022.wav"

    if sample_audio:
        print(f"\nAnalyzing sample audio file: {sample_audio}")
        fig, tensor = visualize_tensor_structure(audio_path=sample_audio)

        # Show actual tensor values
        print_tensor_values_sample(tensor)

        # Save the plot
        plt.savefig("tensor_visualization.png", dpi=300, bbox_inches="tight")
        print(f"\nVisualization saved as: tensor_visualization.png")

        # Show the plot
        plt.show()

    else:
        print("\nNo audio files found in dataset directory.")
        print("Place a .wav file in the dataset directory to run visualization.")

    print(f"\nSUMMARY:")
    print(f"Your tensor shape [2, 64, 3] represents:")
    print(f"  • 2 stereo channels (left and right audio)")
    print(f"  • 64 frequency bins (mel-scale, like musical notes)")
    print(f"  • 3 time frames (slices of your 40ms audio)")
    print(f"This is completely different from RGB images!")
