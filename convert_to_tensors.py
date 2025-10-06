from mel_spec_tensor import generate_mel_spectrogram_torch_tensor
import pandas as pd
import os


def convert_directory_to_tensors(
    csv_path="dataset/dataset_metadata.csv",
    dataset_dir="dataset",
    output_dir="output_tensors",
    output_csv_path="tensor_metadata.csv",
):
    """
    Convert audio files from dataset directory to tensors and create metadata CSV.

    Args:
        csv_path: Path to the dataset metadata CSV file
        dataset_dir: Directory containing the .wav files
        output_dir: Directory to save the generated tensor files
        output_csv_path: Path for the output CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read input CSV
    csv_data = pd.read_csv(csv_path)

    # Prepare output data list
    output_data = []

    for index, row in csv_data.iterrows():
        # Get the input wav file path
        wav_filename = row["name_file"]
        input_wav_path = os.path.join(dataset_dir, wav_filename)

        # Check if the wav file exists
        if not os.path.exists(input_wav_path):
            print(f"Warning: File {input_wav_path} not found, skipping...")
            continue

        # Generate output tensor filename
        tensor_filename = wav_filename.replace(".wav", ".pt")
        output_tensor_path = os.path.join(output_dir, tensor_filename)

        try:
            # Generate tensor from wav file
            generate_mel_spectrogram_torch_tensor(input_wav_path, output_tensor_path)

            # Add to output data with same format as input but updated file path
            output_data.append(
                {
                    "name_file": tensor_filename,  # Now points to .pt file
                    "classes": row["classes"],
                    "x": row["x"],
                    "y": row["y"],
                    "z": row["z"],
                    "num_classes": row["num_classes"],
                }
            )

            # Print progress
            if (index + 1) % 100 == 0:
                print(f"Processed {index + 1}/{len(csv_data)} files")

        except Exception as e:
            print(f"Error processing {input_wav_path}: {str(e)}")
            continue

    # Create output CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Output CSV saved to: {output_csv_path}")
    print(f"Total files processed: {len(output_data)}")
    print(f"Tensors saved to: {output_dir}")


if __name__ == "__main__":
    # Use the existing dataset metadata from the dataset directory
    convert_directory_to_tensors(
        csv_path="dataset_parallel_100ms/dataset_metadata.csv",
        dataset_dir="dataset_parallel_100ms",
        output_dir="output_tensors_100ms",
        output_csv_path="tensor_metadata_100ms.csv",
    )
