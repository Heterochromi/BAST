from mel_spec_tensor import generate_mel_spectrogram_torch_tensor
import pandas as pd
import os

## column shape: created_at,input_file,output_file,subject_id,ir_type,speaker_layout,sample_rate,level,reverb,azimuth_deg,elevation_deg



##output column shape: input_file,azimuth_deg,elevation_deg,class




def convert_directory_to_tensors(csv_path, output_dir, output_csv_path="tensor_metadata.csv"):
    # Read input CSV
    csv = pd.read_csv(csv_path)

    # Prepare output data list
    output_data = []

    for index, row in csv.iterrows():
        # Extract class from input_file path (e.g., "footsteps", "weapons", "grenade")
        input_file_path = row["input_file"]
        # Split path and get the category (second part after "sounds/")
        path_parts = input_file_path.split("/")
        class_name = path_parts[path_parts.index("sounds") + 1] if "sounds" in path_parts else "unknown"
        # Generate tensor
        output_file_name = row["output_file"].split("/")[-1].replace(".wav", ".pt")
        output_location = os.path.join(output_dir, output_file_name)
        generate_mel_spectrogram_torch_tensor(row["output_file"], output_location)

        # Add to output data
        output_data.append({
            'input_file': output_location,
            'azimuth_deg': row["azimuth_deg"],
            'elevation_deg': row["elevation_deg"],
            'class': class_name
        })

        # Print progress
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(csv)} files")

    # Create output CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Output CSV saved to: {output_csv_path}")
    print(f"Total files processed: {len(output_data)}")


if __name__ == "__main__":
    convert_directory_to_tensors("random_sample.csv", "output_tensors", "tensor_metadata.csv")
