import os
import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiSourceSpectrogramDataset(Dataset):
    """
    Like SpectrogramDataset but returns ALL sources (not averaged).
    Assumes positional alignment:
      classes: "c1,c2,c3"
      azimuth: "10,200,355"
      elevation: "5,-3,12"
    -> Source i has (class_i, azimuth_i, elevation_i)

    Returns:
      spec: [2,F,T]
      loc_targets: [N,2] (az_deg, el_deg)
      cls_targets: [N,C] multi-hot per source (usually single 1)
      num_sources: int
    """
    def __init__(self,
                 csv_path: str,
                 tensor_dir: str = "output_tensors",
                 class_map: dict | None = None):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.tensor_dir = tensor_dir

        required_columns = ['name_file', 'classes', 'azimuth', 'elevation', 'num_classes']
        missing = [c for c in required_columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        self.df = self.df[self.df['name_file'].str.endswith('.pt')].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("No valid .pt tensor files found in CSV")

        # Build class mapping
        all_classes = set()
        for classes_str in self.df['classes']:
            if pd.notna(classes_str):
                for c in str(classes_str).split(','):
                    all_classes.add(c.strip())

        if class_map is None:
            unique_classes = sorted(list(all_classes))
            self.class_to_index = {c: i for i, c in enumerate(unique_classes)}
        else:
            self.class_to_index = dict(class_map)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.num_classes = len(self.class_to_index)

    def __len__(self):
        return len(self.df)

    def _load_spec(self, filename: str) -> torch.Tensor:
        path = os.path.join(self.tensor_dir, filename)
        tensor = torch.load(path).float()
        if tensor.ndim != 3 or tensor.shape[0] != 2:
            raise ValueError(f"Expected [2,F,T], got {tuple(tensor.shape)} at {path}")
        return tensor

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        spec = self._load_spec(row['name_file'])

        classes_raw = [] if pd.isna(row['classes']) else [c.strip() for c in str(row['classes']).split(',')]
        az_list = [] if pd.isna(row['azimuth']) else [float(x.strip()) for x in str(row['azimuth']).split(',')]
        el_list = [] if pd.isna(row['elevation']) else [float(x.strip()) for x in str(row['elevation']).split(',')]

        # Sanity: lengths should match (otherwise we drop extras to min length)
        n = min(len(classes_raw), len(az_list), len(el_list))
        classes_raw = classes_raw[:n]
        az_list = az_list[:n]
        el_list = el_list[:n]

        if n == 0:
            # No sources: we still return empty tensors
            loc_targets = torch.zeros(0, 2, dtype=torch.float32)
            cls_targets = torch.zeros(0, self.num_classes, dtype=torch.float32)
            return spec, loc_targets, cls_targets, 0

        loc_targets = torch.stack([torch.tensor([az_list[i], el_list[i]], dtype=torch.float32)
                                   for i in range(n)], dim=0)  # [N,2]

        cls_targets = torch.zeros(n, self.num_classes, dtype=torch.float32)
        for i, cname in enumerate(classes_raw):
            if cname in self.class_to_index:
                cls_targets[i, self.class_to_index[cname]] = 1.0

        return spec, loc_targets, cls_targets, n



if __name__ == "__main__":
    dataset = MultiSourceSpectrogramDataset("tensor_metadata.csv")
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of unique classes: {len(dataset.class_to_index)}")
    print(f"Classes: {list(dataset.class_to_index.keys())}")

    # Test first sample
    for i in range(5):
        spec, loc_targets, cls_targets, num_sources = dataset[i]
        print(f"\nSample {i}:")
        print(f"Spectrogram shape: {spec.shape}")
        print(f"Num sources: {num_sources}")
        print(f"Locations (az, el): {loc_targets}")          # each row: [az_deg, el_deg]
        print(f"Class target shape: {cls_targets.shape}")
        print(f"Per-source class one-hot rows:\n{cls_targets}")
        if num_sources > 0:
            az_list = loc_targets[:, 0].tolist()
            el_list = loc_targets[:, 1].tolist()
            print(f"Azimuths: {az_list}")
            print(f"Elevations: {el_list}")
