import os
import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """
    A simple dataset that loads precomputed log-mel spectrograms and multitask labels
    from a CSV index. Expected CSV columns:
      - input_file: path to the spectrogram file for the stereo sample
                    (either a .npy saved array or a .pt tensor)
      - class: integer or string class label (will be mapped to int)
      - azimuth_deg: float degrees in [-180, 180] or [0, 360)
      - elevation_deg: float degrees in [-90, 90]

    Expected spectrogram shape per sample when loaded: [2, F, T]
      - channel 0: left, channel 1: right

    Optionally, you may store left/right as two files; in that case provide two columns
    named input_file_left and input_file_right. If present, they will be loaded and stacked.
    """

    def __init__(self,
                 csv_path: str,
                 class_map: dict | None = None,
                 normalize: bool = True,
                 allowed_exts: tuple = ('.npy', '.pt'),
                 keep_columns: tuple = ('input_file', 'azimuth_deg', 'elevation_deg', 'class'),
                 ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        # Allow either 'class' or 'Class' naming
        if 'class' not in self.df.columns:
            if 'Class' in self.df.columns:
                self.df = self.df.rename(columns={'Class': 'class'})
            else:
                raise ValueError("CSV must include a 'class' column (or 'Class').")
        # Backward compatibility with provided sample CSV
        if 'input_file' not in self.df.columns:
            if 'spectrogram_path' in self.df.columns:
                self.df = self.df.rename(columns={'spectrogram_path': 'input_file'})
            else:
                raise ValueError("CSV must include an 'input_file' column with spectrogram paths.")
        missing = [c for c in ['input_file', 'azimuth_deg', 'elevation_deg', 'class'] if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Filter invalid extensions early
        def _valid_path(p: str) -> bool:
            try:
                return isinstance(p, str) and p.lower().endswith(allowed_exts)
            except Exception:
                return False

        self.df = self.df[self.df['input_file'].apply(_valid_path)].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("No valid spectrogram paths found after filtering by extensions .npy/.pt")

        # Build class mapping
        if class_map is None:
            # Stable sort unique values to create indices
            unique_classes = sorted(self.df['class'].astype(str).unique())
            self.class_to_index = {c: i for i, c in enumerate(unique_classes)}
        else:
            self.class_to_index = dict(class_map)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

        self.normalize = normalize

    def __len__(self):
        return len(self.df)

    def _load_spec(self, path: str) -> torch.Tensor:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            arr = np.load(path)
            tensor = torch.from_numpy(arr)
        elif ext == '.pt':
            tensor = torch.load(path)
        else:
            raise ValueError(f"Unsupported spectrogram file extension: {ext}")
        if tensor.ndim != 3 or tensor.shape[0] != 2:
            raise ValueError(f"Expected spectrogram shape [2, F, T], got {tuple(tensor.shape)} for {path}")
        tensor = tensor.float()
        # if self.normalize:
        #     # Per-sample, per-channel standardization across all bins/time
        #     mean = tensor.view(2, -1).mean(dim=1).view(2, 1, 1)
        #     std = tensor.view(2, -1).std(dim=1).view(2, 1, 1)
        #     std = torch.clamp(std, min=1e-6)
        #     tensor = (tensor - mean) * (1.0 / std)
        return tensor

    @staticmethod
    def _get_az_el_deg(az_deg: float, el_deg: float) -> torch.Tensor:
        # Return native azimuth and elevation in degrees
        return torch.tensor([az_deg, el_deg], dtype=torch.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        spec_path = row['input_file']
        spec = self._load_spec(spec_path)  # [2, F, T]
        class_raw = str(row['class'])
        class_idx = self.class_to_index[class_raw]
        az = float(row['azimuth_deg'])
        el = float(row['elevation_deg'])
        # Multi-task targets:
        # - localization: [azimuth_deg, elevation_deg]
        # - classification: integer class
        # - raw az/el (duplicate for convenience): regression in degrees
        loc_target = self._get_az_el_deg(az, el)  # [2]
        cls_target = torch.tensor([class_idx], dtype=torch.long)
        az_el_deg = torch.tensor([az, el], dtype=torch.float32)
        return spec, loc_target, cls_target, az_el_deg


def create_splits(csv_path: str,
                  train_ratio: float = 0.8,
                  seed: int = 42):
    df = pd.read_csv(csv_path)
    rng = np.random.RandomState(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    split = int(len(indices) * train_ratio)
    return indices[:split], indices[split:]



if __name__ == "__main__":
    dataset = SpectrogramDataset("tensor_metadata.csv")
    print(dataset[0])
    print(f"Number of classes: {len(dataset.class_to_index)}")
