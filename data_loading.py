import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiSourceSpectrogramDataset(Dataset):
    """
    Like SpectrogramDataset but returns ALL sources (not averaged).
    Assumes positional alignment:
      classes: "c1,c2,c3"
      x: "x1,x2,x3"
      y: "y1,y2,y3"
      z: "z1,z2,z3"
    -> Source i has (class_i, x_i, y_i, z_i)

    Returns:
      spec: [2, F, T]
      loc_targets: [N, 3] (x, y, z)
      cls_targets: [N, C] multi-hot per source (usually single 1)
      num_sources: int
    """

    def __init__(
        self,
        csv_path: str,
        tensor_dir: str = "output_tensors",
        class_map: dict | None = None,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.tensor_dir = tensor_dir

        required_columns = ["name_file", "classes", "x", "y", "z", "num_classes"]
        missing = [c for c in required_columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        self.df = self.df[
            self.df["name_file"].astype(str).str.endswith(".pt")
        ].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("No valid .pt tensor files found in CSV")

        # Build class mapping
        all_classes = set()
        for classes_str in self.df["classes"]:
            if pd.notna(classes_str):
                for c in str(classes_str).split(","):
                    c = c.strip()
                    if c:
                        all_classes.add(c)

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

    @staticmethod
    def _parse_float_list(cell) -> list[float]:
        """
        Parse a cell that may be:
          - NaN
          - a single float (e.g., 0.412)
          - a string of comma-separated floats (e.g., "0.1,-0.2,0.3")
        Returns list[float].
        """
        if pd.isna(cell):
            return []
        s = str(cell).strip()
        if not s:
            return []
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        try:
            return [float(p) for p in parts]
        except ValueError as e:
            raise ValueError(f"Failed to parse floats from cell value '{cell}'") from e

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        spec = self._load_spec(row["name_file"])

        classes_raw = (
            []
            if pd.isna(row["classes"])
            else [c.strip() for c in str(row["classes"]).split(",") if c.strip() != ""]
        )
        x_list = self._parse_float_list(row["x"])
        y_list = self._parse_float_list(row["y"])
        z_list = self._parse_float_list(row["z"])

        # Sanity: lengths should match (otherwise we drop extras to min length)
        n = min(len(classes_raw), len(x_list), len(y_list), len(z_list))
        classes_raw = classes_raw[:n]
        x_list = x_list[:n]
        y_list = y_list[:n]
        z_list = z_list[:n]

        if n == 0:
            # No sources: we still return empty tensors
            loc_targets = torch.zeros(0, 3, dtype=torch.float32)
            cls_targets = torch.zeros(0, self.num_classes, dtype=torch.float32)
            return spec, loc_targets, cls_targets, 0

        # Locations: [N, 3] => (x, y, z)
        loc_targets = torch.stack(
            [
                torch.tensor([x_list[i], y_list[i], z_list[i]], dtype=torch.float32)
                for i in range(n)
            ],
            dim=0,
        )

        # Per-source one-hot classes: [N, C]
        cls_targets = torch.zeros(n, self.num_classes, dtype=torch.float32)
        for i, cname in enumerate(classes_raw):
            if cname in self.class_to_index:
                cls_targets[i, self.class_to_index[cname]] = 1.0

        return spec, loc_targets, cls_targets, n


if __name__ == "__main__":
    dataset = MultiSourceSpectrogramDataset(
        "tensor_metadata_100ms.csv", tensor_dir="output_tensors_100ms"
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of unique classes: {len(dataset.class_to_index)}")
    print(f"Classes: {list(dataset.class_to_index.keys())}")

    # Test first samples
    for i in range(min(5, len(dataset))):
        spec, loc_targets, cls_targets, num_sources = dataset[i]
        print(f"\nSample {i}:")
        print(f"Spectrogram shape: {spec.shape}")
        print(f"Num sources: {num_sources}")
        print(f"Locations (x, y, z): {loc_targets}")  # each row: [x, y, z]
        print(f"Class target shape: {cls_targets.shape}")
        print(f"Per-source class one-hot rows:\n{cls_targets}")
        if num_sources > 0:
            x_list = loc_targets[:, 0].tolist()
            y_list = loc_targets[:, 1].tolist()
            z_list = loc_targets[:, 2].tolist()
            print(f"x: {x_list}")
            print(f"y: {y_list}")
            print(f"z: {z_list}")
