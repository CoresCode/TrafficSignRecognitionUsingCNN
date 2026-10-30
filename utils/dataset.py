from pathlib import Path
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset


class TrafficSignDataset(Dataset):
    """
    GTSRB Dataset Loader

    Reads every class folder, loads GT-xxxxx.csv,
    and returns (image, label).
    """

    def __init__(self, dataset_folder, transform=None):

        self.dataset_folder = Path(dataset_folder)
        self.transform = transform

        self.samples = []

        self.index_to_name = {}

        # -------------------------------------------------------
        # Read every class folder
        # -------------------------------------------------------

        class_folders = sorted(
            [
                folder
                for folder in self.dataset_folder.iterdir()
                if folder.is_dir()
            ]
        )

        for folder in class_folders:

            class_id = int(folder.name)

            self.index_to_name[class_id] = f"Class {class_id}"

            csv_file = folder / f"GT-{folder.name}.csv"

            annotations = pd.read_csv(
                csv_file,
                sep=";"
            )

            for _, row in annotations.iterrows():

                self.samples.append(
                    {
                        "image_path": folder / row["Filename"],
                        "label": int(row["ClassId"])
                    }
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = cv2.imread(str(sample["image_path"]))

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        if self.transform:

            image = self.transform(image=image)["image"]

        label = torch.tensor(
            sample["label"],
            dtype=torch.long
        )

        return image, label

    def get_class_names(self):

        return self.index_to_name