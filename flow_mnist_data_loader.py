from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


class FlowMnistDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_name, direction, duration, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_name = dataset_name
        self.dataset_base_dir_path = Path(f"/home/eyal/dev/eyal-thesis/thesis-datasets-creator/{self.dataset_name}_npy")

        self.direction = direction
        if self.direction == "up":
            self.direction_dir_name = "FlowMnistUpTrimAndTps"
        elif self.direction == "down":
            self.direction_dir_name = "FlowMnistDownTrimAndTps"
        elif self.direction == "both":
            self.direction_dir_name = "BiFlowMnistTrimAndTps"
        else:
            raise ValueError(f"direction should be: up/down/both, but given {self.direction}")

        self.duration = duration
        self.x_path = self.dataset_base_dir_path / f"{self.direction_dir_name}/X_trim_and_tps_{self.duration}sec.npy"
        self.y_path = self.dataset_base_dir_path / f"{self.direction_dir_name}/y_trim_and_tps_{self.duration}sec.npy"
        self.data_npy = np.load(self.x_path)
        self.target_npy = np.load(self.y_path)
        self.transform = transform

        self.index_to_label = {0: 'g_search', 1: 'g_drive', 2: 'g_doc', 3: 'g_music', 4: 'youtube'}

    def __len__(self):
        return len(self.data_npy)

    def __getitem__(self, idx):
        sample = {"image": self.data_npy[idx], "label": self.target_npy[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    duration = 15
    ds_up = FlowMnistDataset("ucdavis", "up", duration)
    ds_down = FlowMnistDataset("ucdavis", "down", duration)
    ds_both = FlowMnistDataset("ucdavis", "both", duration)
    data_sets = [ds_up, ds_down, ds_both]
    ds = data_sets[0]
    start_indices = [ds.target_npy.tolist().index(i) for i in (np.unique(ds.target_npy))]

    nrows, ncols = len(data_sets), len(start_indices)
    for i in range(nrows):
        for j in range(ncols):
            ds = data_sets[i]
            ds_idx = start_indices[j]
            sample = data_sets[i][ds_idx]
            image = sample['image']
            label = sample['label']
            plt_idx = i * ncols + j + 1
            print(plt_idx, image.shape, label)

            ax = plt.subplot(nrows, ncols, plt_idx)
            plt.imshow(image, cmap=plt.get_cmap('binary'), extent=[0, ds.duration, ds.duration, 0])
            plt.xticks(np.arange(0, ds.duration, (ds.duration-0.1)//2))
            plt.xlabel("time[s]")
            plt.ylabel("pkt_size")
            plt.yticks([])
            plt.gca().invert_yaxis()
            plt.title(f"{ds.index_to_label[label]}:{ds.direction}")

    plt.tight_layout()
    plt.show()
