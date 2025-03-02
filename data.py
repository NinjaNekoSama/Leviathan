import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
from utils import load_config, plot_spectrogram
from graphbuilder import SpectrogramToGraph

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='Baseline Training')
parser.add_argument('--config', default='config/default.yaml', type=str, metavar='PATH',
                    help='path to training data')


class KillerWhaleDataset(Dataset):
    def __init__(self, cfg, feature_dir,
                 feature_type=None, indices=None,
                 cache_all=True):
        """
        Initialize the dataset.

        Args:
            cfg (dict): Configuration dictionary.
            feature_dir (str): Directory containing audio files.
            feature_type (str, optional): Type of feature ("wav" or other). Defaults to "wav".
            indices (list, optional): Indices to subset the dataset. Defaults to None.
            cache_all (bool, optional): Whether to cache all data in memory. Defaults to True.
        """
        # Load the file containing all annotations
        annotations_in = pd.read_csv(cfg["annotation_file"])

        # Subset annotations if indices are provided
        if indices is not None:
            annotations_in = annotations_in.loc[indices].reset_index(drop=True)

        self.feature_dir = feature_dir
        self.feature_type = feature_type if feature_type is not None else "wav"
        self.cache_all = cache_all

        # Get a list of all existing files in the feature directory
        existing_files = self._get_existing_files()

        # Filter annotations to only include rows corresponding to existing files
        self.annotations = self._filter_annotations_by_files(annotations_in, existing_files)

        # Cache data if required
        self.cache_list = []
        if self.cache_all:
            for i in range(len(self.annotations)):
                file_name = self._get_file_path(self.annotations.loc[i])
                feature, _ = torchaudio.load(file_name)
                self.cache_list.append(feature.to(DEVICE))

    def _get_existing_files(self):
        return set(os.listdir(self.feature_dir))

    def _filter_annotations_by_files(self, annotations, existing_files):
        valid_indices = []
        for i in range(len(annotations)):
            file_name = self._get_file_name(annotations.loc[i])
            if file_name in existing_files:
                valid_indices.append(i)

        print(f"Number of valid entries: {len(valid_indices)}")  # Debug: Print number of valid entries
        return annotations.loc[valid_indices].reset_index(drop=True)

    def _get_file_name(self, ann_idx):
        month = str(ann_idx['Month']).zfill(2)
        day = str(ann_idx['Day']).zfill(2)
        hour = str(ann_idx['Hour']).zfill(2)
        minute = str(ann_idx['Minute']).zfill(2)
        second = str(ann_idx['Second']).zfill(2)

        return (
            f"DFOCRP.{ann_idx['Deploy.ID']}-{ann_idx['Loc.ID']}.SM2M-3."
            f"{ann_idx['Year']}{month}{day}_"
            f"{hour}{minute}{second}Z.wav"
        )

    def _get_file_path(self, ann_idx):
        file_name = self._get_file_name(ann_idx)
        return os.path.join(self.feature_dir, file_name)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann_idx = self.annotations.loc[idx]
        file_name = self._get_file_path(ann_idx)

        if self.cache_all:
            feature = self.cache_list[idx]  # Use pre-loaded data
        else:
            feature, _ = torchaudio.load(file_name)

        return feature, {
            "timestamp": ann_idx.UTC,
            "path": file_name,
            "detection_type": ann_idx.detectionType,
            "species": ann_idx["Sound.ID.Species"],
            "recording_start": ann_idx["UTC.recording.start.time"]
        }


def main():
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Load dataset
    kwdb = KillerWhaleDataset(cfg=cfg, feature_dir='POC', cache_all=False)

    # Load an audio sample
    audio, metadata = kwdb[0]
    print(f"Loaded audio with shape: {audio.shape}")
    print(f"Metadata: {metadata}")

    # Plot the spectrogram
    plot_spectrogram(audio, title=f"Spectrogram of {metadata['path']}")

    # Convert spectrogram to graph and visualize
    converter = SpectrogramToGraph(sample_rate=22050, n_fft=1024)
    converter.visualize_spectrogram_as_graph(audio, title=f"Graph Representation of {metadata['path']}")


if __name__ == '__main__':
    main()