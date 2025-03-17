import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from typing import Literal
import os
from utils import load_config, plot_spectrogram, load_audio,play_audio_segment,plot_mel_spectrogram
from graphbuilder import SpectrogramToGraph
from datetime import datetime


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_SOURCES= Literal["wc", "nm", "both"]
parser = argparse.ArgumentParser(description='Baseline Training')
parser.add_argument('--config', default='config/default.yaml', type=str, metavar='PATH',
                    help='path to training data')


class KillerWhaleDataset(Dataset):
    def __init__(self, cfg,classes,annotations_in,
                 feature_type=None, indices=None,
                 cache_all=True):

        if indices is not None:
            annotations_in = annotations_in.loc[indices].reset_index(drop=True)
        self.cfg = cfg
        self.feature_dir = cfg["data_dir"]
        self.source = cfg["source"]
        self.data_sources = {
            "wc": [self.cfg["west_coast"]],
            "nm": [self.cfg["north_mainland"]],
            "both": [self.cfg["west_coast"], self.cfg["north_mainland"]],
        }.get(self.source)
        self.feature_type = feature_type if feature_type is not None else "wav"
        self.cache_all = cache_all
        self.cuda = True if((DEVICE=="cuda") and (cfg["num_gpu"]>0)) else False
        self.classes =  classes
        existing_files = self._get_existing_files(annotations_in)

        self.annotations = self._filter_annotations_by_files(annotations_in, existing_files)

        self.cache_list = []
        if self.cache_all:
            for i in range(len(self.annotations)):
                file_name = self._get_file_path(self.annotations.loc[i])
                feature, _ = torchaudio.load(file_name)
                if self.cuda:
                    self.cache_list.append(feature.to(DEVICE))

    def _get_existing_files(self, annotations_in):
        if self.source not in {"wc", "nm", "both"}:
            raise ValueError(f"Invalid source '{self.source}'. Expected one of: 'wc', 'nm', 'both'.")

        expected_files = set(annotations_in.apply(self._get_file_name, axis=1))  

        existing_files = set()
        scanned_files = set()

        for sub_dir in self.data_sources:
            full_path = os.path.join(self.feature_dir, sub_dir)

            if os.path.isdir(full_path):  
                with os.scandir(full_path) as entries:
                    for e in entries:
                        if e.is_file():
                            existing_files.add(e.name)
                            scanned_files.add(e.name)

        # Determine truly missing files
        not_existing_files = expected_files - scanned_files

        print(f"Missing files ({len(not_existing_files)}): {list(not_existing_files)[:10]}...")  # Show only first 10

        return list(existing_files)


    def _filter_annotations_by_files(self, annotations, existing_files):
        valid_indices = []
        for i in range(len(annotations)):
            file_name = self._get_file_name(annotations.loc[i])
            if file_name in existing_files:
                valid_indices.append(i)
        return annotations.loc[valid_indices].reset_index(drop=True)
    def _get_file_name(self, ann_idx):
        try:
            timestamp = pd.to_datetime(ann_idx["UTC.recording.start.time"])  # Convert to datetime object
        except Exception as e:
            raise ValueError(f"Invalid UTC timestamp '{ann_idx['UTC.recording.start.time']}': {e}")

        recorders = {
            "WCV1": "AURAL117LF",
            "NML1": "SM2M-3"
        }
        recorder = recorders.get(ann_idx['Loc.ID'])
        if not recorder:
            raise ValueError(f"Invalid file '{ann_idx}', unrecognized recorder")

        return (
            f"DFOCRP.{ann_idx['Deploy.ID']}-{ann_idx['Loc.ID']}.{recorder}."
            f"{timestamp.strftime('%Y%m%d_%H%M%SZ')}.wav"
        )



    def _get_file_path(self, ann_idx):
        file_names = self._get_file_name(ann_idx)

        if isinstance(file_names, str):  # Single source case (wc or nm)
            file_names = [file_names]  # Convert to list for consistency

        paths = []
        for file_name in file_names:
            for source_dir in self.data_sources:
                path = os.path.join(self.feature_dir, source_dir, file_name)
                if os.path.exists(path):
                    paths.append(path)

        if paths:
            return paths if len(paths) > 1 else paths[0]  
        else:
            raise FileNotFoundError(f"None of the files {file_names} exist in both sources.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann_idx = self.annotations.loc[idx]
        file_name = self._get_file_path(ann_idx)

        if self.cache_all:
            feature = self.cache_list[idx] 
        else:
            feature, _ = torchaudio.load(file_name)

        return feature, {
            "timestamp": ann_idx.UTC,
            "path": file_name,
            "detection_type": ann_idx.detectionType,
            "species": ann_idx["Sound.ID.Species"],
            "recording_start": ann_idx["UTC.recording.start.time"],
            "duration": ann_idx["duration.sec"],
            "low_freq": ann_idx['lowFreq'],
            "high_freq":ann_idx["highFreq"]
        }


def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    indices_KW = []
    indices_no_KW = []
   
    if cfg["source"]=="nm":
        ann_file = pd.read_csv(cfg['annotation_file_nm'])
    elif cfg["source"] == "wc":
        ann_file =  pd.read_csv(cfg['annotation_file_wc'])
    elif cfg["source"] == "both":
        nm_ann_file = pd.read_csv(cfg['annotation_file_nm'])
        wc_ann_file =  pd.read_csv(cfg['annotation_file_wc'])
        # print(nm_ann_file.columns)
        ann_file = pd.concat([nm_ann_file, wc_ann_file], ignore_index=True)
    for i, row in ann_file.iterrows():
        if row["Sound.ID.Species"] == "KW":
            indices_KW.append(i)
        else:
            indices_no_KW.append(i)       
    classes = sorted(ann_file["Sound.ID.Species"].fillna("Unknown").astype(str).unique())
    kwdb = KillerWhaleDataset(cfg=cfg,classes=classes,cache_all=False,indices=indices_KW,annotations_in=ann_file)
    print(f"Number of Killer Whales in the combined NM and WC  according to annotation files: {len(indices_KW)}")

    print(f"Actual Number of Killer Whales in the combined NM and WC as wav files: {len(kwdb)}")
    # print("First few annotations:\n", kwdb.annotations.head())


    print(kwdb[0])
    # Load an audio sample
    # audio, metadata = kwdb[0]
    # print(f"Loaded audio with shape: {audio.shape}")
    # print(f"Metadata: {metadata}")
    # print(kwdb.classes)
    # waveform, sample_rate = load_audio(metadata["path"])
    # plot_spectrogram(waveform, sample_rate, metadata)
    # plot_mel_spectrogram(waveform, sample_rate, metadata)
    # play_audio_segment(audio, sample_rate)
    # # Convert spectrogram to graph and visualize
    # converter = SpectrogramToGraph(sample_rate=22050, n_fft=1024)
    # converter.visualize_spectrogram_as_graph(audio, title=f"Graph Representation of {metadata['path']}")


if __name__ == '__main__':
    main()