import pandas as pd
import os
import pathlib
from sklearn import preprocessing
from torch.utils.data import Dataset as TorchDataset
import torch
import torchaudio
from torch.hub import download_url_to_file
import numpy as np
import librosa

dataset_dir = './dataset/dataset/'
assert dataset_dir is not None, "Specify 'TAU Urban Acoustic Scenes 2022 Mobile dataset' location in variable " \
                                "'dataset_dir'. The dataset can be downloaded from this URL:" \
                                " https://zenodo.org/record/6337421"

dataset_config = {
    "dataset_name": "tau24",
    "meta_csv": os.path.join(dataset_dir, "meta.csv"),
    "split_path": os.path.join(dataset_dir, "split_setup"),
    "split_url": "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/",
    "test_split_csv": "test.csv",
    "dirs_path": os.path.join(dataset_dir, "dirs"),
    "eval_dir": os.path.join(dataset_dir, "TAU-urban-acoustic-scenes-2024-mobile-evaluation"),
    "eval_meta_csv": os.path.join(dataset_dir, "TAU-urban-acoustic-scenes-2024-mobile-evaluation", "evaluation_setup", "fold1_test.csv")
}


class BasicDCASE24Dataset(TorchDataset):
    """
    Basic DCASE'24 Dataset: loads data from files
    """

    def __init__(self, meta_csv):
        """
        @param meta_csv: meta csv file for the dataset
        return: waveform, file, label, device and city
        """
        df = pd.read_csv(meta_csv, sep="\t")
        le = preprocessing.LabelEncoder()
        self.labels = torch.from_numpy(le.fit_transform(df[['scene_label']].values.reshape(-1)))
        self.devices = le.fit_transform(df[['source_label']].values.reshape(-1))
        self.cities = le.fit_transform(df['identifier'].apply(lambda loc: loc.split("-")[0]).values.reshape(-1))
        self.files = df[['filename']].values.reshape(-1)

    def __getitem__(self, index):
        sig, _ = torchaudio.load(os.path.join(dataset_dir, self.files[index]))
        return sig, self.files[index], self.labels[index], self.devices[index], self.cities[index]

    def __len__(self):
        return len(self.files)
    
    # Consistent reasons attribution prior
    def prepare_label_extraction(self, available_indices):
        available_indices = torch.Tensor(available_indices)
        labels = torch.unique(self.labels)
        self.label_idcs = {}
        for label in labels:
            label_idc = (self.labels == label).nonzero(as_tuple=True)[0]
            mask = torch.isin(label_idc, available_indices)
            # Use the mask to filter out elements from indices
            filtered_indices = label_idc[mask]
            self.label_idcs[int(label)] = filtered_indices

    def extract_by_labels(self, labels):
        tensors = []
        for label in labels:
            tensors.append(self[np.random.choice(self.label_idcs[int(label)])][0].float())
        return torch.stack(tensors)



class SimpleSelectionDataset(TorchDataset):
    """A dataset that selects a subsample from a dataset based on a set of sample ids.
        Supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, available_indices):
        """
        @param dataset: dataset to load data from
        @param available_indices: available indices of samples for different splits
        return: waveform, file, label, device, city
        """
        self.available_indices = available_indices
        self.dataset = dataset

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[self.available_indices[index]]
        return x, file, label, device, city

    def __len__(self):
        return len(self.available_indices)
    
    def prepare_label_extraction(self):
        self.dataset.prepare_label_extraction(self.available_indices)
    
    def extract_by_labels(self, labels):
        return self.dataset.extract_by_labels(labels)

    

class DIRAugmentDataset(TorchDataset):
    """
    From https://github.com/fschmid56/cpjku_dcase23/blob/main/datasets/dcase23.py
    """

    def __init__(self, ds, dirs, prob):
        self.ds = ds
        self.dirs = dirs
        self.prob = prob

    def __getitem__(self, index):
        x, file, label, device, city = self.ds[index]

        fsplit = file.rsplit("-", 1)
        device = fsplit[1][:-4]

        if device == 'a' and torch.rand(1) < self.prob:
            # choose a DIR at random
            dir_idx = int(np.random.randint(0, len(self.dirs)))
            dir = self.dirs[dir_idx]

            x = torchaudio.functional.fftconvolve(x, dir, 'full')[:, :x.shape[1]]
        return x, file, label, device, city

    def __len__(self):
        return len(self.ds)
    
    def prepare_label_extraction(self):
        self.ds.prepare_label_extraction()
    
    def extract_by_labels(self, labels):
        return self.ds.extract_by_labels(labels)


def load_dirs(dirs_path, resample_rate):
    all_paths = [path for path in pathlib.Path(os.path.expanduser(dirs_path)).rglob('*.wav')]
    all_paths = sorted(all_paths)
    all_paths_name = [str(p).rsplit("/", 1)[-1] for p in all_paths]

    print("Augment waveforms with the following device impulse responses:")
    for i in range(len(all_paths_name)):
        print(i, ": ", all_paths_name[i])

    def process_func(dir_file):
        sig, _ = librosa.load(dir_file, sr=resample_rate, mono=True)
        sig = torch.from_numpy(sig[np.newaxis])
        return sig

    return [process_func(p) for p in all_paths]



class RollDataset(TorchDataset):
    """A dataset implementing time rolling of waveforms.
    """

    def __init__(self, dataset: TorchDataset, shift_range: int, axis=1):
        """
        @param dataset: dataset to load data from
        @param shift_range: maximum shift range
        return: waveform, file, label, device, city
        """
        self.dataset = dataset
        self.shift_range = shift_range
        self.axis = axis

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[index]
        sf = int(np.random.random_integers(-self.shift_range, self.shift_range))
        return x.roll(sf, self.axis), file, label, device, city

    def __len__(self):
        return len(self.dataset)
    
    def prepare_label_extraction(self):
        self.dataset.prepare_label_extraction()
    
    def extract_by_labels(self, labels):
        return self.dataset.extract_by_labels(labels)



def get_training_set(split=100, roll=False, resample_rate=32000, dir_p=0.0):
    assert str(split) in ("5", "10", "25", "50", "100"), "Parameters 'split' must be in [5, 10, 25, 50, 100]"
    os.makedirs(dataset_config['split_path'], exist_ok=True)
    subset_fname = f"split{split}.csv"
    subset_split_file = os.path.join(dataset_config['split_path'], subset_fname)
    if not os.path.isfile(subset_split_file):
        # download split{x}.csv (file containing all audio snippets for respective development-train split)
        subset_csv_url = dataset_config['split_url'] + subset_fname
        print(f"Downloading file: {subset_fname}")
        download_url_to_file(subset_csv_url, subset_split_file)
    ds = get_base_training_set(dataset_config['meta_csv'], subset_split_file)
    if roll:
        ds = RollDataset(ds, shift_range=roll)
    if dir_p > 0.0:
        ds = DIRAugmentDataset(ds, load_dirs(dataset_config['dirs_path'], resample_rate), dir_p)
    return ds


def get_base_training_set(meta_csv, train_files_csv):
    meta = pd.read_csv(meta_csv, sep="\t")
    train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
    train_subset_indices = list(meta[meta['filename'].isin(train_files)].index)
    ds = SimpleSelectionDataset(BasicDCASE24Dataset(meta_csv),
                                train_subset_indices)
    return ds


def get_test_set():
    os.makedirs(dataset_config['split_path'], exist_ok=True)
    test_split_csv = os.path.join(dataset_config['split_path'], dataset_config['test_split_csv'])
    if not os.path.isfile(test_split_csv):
        # download test.csv (file containing all audio snippets for development-test split)
        test_csv_url = dataset_config['split_url'] + dataset_config['test_split_csv']
        print(f"Downloading file: {dataset_config['test_split_csv']}")
        download_url_to_file(test_csv_url, test_split_csv)
    ds = get_base_test_set(dataset_config['meta_csv'], test_split_csv)
    return ds


def get_base_test_set(meta_csv, test_files_csv):
    meta = pd.read_csv(meta_csv, sep="\t")
    test_files = pd.read_csv(test_files_csv, sep='\t')['filename'].values.reshape(-1)
    test_indices = list(meta[meta['filename'].isin(test_files)].index)
    ds = SimpleSelectionDataset(BasicDCASE24Dataset(meta_csv), test_indices)
    return ds


class BasicDCASE24EvalDataset(TorchDataset):
    """
    Basic DCASE'24 Dataset: loads eval data from files
    """

    def __init__(self, meta_csv, eval_dir):
        """
        @param meta_csv: meta csv file for the dataset
        @param eval_dir: directory containing evaluation set
        return: waveform, file
        """
        df = pd.read_csv(meta_csv, sep="\t")
        self.files = df[['filename']].values.reshape(-1)
        self.eval_dir = eval_dir

    def __getitem__(self, index):
        sig, _ = torchaudio.load(os.path.join(self.eval_dir, self.files[index]))
        return sig, self.files[index]

    def __len__(self):
        return len(self.files)


def get_eval_set():
    assert os.path.exists(dataset_config['eval_dir']), f"No such folder: {dataset_config['eval_dir']}"
    ds = get_base_eval_set(dataset_config['eval_meta_csv'], dataset_config['eval_dir'])
    return ds


def get_base_eval_set(meta_csv, eval_dir):
    ds = BasicDCASE24EvalDataset(meta_csv, eval_dir)
    return ds

