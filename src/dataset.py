"""
Dataset & Feature Extraction Module
Downloads BBBP dataset, extracts 2048-bit Morgan Fingerprints,
silences RDKit C++ logs, and applies randomized Scaffold Splitting.
"""
from collections import defaultdict
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

# Mute RDKit C++ stderr spam
RDLogger.DisableLog("rdApp.*")

MOLECULE_NET_BBBP_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
)


def load_raw_data(url: str = MOLECULE_NET_BBBP_URL) -> pd.DataFrame:
    df = pd.read_csv(url)
    df = df.dropna(subset=["smiles", "p_np"])
    df = df[["smiles", "p_np", "name"]].reset_index(drop=True)
    return df


def smiles_to_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype="float32")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    fp_array = generator.GetCountFingerprintAsNumPy(mol)
    return (fp_array > 0).astype("float32")


def generate_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_split(df: pd.DataFrame, train_ratio: float = 0.8, seed: int = 42):
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        scaffold = generate_scaffold(row["smiles"])
        scaffolds[scaffold].append(idx)

    scaffold_sets = list(scaffolds.values())
    random.seed(seed)
    random.shuffle(scaffold_sets)

    train_cutoff = int(len(df) * train_ratio)
    train_indices, val_indices = [], []

    for group in scaffold_sets:
        if len(train_indices) + len(group) <= train_cutoff:
            train_indices.extend(group)
        else:
            val_indices.extend(group)

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    return train_df, val_df


class BBBPDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        features, targets = [], []

        for row in df.itertuples():
            mfp = smiles_to_morgan_fingerprint(row.smiles)
            features.append(mfp)
            targets.append(row.p_np)

        self.X = torch.tensor(np.array(features), dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]