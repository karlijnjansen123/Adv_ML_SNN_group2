import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS

from preprocessing import Preprocessor


class SpeechCommands(Dataset):
    """
    This is the class for the full SpeechCommands dataset (I excluded background noise files).
    The preprocesser is flexible you cna add a different kind of preprocessing technique (if we wish later)
    """
    def __init__(self, root: str = "./data", preprocessor: Optional[Preprocessor] = None):
        super().__init__()

        self.base = SPEECHCOMMANDS(root=root, download=True)

        class_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, rel in enumerate(self.base._walker):
            label = os.path.basename(os.path.split(rel)[0])

            if label == "_background_noise_":
                continue

            class_to_indices[label].append(i)

        self.indices: List[int] = [i for _, idxs in class_to_indices.items() for i in idxs]
        self.classes: List[str] = sorted(class_to_indices.keys())
        self.label2index: Dict[str, int] = {c: j for j, c in enumerate(self.classes)}
        self.num_classes: int = len(self.classes)

        self.pre = preprocessor or Preprocessor(sample_rate=16000, n_mfcc=40, n_mels=64, cmvn=True, fixed_frames=100)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        waveform, sr, label, *_ = self.base[self.indices[idx]]
        mfcc = self.pre(waveform, int(sr))
        y = self.label2index[label]
        return mfcc, y



