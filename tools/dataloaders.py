import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler


def get_dataloader(dataset: Dataset, start_fraction: float = 0, stop_fraction: float = 1,
                   shuffle=False, seed=1, **kwargs) -> DataLoader:
    assert 0 <= start_fraction < stop_fraction <= 1

    if start_fraction == 0 and stop_fraction == 1:
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
    else:
        indices = torch.arange(len(dataset))

        if shuffle:
            rng = np.random.RandomState(seed=seed)
            np_indices = indices.detach().numpy()
            rng.shuffle(np_indices)
            indices = torch.tensor(np_indices)

        start_idx = int(start_fraction * len(dataset))
        stop_idx = int(stop_fraction * len(dataset))
        selected_indices = indices[start_idx:stop_idx]

        sampler = SubsetRandomSampler(indices=selected_indices)

    loader = DataLoader(dataset, sampler=sampler, **kwargs)

    return loader
