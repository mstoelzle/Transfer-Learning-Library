import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler, RandomSampler

from torchpercentile import Percentile


class VWHL:
    def __init__(self, purpose: str, dataloader: DataLoader,
                 percentile_rank: float = 0, certainty_threshold: float = 0, weight_inferred_dataset: float = None):
        self.purpose = purpose
        self.dataloader = dataloader

        self.percentile_rank = percentile_rank
        self.certainty_threshold = certainty_threshold
        self.weight_inferred_dataset = weight_inferred_dataset

        self.max_pred_probabilities = None
        self.max_predictions = None

        self.start_idx = 0

    def reset(self):
        self.max_pred_probabilities = None
        self.max_predictions = None

        self.start_idx = 0

    def add_predictions(self, pred: torch.Tensor):
        pred = pred.detach()

        # we need to apply softmax to get normalized class probabilities
        softmax = torch.nn.Softmax(dim=1)
        pred_softmaxed = softmax(pred)

        # returns the indice of the class with the highest probability and its associated certainty for
        # all images in the batch
        batch_max_pred_probabilities, batch_max_predictions = torch.max(pred_softmaxed, 1)

        if self.max_pred_probabilities is None:
            self.max_pred_probabilities = batch_max_pred_probabilities.new_zeros(size=(len(self.dataloader.dataset),))
        self.max_pred_probabilities[self.start_idx:self.start_idx + batch_max_pred_probabilities.size(0)] \
            = batch_max_pred_probabilities

        if self.max_predictions is None:
            self.max_predictions = batch_max_predictions.new_zeros(size=(len(self.dataloader.dataset),))
        self.max_predictions[self.start_idx:self.start_idx + batch_max_predictions.size(0)] \
            = batch_max_predictions

        self.start_idx += pred.size(0)

    def get_semi_supervised_dataloader(self) -> DataLoader:
        percentiles = Percentile()(self.max_pred_probabilities, [self.percentile_rank])
        threshold: float = max(self.certainty_threshold, percentiles[0].item())
        self.max_predictions[self.max_pred_probabilities < threshold] = -1

        inferred_labels = self.max_predictions
        self.reset()

        inferred_dataset = _SemiSupervisedDataset(self.dataloader.dataset, inferred_labels)

        labeled_dataset = self.dataloader.dataset
        semi_supervised_dataset = ConcatDataset([inferred_dataset, labeled_dataset])

        if self.weight_inferred_dataset is None:
            if len(inferred_dataset) > 0:
                weight_inferred_dataset = len(labeled_dataset) / len(inferred_dataset)
            else:
                weight_inferred_dataset = 0.0
            # we want to limit the weight if the length of the inferred dataset is very short
            weight_inferred_dataset = min(weight_inferred_dataset, 100)
        else:
            weight_inferred_dataset = self.weight_inferred_dataset

        print(f"Using a weight of {self.weight_inferred_dataset} for {self.purpose} inferred dataset.")

        sampler = None
        if weight_inferred_dataset is not 1:
            weights_inferred_dataset = weight_inferred_dataset * torch.ones(size=(len(inferred_dataset),))
            weights_labeled_dataset = torch.ones(size=(len(labeled_dataset),))
            weights = torch.cat((weights_inferred_dataset, weights_labeled_dataset), dim=0)

            # num_samples: int = max(len(inferred_dataset), len(labeled_dataset))
            num_samples: int = len(semi_supervised_dataset)
            sampler = WeightedRandomSampler(weights, num_samples, replacement=True)

        if sampler is None:
            sampler = RandomSampler(semi_supervised_dataset)

        if inferred_dataset is not None:
            print(f"Constructed semi-supervised {self.purpose} dataset from "
                  f"{len(inferred_dataset)} inferred labels with length {len(semi_supervised_dataset)} "
                  f"and the RandomSampler uses {len(sampler)} samples.")

        return DataLoader(semi_supervised_dataset,
                          sampler=sampler,
                          batch_size=self.dataloader.batch_size, num_workers=self.dataloader.num_workers)


class _SemiSupervisedDataset(Dataset):
    def __init__(self, source_dataset: Dataset, inferred_labels: torch.Tensor):
        self.source_dataset = source_dataset

        cond = ~inferred_labels.eq(-1).detach().cpu()

        self.indices = torch.nonzero(cond, as_tuple=False).squeeze()
        self.targets = inferred_labels[cond].detach().cpu()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        feature, source_target = self.source_dataset[self.indices[idx]]

        return feature, self.targets[idx].item()
