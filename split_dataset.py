import tqdm
import math
import torch
import numpy as np


def split_to_train_test_set(
    train_ratio: float,
    origin_dataset: torch.utils.data.Dataset,
    num_classes: int,
    random_split: bool = False,
):
    """
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.random.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    """
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(tqdm.tqdm(origin_dataset)):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0:pos])

    # return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)
    return train_idx


if __name__ == "__main__":
    from timm.data import create_dataset

    dataset = create_dataset(
        "imagenet",
        root="/data1/ligq/imagenet1-k",
        split="train",
        is_training=True,
        batch_size=1,
        # download=True,
    )
    train_idx = split_to_train_test_set(
        1 / 8, origin_dataset=dataset, num_classes=1000, random_split=True
    )
    np.save("train_idx.npy", np.array(train_idx))
