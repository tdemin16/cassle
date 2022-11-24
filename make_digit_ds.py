import os
from torchvision.datasets import MNIST, USPS, SVHN

def make_if_needed(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dataset(ds, train):
    if ds == "mnist":
        dataset_class = MNIST(root="data", train=train, download=True)
    elif ds == "usps":
        dataset_class = USPS(root="data", train=train, download=True)
    elif ds == "svhn":
        dataset_class = SVHN(root="data", split="train" if train else "test", download=True)
    else:
        raise AssertionError

    dataset = []
    for x, y in dataset_class:
        dataset.append((x, y))

    dataset.sort(key=lambda x: x[1])

    crurr_label = 0
    counter = 0
    association = []
    for image, label in dataset:
        if label != crurr_label:
            counter = 0
            crurr_label = label
        
        path = f"datasets/digits/{ds}/{'train' if train else 'val'}/{label}/{str(counter).zfill(4)}.jpg"
        make_if_needed(os.path.join(*(path.split('/')[:-1])))
        association.append((os.path.join(*(path.split('/')[2:])), label))
        image.save(path)

        counter += 1

    with open(f"datasets/digits/{ds}_{'train' if train else 'val'}.txt", 'w') as f:
        for line in association:
            f.write(f"{line[0]} {line[1]}\n")


if __name__ == "__main__":
    for dataset in ["mnist", "svhn", "usps"]:
        for split in [True, False]:
            make_dataset(dataset, split)