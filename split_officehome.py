# TODO: Add download of dataset
import os
import random as rand
import shutil

SOURCE = "cassle/datasets/officehome"
CLASSES = [
    "Art",
    "Clipart",
    "Product",
    "Real World"
]
SEED = 0
RATIO = 0.8

rand.seed(SEED)


def check_and_create(path):
    if not os.path.exists(path):
        os.makedirs(path)


def split(path):
    data = os.listdir(path)
    rand.shuffle(data)
    idx = int(RATIO * len(data))
    train = data[:idx]
    test = data[idx:]
    return train, test


def move_to_dir(data, source, target):
    img_path_list = []
    check_and_create(target)
    for img in data:
        img_path = os.path.join(source, img)
        shutil.move(img_path, target)

        img_path_target = os.path.join(target, img)
        img_path_list.append(img_path_target)

    return img_path_list


def add_in_pos(path, mid, pos):
    target = path.split('/')
    target.insert(pos, mid)
    target = os.path.join(*target)
    return target


def save_list(l, path):
    with open(path, 'w') as fp:
        for line in l:
            fp.write(line + '\n')


def main():
    train_dir = os.path.join(SOURCE, "train")
    test_dir = os.path.join(SOURCE, "test")
    
    # create train and test dir
    check_and_create(train_dir)
    check_and_create(test_dir)

    for domain in os.listdir(SOURCE):
        # skip if not class domain
        if domain not in CLASSES:
            continue    
        
        # path of each image in the domain
        domain_train = []
        domain_test = []

        # path from root to domain
        domain_path = os.path.join(SOURCE, domain)
        for class_ in os.listdir(domain_path):
            # path from root to class
            class_path = os.path.join(domain_path, class_)
            # train, test split
            train, test = split(class_path)
            
            # create train and test target path for the image
            train_target = add_in_pos(class_path, "train", -2)
            test_target = add_in_pos(class_path, "test", -2)

            # move images
            train_files = move_to_dir(train, class_path, train_target)
            test_files = move_to_dir(test, class_path, test_target)

            domain_train.extend(train_files)
            domain_test.extend(test_files)

            shutil.rmtree(class_path)
        
        # create save path
        train_file_path = os.path.join(add_in_pos(domain_path, "train", -1), 'dataset.txt')
        test_file_path = os.path.join(add_in_pos(domain_path, "test", -1), 'dataset.txt')

        # sorting to make things easier to check
        domain_train.sort()
        domain_test.sort()

        save_list(domain_train, train_file_path)
        save_list(domain_test, test_file_path)

        shutil.rmtree(domain_path)


if __name__ == "__main__":
    main()