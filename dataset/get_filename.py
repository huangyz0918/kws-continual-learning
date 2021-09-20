import argparse
from pathlib import Path
from random import randrange

from os import listdir
from os.path import isfile, join

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input optional guidance for training")
    parser.add_argument("--dpath", default="./data", type=str, help="The path of dataset")
    parser.add_argument("--train", default=0.8, type=float, help="The ratio of splited training data")
    parser.add_argument("--silence", default=0, type=float,
                        help="The ratio of splited silence data in all training data")
    args = parser.parse_args()

    class_list = [
        "yes", "no", "nine", "three", "bed",
        "up", "down", "wow", "happy", "four",
        "left", "right", "seven", "six", "marvin",
        "on", "off", "house", "zero", "sheila",
        "stop", "go", "dog", "cat", "two",
        "bird", "eight", "five", "tree", "one"
    ]

    splits_path = "./splits"
    Path(splits_path).mkdir(parents=True, exist_ok=True)

    # create the training data index file.
    with open(f'{splits_path}/train.txt', 'w') as f:
        for keyword in class_list:
            print(f"indexing the training data of keyword: {keyword}...")
            path = args.dpath + '/' + keyword
            file_names = [f for f in listdir(path) if isfile(join(path, f))]
            file_len = len(file_names)
            train_len = int(file_len * args.train)
            silence_len = int(train_len * args.silence)

            for index in range(train_len):
                if randrange(train_len) < silence_len:
                    print(f"_silence_/{file_names[index]}", file=f)
                else:
                    print(f"{keyword}/{file_names[index]}", file=f)

    # create the training data index file.
    with open(f'{splits_path}/valid.txt', 'w') as f:
        for keyword in class_list:
            print(f"indexing the training data of keyword: {keyword}...")
            path = args.dpath + '/' + keyword
            file_names = [f for f in listdir(path) if isfile(join(path, f))]
            file_len = len(file_names)
            train_len = int(file_len * args.train)
            silence_len = int(train_len * args.silence)

            for index in range(train_len, file_len):
                if randrange(train_len) < silence_len:
                    print(f"_silence_/{file_names[index]}", file=f)
                else:
                    print(f"{keyword}/{file_names[index]}", file=f)
