"""
Preprocess the Questa dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/questa", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--train_val_frac", default=0.9, help="split for validation set."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "foreverlasting1202/QuestA"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "default")
    else:
        dataset = datasets.load_dataset(data_source, "default")

    whole_dataset = dataset["train"]
    num_train = int(len(whole_dataset) * args.train_val_frac)
    num_val = len(whole_dataset) - num_train
    train_dataset = whole_dataset.select(range(num_train))
    test_dataset = whole_dataset.select(range(num_train, num_train + num_val))

    print("Number of training: {}".format(len(train_dataset)))
    print("Number of test: {}".format(len(test_dataset)))

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")

            question = question_raw + " " + instruction_following

            answer = example.pop("answer")
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
