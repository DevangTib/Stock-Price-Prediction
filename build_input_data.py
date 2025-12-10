import numpy as np
import os
import random

# make needed directories
def make_dir():
    base = "./Data/"
    for sub in ["train_price", "train_label", "train_text", "test_price", "test_label", "test_text"]:
        path = os.path.join(base, sub)
        os.makedirs(path, exist_ok=True)

# get valid company list
def get_company_list():
    with open("./valid_company.txt", 'r') as f:
        return [line.strip() for line in f if line.strip()]

# build data by sampling each company independently
def build_data(company_list, data_size, train_rate, price_in, text_in, label_in, price_out, text_out, label_out):
    """
    For each sample index, randomly pick a date file for each company.
    Splits into train/test based on index and train_rate.
    """
    for i in range(data_size):
        price_list, tweet_list, label_list = [], [], []
        for comp in company_list:
            all_files = os.listdir(os.path.join(price_in, comp))
            # split available files by train_rate
            total = len(all_files)
            split_idx = int(total * train_rate)
            if price_out.endswith('train_price'):
                choices = all_files[:split_idx]
            else:
                choices = all_files[split_idx:]
            if not choices:
                choices = all_files
            selected = random.choice(choices)
            date = os.path.splitext(selected)[0]
            # load data arrays
            price_list.append(np.load(f"{price_in}/{comp}/{selected}"))
            tweet_list.append(np.load(f"{text_in}/{comp}/{selected}"))
            label_list.append(np.load(f"{label_in}/{comp}/{selected}"))
        # stack and save
        np.save(f"{price_out}/{i:010d}.npy", np.stack(price_list))
        np.save(f"{text_out}/{i:010d}.npy", np.stack(tweet_list))
        np.save(f"{label_out}/{i:010d}.npy", np.stack(label_list))
        if i % 20 == 0:
            print(f"Processed {i}/{data_size}")

if __name__ == '__main__':
    TRAIN_SIZE = 300
    TEST_SIZE = 90
    TRAIN_RATE = 0.7

    make_dir()
    companies = get_company_list()

    # build training data
    build_data(
        companies, TRAIN_SIZE, TRAIN_RATE,
        price_in="./Data/raw_data/price",
        text_in="./Data/raw_data/text",
        label_in="./Data/raw_data/label",
        price_out="./Data/train_price",
        text_out="./Data/train_text",
        label_out="./Data/train_label"
    )
    # build test data
    build_data(
        companies, TEST_SIZE, TRAIN_RATE,
        price_in="./Data/raw_data/price",
        text_in="./Data/raw_data/text",
        label_in="./Data/raw_data/label",
        price_out="./Data/test_price",
        text_out="./Data/test_text",
        label_out="./Data/test_label"
    )
    print("Dataset build complete.")
