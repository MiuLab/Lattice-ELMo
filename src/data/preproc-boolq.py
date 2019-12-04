import argparse
import re
import csv
import json

from tqdm import tqdm


def preproc(dataset_file, out_file):
    data = []
    for line in open(dataset_file):
        row = json.loads(line.strip())
        row["label"] = row.pop("answer")
        row["text"] = row.pop("passage")
        row["text2"] = row.pop("question")
        data.append(row)

    with open(out_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, sorted(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_file")
    parser.add_argument("out_file")
    args = parser.parse_args()
    preproc(args.dataset_file, args.out_file)
