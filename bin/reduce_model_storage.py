from argparse import ArgumentParser
from glob import glob
import os
from os.path import expanduser
import re
from typing import Union

StrOrBytesPath = Union[str, bytes, os.PathLike]


def parse_arguments():
    parser = ArgumentParser(
        description="Utility that helps keep the size of your checkpoints storage trackable by deleting irrelevant stored models."
    )
    parser.add_argument(
        "--prefix",
        type=expanduser,
        default="checkpoints",
        help="A folder containing DCP experiments",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=50,
        help="Preserve model at every i-th epoch. Note: the last model is always kept",
    )
    return parser.parse_args()


def files_to_delete(prefix: StrOrBytesPath, interval: int):

    pattern = re.compile(r".*model\.([0-9]+).t7")

    files = []
    for experiment in sorted(glob(os.path.join(prefix, "*"))):

        # populate full list of models available
        models_to_remove = dict()
        for model in sorted(
            glob(os.path.join(experiment, "models", "model.[0-9]*.t7"))
        ):
            ret = pattern.search(model)
            if not ret:
                continue

            models_to_remove[int(ret[1])] = model

        if not len(models_to_remove):
            continue

        # remove selected models from this list
        min_n, max_n = min(models_to_remove), max(models_to_remove)
        del models_to_remove[max_n]
        for i in range(min_n, max_n, interval):
            if i in models_to_remove:
                del models_to_remove[i]
        files.extend(models_to_remove.values())
    return files


def main():
    args = parse_arguments()

    files = files_to_delete(args.prefix, args.interval)
    if len(files):
        input(
            f"Press ENTER to proceed to delete {len(files)} models in folder {os.path.realpath(args.prefix)}:"
        )

    for file in files:
        os.remove(file)


if __name__ == "__main__":
    main()
