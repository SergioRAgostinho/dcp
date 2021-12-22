from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
from pathlib import Path
from typing import Union

import numpy as np
from tqdm import tqdm


def scatter_mean(data, indices):
    inverse, counts = np.unique(indices, return_inverse=True, return_counts=True)[1:3]
    idx_sorted = np.argsort(inverse)

    reduce_idx = np.zeros(len(counts), dtype=counts.dtype)
    np.add.accumulate(counts[:-1], out=reduce_idx[1:])
    out = np.add.reduceat(data[idx_sorted], reduce_idx) / counts[:, None]
    return out


def grid_filter_numpy(pts, size):

    # compute unique cluster indices
    pool_coords = np.floor(pts / size).astype(int)
    pool_id, counts = np.unique(
        pool_coords, return_inverse=True, return_counts=True, axis=0
    )[1:]

    # compose pool information
    idx = np.repeat(np.arange(len(counts)), counts)
    values = np.argsort(pool_id)

    # compute scatter mean
    reduce_idx = np.zeros_like(counts)
    np.add.accumulate(counts[:-1], out=reduce_idx[1:])
    out = np.add.reduceat(pts[values], reduce_idx) / counts[:, None]
    return out, dict(idx=idx, values=values)


def search_optimal_grid_size(
    points: np.array, target_points: int, max_iters: int = 100
):

    # first grid size estimate
    span = np.max(points, axis=0) - np.min(points, axis=0)
    size = np.max(span / target_points)

    #
    i = 0
    while True:
        n = len(grid_filter_numpy(points, size)[0])
        # print(i, n)
        i += 1
        rel = abs(n - target_points) / target_points
        if n >= target_points and (rel < 0.05 or i >= max_iters):
            break

        size *= 1.1 if n > target_points else 0.9

    return size


def process_pointcloud(points: np.array, colors: np.array, max_points: int):

    # if point cloud is already small do nothing
    if len(points) <= max_points:
        return points, colors

    # find right grid size, right above max_points
    grid_size = search_optimal_grid_size(points, max_points)

    # filter the grid out
    points_r, cells = grid_filter_numpy(points, grid_size)
    colors_r = scatter_mean(colors[cells["values"]], cells["idx"])

    # subsample max number
    idx = np.random.choice(len(points_r), max_points, replace=False)
    return points_r[idx], colors_r[idx]


def process_folder(prefix: Union[str, Path], output: Union[str, Path], max_points: int):

    # create output folder unconditionally
    os.makedirs(output, exist_ok=True)

    # Figure out how many files we need to process
    n = len(list(os.scandir(prefix)))

    # iterate all file in prefix
    for i, item in enumerate(tqdm(os.scandir(prefix), total=n)):

        # sample 113 had no points
        # if i < 113:
        #     continue

        if not item.is_file():
            continue

        # point clouds files. down sample
        if item.name.endswith(".npz"):
            data = np.load(item.path)
            points, colors = process_pointcloud(data["pcd"], data["color"], max_points)
            np.savez_compressed(
                os.path.join(output, item.name), pcd=points, color=colors
            )

        # symlink sequence files
        elif item.name.endswith(".txt"):
            src = item.path
            dst = os.path.join(output, item.name)

            # if both are relative. symlink relative
            if not (os.path.isabs(src) or os.path.isabs(dst)):
                src = os.path.relpath(src, output)

            try:
                os.symlink(src, dst)
            except FileExistsError:
                pass


def parse_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("output", help="The desired output directory")
    parser.add_argument(
        "--max_points", type=int, default=1024, help="Max number of points to retain"
    )
    parser.add_argument(
        "--prefix",
        default="pointclouds",
        help="Allows manually specifying a prefix for the input folder.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=20,
        help="Number of iterations spent trying the find optimal grid size",
    )
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_arguments()

    process_folder(args.prefix, args.output, args.max_points)


if __name__ == "__main__":
    main()
