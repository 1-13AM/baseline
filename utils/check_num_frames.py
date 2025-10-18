import os
import glob
import argparse
import shutil
def remove_folders_with_few_files(base_path, n):
    """
    Removes folders that have fewer than 'n' files inside.

    Args:
        base_path (str): Path to the base directory containing folders.
        n (int): Minimum number of files a folder must have to avoid deletion.
    """
    folders = glob.glob(os.path.join(base_path, '*/*'))
    for folder in folders:
        files = glob.glob(os.path.join(folder, '*'))
        if len(files) < n:
            print(f"Removing folder: {folder}")
            shutil.rmtree(folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove folders with fewer than 'n' files.")
    parser.add_argument("--base_path", type=str, help="Base path to the directory containing folders.")
    parser.add_argument("--n", type=int, default=30, help="Minimum number of files a folder must have to avoid deletion.")
    args = parser.parse_args()

    remove_folders_with_few_files(args.base_path, args.n)
