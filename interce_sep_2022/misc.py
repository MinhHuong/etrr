import os
import time
import pandas as pd
import numpy as np


def move_file(inrepo, outrepo, fname):
    src = os.path.join(inrepo, fname)
    dest = os.path.join(outrepo, fname)
    os.rename(src, dest)


def move_file_main():
    inpath = 'E:\\PhD\\Data\\clean\\2021\\NAT_decoded'
    outpath = 'E:\\PhD\\Data\\clean\\2021\\NAT_decoded_marche'
    atts = ['pmot', 'umot', 'imot']
    n_moved = 0
    n_files = 0

    start = time.perf_counter()
    # ------------------------------------------------------------ #
    # MOVE FILES THAT CONTAIN ONLY FOOTSTEP DATA TO ANOTHER FOLDER #
    # ------------------------------------------------------------ #
    with os.scandir(inpath) as it:
        for entry in it:
            # skip if it's not a file
            if entry.name.startswith('.') or not entry.is_file():
                continue

            # print to keep track
            n_files += 1
            if n_files % 1000 == 0:
                print(f'Processed {n_files} files...')

            # get the base filename
            inp = entry.name

            # read the file
            data = pd.read_pickle(os.path.join(inpath, inp))

            # if the file has no data, move it
            N = len(data)
            if N == 0:
                move_file(inpath, outpath, inp)
                n_moved += 1
                continue

            # if the data pmot, umot, imot have no signals, move the file
            if np.count_nonzero(data[atts].values) == 0:
                move_file(inpath, outpath, inp)
                n_moved += 1
                continue

            # separate door and footstep data
            mapped_id = np.zeros(N)
            for i in range(N):
                row = data.iloc[i]
                ID = row['id']
                if ID == 32 or ID == 33:
                    mapped_id[i] = 1  # door
                if 34 <= ID <= 37:
                    mapped_id[i] = 2  # footstep
                if ID == 16:
                    mapped_id[i] = mapped_id[i - 1] if i > 0 else mapped_id[i + 1]
            data['mapped_id'] = mapped_id
            # separate porte and marche data
            idx_p = np.where(data['mapped_id'] == 1)[0]
            # if the data contain no door signal, move that file to another folder
            if len(idx_p) == 0:
                move_file(inpath, outpath, inp)
                n_moved += 1
                continue

    print(f'\n{n_moved} files moved to {outpath}.\n')

    end = time.perf_counter()
    print(f'\nTOTAL TIME: {(end - start) / 60} mins')


if __name__ == '__main__':
    inpath = 'E:\\PhD\\Data\\clean\\2021\\NAT_decoded'
    limit = 100
    n_files = 0
    for entry in os.scandir(inpath):
        print(entry.name)
        # print to keep track
        n_files += 1
        if n_files % 1000 == 0:
            print(f'Processed {n_files} files...')

        if n_files == limit:
            break
