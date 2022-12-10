#!/usr/bin/env python
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from argparse import ArgumentParser

import numpy as np
import utils
import sys

def preproc_kernel_data(inputfile, repfile, allfile):
    cols = utils.stats_of_interest[1:] + ['gpc__cycles_elapsed.max']
    names, data = utils.read_ncu_raw_file_numpy(inputfile, utils.stats_of_interest[1:])
    _, data_full = utils.read_ncu_raw_file_numpy(inputfile, cols)

    with open(allfile, 'w') as af:
        af.write('\t'.join(['Index', 'Kernel'] + cols) + '\n')
        for idx, (name, row) in enumerate(zip(names, data_full)):
            af.write('\t'.join([str(idx), name] + [str(x) for x in row]) + '\n')

    packed_dists = pdist(data, 'cosine')
    dists = squareform(packed_dists)


    rep_list = utils.pick_clusters(dists)
    print(f'# Total Rep Kernels: {len(rep_list)}')
    print('Rep Kernel List', rep_list)

    with open(repfile, 'w') as rf:
        rf.write('\t'.join(['Index', 'Kernel'] + cols) + '\n')
        for idx in sorted(rep_list):
            rf.write('\t'.join([str(idx), names[idx]] + [str(x) for x in data_full[idx]]) + '\n')

def main():
    parser = ArgumentParser(
        prog='kernel similarity tool',
        description='read ncu and dump kernel stats',
        epilog='read the code'
    )
    parser.add_argument('-i', '--input')
    parser.add_argument('-r', '--rep-output', default='kernel_signatures_rep.csv')
    parser.add_argument('-a', '--all-output', default='kernel_signatures_all.csv')
    
    args = parser.parse_args()
    if not args.input:
        exit()
    
    preproc_kernel_data( args.input, args.rep_output, args.all_output )


if __name__ == '__main__':
    main()
