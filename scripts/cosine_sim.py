#!/usr/bin/env python
from argparse import ArgumentParser
from scipy.spatial import distance
import csv

'''
Example Usage:
python scripts/cosine_sim.py --input kernel_signatures_rep.csv --ref scripts/kernel_signatures_rep.csv --test sqrt

- finds all kernels with the keyword sqrt in them in the input file, and compares with all kernels with keyword sqrt in the reference (NERF) file

'''

import numpy as np
import utils
import sys
def parse_proc_line(line):
    return line.split('\t')[2:-1]

def parse_original_line(line):
    return line.split(',')[-17:]

def to_float(l):
    return [float(x) for x in l]

def main():
    parser = ArgumentParser(
        prog='cosine similarity calculator',
        description='read the code',
        epilog='read the code'
    )
    parser.add_argument('-i', '--input')
    parser.add_argument('-r', '--ref', default='kernel_signatures_rep.csv')
    parser.add_argument('-t', '--test', default='sin')
    
    args = parser.parse_args()
    if not args.input:
        exit()

    input_list = []
    ref_list = []
    with open(args.input, 'r') as ip:
        for line in ip:
            line_s = line.strip()
            if args.test in line_s:
                input_list.append((line_s, to_float(parse_proc_line(line_s))))

    with open(args.ref, 'r') as rf:
        for line in rf:
            line_s = line.strip()
            if args.test in line_s:
                ref_list.append((line_s, to_float(parse_proc_line(line_s))))
    
    for line, val in input_list:
        for ref, refval in ref_list:
            dist = distance.cosine(val, refval) 
            print(distance.cosine(val, refval))
            if dist <= 0.2:
                print("SYNTH:", "[{}]".format(line.split('\t')[0]), line.split('\t')[1])
                print("NERF:", "[{}]".format(ref.split('\t')[0]), ref.split('\t')[1])
                print("MATCH:", line.split('\t')[1] == ref.split('\t')[1])
                print()

if __name__ == '__main__':
    main()
