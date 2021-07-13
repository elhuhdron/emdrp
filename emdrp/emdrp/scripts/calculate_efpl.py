from emdrp.utils.efpl import calc_supervoxel_eftpl
import numpy as np

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument('--csv_results', type=str, help="file path for result csv")
    parser.add_argument('--nml_file', type=str, help="file path for nml file")
    parser.add_argument('--h5_file', type=str, help="file path for hdf5 file")
    parser.add_argument('--size', type=int, nargs='+', help="size of region of interest")
    parser.add_argument('--dataset_start', type=int, nargs='+', help="start indices of region of interest")

    return parser, parser.parse_args()

def main():
    parser, args = parse_args()

    eftpl, params = calc_supervoxel_eftpl(args.nml_file, args.h5_file, size=args.size, dset_start=args.dataset_start)

    results = np.asarray([eftpl, params ])
    np.savetxt(args.csv_results, results.transpose(), fmt='%s', delimiter=',', header=','.join(['eftpl', 'param']))

if __name__ == '__main__':
    main()