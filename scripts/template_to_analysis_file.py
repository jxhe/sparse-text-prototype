import sys
import os
import argparse


parser = argparse.ArgumentParser(
    description='take template file to a tab splitted prototype-example file')
parser.add_argument('--input', type=str, default='templates_eval_valid.txt', help='the file name')
parser.add_argument('--exp-dir', type=str, help='the prefix to the saving file')

args = parser.parse_args()
fout = open(os.path.join(args.exp_dir, 'inference_analysis_input.txt'), 'w')

with open(os.path.join(args.exp_dir, args.input)) as fin:
    while True:
        line = fin.readline()
        if not line:
            break
        if line.startswith('src:'):
            example = ' '.join(line.rstrip().split()[1:])
            while True:
                tmp = fin.readline()
                if not tmp:
                    raise ValueError

                if '-top K templates-' in tmp:
                    prototype = fin.readline()
                    prototype = prototype.rstrip().split('\t')[2]
                    break
            fout.write('{}\t{}\n'.format(prototype, example))

fout.close()
