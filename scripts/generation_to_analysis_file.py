import sys
import os
import argparse


parser = argparse.ArgumentParser(
    description='take generation file to a tab splitted prototype-generation file')
parser.add_argument('--input', type=str, default='eval_gen_sample_beam.log', help='the file name')
parser.add_argument('--exp-dir', type=str, help='exp dir')

args = parser.parse_args()
fout = open(os.path.join(args.exp_dir, 'generation_analysis_input.txt'), 'w')

with open(os.path.join(args.exp_dir, args.input)) as fin:
    while True:
        line = fin.readline()
        if not line:
            break
        if line.startswith('S-'):
            prototype = line.rstrip().split('\t')[1]
            examples = []
            while True:
                tmp = fin.readline()
                if not tmp or tmp == '\n':
                    break

                if '-generations-' in tmp:
                    continue

                if len(tmp.rstrip().split('\t')) == 3:
                    example = tmp.rstrip().split('\t')[2]
                    examples.append(example)

            for example in examples:
                fout.write('{}\t{}\n'.format(prototype, example))

fout.close()