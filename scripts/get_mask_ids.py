import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pre-compute mask id files')
    parser.add_argument('--train', type=str, help='the path to training file')
    parser.add_argument('--template', type=str, help='the path to template file')
    parser.add_argument('--outdir', type=str, help='output dir')

    args = parser.parse_args()

    template2id = {}
    with open(args.template) as fin:
        for i, line in enumerate(fin):
            template2id[line.strip()] = i


    with open(args.train) as fin, \
         open(os.path.join(args.outdir, 'mask_id.txt'), 'w') as fout:
        for i, line in enumerate(fin):
            fout.write('{}\n'.format(template2id.get(line.rstrip(), -1)))

            if i % 10000 == 0:
                print("processed {} lines".format(i))
