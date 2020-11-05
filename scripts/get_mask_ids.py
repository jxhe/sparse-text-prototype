import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pre-compute mask id files')
    parser.add_argument('dataset', type=str, help='the path to training file')

    args = parser.parse_args()

    template2id = {}
    with open(f'datasets/{args.dataset}/template.txt') as fin:
        for i, line in enumerate(fin):
            template2id[line.strip()] = i


    with open(f'datasets/{args.dataset}/train.txt') as fin, \
         open(f'data-bin/{args.dataset}/mask_id.txt', 'w') as fout:
        for i, line in enumerate(fin):
            fout.write(f'{template2id.get(line.rstrip(), -1)}\n')
            if i % 10000 == 0:
                print("processed {} lines".format(i))

