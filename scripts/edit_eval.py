import os
import argparse
import subprocess
import random
import edlib
from typing import List
from collections import Counter

import stanza

def flat_cigar(cigar):
    r = []
    pointer = 0

    while pointer < len(cigar):
        num = []
        while cigar[pointer].isdigit():
            num.append(cigar[pointer])
            pointer += 1
        num = int(''.join(num))

        r.extend([cigar[pointer]] * num)
        pointer += 1

    return r

class ExtractMetric(object):
    """used for precision recall"""
    def __init__(self, nume=0, denom_p=0, denom_r=0, precision=0, recall=0, f1=0):
        super(ExtractMetric, self).__init__()
        self.nume = nume
        self.denom_p = denom_p
        self.denom_r = denom_r
        self.precision = precision
        self.recall = recall
        self.f1 = f1

def read_file(fname):
    res1, res2 = [], []
    with open(fname) as fin:
        for line in fin:
            x, y = line.rstrip().split('\t')
            res1.append(x)
            res2.append(y)

    return res1, res2

def write_file(fname: str, data: List[str]):
    with open(fname, 'w') as fout:
        for sent in data:
            if isinstance(sent, list):
                fout.write('{}\n'.format(' '.join(sent)))
            else:
                fout.write('{}\n'.format(sent))


parser = argparse.ArgumentParser(description='Evaluate analysis metrics')
parser.add_argument('--prefix', type=str, choices=['inference', 'generation'],
    help='prediction file prefix')
parser.add_argument('--exp-dir', type=str, help='output directory')

args = parser.parse_args()

fout = open(os.path.join(args.exp_dir, 'edit_analysis_{}_res.txt'.format(args.prefix)), 'w')

prototypes, examples = read_file(os.path.join(args.exp_dir, '{}_analysis_input.txt'.format(args.prefix)))
examples_rand = random.sample(examples, len(examples))
prototype_path = os.path.join(args.exp_dir, 'prototype.txt')
prototype_pos_path = os.path.join(args.exp_dir, 'prototype_pos.txt')


example_path = os.path.join(args.exp_dir, 'example.txt')
example_rand_path = os.path.join(args.exp_dir, 'example_rand.txt')
example_pos_path = os.path.join(args.exp_dir, 'example_pos.txt')
example_pos_rand_path = os.path.join(args.exp_dir, 'example_pos_rand.txt')



write_file(prototype_path, prototypes)
write_file(example_path, examples)
write_file(example_rand_path, examples_rand)

# surface BLEU
bleu = subprocess.getoutput(
    "./support_prototype/scripts/multi-bleu.perl {} < {}".format(prototype_path, example_rand_path))
print('Regular BLEU (random baseline): \n{}'.format(bleu))
fout.write('Regular BLEU (random baseline): \n{}'.format(bleu))

fout.write('\n\n\n')

bleu = subprocess.getoutput(
    "./support_prototype/scripts/multi-bleu.perl {} < {}".format(prototype_path, example_path))
print('Regular BLEU: \n{}'.format(bleu))
fout.write('Regular BLEU: \n{}'.format(bleu))

fout.write('\n\n\n')

# POS tagging
print('POS tagging')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos', tokenize_pretokenized=True)
prototype_doc = nlp('\n'.join(prototypes))
example_doc = nlp('\n'.join(examples))

prototypes_pos = [[word.upos for word in sent.words] for sent in prototype_doc.sentences]
examples_pos = [[word.upos for word in sent.words] for sent in example_doc.sentences]

example_rand_doc = random.sample(list(example_doc.sentences), len(example_doc.sentences))
examples_pos_rand = [[word.upos for word in sent.words]for sent in example_rand_doc]

write_file(prototype_pos_path, prototypes_pos)
write_file(example_pos_path, examples_pos)
write_file(example_pos_rand_path, examples_pos_rand)


# POS BLEU
bleu = subprocess.getoutput(
    "./support_prototype/scripts/multi-bleu.perl {} < {}".format(prototype_pos_path, example_pos_rand_path))
print('POS BLEU (random baseline): \n{}'.format(bleu))
fout.write('POS BLEU (random baseline): \n{}'.format(bleu))

fout.write('\n\n\n')

bleu = subprocess.getoutput(
    "./support_prototype/scripts/multi-bleu.perl {} < {}".format(prototype_pos_path, example_pos_path))
print('POS BLEU: \n{}'.format(bleu))
fout.write('POS BLEU: \n{}'.format(bleu))

fout.write('\n\n\n')

# break down precision and recall
print("compute precision, recall, f1")
assert len(prototypes) == len(prototypes_pos)
assert len(examples) == len(examples_pos)

res = eval_f1(list(prototype_doc.sentences), example_rand_doc)

res = sorted(res.items(), key=lambda item: -item[1].f1)

fout.write('random baseline precision-recall\n')
fout.write('POS recall precision f1\n')
for k, v in res:
    fout.write('{} {} {} {}\n'.format(k, v.recall, v.precision, v.f1))

fout.write('\n\n\n')

res = eval_f1(list(prototype_doc.sentences), list(example_doc.sentences))
res = sorted(res.items(), key=lambda item: -item[1].f1)

fout.write('precision-recall\n')
fout.write('POS recall precision f1\n')
for k, v in res:
    fout.write('{} {} {} {}\n'.format(k, v.recall, v.precision, v.f1))

fout.close()
