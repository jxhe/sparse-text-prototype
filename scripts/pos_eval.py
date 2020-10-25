import os
import argparse
import subprocess
import random
import edlib
from typing import List
from collections import Counter

import stanza

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

def read_file(fname, len_cut):
    res1, res2 = [], []
    with open(fname) as fin:
        for line in fin:
            x, y = line.rstrip().split('\t')
            if len(x.split()) > len_cut or len(y.split()) > len_cut:
                continue
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

def eval_edit(prototype, example):

    def flat_cigar(cigar):
        """flatten the result path returned by edlib.align
        """
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


    res = {}
    for p_sent, e_sent in zip(prototype, example):
        p_pos = [x.upos for x in p_sent.words]
        e_pos = [x.upos for x in e_sent.words]

        p_text = [x.text for x in p_sent.words]
        e_text = [x.text for x in e_sent.words]

        edit_operation = edlib.align(e_text, p_text, task='path')        
        edit_operation = flat_cigar(edit_operation['cigar'])

        new_p_text = []
        new_e_text = []
        new_p_pos = []
        new_e_pos = []
        src_cur = tgt_cur = 0

        for edit in edit_operation:
            if edit == '=' or edit == 'X':
                new_p_text.append(p_text[src_cur])
                new_p_pos.append(p_pos[src_cur])
                new_e_text.append(e_text[tgt_cur])
                new_e_pos.append(e_pos[tgt_cur])
                src_cur += 1
                tgt_cur += 1
            elif edit == 'I':
                new_p_text.append(-1)
                new_p_pos.append(-1)
                new_e_text.append(e_text[tgt_cur])
                new_e_pos.append(e_pos[tgt_cur])
                tgt_cur += 1
            elif edit == 'D':
                new_p_text.append(p_text[src_cur])
                new_p_pos.append(p_pos[src_cur])
                new_e_text.append(-1)
                new_e_pos.append(-1)
                src_cur += 1
            else:
                raise ValueError('{} edit operation is invalid!'.format(edit))  

        for i, edit in enumerate(edit_operation):
            if edit not in res:
                res[edit] = Counter()

            if edit == '=':
                res[edit]['{}={}'.format(new_p_pos[i], new_e_pos[i])] += 1
            elif edit == 'X':
                res[edit]['{}->{}'.format(new_p_pos[i], new_e_pos[i])] += 1 
            elif edit == 'I':
                res[edit]['+{}'.format(new_e_pos[i])] += 1
            elif edit == 'D':
                res[edit]['-{}'.format(new_p_pos[i])] += 1
            else:
                raise ValueError

    return res





def eval_f1(prototype, example):
    res = {}
    for p_sent, e_sent in zip(prototype, example):
        p_pos = [x.upos for x in p_sent.words]
        e_pos = [x.upos for x in e_sent.words]

        p_text = [x.text for x in p_sent.words]
        e_text = [x.text for x in e_sent.words]

        e_word_counter = Counter(e_text)
        for word, pos in zip(p_text, p_pos):
            if pos not in res:
                res[pos] = ExtractMetric(
                    nume=0, 
                    denom_p=0,
                    denom_r=0,
                    precision=0,
                    recall=0,
                    f1=0
                    )

            res[pos].denom_r += 1
            if e_word_counter[word] > 0:
                e_word_counter[word] -= 1
                res[pos].nume += 1

        e_pos_counter = Counter(e_pos)
        for k, v in e_pos_counter.items():
            if k not in res:
                res[k] = ExtractMetric(
                    nume=0, 
                    denom_p=0,
                    denom_r=0,
                    precision=0,
                    recall=0,
                    f1=0
                    ) 

            res[k].denom_p += v

    for k, v in res.items():
        if res[k].denom_p != 0 and res[k].denom_r != 0 and res[k].nume != 0:
            res[k].precision = res[k].nume / res[k].denom_p         
            res[k].recall = res[k].nume / res[k].denom_r        
            res[k].f1 = 2 * res[k].precision * res[k].recall / (res[k].precision + res[k].recall)

    return res


def sentence_bleu(ref_path, hypo_path):
    sent_bleu = subprocess.getoutput(
        "fairseq-score --ref {} --sys {} --sentence-bleu".format(ref_path, hypo_path))    
    bleu_list = [float(line.split()[3].rstrip(',')) for line in sent_bleu.split('\n')[1:]]
    return sum(bleu_list) / len(bleu_list)

def generate_rand_prototype(exp_dir, num):
    dataset_to_template = {
        "coco40k": "support_prototype/datasets/coco/coco.template.40k.txt",
        "yelp": "support_prototype/datasets/yelp_data/yelp.template.50k.lower.txt",
        "yelp_large": "support_prototype/datasets/yelp_large_data/yelp_large.template.100k.txt",
    }

    def parse_exp_dir(name):
        dataset = name.rstrip('/').split('/')[-1].split('_')[0]
        return dataset

    dataset = parse_exp_dir(exp_dir)

    return subprocess.getoutput(
        "shuf -n {} {}".format(num, dataset_to_template[dataset])).split('\n')


parser = argparse.ArgumentParser(description='Evaluate analysis metrics')
parser.add_argument('--prefix', type=str, choices=['inference', 'generation'], 
    help='prediction file prefix')
parser.add_argument('--exp-dir', type=str, help='output directory')

args = parser.parse_args()

fout = open(os.path.join(args.exp_dir, 'analysis_{}_res.txt'.format(args.prefix)), 'w')
len_cut = 1000
prototypes, examples = read_file(os.path.join(args.exp_dir, '{}_analysis_input.txt'.format(args.prefix)), len_cut=len_cut)
prototype_path = os.path.join(args.exp_dir, 'prototype.txt')
prototype_pos_path = os.path.join(args.exp_dir, 'prototype_pos.txt')

prototype_rand_path = os.path.join(args.exp_dir, 'prototype_rand.txt')
prototype_pos_rand_path = os.path.join(args.exp_dir, 'prototype_pos_rand.txt')

example_path = os.path.join(args.exp_dir, 'example.txt')
example_pos_path = os.path.join(args.exp_dir, 'example_pos.txt')

prototypes_rand = generate_rand_prototype(args.exp_dir, len(examples))

write_file(prototype_path, prototypes)
write_file(example_path, examples)
write_file(prototype_rand_path, prototypes_rand)

# surface BLEU
# bleu = subprocess.getoutput(
#     "./support_prototype/scripts/multi-bleu.perl {} < {}".format(prototype_path, example_rand_path))
bleu = sentence_bleu(prototype_rand_path, example_path)
print('Regular BLEU (random baseline): \n{}'.format(bleu))
fout.write('Regular BLEU (random baseline): \n{}'.format(bleu))

fout.write('\n\n\n')

# bleu = subprocess.getoutput(
#     "./support_prototype/scripts/multi-bleu.perl {} < {}".format(prototype_path, example_path))
bleu = sentence_bleu(prototype_path, example_path)
print('Regular BLEU: \n{}'.format(bleu))
fout.write('Regular BLEU: \n{}'.format(bleu))

fout.write('\n\n\n')

# POS tagging
print('POS tagging')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos', tokenize_pretokenized=True)
prototype_doc = nlp('\n'.join(prototypes))
example_doc = nlp('\n'.join(examples))
prototype_rand_doc = nlp('\n'.join(prototypes_rand))

prototypes_pos = [[word.upos for word in sent.words] for sent in prototype_doc.sentences]
examples_pos = [[word.upos for word in sent.words] for sent in example_doc.sentences]

prototypes_pos_rand = [[word.upos for word in sent.words]for sent in prototype_rand_doc.sentences]

write_file(prototype_pos_path, prototypes_pos)
write_file(example_pos_path, examples_pos)
write_file(prototype_pos_rand_path, prototypes_pos_rand)


# POS BLEU
# bleu = subprocess.getoutput(
#     "./support_prototype/scripts/multi-bleu.perl {} < {}".format(prototype_pos_path, example_pos_rand_path))
bleu = sentence_bleu(prototype_pos_rand_path, example_pos_path)
print('POS BLEU (random baseline): \n{}'.format(bleu))
fout.write('POS BLEU (random baseline): \n{}'.format(bleu))

fout.write('\n\n\n')

# bleu = subprocess.getoutput(
#     "./support_prototype/scripts/multi-bleu.perl {} < {}".format(prototype_pos_path, example_pos_path))
bleu = sentence_bleu(prototype_pos_path, example_pos_path)
print('POS BLEU: \n{}'.format(bleu))
fout.write('POS BLEU: \n{}'.format(bleu))

fout.write('\n\n\n')

# break down precision and recall
print("compute precision, recall, f1")
assert len(prototypes) == len(prototypes_pos)
assert len(examples) == len(examples_pos)

res = eval_f1(list(prototype_rand_doc.sentences), list(example_doc.sentences))

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


fout.write('\n\n\n')

# edit operations
print("edit analysis")
res = eval_edit(list(prototype_doc.sentences), list(example_doc.sentences))
total = sum([sum(v.values()) for k, v in res.items()])
fout.write('total: {}\n'.format(total))
res = sorted(res.items(), key=lambda item: (-sum(item[1].values())))
for k, v in res:
    fout.write('{}: {}\n'.format(k, sum(v.values())))
    for k1, v1 in v.most_common():
        fout.write('{}: {} ({:.3f}), '.format(k1, v1, v1 / sum(v.values())))
    fout.write('\n\n')

fout.close()

