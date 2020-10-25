import argparse
import numpy as np

from collections import namedtuple

LL=namedtuple('Sent', ['tokens', 'll'])


def read_file(fname, guu_lm=False, length=None):
    res = {}
    with open(fname) as fin:
        for i, line in enumerate(fin):
            if not guu_lm:
                id_, tokens, ll = line.rstrip().split()
                id_ = int(id_)
                tokens = int(tokens)
                ll = float(ll)
                assert id_ not in res
            else:
                id_ = i
                tokens = 0
                ll = -float(line.rstrip()) * np.log(10)
            res[id_] = LL(tokens=tokens, ll=ll)

            if length is not None and (i+1) == length:
                break

    return res, len(res)

def compute_ppl(res, input_tokens=None):
    ntokens = 0
    ll = 0
    for k,v in res.items():
        ntokens += v.tokens
        ll += v.ll

    if ntokens == 0:
        ntokens = input_tokens

    return -ll, ntokens, np.exp(-ll/ntokens)

def combined_ppl(data1, data2, discount):
    ntokens = 0
    ll_new = []
    for k in data1:
        if data2[k].tokens != 0:
            assert data1[k].tokens == data2[k].tokens
        ntokens += data1[k].tokens
        ll_new.append(np.logaddexp(data1[k].ll + np.log(discount),
            data2[k].ll + np.log(1. - discount)))

    return np.exp(-np.sum(ll_new) / ntokens)





parser = argparse.ArgumentParser(description='computer ppl with interpolation')
parser.add_argument('--input1', type=str, help='the path to the input text file')
parser.add_argument('--input2', type=str, help='the prefix to the saving file')
parser.add_argument('--discount', type=float, help='the prefix to the saving file')
parser.add_argument('--guu_lm', action='store_true', default=False, help='input2 is guu query file')

args = parser.parse_args()

data1, length = read_file(args.input1)
data2, _ = read_file(args.input2, args.guu_lm, length)

total_loss, ntokens, ppl = compute_ppl(data1)
print("input 1 total loss: {}, ntokens: {}, ppl: {}".format(total_loss, ntokens, ppl))

total_loss, ntokens, ppl = compute_ppl(data2, ntokens)
print("input 2 total loss: {}, ntokens: {}, ppl: {}".format(total_loss, ntokens, ppl))

print("interpolated ppl: {}".format(combined_ppl(data1, data2, args.discount)))







