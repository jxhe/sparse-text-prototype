import argparse
import numpy as np


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.

    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path) as f_embed:
        # next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = pieces[1:]

    return embed_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simplify glove word embedding')
    parser.add_argument('--embed-path', type=str, help='the original glove embed path')
    parser.add_argument('--dict-path', type=str, default='dict path')

    args = parser.parse_args()

    embed_dict = parse_embedding(args.embed_path)
    sample = embed_dict['a']
    embed_dim = len(sample)
    with open(args.dict_path) as fin:
        vocab_size = len(fin.readlines())

    print('{} {}'.format(vocab_size, embed_dim))
    with open(args.dict_path) as fin:
        for line in fin:
            word = line.split()[0]
            if word in embed_dict:
                print('{} {}'.format(word, ' '.join(embed_dict[word])))
            else:
                print('{} {}'.format(word, ' '.join(['0'] * embed_dim)))


