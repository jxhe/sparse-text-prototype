import os
import argparse
import h5py
import torch

from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
from transformers import *



def get_sentence_features(batches, tokenizer, model, device):
    features = tokenizer.batch_encode_plus(batches, padding=True,
        return_attention_mask=True, return_token_type_ids=True)
    attention_mask = torch.tensor(features['attention_mask'], device=device)
    input_ids = torch.tensor(features['input_ids'], device=device)
    token_type_ids=torch.tensor(features['token_type_ids'], device=device)

    # (batch, seq_len, nfeature)
    token_embeddings = model(input_ids=input_ids, 
        attention_mask=attention_mask,
        token_type_ids=token_type_ids)[0]

    # mean of embeddings as sentence embeddings
    embeddings = (attention_mask.unsqueeze(-1) * token_embeddings).sum(1) / attention_mask.sum(1).unsqueeze(-1)

    return embeddings


def hdf5_create_dataset(group, input_file):
    global tokenizer, model, device

    print(f'precompute embeddings for {input_file}')
    gname = group.name[1:]
    pbar = tqdm()
    with open(input_file, 'r') as fin:
        batches = []
        cur = 0
        for i, line in enumerate(fin):
            batches.append(line.strip())
            if (i+1) % batch_size == 0:
                with torch.no_grad():
                    embeddings = get_sentence_features(batches, tokenizer, model, device)

                for j, embed in enumerate(embeddings):
                    group.create_dataset(f'{gname}_{cur}', embed.shape, dtype='float32', data=embed.cpu())
                    cur += 1

                pbar.update(len(batches))
                batches = []

        if len(batches) > 0:
            with torch.no_grad():
                embeddings = get_sentence_features(batches, tokenizer, model, device)

            for j, embed in enumerate(embeddings):
                group.create_dataset(f'{gname}_{cur}', embed.shape, dtype='float32', data=embed.cpu())
                cur += 1

MODELS = [BertModel, BertTokenizer, 'bert-base-uncased']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pre-compute the Bert embeddings')
    parser.add_argument('dataset', type=str, help='the path to the dataset name')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    save_dir = "precompute_embedding_datasets"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = "cuda" if args.cuda else "cpu"

    model_class, tokenizer_class, pretrained_weights = MODELS
    model = model_class.from_pretrained(pretrained_weights)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    model.to(device)
    model.eval()


    batch_size = 128
    with h5py.File(os.path.join(save_dir, f'{args.dataset}.bert.hdf5'), 'w') as fout:
        for gname in ['valid', 'test', 'template', 'train']:
            if os.path.isfile(f'datasets/{args.dataset}/{gname}.txt'):
                group = fout.create_group(gname)
                hdf5_create_dataset(group, os.path.join(f'datasets/{args.dataset}/{gname}.txt'))
            elif gname != 'test':
                raise ValueError(f'{gname} file must exist')
