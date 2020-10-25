import os
import argparse
import h5py
import torch

from tqdm import tqdm

from sentence_transformers import SentenceTransformer



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pre-compute the Sentence Bert embedding')
    parser.add_argument('--retrieve-path', type=str, help='the path to the retrieve pool file')
    parser.add_argument('--prefix', type=str, help='the path to the retrieve pool file')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    save_dir = "pretrained_sent_embeddings"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = "cuda" if args.cuda else "cpu"
    model_name = 'bert-base-nli-mean-tokens'
    model = SentenceTransformer(model_name, device=device)
    model.eval()

    batch_size = 128
    pbar = tqdm()
    with h5py.File(os.path.join(save_dir, '{}.sentbert.hdf5'.format(args.prefix)), 'w') as fout:
        with open(args.retrieve_path, 'r') as fin:
            batches = []
            cur = 0
            for i, line in enumerate(fin):
                batches.append(line.strip())
                if (i+1) % (batch_size * 10) == 0:
                    embeddings = model.encode(batches, batch_size=batch_size)

                    for j, embed in enumerate(embeddings):
                        fout.create_dataset(str(cur), embed.shape, dtype='float32', data=embed)
                        cur += 1

                    pbar.update(len(batches))
                    batches = []

            if len(batches) > 0:
                embeddings = model.encode(batches, batch_size=batch_size)

                for j, embed in enumerate(embeddings):
                    fout.create_dataset(str(cur), embed.shape, dtype='float32', data=embed)
                    cur += 1

