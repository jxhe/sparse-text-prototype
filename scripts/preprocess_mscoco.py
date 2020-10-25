import os
import h5py
import numpy as np

if __name__ == '__main__':

    data_dir = 'support_prototype/datasets/coco'
    emb_data = np.load(os.path.join(data_dir, 'emb_and_id.npz'))

    save_dir = 'pretrained_sent_embeddings'

    embed_dict = {}
    for id_, embed in zip(emb_data['id'], emb_data['t_emb']):
        embed_dict[id_] = embed

    ftrain = open(os.path.join(data_dir, 'coco.template.40k.txt'), 'w')
    cur = 0
    with h5py.File(os.path.join(save_dir, 'coco40k.template.hdf5'), 'w') as fout:
        with open(os.path.join(data_dir, 'coco.train.shuf40k.deduplicate')) as fin:
            for line in fin:
                id_, text = line.split('\t')
                if id_ in embed_dict:
                    embed = embed_dict[id_]
                    ftrain.write(text)
                    fout.create_dataset(str(cur), embed.shape, dtype='float32', data=embed)
                    cur += 1

    ftrain.close()
    print('finish preprocessing template data')

    # ftrain = open(os.path.join(data_dir, 'coco.train.40k.txt'), 'w')
    # cur = 0
    # with h5py.File(os.path.join(save_dir, 'mscoco.train.hdf5'), 'w') as fout:
    #     with open(os.path.join(data_dir, 'coco.train.shuf40k')) as fin:
    #         for line in fin:
    #             id_, text = line.split('\t')
    #             if id_ in embed_dict:
    #                 embed = embed_dict[id_]
    #                 ftrain.write(text)
    #                 fout.create_dataset(str(cur), embed.shape, dtype='float32', data=embed)
    #                 cur += 1

    # ftrain.close()
    # print('finish preprocessing training data')

    # fval = open(os.path.join(data_dir, 'coco.valid.4k.txt'), 'w')
    # cur = 0
    # with h5py.File(os.path.join(save_dir, 'mscoco.valid.hdf5'), 'w') as fout:
    #     with open(os.path.join(data_dir, 'coco.valid.shuf4k')) as fin:
    #         for line in fin:
    #             id_, text = line.split('\t')
    #             if id_ in embed_dict:
    #                 embed = embed_dict[id_]
    #                 fval.write(text)
    #                 fout.create_dataset(str(cur), embed.shape, dtype='float32', data=embed)
    #                 cur += 1

    # fval.close()
    # print('finish preprocessing valid data')
