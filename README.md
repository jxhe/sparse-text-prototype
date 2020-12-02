# Sparse Neural Editor
This repo is the PyTorch implementation of this [paper](https://arxiv.org/abs/2006.16336):

```
Learning Sparse Prototypes for Text Generation
Junxian He, Taylor Berg-Kirkpatrick, Graham Neubig
NeurIPS 2020
```

In this repo, we implement a generative model of text that generates sentences by editying non-parametric prorotypes. The prototype support set is encouraged to be sparse during training to improve the memory/time efficiency at test time.


## Dependencies

The code mainly requires [PyTorch](https://pytorch.org/) (>=1.4.0) and [fairseq](https://github.com/pytorch/fairseq) (we run our experiments based on this specific [commit](https://github.com/pytorch/fairseq/commit/b65a85b692544e36f9e83ada91cf4ef529791c69)).

Install dependencies:

```bash
# install fairseq from a specific commit
git clone git@github.com:pytorch/fairseq.git fairseq_local
cd fairseq_local
git reset --hard b65a85b

# a modified sequence_generator.py to use edit vectors
cp ../sparse_prototype/sequence_generator.py fairseq

pip install --editable ./

cd ..

# install additional dependencies
pip install -r requirements.txt
```



## Prepare Data 

```bash
# download coco data
gdown https://drive.google.com/uc?id=1fMBZnMZz46qC0Im6y53MnDDQGRuwoC_M

# download yelp medium data
gdown https://drive.google.com/uc?id=1Z6wc4n5UBghwyNOo-C41vXEdNG5CE1Pa

# download yelp large data
gdown https://drive.google.com/uc?id=1Bgk94NZeoexdCWF_WPMoIFPLRjJsbuBF

mkdir datasets

# take coco dataset as an example
tar -xvzf coco40k.tar.gz -C datasets

# binarize dataset for fairseq
bash scripts/binarize_data.sh coco40k

# generate a mask file which is used to avoid selecting 
# exactly the same example as prototype during training
python scripts/get_mask_ids.py coco40k
```



## Training

We first pre-compute the sentence embeddings for all data examples offline and save them in memory-mapped files using `np.memmap`. During training/evaluation, a bilinear transformation is applied between these data embeddings and prototype embeddings to obtain the retrieval distribution. Here we use BERT as the offline  encoder:

```bash
# embeddings are saved into pretrained_sent_embeddings/[dataset name]
CUDA_VISIBLE_DEVICES=xx python scripts/precompute_bert.py [dataset name]
```



[GloVe](https://github.com/stanfordnlp/GloVe) embeddings are used in the paper to initialize word embeddings:

```bash
wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
mkdir glove_embeddings
unzip glove.6B.zip -d glove_embeddings

# compress glove embeddings to generate a new embedding file
# that only contains the dictionary of the dataset
python scripts/compress_glove.py \
		--embed-path glove_embeddings/glove.6B.300d.txt \
		--dict-path data-bin/[dataset_name]/dict.txt \
		> glove_embeddings/[dataset_name]_glove.txt
```



##### Train the model:

```bash
# train the sparse neural editor
# [GPUs] can be multiple ids to perform data-parallel training
# some hyperparameters can be specified (e.g. -a [alpha]), see 
# details in the script
bash scripts/train.sh -g [GPUs] -d [dataset name]

# train lm baseline
bash scripts/train.sh -g [GPUs] -c lm_baseline -d [dataset name]
```



## Evaluation

##### compute ppl:

```bash
# approximate importance-weighted ppl
bash scripts/train.sh -g [GPUs] -d [dataset name] -e iw -p [checkpoint directory]

# pruning prototypes can be performed at eval time
# [prune num] is the number of prototypes kept
bash scripts/train.sh -g [GPUs] -d [dataset name] -u [prune num] -e iw -p [checkpoint directory] 
```

## Template-based Generation
See the notebook `generate_demo.ipynb`(mainly the `sample_from_cluster` function) for examples to load the pretrained model and generate based on given templates.


## Citation

```
@inproceedings{he2020learning,
  title={Learning Sparse Prototypes for Text Generation},
  author={He, Junxian and Berg-Kirkpatrick, Taylor and Neubig, Graham},
  booktitle={Proceedings of NeurIPS},
  year={2020}
}
```

