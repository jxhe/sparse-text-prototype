#! /bin/bash
#
# preprocess.sh
# Copyright (C) 2020-02-09 Junxian <He>
#
# Distributed under terms of the MIT license.
#

dataset=$1
datadir=datasets/${dataset}
bpeprefix=bpe20000

fairseq-preprocess \
        --only-source \
        --trainpref ${datadir}/${bpeprefix}.train.txt \
        --validpref ${datadir}/${bpeprefix}.valid.txt,${datadir}/${bpeprefix}.template.txt \
        --testpref ${datadir}/${bpeprefix}.test.txt \
        --destdir data-bin/${dataset} \
        --workers 20 \
        # --nwordssrc 10000 \

