#! /bin/bash
#
# preprocess.sh
# Copyright (C) 2020-02-09 Junxian <He>
#
# Distributed under terms of the MIT license.
#

dataset=$1
datadir=datasets/${dataset}

fairseq-preprocess \
        --only-source \
        --trainpref ${datadir}/train.txt \
        --validpref ${datadir}/valid.txt,${datadir}/template.txt \
        --testpref ${datadir}/test.txt \
        --destdir data-bin/${dataset} \
        --nwordssrc 10000 \
        --workers 20 \

