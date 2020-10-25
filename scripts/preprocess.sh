#! /bin/bash
#
# preprocess.sh
# Copyright (C) 2020-02-09 Junxian <He>
#
# Distributed under terms of the MIT license.
#

NAME=yelp
TEXT=datasets/yelp_data
shuf=shuf4

fairseq-preprocess \
        --only-source \
        --trainpref $TEXT/${NAME}.train.lower.txt \
        --validpref $TEXT/${NAME}.valid.small.lower.txt,$TEXT/${NAME}.template.50k.${shuf}.txt\
        --testpref $TEXT/${NAME}.test.split200.txt \
        --destdir data-bin/yelp_${shuf} \
        --nwordssrc 10000 \
        --workers 20 \
        --srcdict data-bin/yelp/dict.txt \

# cp datasets/${NAME}_data/mask_id.txt data-bin/${NAME}/

