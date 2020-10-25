#! /bin/bash
#
# setup_on_new_cluster.sh
# Copyright (C) 2020-04-14 Junxian <He>
#
# Distributed under terms of the MIT license.
#


# copy data files from SPLICER to this machine

pip install edlib
pip install tensorboardX
pip install transformers

scp -r $SPLICER:/projects/junxianh/support-prototype/support_prototype/data-bin ./support_prototype/
scp -r $SPLICER:/projects/junxianh/support-prototype/glove_embeddings ./
scp -r $SPLICER:/projects/junxianh/support-prototype/pretrained_sent_embeddings ./

pip install --editable .

# cd support_prototype/sentence-transformers/
# pip install -e .
# cd ../../
