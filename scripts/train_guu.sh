#! /bin/bash
#
# train.sh
# Copyright (C) 2020-02-09 Junxian <He>
#
# Distributed under terms of the MIT license.
#

DATE=`date +%Y%m%d`

kappa=30
alpha=0.01
linear_bias=0
stop_bert_grad=1
freeze_retriever=1
reinforce=0
temperature=1
grad_lambda=0
data_bin="ptb"
momentum=1
copy=0
inv_editor='levenshtein'
edit_embed_dim=10

entropy_w=0
term2_w=0

glove_path=glove_embeddings/${data_bin}_glove.txt

if [ "$data_bin" = "ptb" ];
then
    num_class=42068
    if [ "$copy" = 1 ];
    then
        max_tokens=2048
    else
        max_tokens=4096
    fi
    save_interval_updates=0
    warmup_updates=2000
    ns=5
elif [ "$data_bin" = "ptb10" ];
then
    num_class=5703
    max_tokens=512
    save_interval_updates=0
    warmup_updates=800
    ns=50
else
    num_class=0
    max_tokens=0
    save_interval_updates=0
    warmup_updates=0
    ns=0
fi

while getopts ":g:a:p:k:e:" arg; do
  case $arg in
    g) GPU="$OPTARG"
    ;;
    a) alpha="$OPTARG"
    ;;
    p) LOADDIR="$OPTARG"
    ;;
    k) kappa="$OPTARG"
    ;;
    e) edit_embed_dim="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done


GPUSTR=$(printf "$GPU" | tr , _)

SAVE_ROOT=checkpoint/${data_bin}/${DATE}/guu_${data_bin}_alpha${alpha}_kappa${kappa}_m${momentum}_bias${linear_bias}_rf${reinforce}_ns${ns}_t${temperature}_gl${grad_lambda}_sb${stop_bert_grad}_fr${freeze_retriever}_ew${entropy_w}_tw${term2_w}_copy${copy}_iv${inv_editor}_editdim${edit_embed_dim}_gpu${GPU}

SAVE=${SAVE_ROOT}
TENSORBOARD=${SAVE}/tensorboard

rm -r ${SAVE}; mkdir -p ${SAVE} ${TENSORBOARD}

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    support_prototype/data-bin/${data_bin} \
    --arch ${data_bin} --task support_prototype \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates ${warmup_updates} \
    --warmup-init-lr '1e-03' \
    --dropout 0.3 --weight-decay 0.0001 \
    --decoder-dropout-in 0.5 --decoder-dropout-out 0.5 --decoder-layers 1 \
    --edit-embed-dim ${edit_embed_dim} \
    --encoder-layers 1 --retrieve-embed pretrained_sent_embeddings/${data_bin}.train.hdf5 \
    --reinforce ${reinforce} --infer-ns ${ns} --reinforce-temperature ${temperature} \
    --freeze-retriever ${freeze_retriever} --decoder-copy ${copy} \
    --inveditor-embed-path ${glove_path} --encoder-embed-path ${glove_path} --decoder-embed-path ${glove_path} \
    --encoder-embed-dim 300 --decoder-embed-dim 300 \
    --grad-lambda ${grad_lambda} --entropy-weight ${entropy_w} --term2-weight ${term2_w} \
    --user-dir support_prototype \
    --lambda-momentum ${momentum} --retrieve-split train --alpha ${alpha} --vmf-kappa ${kappa} \
    --linear-bias ${linear_bias} --stop-bert-grad ${stop_bert_grad} \
    --criterion guu_elbo --label-smoothing 0. --num-workers 0 \
    --max-tokens ${max_tokens} --num-class ${num_class} \
    --log-format simple --log-interval 20 \
    --retriever sentence-bert --inv-editor ${inv_editor} \
    --validate-interval 1 --best-checkpoint-metric ppl --no-epoch-checkpoints \
    --no-last-checkpoints \
    --save-interval-updates ${save_interval_updates} --keep-interval-updates 1 \
    --save-dir ${SAVE} --tensorboard-logdir ${TENSORBOARD} \
    | tee -a ${SAVE}/stdout.log
    # --restore-file checkpoint_best.pt \

