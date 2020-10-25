#! /bin/bash
#
# train.sh
# Copyright (C) 2020-02-09 Junxian <He>
#
# Distributed under terms of the MIT license.
#

# evaluation script
# Usage:
# bash eval.sh -g [GPU] -p [model_dir]

iw_nsamples=1000
valid_subset="valid"


DATE=`date +%Y%m%d`
data_bin="yelp_large"
emb_type="bert"
eval_mode='gen_interpolation'
decode_strategy='beam'
kappa=30
max_epoch=-1
retriever=pretrained_embed
linear_bias=0
stop_bert_grad=1
freeze_retriever=0
reinforce=1
temperature=1
grad_lambda=0
forget_rate=0.8
decay_rate=1
copy=0
inv_editor='levenshtein'
edit_embed_dim=10
entropy_w=1
term2_w=1
rescale_factor=1.
criterion=sp_elbo
log_format=simple

load_name=''

retrieve_split=train

glove_path=glove_embeddings/${data_bin}_glove.txt
opt=adam
template_emb_file=pretrained_sent_embeddings/${data_bin}.template.${emb_type}.hdf5

if [ "$data_bin" = "ptb" ];
then
    num_class=41088
    max_tokens=2048
    save_interval_updates=100
    warmup_updates=1000
    retrieve_split=valid1
    ns=5
elif [ "$data_bin" = "ptb10" ];
then
    num_class=5703
    max_tokens=512
    save_interval_updates=0
    warmup_updates=800
    ns=20
elif [ "$data_bin" = "coco40k" ];
then
    num_class=39577
    max_tokens=2048
    save_interval_updates=0
    warmup_updates=1000
    max_epoch=30
    retrieve_split=valid1
    ns=10

    if [ "$retriever" = "sentence-bert" ];
    then
        template_emb_file=pretrained_sent_embeddings/${data_bin}.template.bert.hdf5
        echo "read bert embeddings"
    fi
elif [ "$data_bin" = "yelp" ];
then
    num_class=50000
    max_tokens=2048
    save_interval_updates=0
    warmup_updates=10000
    max_update=300000
    retrieve_split=valid1
    log_interval=100
    retriever=bert
    ns=10
    log_format=tqdm
elif [ "$data_bin" = "yelp_large" ];
then
    num_class=100000
    max_tokens=1024 # distributed on two gpus
    save_interval_updates=5000
    warmup_updates=150000
    max_update=500000
    kappa=40
    lambda_config="0:0,150000:1"
    retrieve_split=valid1
    log_interval=100
    retriever=bert
    validate_interval=1000
    ns=10
else
    num_class=0
    max_tokens=0
    save_interval_updates=0
    warmup_updates=0
    ns=0
fi

if [ "$opt" = "adam" ];
then
    warmup_init_lr='1e-03'
    # add_opt_string="--adam-betas '(0.9, 0.98)'"
    add_opt_string=''
    lr=0.001
else
    warmup_init_lr='1'
    add_opt_string=""
    warmup_updates=8000
    lr=1.0
fi



GPU=0
alpha=0.01
separate="1"
prune_num="-1"
gen_prototype=200
gen_nz=10

while getopts ":g:a:p:k:e:l:t:s:r:u:c:n:z:" arg; do
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
    l) lambda_momentum="$OPTARG"
    ;;
    t) temperature="$OPTARG"
    ;;
    s) separate="$OPTARG"
    ;;
    r) rescale_factor="$OPTARG"
    ;;
    u) prune_num="$OPTARG"
    ;;
    c) criterion="$OPTARG"
    ;;
    n) gen_prototype="$OPTARG"
    ;;
    z) gen_nz="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

max_tokens=$(( max_tokens * ns / gen_nz ))

if [ "$criterion" = "lm_baseline" ];
then
    ns=1
    eval_mode='none'
fi

GPUSTR=$(printf "$GPU" | tr , _)

if [[ -v LOADDIR ]];
then
    # add_load_string="--reset-meters"
    add_load_string="--reset-meters --reset-optimizer"
    cstring="_continue"
    restore_file=checkpoint_load.pt
    save_interval_updates=50
    if [ "$opt" = "adam" ];
    then
        lr=0.0003
        warmup_init_lr=0.0003
        warmup_updates=8000
    fi
    if [ "$stop_bert_grad" = 0 ];
    then
        max_tokens=1024
        ns=10
        lr=0.1
        warmup_init_lr='1e-3'
        warmup_updates=5000
    fi
else
    add_load_string=""
    cstring=""
    restore_file=null.pt
fi

if [ "$eval_mode" = "entropy" ];
then
    LOADDIR="tmp"
fi

enc_opt_freq=10
dec_opt_freq=1

if [ "$separate" = "0" ];
then
    separate_str=""
    train_separate_str=""
    train_script=train.py
else
    echo "separate training"
    separate_str="_sep_eof${enc_opt_freq}_dof${dec_opt_freq}"
    train_separate_str="--dec-opt-freq ${dec_opt_freq} --enc-opt-freq ${enc_opt_freq}"
    train_script=train_junxian.py
    warmup_updates=$(( warmup_updates*2 ))
    max_epoch=$(( max_epoch*2 ))
fi

if [ "$prune_num" != "-1" ];
then
    prune_str="_prune${prune_num}"
else
    prune_str=""
fi
# declare -a rescale_list=("0.01")
# declare -a alpha_list=("0.01")

if [ "$decode_strategy" = "beam" ];
then
    decode_str=""
elif [ "$decode_strategy" = "sample" ];
then
    decode_str="--sampling --sampling-topk 100 --beam 1"
else
    decode_str=""
fi

echo "start evaluation"


CUDA_VISIBLE_DEVICES=${GPU} python generate_junxian.py \
    support_prototype/data-bin/${data_bin} \
    --arch ${data_bin} --task support_prototype \
    --dropout 0.3 \
    --edit-embed-dim ${edit_embed_dim} --embed-init-rescale ${rescale_factor} \
    ${train_separate_str} \
    --retrieve-embed ${template_emb_file} \
    --train-embed pretrained_sent_embeddings/${data_bin}.train.${emb_type}.hdf5 \
    --valid-embed pretrained_sent_embeddings/${data_bin}.${valid_subset}.${emb_type}.hdf5 \
    --reinforce ${reinforce} --infer-ns ${ns} --reinforce-temperature ${temperature} \
    --freeze-retriever ${freeze_retriever}  --decoder-copy ${copy} \
    --inveditor-embed-path ${glove_path} --encoder-embed-path ${glove_path} --decoder-embed-path ${glove_path} \
    --encoder-embed-dim 300 --decoder-embed-dim 300 \
    --grad-lambda ${grad_lambda} --entropy-weight ${entropy_w} --term2-weight ${term2_w} \
    --user-dir support_prototype \
    --forget-rate ${forget_rate} --decay-rate ${decay_rate} --retrieve-split ${retrieve_split} --alpha ${alpha} --vmf-kappa ${kappa} \
    --linear-bias ${linear_bias} --stop-bert-grad ${stop_bert_grad} \
    --criterion ${criterion} --label-smoothing 0. --num-workers 0 \
    --max-tokens ${max_tokens} --num-class ${num_class} \
    --log-format ${log_format} --log-interval 5 \
    --retriever ${retriever} --inv-editor ${inv_editor} \
    --path ${LOADDIR}/checkpoint_best.pt \
    --eval-mode ${eval_mode} ${decode_str} --gen-np ${gen_prototype} --gen-nz ${gen_nz}\
    > ${LOADDIR}/eval_${eval_mode}_${decode_strategy}.log
