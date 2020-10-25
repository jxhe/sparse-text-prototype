#! /bin/bash
#
# train.sh
# Copyright (C) 2020-02-09 Junxian <He>
#
# Distributed under terms of the MIT license.
#

DATE=`date +%Y%m%d`

data_bin="yelp_shuf4"
emb_type="bert"
kappa=30
lambda_config="0:0,1250:1"
max_update=30000
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
criterion="sp_elbo"
iw_nsamples=1000
free_bits=0
embed_transform_nlayer=0
log_interval=5
validate_interval=1
weight_decay=0.0001

load_name=''

retrieve_split=train

glove_path=glove_embeddings/yelp_glove.txt
opt=adam
template_emb_file=pretrained_sent_embeddings/${data_bin}.template.${emb_type}.hdf5

if [ "$criterion" = "topk_elbo" ];
then
    reinforce=0
fi

if [ "$data_bin" = "ptb" ];
then
    num_class=41088
    max_tokens=2048
    save_interval_updates=0
    warmup_updates=1000
    max_update=30000
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
    max_update=15000
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
    save_interval_updates=5000
    warmup_updates=20000
    max_update=300000
    lambda_config="0:0,20000:1"
    retrieve_split=valid1
    log_interval=100
    retriever=bert
    validate_interval=1000
    ns=10
elif [ "$data_bin" = "yelp_shuf1" ];
then
    num_class=50000
    max_tokens=2048
    save_interval_updates=5000
    warmup_updates=20000
    max_update=300000
    lambda_config="0:0,20000:1"
    retrieve_split=valid1
    log_interval=100
    retriever=bert
    validate_interval=1000
    ns=10
elif [ "$data_bin" = "yelp_shuf2" ];
then
    num_class=50000
    max_tokens=2048
    save_interval_updates=5000
    warmup_updates=20000
    max_update=300000
    lambda_config="0:0,20000:1"
    retrieve_split=valid1
    log_interval=100
    retriever=bert
    validate_interval=1000
    ns=10
elif [ "$data_bin" = "yelp_shuf3" ];
then
    num_class=50000
    max_tokens=2048
    save_interval_updates=5000
    warmup_updates=20000
    max_update=300000
    lambda_config="0:0,20000:1"
    retrieve_split=valid1
    log_interval=100
    retriever=bert
    validate_interval=1000
    ns=10
elif [ "$data_bin" = "yelp_shuf4" ];
then
    num_class=50000
    max_tokens=2048
    save_interval_updates=5000
    warmup_updates=20000
    max_update=300000
    lambda_config="0:0,20000:1"
    retrieve_split=valid1
    log_interval=100
    retriever=bert
    validate_interval=1000
    ns=10
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
separate="0"

while getopts ":g:a:p:k:e:l:t:s:r:f:c:" arg; do
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
    f) free_bits="$OPTARG"
    ;;
    c) criterion="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

if [ "$criterion" = "lm_baseline" ];
then
    ns=1
fi

GPUSTR=$(printf "$GPU" | tr , _)

lambda_conifg_str=$(printf "$lambda_config" | tr , _)
lambda_conifg_str=$(printf "$lambda_conifg_str" | tr : _)

if [[ -v LOADDIR ]];
then
    # add_load_string="--reset-meters"
    # add_load_string="--reset-meters --reset-optimizer"
    add_load_string=""
    cstring="_continue"
    restore_file=checkpoint_load.pt
    # save_interval_updates=50
    # if [ "$opt" = "adam" ];
    # then
    #     lr=0.0003
    #     warmup_init_lr=0.0003
    #     warmup_updates=8000
    # fi
    # if [ "$stop_bert_grad" = 0 ];
    # then
    #     max_tokens=1024
    #     ns=10
    #     lr=0.1
    #     warmup_init_lr='1e-3'
    #     warmup_updates=5000
    # fi
else
    add_load_string=""
    cstring=""
    restore_file=null.pt
fi

enc_opt_freq=1
dec_opt_freq=1

if [ "$separate" = "0" ];
then
    separate_str=""
    train_separate_str=""
    train_script=train.py
else
    echo "separate training"
    criterion="ppo_elbo"
    separate_str="_sep_eof${enc_opt_freq}_dof${dec_opt_freq}"
    train_separate_str="--dec-opt-freq ${dec_opt_freq} --enc-opt-freq ${enc_opt_freq}"
    train_script=train_junxian.py
    warmup_updates=$(( warmup_updates*2 ))
fi

# declare -a rescale_list=("0.01" "0.03" "0.05" "0.07" "0.1")
# declare -a alpha_list=("0.01" "0.05" "0.1" "1" "10")

declare -a rescale_list=("0.3")
declare -a alpha_list=("0.5")


for alpha in "${alpha_list[@]}"
do

for rescale_factor in "${rescale_list[@]}"
do

    SAVE_ROOT=checkpoint/${data_bin}/${DATE}/${data_bin}_alpha${alpha}_kappa${kappa}_lm${lambda_momentum}_bias${linear_bias}_rf${reinforce}_ns${ns}_t${temperature}_gl${grad_lambda}_sb${stop_bert_grad}_fr${freeze_retriever}_ew${entropy_w}_tw${term2_w}_${opt}_copy${copy}_iv${inv_editor}_editdim${edit_embed_dim}_rtr${retriever}_emblayer${embed_transform_nlayer}_fr${forget_rate}_dr${decay_rate}_rf${rescale_factor}_fb${free_bits}_embt${emb_type}_lc${lambda_conifg_str}_gpu${GPUSTR}_c${criterion}${separate_str}${cstring}

    SAVE=${SAVE_ROOT}
    TENSORBOARD=${SAVE}/tensorboard

    rm -r ${SAVE}; mkdir -p ${SAVE} ${TENSORBOARD}

    if [[ -v LOADDIR ]];
    then
        cp ${LOADDIR}/checkpoint_best.pt ${SAVE}/checkpoint_load.pt
    fi

    CUDA_VISIBLE_DEVICES=${GPU} python ${train_script} \
        support_prototype/data-bin/${data_bin} \
        --arch yelp --task support_prototype \
        --optimizer ${opt} ${add_opt_string} --adam-betas '(0.9, 0.98)' \
        --lr ${lr} --lr-scheduler inverse_sqrt --warmup-updates ${warmup_updates} \
        --warmup-init-lr ${warmup_init_lr} \
        --dropout 0.3 --weight-decay ${weight_decay} \
        --edit-embed-dim ${edit_embed_dim} --embed-init-rescale ${rescale_factor} \
        ${train_separate_str} --free-bits ${free_bits} --lambda-t-config ${lambda_config} \
        --retrieve-embed ${template_emb_file} \
        --train-embed pretrained_sent_embeddings/${data_bin}.train.${emb_type}.hdf5 \
        --valid-embed pretrained_sent_embeddings/${data_bin}.valid.${emb_type}.hdf5 \
        --reinforce ${reinforce} --infer-ns ${ns} --reinforce-temperature ${temperature} \
        --freeze-retriever ${freeze_retriever}  --decoder-copy ${copy} \
        --inveditor-embed-path ${glove_path} --encoder-embed-path ${glove_path} --decoder-embed-path ${glove_path} \
        --encoder-embed-dim 300 --decoder-embed-dim 300 \
        --grad-lambda ${grad_lambda} --entropy-weight ${entropy_w} --term2-weight ${term2_w} \
        --user-dir support_prototype \
        --forget-rate ${forget_rate} --decay-rate ${decay_rate} --retrieve-split ${retrieve_split} --alpha ${alpha} --vmf-kappa ${kappa} \
        --linear-bias ${linear_bias} --stop-bert-grad ${stop_bert_grad} --embed-transform-nlayer ${embed_transform_nlayer}\
        --criterion ${criterion} --label-smoothing 0. --num-workers 0 \
        --max-tokens ${max_tokens} --num-class ${num_class} \
        --log-format simple --log-interval ${log_interval} \
        --retriever ${retriever} --inv-editor ${inv_editor} \
        --max-update ${max_update} \
        --validate-interval ${validate_interval} --best-checkpoint-metric ppl --no-epoch-checkpoints \
        --no-last-checkpoints \
        --save-interval-updates ${save_interval_updates} --keep-interval-updates 1 \
        --save-dir ${SAVE} --tensorboard-logdir ${TENSORBOARD} \
        ${add_load_string} --restore-file ${SAVE}/checkpoint_load.pt \
        --eval-mode iw --iw-nsamples ${iw_nsamples} \
        | tee -a ${SAVE}/stdout.log

done
done

