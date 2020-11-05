#! /bin/bash
#
# train.sh
# Copyright (C) 2020-02-09 Junxian <He>
#
# Distributed under terms of the MIT license.
#

DATE=`date +%Y%m%d`

# general configuration
data_bin="coco40k"
emb_type="bert"
kappa=30
copy=0  # whether to have copy mechanism in the decoder
inv_editor='levenshtein'
edit_embed_dim=10
retrieve_split=valid1
criterion="sp_elbo"


# optimization hyperparameters
warmup_init_lr='1e-03'
lr=0.001
max_update=30000
retriever=precompute_emb
linear_bias=0
weight_decay=0.0001
stop_bert_grad=1
freeze_retriever=0
reinforce=1
opt=adam
update_freq=1

# hyperparameters for lambda update
forget_rate=0.8
decay_rate=1

# some important hyperparameters that we may need to change
rescale_factor=0.3 # rescaling factor for reinforce sampling
free_bits=0       # KL free bits for mitigating posterior collapse
alpha=0.1        # Dirichlet hyperparameters
lambda_config="0:0,1500:1"  # KL weights annealing schedule
GPU=0


# evaluation parameters, only used during evaluationa after training
eval_mode="none"  # perform training by default
prune_num="-1"
valid_subset="valid" # use "valid" to test on valid set
iw_nsamples=100

while getopts ":g:a:p:k:r:f:c:u:e:d:" arg; do
  case $arg in
    g) GPU="$OPTARG"
    ;;
    a) alpha="$OPTARG"
    ;;
    p) LOADDIR="$OPTARG"
    ;;
    k) kappa="$OPTARG"
    ;;
    r) rescale_factor="$OPTARG"
    ;;
    f) free_bits="$OPTARG"
    ;;
    c) criterion="$OPTARG"
    ;;
    u) prune_num="$OPTARG"
    ;;
    e) eval_mode="$OPTARG"
    ;;
    d) data_bin="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

glove_path=glove_embeddings/${data_bin}_glove.txt
emb_dataset_file=precompute_embedding_datasets/${data_bin}/${data_bin}.${emb_type}


if [ "$eval_mode" = "none" ];
then
    stdout="stdout.log"
    eval_params="--eval-mode none"
else
    stdout="eval_${valid_subset}_${eval_mode}_prune${prune_num}.log"
    eval_params="--eval-mode ${eval_mode} --iw-nsamples ${iw_nsamples} \
    --valid-subset ${valid_subset} --prune-num ${prune_num} \
    --reset-meters --reset-optimizer --write-loss-path loss_per_sent_${valid_subset}.txt"
fi

if [ "$criterion" = "topk_elbo" ];
then
    reinforce=0
fi

if [ "$data_bin" = "coco40k" ];
then
    max_tokens=2048
    save_interval_updates=0
    warmup_updates=1000
    max_update=15000
    log_interval=20
    validate_interval=1
    ns=10
elif [ "$data_bin" = "yelp" ];
then
    max_tokens=2048
    save_interval_updates=5000
    warmup_updates=20000
    max_update=300000
    lambda_config="0:0,20000:1"
    log_interval=100
    validate_interval=1000
    ns=10
elif [ "$data_bin" = "yelp_large" ];
then
    max_tokens=2048 # distributed on two gpus
    save_interval_updates=5000
    warmup_updates=150000
    max_update=500000
    kappa=40
    lambda_config="0:0,150000:1"
    log_interval=100
    validate_interval=1000
    ns=10
else
    max_tokens=0
    save_interval_updates=0
    warmup_updates=0
    ns=0
fi


if [ "$criterion" = "lm_baseline" ];
then
    ns=1
fi

GPUSTR=$(printf "$GPU" | tr , _)

lambda_conifg_str=$(printf "$lambda_config" | tr , _)
lambda_conifg_str=$(printf "$lambda_conifg_str" | tr : _)

if [[ -v LOADDIR && eval_mode = "none" ]];
then
    add_load_string=""
    cstring="_continue"
    restore_file=checkpoint_load.pt
else
    add_load_string=""
    cstring=""
    restore_file=null.pt
fi


if [[ -v LOADDIR && eval_mode != "none" ]];
then
    SAVE=${LOADDIR}
    TENSORBOARD=${SAVE}/tensorboard
else
    SAVE_ROOT="checkpoint/${data_bin}/${DATE}/${data_bin}_${opt}_noeditvec_alpha${alpha}_kappa${kappa}"
    SAVE_ROOT="${SAVE_ROOT}_ns${ns}"
    SAVE_ROOT="${SAVE_ROOT}_editdim${edit_embed_dim}"
    SAVE_ROOT="${SAVE_ROOT}_rtr${retriever}_fr${forget_rate}_dr${decay_rate}_rf${rescale_factor}_fb${free_bits}"
    SAVE_ROOT="${SAVE_ROOT}_embt${emb_type}_lc${lambda_conifg_str}_uf${update_freq}_gpu${GPUSTR}_c${criterion}${cstring}"

    SAVE=${SAVE_ROOT}
    TENSORBOARD=${SAVE}/tensorboard

    rm -r ${SAVE}; mkdir -p ${SAVE} ${TENSORBOARD}
fi

if [[ -v LOADDIR ]];
then
    cp ${LOADDIR}/checkpoint_best.pt ${SAVE}/checkpoint_load.pt
fi

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    data-bin/${data_bin} \
    --arch ${data_bin} --task sparse_prototype \
    --optimizer ${opt} --adam-betas '(0.9, 0.98)' \
    --lr ${lr} --lr-scheduler inverse_sqrt --warmup-updates ${warmup_updates} \
    --warmup-init-lr ${warmup_init_lr} \
    --weight-decay ${weight_decay} \
    --edit-embed-dim ${edit_embed_dim} --embed-init-rescale ${rescale_factor} \
    --free-bits ${free_bits} --lambda-t-config ${lambda_config} \
    --emb-dataset-file ${emb_dataset_file} \
    --reinforce ${reinforce} --infer-ns ${ns} \
    --freeze-retriever ${freeze_retriever}  --decoder-copy ${copy} \
    --inveditor-embed-path ${glove_path} --encoder-embed-path ${glove_path} --decoder-embed-path ${glove_path} \
    --encoder-embed-dim 300 --decoder-embed-dim 300 \
    --user-dir sparse_prototype \
    --forget-rate ${forget_rate} --decay-rate ${decay_rate} --retrieve-split ${retrieve_split} --alpha ${alpha} --vmf-kappa ${kappa} \
    --linear-bias ${linear_bias} --stop-bert-grad ${stop_bert_grad} \
    --criterion ${criterion} --label-smoothing 0. --num-workers 0 \
    --max-tokens ${max_tokens} \
    --log-format simple --log-interval ${log_interval} \
    --retriever ${retriever} --inv-editor ${inv_editor} \
    --max-update ${max_update} --update-freq ${update_freq} \
    --validate-interval ${validate_interval} --best-checkpoint-metric ppl --no-epoch-checkpoints \
    --no-last-checkpoints \
    --save-interval-updates ${save_interval_updates} --keep-interval-updates 1 \
    --save-dir ${SAVE} --tensorboard-logdir ${TENSORBOARD} \
    ${add_load_string} --restore-file ${SAVE}/checkpoint_load.pt \
    ${eval_params} \
    | tee -a ${SAVE}/${stdout}
