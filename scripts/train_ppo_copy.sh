#! /bin/bash
#
# train.sh
# Copyright (C) 2020-02-09 Junxian <He>
#
# Distributed under terms of the MIT license.
#

DATE=`date +%Y%m%d`

kappa=30
max_epoch=100
retriever=pretrained_embed
linear_bias=0
stop_bert_grad=1
freeze_retriever=0
reinforce=1
temperature=1
grad_lambda=0
data_bin="coco40k"
forget_rate=0.8
decay_rate=1
copy=0
inv_editor='levenshtein'
edit_embed_dim=10
entropy_w=1
term2_w=1
rescale_factor=1.

load_name=''

retrieve_split=train

glove_path=glove_embeddings/${data_bin}_glove.txt
opt=adam
template_emb_file=pretrained_sent_embeddings/${data_bin}.template.hdf5

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

while getopts ":g:a:p:k:e:l:t:s:r:" arg; do
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
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

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
fi

# declare -a rescale_list=("0.01" "0.03" "0.05" "0.07" "0.1")
# declare -a alpha_list=("0.01" "0.05" "0.1" "1" "10")

declare -a rescale_list=("0.01")
declare -a alpha_list=("0.01" "0.1" "1")


for alpha in "${alpha_list[@]}"
do

for rescale_factor in "${rescale_list[@]}"
do

    SAVE_ROOT=checkpoint/${data_bin}/${DATE}/${data_bin}_alpha${alpha}_kappa${kappa}_lm${lambda_momentum}_bias${linear_bias}_rf${reinforce}_ns${ns}_t${temperature}_gl${grad_lambda}_sb${stop_bert_grad}_fr${freeze_retriever}_ew${entropy_w}_tw${term2_w}_${opt}_copy${copy}_iv${inv_editor}_editdim${edit_embed_dim}_rtr${retriever}_fr${forget_rate}_dr${decay_rate}_rf${rescale_factor}_ppo_gpu${GPU}${separate_str}${cstring}

    SAVE=${SAVE_ROOT}
    TENSORBOARD=${SAVE}/tensorboard

    rm -r ${SAVE}; mkdir -p ${SAVE} ${TENSORBOARD}

    if [[ -v LOADDIR ]];
    then
        cp ${LOADDIR}/checkpoint_best.pt ${SAVE}/checkpoint_load.pt
    fi

    CUDA_VISIBLE_DEVICES=${GPU} python ${train_script} \
        support_prototype/data-bin/${data_bin} \
        --arch ${data_bin} --task support_prototype \
        --optimizer ${opt} ${add_opt_string} --adam-betas '(0.9, 0.98)' \
        --lr ${lr} --lr-scheduler inverse_sqrt --warmup-updates ${warmup_updates} \
        --warmup-init-lr ${warmup_init_lr} \
        --dropout 0.3 --weight-decay 0.0001 \
        --decoder-dropout-in 0.5 --decoder-dropout-out 0.5 --decoder-layers 1 \
        --edit-embed-dim ${edit_embed_dim} --embed-init-rescale ${rescale_factor} \
        ${train_separate_str} \
        --encoder-layers 1 --retrieve-embed ${template_emb_file} \
        --train-embed pretrained_sent_embeddings/${data_bin}.train.hdf5 \
        --valid-embed pretrained_sent_embeddings/${data_bin}.valid.hdf5 \
        --reinforce ${reinforce} --infer-ns ${ns} --reinforce-temperature ${temperature} \
        --freeze-retriever ${freeze_retriever}  --decoder-copy ${copy} \
        --inveditor-embed-path ${glove_path} --encoder-embed-path ${glove_path} --decoder-embed-path ${glove_path} \
        --encoder-embed-dim 300 --decoder-embed-dim 300 \
        --grad-lambda ${grad_lambda} --entropy-weight ${entropy_w} --term2-weight ${term2_w} \
        --user-dir support_prototype \
        --forget-rate ${forget_rate} --decay-rate ${decay_rate} --retrieve-split ${retrieve_split} --alpha ${alpha} --vmf-kappa ${kappa} \
        --linear-bias ${linear_bias} --stop-bert-grad ${stop_bert_grad} \
        --criterion ppo_elbo --label-smoothing 0. --num-workers 0 \
        --max-tokens ${max_tokens} --num-class ${num_class} \
        --log-format simple --log-interval 5 \
        --retriever ${retriever} --inv-editor ${inv_editor} \
        --max-epoch ${max_epoch} \
        --validate-interval 1 --best-checkpoint-metric ppl --no-epoch-checkpoints \
        --no-last-checkpoints \
        --save-interval-updates ${save_interval_updates} --keep-interval-updates 1 \
        --save-dir ${SAVE} --tensorboard-logdir ${TENSORBOARD} \
        ${add_load_string} --restore-file ${SAVE}/checkpoint_load.pt \
        | tee -a ${SAVE}/stdout.log
        # --restore-file checkpoint_best.pt \
done
done

