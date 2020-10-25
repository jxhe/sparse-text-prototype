#! /bin/bash
#
# train.sh
# Copyright (C) 2020-02-09 Junxian <He>
#
# Distributed under terms of the MIT license.
#

DATE=`date +%Y%m%d`

kappa=30
linear_bias=0
stop_bert_grad=1
freeze_retriever=0
reinforce=1
temperature=1
grad_lambda=1
data_bin="ptb"
lambda_momentum=0.9
copy=0
inv_editor='levenshtein'

edit_embed_dim=10
latent_embed_dim=100

entropy_w=1
term2_w=1

glove_path=glove_embeddings/${data_bin}_glove.txt
opt=adam
enc_opt_freq=1
dec_opt_freq=30

if [ "$data_bin" = "ptb" ];
then
    num_class=41088
    max_tokens=2048
    save_interval_updates=0
    warmup_updates=10000
    ns=5
elif [ "$data_bin" = "ptb10" ];
then
    num_class=5703
    max_tokens=512
    save_interval_updates=0
    warmup_updates=800
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

while getopts ":g:a:p:k:e:l:" arg; do
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
        lr=0.001
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

SAVE_ROOT=checkpoint/separate_opt/${data_bin}/${DATE}/${data_bin}_alpha${alpha}_kappa${kappa}_lm${lambda_momentum}_bias${linear_bias}_rf${reinforce}_ns${ns}_t${temperature}_gl${grad_lambda}_sb${stop_bert_grad}_fr${freeze_retriever}_ew${entropy_w}_tw${term2_w}_${opt}_dof${dec_opt_freq}_eof${enc_opt_freq}_wu${warmup_updates}_copy${copy}_iv${inv_editor}_editdim${edit_embed_dim}_ld${latent_embed_dim}_gpu${GPU}${cstring}

SAVE=${SAVE_ROOT}
TENSORBOARD=${SAVE}/tensorboard

rm -r ${SAVE}; mkdir -p ${SAVE} ${TENSORBOARD}

if [[ -v LOADDIR ]];
then
    cp ${LOADDIR}/checkpoint_best.pt ${SAVE}/checkpoint_load.pt
fi

CUDA_VISIBLE_DEVICES=${GPU} python train_junxian.py \
    support_prototype/data-bin/${data_bin} \
    --arch ${data_bin} --task support_prototype \
    --optimizer ${opt} ${add_opt_string} --adam-betas '(0.9, 0.98)'\
    --lr ${lr} --lr-scheduler inverse_sqrt --warmup-updates ${warmup_updates} \
    --warmup-init-lr ${warmup_init_lr} \
    --dropout 0.3 --weight-decay 0.0001 \
    --decoder-dropout-in 0.5 --decoder-dropout-out 0.5 --decoder-layers 1 \
    --edit-embed-dim ${edit_embed_dim} --latent-dim ${latent_embed_dim} \
    --encoder-layers 1 --retrieve-embed pretrained_sent_embeddings/${data_bin}.template.hdf5 \
    --reinforce ${reinforce} --infer-ns ${ns} --reinforce-temperature ${temperature} \
    --dec-opt-freq ${dec_opt_freq} --enc-opt-freq ${enc_opt_freq} \
    --freeze-retriever ${freeze_retriever} --decoder-copy ${copy} \
    --inveditor-embed-path ${glove_path} --encoder-embed-path ${glove_path} --decoder-embed-path ${glove_path} \
    --encoder-embed-dim 300 --decoder-embed-dim 300 \
    --grad-lambda ${grad_lambda} --entropy-weight ${entropy_w} --term2-weight ${term2_w} \
    --user-dir support_prototype \
    --lambda-momentum ${lambda_momentum} --retrieve-split valid1 --alpha ${alpha} --vmf-kappa ${kappa} \
    --linear-bias ${linear_bias} --stop-bert-grad ${stop_bert_grad} \
    --criterion sp_elbo --label-smoothing 0. --num-workers 0 \
    --max-tokens ${max_tokens} --num-class ${num_class} \
    --log-format simple --log-interval 5 \
    --retriever sentence-bert --inv-editor ${inv_editor} \
    --validate-interval 1 --best-checkpoint-metric ppl --no-epoch-checkpoints \
    --no-last-checkpoints \
    --save-interval-updates ${save_interval_updates} --keep-interval-updates 1 \
    --save-dir ${SAVE} --tensorboard-logdir ${TENSORBOARD} \
    ${add_load_string} --restore-file ${SAVE}/checkpoint_load.pt \
    | tee -a ${SAVE}/stdout.log
    # --restore-file checkpoint_best.pt \

