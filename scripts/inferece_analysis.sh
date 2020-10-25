
gpu=$1
exp_dir=$2

python support_prototype/scripts/template_to_analysis_file.py --exp-dir ${exp_dir}
CUDA_VISIBLE_DEVICES=${gpu} python support_prototype/scripts/pos_eval.py --prefix inference --exp-dir ${exp_dir}
