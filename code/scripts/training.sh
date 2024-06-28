cd ../
CUDA_VISIBLE_DEVICES=$1  python run.py --dataset=$2 --pretrained_bert_name $3
