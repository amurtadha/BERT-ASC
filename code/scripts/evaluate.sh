cd ../
CUDA_VISIBLE_DEVICES=$1  python evluate.py --dataset=$2 --pretrained_bert_name $3
