export CUDA_VISIBLE_DEVICES=2,3
source activate tf_1.10

# for train and predict
# bert-base-chinese
# /home/delaiq/data/bert/wwm
index=6
nohup python run_cail.py \
--bert_model /home/delaiq/data/bert/wwm \
--do_train \
--do_lower_case \
--version_2_with_negative \
--train_file /home/delaiq/data/cail/big_train_case_$index.json \
--dev_file /home/delaiq/data/cail/big_dev_case_$index.json \
--train_batch_size 12 \
--learning_rate 2e-5 \
--num_train_epochs 5.0 \
--max_seq_length 512 \
--doc_stride 128 \
--max_answer_length 128 \
--output_dir /home/delaiq/ouput/cail_base_$index >log_$index.txt 2>&1 &
