export CUDA_VISIBLE_DEVICES=3
source activate tf_1.10

# for train and predict
python run_cail.py \
--bert_model bert-base-chinese \
--do_test \
--do_lower_case \
--test_file /home/delaiq/data/cail/big_dev_case_1.json \
--max_seq_length 512 \
--doc_stride 128 \
--max_answer_length 128 \
--output_dir /home/delaiq/ouput/cail_base_1 \
--version_2_with_negative \
# --null_score_diff_threshold -11.0


python evaluate.py --data-file /home/delaiq/data/cail/big_dev_case_1.json --pred-file /home/delaiq/ouput/cail_base_1/predictions_test.json


