import collections
import logging
import os
import random

import numpy as np
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)

from CailExample import read_squad_examples, convert_examples_to_features, write_predictions_test_ensemble, write_predictions_test
from CailModel import CailModel
from config_test import config
import json
from answer_verified import *

logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits",
                                    "unk_logits", "yes_logits", "no_logits"])


def load_test_features(args, tokenizer):
    test_examples = read_squad_examples(
        input_file=args.test_file, is_training=False, version_2_with_negative=args.version_2_with_negative)
    # test_examples = test_examples[:100]
    test_features = convert_examples_to_features(
        examples=test_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.predict_batch_size)

    logger.info("***** Test *****")
    logger.info("  Num orig examples = %d", len(test_examples))
    logger.info("  Num split examples = %d", len(test_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    return test_dataloader, test_examples, test_features


def _test(args, output_dir, device, n_gpu, answer_verification=True):
    model = CailModel.from_pretrained(output_dir, answer_verification=answer_verification)
    tokenizer = BertTokenizer.from_pretrained(output_dir, do_lower_case=args.do_lower_case)

    test_dataloader, test_examples, test_features = load_test_features(args, tokenizer)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    logger.info("Start evaluating")
    all_results = []
    for input_ids, input_mask, segment_ids, example_indices in test_dataloader:
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits, \
            batch_unk_logits, batch_yes_logits, batch_no_logits = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            unk_logits = batch_unk_logits[i].detach().cpu().tolist()
            yes_logits = batch_yes_logits[i].detach().cpu().tolist()
            no_logits = batch_no_logits[i].detach().cpu().tolist()
            test_feature = test_features[example_index.item()]
            unique_id = int(test_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits,
                                         unk_logits=unk_logits,
                                         yes_logits=yes_logits,
                                         no_logits=no_logits))

    # all_preds, all_nbest = write_predictions_test_ensemble(test_examples, test_features,
    #                                                        all_results, args.n_best_size, args.max_answer_length,
    #                                                        args.do_lower_case, args.verbose_logging,
    #                                                        args.version_2_with_negative, args.null_score_diff_threshold)
    #
    # return all_preds, all_nbest, test_examples
    return all_results, test_examples, test_features


def find_correct_the_insured(question, passage_all):
    pred_answer = ''
    if question.find('被保险人是谁') >= 0 or (question.find('被保险人是') >= 0 and question.find('被保险人是否') < 0):
        # 还有一种情况，被保险人xxx，但是这种很难匹配因为文章可能出现多次，所以交给模型来预测
        if passage_all.find('被保险人是') >= 0:
            start_index = passage_all.find('被保险人是')
            for ch in passage_all[start_index + 5:]:
                if ch == '，' or ch == '；' or ch == '(' or ch == ',' or ch == ';':
                    break
                else:
                    pred_answer += ch
        elif passage_all.find('被保险人为') >= 0:
            start_index = passage_all.find('被保险人为')
            for ch in passage_all[start_index + 5:]:
                if ch == '，' or ch == '；' or ch == '(' or ch == ',' or ch == ';':
                    break
                else:
                    pred_answer += ch
        if pred_answer != '' and question.find("被保险人是" + pred_answer) > 0:
            pred_answer = 'YES'

    if question.find('投保人是谁') >= 0:
        start_index = passage_all.find('投保人为')
        for ch in passage_all[start_index + 4:]:
            if ch == '，' or ch == '；' or ch == '(' or ch == ',' or ch == ';':
                break
            else:
                pred_answer += ch

    # 如果 pred_answer ==''说明文章中找不到，以模型预测出的结果为准
    return pred_answer


def vote_max_prob(all_preds_dict, all_probs_dict, model_nums):
    result = {}
    for key, preds in all_preds_dict.items():
        probs = all_probs_dict[key]
        preds_dict = {}

        for pred in preds:
            if pred in preds_dict:
                preds_dict[pred] += 1
            else:
                preds_dict[pred] = 1

        order_preds_dict = sorted(preds_dict.items(), key=lambda x: x[0], reverse=True)

        # if order_preds_dict[0][1] == 1:
        #     result[key] = preds[np.argmax(probs)]
        # else:
        #     candidate = [order_preds_dict[0][0]]
        #     for v in order_preds_dict[1:]:
        #         if v[1] == order_preds_dict[0][1]:
        #             candidate.append(v[0])

        #     scores = {}
        #     for cand in candidate:
        #         for i, v in enumerate(preds):
        #             if v == cand:
        #                 if v in scores:
        #                     scores[v].append(probs[i])
        #                 else:
        #                     scores[v] = [probs[i]]
        #     for v in scores:
        #         scores[v] = sum(scores[v])/len(scores[v])
        #     max_v = ('', -1)
        #     for v in scores:
        #         if scores[v] > max_v[1]:
        #             max_v = (v, scores[v])
        #     result[key] = max_v[0]
        flag = False

        for pred, value in preds_dict.items():
            if value > model_nums // 2:
                result[key] = pred
                flag = True
                break

        if not flag:
            result[key] = preds[np.argmax(probs)]

    return result


def ensemble(all_preds_list, all_nbest_list, output_dir, all_examples):
    result = {}

    # {key:[a_1, a_2,...]}
    all_preds_dict = {}
    for key in all_preds_list[0]:
        for preds_list in all_preds_list:
            if key not in all_preds_dict:
                all_preds_dict[key] = [preds_list[key]]
            else:
                all_preds_dict[key].append(preds_list[key])

    all_probs_dict = {}
    for key in all_nbest_list[0]:
        for nbest_list in all_nbest_list:
            if key not in all_probs_dict:
                all_probs_dict[key] = [nbest_list[key][0]['probability']]
            else:
                all_probs_dict[key].append(nbest_list[key][0]['probability'])

    # all_predictions = {}
    # for key in all_preds_dict:
    #     # print(np.argmax(all_probs_dict[key]))
    #     # probs = [
    #     #     all_probs_dict[key][0]*0.2,
    #     #     all_probs_dict[key][1]*0.2,
    #     #     all_probs_dict[key][2] * 0.4,
    #     #     all_probs_dict[key][3] * 0.2,
    #     # ]

    #     all_predictions[key] = all_preds_dict[key][np.argmax(all_probs_dict[key])]
    all_predictions = vote_max_prob(all_preds_dict, all_probs_dict, 8)

    yes_id = []
    the_insured = {}
    null_id = []
    doc_len = {}
    unk_id = []
    long_answer = {}
    time_id = {}
    occur_time = {}
    repair_r = {}
    insurant_person_id = {}
    insurant_company_id = {}
    for example in all_examples:
        if example.question_text.find('是否') >= 0:
            yes_id.append(example.qas_id)

        if example.question_text.find('吗？') >= 0:
            null_id.append(example.qas_id)

        if find_correct_the_insured(example.question_text, "".join(example.doc_tokens)) != '':
            the_insured[example.qas_id] = \
                find_correct_the_insured(example.question_text, "".join(example.doc_tokens))
        doc_len[example.qas_id] = len(example.doc_tokens)

        if example.question_text in [
            '被告人有无存在其他犯罪记录？', '哪个法院受理了此案？',
            '双方有没有达成一致的调解意见？', '被告人最终判刑情况？',
            '被告人是如何归案的？', '本案诉讼费是多少钱？',
            '双方有没有达成一致的协调意见？', '本案事实有无证据证实？',
            '本案所述事故发生原因是什么？', '事故发生原因是什么？',
            '被告为何要变更企业名称？', '原告的工资水平如何？',
            '被告人被判刑情况？', '借款人借的钱用来做什么了？',
        ]:
            unk_id.append(example.qas_id)
        if example.question_text.find("案件发生经过是怎样的") >= 0:
            long_answer[example.qas_id] = find_long_answer(all_predictions[example.qas_id], "".join(example.doc_tokens),
                                                           example.question_text)
            print('long_answer')
            print('r', long_answer[example.qas_id])
            print('pred', all_predictions[example.qas_id])

        if example.question_text.find('有效时间是多久') >= 0:
            time_id[example.qas_id] = find_time_span(example.question_text, all_predictions[example.qas_id])

            print('time_id')
            print('r', time_id[example.qas_id])
            print('pred', all_predictions[example.qas_id])

        if example.question_text.find('事故发生时间是什么时候？') >= 0:
            occur_time[example.qas_id] = repair_time(example.question_text, all_predictions[example.qas_id])
            print('occur_time')
            print('r', occur_time[example.qas_id])
            print('pred', all_predictions[example.qas_id])

        if example.question_text.find('事故结果如何') >= 0:
            repair_r[example.qas_id] = repair_result("".join(example.doc_tokens),
                                                     example.question_text, all_predictions[example.qas_id])

            print('occur_time')
            print('r', repair_r[example.qas_id])
            print('pred', all_predictions[example.qas_id])

        if example.question_text.find('投保的人是谁') >= 0 or example.question_text.find('投保人是谁') >= 0:
            per = get_insurant_person("".join(example.doc_tokens), example.question_text)
            if per:
                insurant_person_id[example.qas_id] = per
                print('ins_per')
                print('r', insurant_person_id[example.qas_id])
                print('pred', all_predictions[example.qas_id])

        if example.question_text.find('向什么公司投保') >= 0:
            cmpa = get_insurant_company("".join(example.doc_tokens))
            if cmpa:
                insurant_company_id[example.qas_id] = cmpa
                print('ins_cmp')
                print('r', insurant_company_id[example.qas_id])
                print('pred', all_predictions[example.qas_id])

    preds = []
    for key, value in all_predictions.items():
        if key in insurant_company_id:
            preds.append({'id': key, 'answer': insurant_company_id[key]})
        elif key in insurant_person_id:
            preds.append({'id': key, 'answer': insurant_person_id[key]})
        elif key in long_answer:
            preds.append({'id': key, 'answer': long_answer[key]})
        elif key in time_id:
            preds.append({'id': key, 'answer': time_id[key]})
        elif key in occur_time:
            preds.append({'id': key, 'answer': occur_time[key]})
        elif key in repair_r:
            preds.append({'id': key, 'answer': repair_r[key]})
        elif key in unk_id:
            preds.append({'id': key, 'answer': ''})
        elif key in yes_id:
            if value in ['YES', 'NO', '']:
                preds.append({'id': key, 'answer': value})
            elif value.find('未') >= 0 or value.find('没有') >= 0 or value.find('不是') >= 0 \
                    or value.find('无责任') >= 0 or value.find('不归还') >= 0 \
                    or value.find('不予认可') >= 0 or value.find('拒不') >= 0 \
                    or value.find('无效') >= 0 or value.find('不是') >= 0 \
                    or value.find('未尽') >= 0 or value.find('未经') >= 0 \
                    or value.find('无异议') >= 0 or value.find('未办理') >= 0 \
                    or value.find('均未') >= 0:
                preds.append({'id': key, 'answer': "NO"})
            else:
                preds.append({'id': key, 'answer': "YES"})
        elif key in the_insured:
            if value != '' and the_insured[key].find(value) >= 0:
                preds.append({'id': key, 'answer': value})
            else:
                preds.append({'id': key, 'answer': the_insured[key]})

        else:
            preds.append({'id': key, 'answer': value})

    with open(output_dir, 'w') as fh:
        json.dump(preds, fh, ensure_ascii=False)


def main():
    args = config()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    output_dir_list = [ './model/', './model_22/', './model_case_6/', './model_case_7/',
                    './model_case_0/','./model_case_1/', './model_case_2/', './model_case_3/']
    # all_preds_list = []
    # all_nbese_list = []
    all_result_list = []
    test_examples = None
    test_features = None
    if args.do_test:
        for output_dir in output_dir_list:
            all_result, test_examples, test_features = _test(args, output_dir, device, n_gpu, True)
            all_result_list.append(all_result)

        """
                    all_results.append(RawResult(
                    unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits,
                                         unk_logits=unk_logits,
                                         yes_logits=yes_logits,
                                         no_logits=no_logits))
        
        
        """
        final_results = []
        for pair in zip(*all_result_list):
            unique_id = None
            cnt = 0
            start_logits = [0]*len(pair[0].start_logits)
            end_logits = [0]*len(pair[0].end_logits)
            unk_logits = [0]*len(pair[0].unk_logits)
            yes_logits = [0]*len(pair[0].yes_logits)
            no_logits = [0]*len(pair[0].no_logits)
            for feat in pair:
                cnt += 1
                unique_id = int(feat.unique_id)
                for i, v in enumerate(feat.start_logits):
                    start_logits[i]+=v
                for i, v in enumerate(feat.end_logits):
                    end_logits[i]+=v
                for i, v in enumerate(feat.unk_logits):
                    unk_logits[i]+=v
                for i, v in enumerate(feat.yes_logits):
                    yes_logits[i]+=v
                for i, v in enumerate(feat.no_logits):
                    no_logits[i]+=v

            start_logits = [v / cnt for v in start_logits]
            end_logits = [v / cnt for v in end_logits]
            unk_logits = [v / cnt for v in unk_logits]
            yes_logits = [v / cnt for v in yes_logits]
            no_logits = [v / cnt for v in no_logits]

            final_results.append(
                RawResult(
                    unique_id=unique_id,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    unk_logits=unk_logits,
                    yes_logits=yes_logits,
                    no_logits=no_logits)
            )

        write_predictions_test(
            test_examples, test_features,
            final_results, args.n_best_size, args.max_answer_length,
            args.do_lower_case, args.output_file,
            args.verbose_logging, args.version_2_with_negative, args.null_score_diff_threshold
        )

        # ensemble(all_preds_list, all_nbese_list, args.output_file, test_examples)


if __name__ == "__main__":
    main()
