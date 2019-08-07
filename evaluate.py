"""Official evaluation script for CJRC.
The code is based partially on CoQA evaluation script.
"""
import argparse
import json
import re
import string
import sys

from collections import Counter, OrderedDict
import logging

OPTS = None

logger = logging.getLogger(__name__)

class CJRCEvaluator():

    def __init__(self, gold_file):
        self.gold_data, self.id_to_domain = CJRCEvaluator.gold_answers_to_dict(gold_file)

    @staticmethod
    def gold_answers_to_dict(gold_file):
        dataset = json.load(open(gold_file))
        gold_dict = {}
        id_to_domain = {}
        for story in dataset['data']:
            qas = story["paragraphs"][0]["qas"]
            for qa in qas:
                qid = qa['id']
                gold_answers = []
                if not qa["answers"]:
                    gold_answers = ['']
                for answer in qa["answers"]:
                    gold_answers.append(answer["text"])
                if qid in gold_dict:
                    sys.stderr.write("Gold file has duplicate stories: {}".format(qid))
                gold_dict[qid] = gold_answers
                id_to_domain[qid] = story["domain"]

        return gold_dict, id_to_domain

    @staticmethod
    def preds_to_dict(pred_file):
        preds = json.load(open(pred_file))
        pred_dict = {}
        for pred in preds:
            pred_dict[pred['id']] = pred['answer']
        return pred_dict

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, storys and extra whitespace."""

        def remove_punc(text):
            return "".join(ch for ch in text if ch.isdigit() or ch.isalpha())

        def lower(text):
            return text.lower()

        return remove_punc(lower(s))

    @staticmethod
    def get_tokens(s):
        if not s: return []
        return list(CJRCEvaluator.normalize_answer(s))

    @staticmethod
    def compute_exact(a_gold, a_pred):
        return int(CJRCEvaluator.normalize_answer(a_gold) == CJRCEvaluator.normalize_answer(a_pred))

    @staticmethod
    def compute_f1(a_gold, a_pred):
        gold_toks = CJRCEvaluator.get_tokens(a_gold)
        pred_toks = CJRCEvaluator.get_tokens(a_pred)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            # if len(gold_toks) == 0:
            #     print("---"*10)
            #     print('gold: ', a_gold)
            #     print('pred: ', a_pred)
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)

        # if f1 < 0.5:
        #     print("---"*10)
        #     print(f1)
        #     print('gold', a_gold)
        #     print('pred', a_pred)
        return f1

    @staticmethod
    def _compute_turn_score(a_gold_list, a_pred):
        f1_sum = 0.0
        em_sum = 0.0
        if len(a_gold_list) > 1:
            for i in range(len(a_gold_list)):
                # exclude the current answer
                gold_answers = a_gold_list[0:i] + a_gold_list[i + 1:]
                em_sum += max(CJRCEvaluator.compute_exact(a, a_pred) for a in gold_answers)
                f1_sum += max(CJRCEvaluator.compute_f1(a, a_pred) for a in gold_answers)
        else:
            em_sum += max(CJRCEvaluator.compute_exact(a, a_pred) for a in a_gold_list)
            f1_sum += max(CJRCEvaluator.compute_f1(a, a_pred) for a in a_gold_list)

        return {'em': em_sum / max(1, len(a_gold_list)), 'f1': f1_sum / max(1, len(a_gold_list))}

    def compute_turn_score(self, qid, a_pred):
        ''' This is the function what you are probably looking for. a_pred is the answer string your model predicted. '''
        a_gold_list = self.gold_data[qid]
        return CJRCEvaluator._compute_turn_score(a_gold_list, a_pred)

    def get_raw_scores(self, pred_data):
        ''''Returns a dict with score'''
        exact_scores = {}
        f1_scores = {}
        civil_yes_all = 0
        civil_yes_right = 0
        civil_no_all = 0
        civil_no_right = 0
        civil_null_all = 0
        civil_null_right = 0
        civil_wrong_cls = 0
        yes_all = 0
        yes_right = 0
        no_all = 0
        no_right = 0
        null_all = 0
        null_right = 0
        wrong_cls = 0
        for qid in self.gold_data:
            domain = self.id_to_domain[qid]
            if qid not in pred_data:
                sys.stderr.write('Missing prediction for {}\n'.format(qid))
                continue
            a_pred = pred_data[qid]
            scores = self.compute_turn_score(qid, a_pred)
            # Take max over all gold answers
            exact_scores[qid] = scores['em']
            f1_scores[qid] = scores['f1']

            if domain == 'civil':
                if self.gold_data[qid][0] == 'YES':
                    civil_yes_all += 1
                    if self.gold_data[qid][0] == a_pred:
                        civil_yes_right += 1
                    # else:
                    #     print(qid)
                    #     print('YES vs '+a_pred)
                if self.gold_data[qid][0] == 'NO':
                    civil_no_all += 1
                    if self.gold_data[qid][0] == a_pred:
                        civil_no_right += 1
                    # else:
                    #     print(qid)
                    #     print('NO vs '+a_pred)
                if self.gold_data[qid][0] == '':
                    civil_null_all += 1
                    if a_pred == '':
                        civil_null_right += 1
                if self.gold_data[qid][0] not in ['YES','NO',''] and a_pred in ['YES','NO','']:
                    civil_wrong_cls += 1


            else:
                if self.gold_data[qid][0] == 'YES':
                    yes_all += 1
                    if self.gold_data[qid][0] == a_pred:
                        yes_right += 1
                    # else:
                    #     print(qid)
                    #     print('YES vs '+a_pred)
                if self.gold_data[qid][0] == 'NO':
                    no_all += 1
                    if self.gold_data[qid][0] == a_pred:
                        no_right += 1
                    # else:
                    #     print(qid)
                    #     print('NO vs '+a_pred)
                if self.gold_data[qid][0] == '':
                    null_all += 1
                    if a_pred == '':
                        null_right += 1
                if self.gold_data[qid][0] not in ['YES', 'NO', ''] and a_pred in ['YES', 'NO', '']:
                    wrong_cls += 1

        print('t')
        logger.info("civil...")
        # logger.info('yes-right:{}, yes-all:{}'.format(civil_yes_right, civil_yes_all))
        logger.info('yes-right:{:0.2f}'.format(civil_yes_right/civil_yes_all))
        # logger.info('no-right:{}, no-all:{}'.format(civil_no_right, civil_no_all))
        logger.info('no-right:{:0.2f}'.format(civil_no_right/civil_no_all))
        # logger.info('null-right:{}, null-all:{}'.format(civil_null_right, civil_null_all))
        logger.info('null-right:{:0.2f}'.format(civil_null_right/civil_null_all))
        logger.info('wrong_cls:{}'.format(civil_wrong_cls))
        logger.info('criminal...')
        # logger.info('yes-right:{}, yes-all:{}'.format(yes_right, yes_all))
        logger.info('yes-right:{:0.2f}'.format(yes_right/yes_all))
        # logger.info('no-right:{}, no-all:{}'.format(no_right, no_all))
        logger.info('no-right:{:0.2f}'.format(no_right/no_all))
        # logger.info('null-right:{}, null-all:{}'.format(null_right, null_all))
        logger.info('null-right:{:0.2f}'.format(null_right/null_all))
        logger.info('wrong_cls:{}'.format(wrong_cls))

        return exact_scores, f1_scores

    def get_raw_scores_human(self):
        ''''Returns a dict with score'''
        exact_scores = {}
        f1_scores = {}
        for qid in self.gold_data:
            f1_sum = 0.0
            em_sum = 0.0
            if len(self.gold_data[qid]) > 1:
                for i in range(len(self.gold_data[qid])):
                    # exclude the current answer
                    gold_answers = self.gold_data[qid][0:i] + self.gold_data[qid][i + 1:]
                    em_sum += max(CJRCEvaluator.compute_exact(a, self.gold_data[qid][i]) for a in gold_answers)
                    f1_sum += max(CJRCEvaluator.compute_f1(a, self.gold_data[qid][i]) for a in gold_answers)
            else:
                exit("Gold answers should be multiple: {}={}".format(qid, self.gold_data[qid]))
            exact_scores[qid] = em_sum / len(self.gold_data[qid])
            f1_scores[qid] = f1_sum / len(self.gold_data[qid])

        return exact_scores, f1_scores

    def human_performance(self):
        exact_scores, f1_scores = self.get_raw_scores_human()
        return self.get_domain_scores(exact_scores, f1_scores)

    def model_performance(self, pred_data):
        exact_scores, f1_scores = self.get_raw_scores(pred_data)
        return self.get_domain_scores(exact_scores, f1_scores)

    def get_domain_scores(self, exact_scores, f1_scores):
        domains = {"civil": Counter(), "criminal": Counter()}
        for qid in self.gold_data:
            domain = self.id_to_domain[qid]
            domains[domain]['em_total'] += exact_scores.get(qid, 0)
            domains[domain]['f1_total'] += f1_scores.get(qid, 0)
            domains[domain]['qa_count'] += 1

        scores = OrderedDict()
        civil_em_total = domains["civil"]["em_total"]
        civil_f1_total = domains["civil"]["f1_total"]
        civil_turn_count = domains["civil"]["qa_count"]

        criminal_em_total = domains["criminal"]["em_total"]
        criminal_f1_total = domains["criminal"]["f1_total"]
        criminal_turn_count = domains["criminal"]["qa_count"]

        em_total = civil_em_total + criminal_em_total
        f1_total = civil_f1_total + criminal_f1_total
        turn_count = civil_turn_count + criminal_turn_count

        scores["civil"] = {'em': round(civil_em_total / max(1, civil_turn_count) * 100, 1),
                           'f1': round(civil_f1_total / max(1, civil_turn_count) * 100, 1),
                           'qas': civil_turn_count}
        scores["criminal"] = {'em': round(criminal_em_total / max(1, criminal_turn_count) * 100, 1),
                              'f1': round(criminal_f1_total / max(1, criminal_turn_count) * 100, 1),
                              'qas': criminal_turn_count}
        scores["overall"] = {'em': round(em_total / max(1, turn_count) * 100, 1),
                             'f1': round(f1_total / max(1, turn_count) * 100, 1),
                             'qas': turn_count}

        return scores


def parse_args():
    parser = argparse.ArgumentParser('Official evaluation script for CJRC based on CoQA.')
    parser.add_argument('--data-file', dest="data_file", help='Input data JSON file.')
    parser.add_argument('--pred-file', dest="pred_file", help='Model predictions.')
    parser.add_argument('--out-file', '-o', metavar='eval.json',
                        help='Write accuracy metrics to file (default is stdout).')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--human', dest="human", action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
  num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
  cur_score = num_no_ans
  best_score = cur_score
  best_thresh = 0.0
  qid_list = sorted(na_probs, key=lambda k: na_probs[k])
  for i, qid in enumerate(qid_list):
    if qid not in scores: continue
    if qid_to_has_ans[qid]:
      diff = scores[qid]
    else:
      if preds[qid]:
        diff = -1
      else:
        diff = 0
    cur_score += diff
    if cur_score > best_score:
      best_score = cur_score
      best_thresh = na_probs[qid]
  return 100.0 * best_score / len(scores), best_thresh


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
  best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
  best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
  main_eval['best_exact'] = best_exact
  main_eval['best_exact_thresh'] = exact_thresh
  main_eval['best_f1'] = best_f1
  main_eval['best_f1_thresh'] = f1_thresh


def main():
    evaluator = CJRCEvaluator(OPTS.data_file)

    if OPTS.human:
        res = evaluator.human_performance()
        print(res)

    if OPTS.pred_file:
        with open(OPTS.pred_file) as f:
            pred_data = CJRCEvaluator.preds_to_dict(OPTS.pred_file)
            res = evaluator.model_performance(pred_data)
        print(res)

    if OPTS.out_file:
        with open(OPTS.out_file, 'w') as f:
            json.dump(res, f)

        return res

    # if OPTS.na_prob_file:
    #     find_all_best_thresh(out_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans)


if __name__ == '__main__':
    OPTS = parse_args()
    main()
