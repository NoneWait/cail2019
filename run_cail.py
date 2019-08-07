import os
import random
import logging
import numpy as np
import collections
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import pickle

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert.tokenization import BertTokenizer

from config import config
from CailExample import read_squad_examples, convert_examples_to_features, write_predictions, write_predictions_test
from CailModel import CailModel
from evaluate import CJRCEvaluator

logger = logging.getLogger(__name__)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits",
                                    "unk_logits", "yes_logits", "no_logits"])


def save_model(args, model, tokenizer):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("save model")
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)


def save_code(path):
    import shutil
    if not os.path.exists(path):
        os.mkdir(path)
    code_path = os.path.join(path+'/code')
    if not os.path.exists(code_path):
        os.mkdir(code_path)
    f_list = os.listdir('./')
    for fileName in f_list:
        if os.path.splitext(fileName)[1] == '.py' or os.path.splitext(fileName)[1] == '.sh':
            shutil.copy(fileName, code_path)


def _test(args, device, n_gpu):
    model = CailModel.from_pretrained(args.output_dir)
    tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

    test_dataloader, test_examples, test_features = load_test_features(args, tokenizer)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    logger.info("Start evaluating")
    all_results = []
    for input_ids, input_mask, segment_ids, example_indices in test_dataloader:
        if len(all_results) % 5000 == 0:
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
    output_prediction_file = os.path.join(args.output_dir, "predictions_test.json")
    write_predictions_test(test_examples, test_features, all_results, args.n_best_size, args.max_answer_length,
                                        args.do_lower_case, output_prediction_file, args.verbose_logging,
                           args.version_2_with_negative, args.null_score_diff_threshold)


def _dev(args, device, model, eval_dataloader, eval_examples, eval_features):
    model.eval()
    logger.info("Start evaluating")
    model.eval()
    all_results = []
    for input_ids, input_mask, segment_ids, example_indices in eval_dataloader:
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits, \
            batch_unk_logits, batch_yes_logits, batch_no_logits= model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            unk_logits = batch_unk_logits[i].detach().cpu().tolist()
            yes_logits = batch_yes_logits[i].detach().cpu().tolist()
            no_logits = batch_no_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits,
                                         unk_logits=unk_logits,
                                         yes_logits=yes_logits,
                                         no_logits=no_logits))
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
    all_predictions = write_predictions(eval_examples, eval_features, all_results,
                      args.n_best_size, args.max_answer_length,
                      args.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                      args.version_2_with_negative, args.null_score_diff_threshold)
    evaluator = CJRCEvaluator(args.dev_file)
    res = evaluator.model_performance(all_predictions)
    result = {'f1': res['overall']['f1']}
    return result


def _train(args, device, n_gpu):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    train_dataloader = load_train_features(args, tokenizer)
    eval_dataloader, eval_examples, eval_features = load_dev_features(args, tokenizer)
    num_train_optimization_steps = int(
        len(train_dataloader) / args.gradient_accumulation_steps * args.num_train_epochs)

    # logger.info('num_train_optimization_steps:', num_train_optimization_steps)
    # logger.info('train_dataloader:', len(train_dataloader))
    # Prepare model

    model = CailModel.from_pretrained(args.bert_model,
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))

    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    global_step = 0

    model.train()
    f1 = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, start_positions, end_positions, \
            unk_mask, yes_mask, no_mask = batch
            loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions,
                         unk_mask, yes_mask, no_mask)
            if n_gpu > 1:
                loss = loss.mean()
                # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            if step % 100 == 0:
                logger.info('step is {} and the loss is {:.2f}...'.format(step, loss))

            if step % 1000 == 0 and step != 0 and epoch != 0:
                # if step % 1000 == 0:
                # ignore the first epoch
                metrics = _dev(args, device, model, eval_dataloader, eval_examples, eval_features)
                if metrics['f1'] > f1:
                    f1 = metrics['f1']
                    save_model(args, model, tokenizer)
                logger.info("epoch is {} ,step is {}, f1 is {:.4f}, current_best is {:.4f}...".
                            format(epoch, step, metrics['f1'], f1))
                model.train()
        metrics = _dev(args, device, model, eval_dataloader, eval_examples, eval_features)
        if metrics['f1'] > f1:
            f1 = metrics['f1']
            save_model(args, model, tokenizer)
        logger.info("epoch is {} , f1 is {:.4f}, current_best is {:.4f}...".
                    format(epoch, metrics['f1'], f1))
        model.train()


def load_dev_features(args, tokenizer):
    cached_dev_features_file = os.path.join(args.output_dir, 'dev_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),
        str(args.max_query_length)))
    eval_examples = read_squad_examples(
        input_file=args.dev_file, is_training=False, version_2_with_negative=args.version_2_with_negative)
    # eval_examples = eval_examples[:20]

    eval_features = None
    try:
        with open(cached_dev_features_file, "rb") as reader:
            eval_features = pickle.load(reader)
    except:
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving dev features into cached file %s", cached_dev_features_file)
            with open(cached_dev_features_file, "wb") as writer:
                pickle.dump(eval_features, writer)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    logger.info("***** Eval *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    return eval_dataloader, eval_examples, eval_features


def load_train_features(args, tokenizer):
    cached_train_features_file = os.path.join(args.output_dir, 'train_{0}_{1}_{2}_{3}'.format(
        list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride),
        str(args.max_query_length)))

    train_examples = read_squad_examples(
        input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative)

    # train_examples = train_examples[:20]
    train_features = None
    try:
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader)
    except:
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True)
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving train features into cached file %s", cached_train_features_file)
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
    logger.info("***** Train *****")
    logger.info("  Num orig examples = %d", len(train_examples))
    logger.info("  Num split examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
    all_unk_mask = torch.tensor([f.unk_mask for f in train_features], dtype=torch.long)
    all_yes_mask = torch.tensor([f.yes_mask for f in train_features], dtype=torch.long)
    all_no_mask = torch.tensor([f.no_mask for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_start_positions, all_end_positions,
                               all_unk_mask, all_yes_mask, all_no_mask)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    return train_dataloader


def load_test_features(args, tokenizer):
    test_examples = read_squad_examples(
        input_file=args.test_file, is_training=False, version_2_with_negative=args.version_2_with_negative)

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

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.do_train:
        save_code(args.output_dir)
        _train(args, device, n_gpu)

    if args.do_test:
        _test(args, device, n_gpu)


if __name__ == "__main__":
    main()
