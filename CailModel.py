from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
VERY_NEGATIVE_NUMBER = -1e29


class CailModel(BertPreTrainedModel):
    def __init__(self, config, answer_verification=True, hidden_dropout_prob=0.3):
        super(CailModel, self).__init__(config)
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.qa_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size*4, 2)
        self.apply(self.init_bert_weights)
        self.answer_verification = answer_verification
        if self.answer_verification:
            self.retionale_outputs = nn.Linear(config.hidden_size*4, 1)
            self.unk_ouputs = nn.Linear(config.hidden_size, 1)
            self.doc_att = nn.Linear(config.hidden_size*4, 1)
            self.yes_no_ouputs = nn.Linear(config.hidden_size*4, 2)
            self.ouputs_cls_3 = nn.Linear(config.hidden_size*4, 3)

            self.beta = 100
        else:
            # self.unk_yes_no_outputs_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.unk_yes_no_outputs = nn.Linear(config.hidden_size, 3)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,
                unk_mask=None, yes_mask=None, no_mask=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=True)
        sequence_output = torch.cat((sequence_output[-4], sequence_output[-3], sequence_output[-2],
                                     sequence_output[-1]), -1)

        if self.answer_verification:
            batch_size = sequence_output.size(0)
            seq_length = sequence_output.size(1)
            hidden_size = sequence_output.size(2)
            sequence_output_matrix = sequence_output.view(batch_size*seq_length, hidden_size)
            rationale_logits = self.retionale_outputs(sequence_output_matrix)
            rationale_logits = F.softmax(rationale_logits)
            # [batch, seq_len]
            rationale_logits = rationale_logits.view(batch_size, seq_length)

            # [batch, seq, hidden] [batch, seq_len, 1] = [batch, seq, hidden]
            final_hidden = sequence_output*rationale_logits.unsqueeze(2)
            sequence_output = final_hidden.view(batch_size*seq_length, hidden_size)

            logits = self.qa_outputs(sequence_output).view(batch_size, seq_length, 2)
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            # [000,11111] 1代表了文章
            # [batch, seq_len] [batch, seq_len]
            rationale_logits = rationale_logits * attention_mask.float()
            # [batch, seq_len, 1] [batch, seq_len]
            start_logits = start_logits*rationale_logits
            end_logits = end_logits*rationale_logits

            # unk
            unk_logits = self.unk_ouputs(pooled_output)

            # doc_attn
            attention = self.doc_att(sequence_output)
            attention = attention.view(batch_size, seq_length)
            attention = attention*token_type_ids.float() + (1-token_type_ids.float())*VERY_NEGATIVE_NUMBER

            attention = F.softmax(attention, 1)
            attention = attention.unsqueeze(2)
            attention_pooled_output = attention*final_hidden
            attention_pooled_output = attention_pooled_output.sum(1)

            yes_no_logits = self.yes_no_ouputs(attention_pooled_output)
            yes_logits, no_logits = yes_no_logits.split(1, dim=-1)

            # unk_yes_no_logits = self.ouputs_cls_3(attention_pooled_output)
            # unk_logits, yes_logits, no_logits = unk_yes_no_logits.split(1, dim=-1)

        else:
            # sequence_output = self.qa_dropout(sequence_output)
            logits = self.qa_outputs(sequence_output)
            # self attention
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            # # unk yes_no_logits
            # pooled_output = self.unk_yes_no_outputs_dropout(pooled_output)
            unk_yes_no_logits = self.unk_yes_no_outputs(pooled_output)
            unk_logits, yes_logits, no_logits= unk_yes_no_logits.split(1, dim=-1)
        # # [batch, 1]
        # unk_logits = unk_logits.squeeze(-1)
        # yes_logits = yes_logits.squeeze(-1)
        # no_logits = no_logits.squeeze(-1)

        new_start_logits = torch.cat([start_logits, unk_logits, yes_logits, no_logits], 1)
        new_end_logits = torch.cat([end_logits, unk_logits, yes_logits, no_logits], 1)

        if self.answer_verification and start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = new_start_logits.size(1)
            start_positions.clamp_(1, ignored_index)
            end_positions.clamp_(1, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(new_start_logits, start_positions)
            end_loss = loss_fct(new_end_logits, end_positions)

            rationale_positions = token_type_ids.float()
            alpha = 0.25
            gamma = 2.
            rationale_loss = -alpha * ((1 - rationale_logits) ** gamma) * rationale_positions * torch.log(
                rationale_logits + 1e-8) - (1 - alpha) * (rationale_logits ** gamma) * (
                                     1 - rationale_positions) * torch.log(1 - rationale_logits + 1e-8)
            rationale_loss = (rationale_loss*token_type_ids.float()).sum() / token_type_ids.float().sum()
            total_loss = (start_loss + end_loss) / 2 + rationale_loss*self.beta
            # total_loss = (start_loss + end_loss) / 2
            return total_loss

        elif start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = new_start_logits.size(1)
            start_positions.clamp_(1, ignored_index)
            end_positions.clamp_(1, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(new_start_logits, start_positions)
            end_loss = loss_fct(new_end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits, unk_logits, yes_logits, no_logits


class MultiLinearLayer(nn.Module):
    def __init__(self, layers, hidden_size, output_size, activation=None):
        super(MultiLinearLayer, self).__init__()
        self.net = nn.Sequential()

        for i in range(layers-1):
            self.net.add_module(str(i)+'linear', nn.Linear(hidden_size, hidden_size))
            self.net.add_module(str(i)+'relu', nn.ReLU(inplace=True))

        self.net.add_module('linear', nn.Linear(hidden_size, output_size))

    def forward(self, x):
        return self.net(x)

