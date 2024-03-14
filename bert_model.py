import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np
import random
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


class BertForMultiLabelClassification(nn.Module):
    def __init__(self, config, num_labels):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_model_name)
        self.dropout = nn.Dropout(config.bert_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, batch):
        outputs = self.bert(batch.text_ids, attention_mask=batch.attention_mask_text)
        # Based on classification token
        out = self.dropout(outputs[1])  # uses [CLS]
        out = self.classifier(out)
        return out


# Models used for contrastive learning experiment below...
class MultiLabelMLPLayer(nn.Module):
    """
    Head for getting multilabel predictions over model's CLS representation.
    """

    def __init__(self, bert, num_labels):
        super().__init__()
        self.classifier = nn.Linear(bert.config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        out = self.classifier(features)
        return out


# All code below is either adapted or directly from Gao et al's SimCSE implementation (https://github.com/princeton-nlp/SimCSE/tree/main)
class MLPLayer(nn.Module):
    """
    Head for getting the representations over model's CLS representation.
    """

    def __init__(self, bert):
        super().__init__()
        self.dense = nn.Linear(bert.config.hidden_size, bert.config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    """
    Cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class MultiLabelBertForCL(nn.Module):
    """
    MultiLabel classification model w/ contrastive loss
    """

    def __init__(self, config, num_labels):
        super(MultiLabelBertForCL, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.bert_model_name, add_pooling_layer=False)
        # self.dropout_self = nn.Dropout(config.bert_dropout)
        # self.dropout_noise = nn.Dropout(config.bert_dropout)
        # self.dropout_neg = nn.Dropout(config.bert_dropout)
        self.cl_representations = MLPLayer(self.bert)
        self.sim = Similarity(temp=0.05)  # todo: change to not be hard-coded in future...
        self.dropout = nn.Dropout(config.bert_dropout)
        self.ml_classifier = MultiLabelMLPLayer(self.bert, num_labels)

    def forward(self, batch, cl_emb=False):
        if cl_emb:
            outputs_orig = self.bert(batch.text_ids, attention_mask=batch.attention_mask_text) # batch size, input tokens, hidden state dimensions
            pooler_output_orig = outputs_orig.last_hidden_state[:, 0, :] # cls token for each batch
            #pooler_output_orig = self.dropout_self(pooler_output_orig)
            z1 = self.cl_representations(pooler_output_orig)  # batch size, hidden state dimensions

            outputs_pos = self.bert(batch.text_ids_pos, attention_mask=batch.attention_mask_text_pos)
            pooler_output_pos = outputs_pos.last_hidden_state[:, 0, :]  # batch size, input tokens, hidden state dimensions
            #pooler_output_pos = self.dropout_noise(pooler_output_pos)
            z2 = self.cl_representations(pooler_output_pos)  # batch size, hidden state dimensions

            outputs_neg = self.bert(batch.text_ids_neg, attention_mask=batch.attention_mask_text_neg)
            pooler_output_neg = outputs_neg.last_hidden_state[:, 0, :]  # batch size, input tokens, hidden state dimensions
            #pooler_output_neg = self.dropout_neg(pooler_output_neg)
            z3 = self.cl_representations(pooler_output_neg)  # batch size, hidden state dimensions

            return z1, z2, z3

        else:
            outputs = self.bert(batch.text_ids, attention_mask=batch.attention_mask_text)
            out = self.dropout(outputs.last_hidden_state[:, 0, :])  # uses [CLS] token
            out = self.ml_classifier(out)
            return out


def self_contrastive_loss(cls, z1, z2, z3):
    """
    Unsupervised contrastive loss from SimCSE; self-comparison using dropout as noise.
    Z1 is the core instance; Z2 is the positive anchor; Z3 is the negative anchor.
    """
    # Calculate similarity between instance and positive anchor
    cos_sim = cls.sim(z1, z2)

    # Hard negative
    z1_z3_cos = cls.sim(z1, z3)
    cos_sim = torch.cat([cos_sim, z1_z3_cos], 0)

    # Calculate loss with hard negatives
    z3_weight = 1.0  # Note that weights are actually logits of weights
    weights = torch.tensor([[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]).to('cuda')
    cos_sim = (cos_sim + weights) *  cls.config.batch_size

    labels = torch.arange(cos_sim.size(0)).long().to('cuda')
    loss_fct = nn.CrossEntropyLoss()

    loss = loss_fct(cos_sim, labels) / cls.config.batch_size  # divide by batch size since we have 1:1 ratio of pos/neg anchors; SimCSE uses 1:mini-batch size
    
    return loss


def strict_contrastive_loss(cls, z1, z2, z3):
    """
    Strict contrastive loss (SCL) for multi-label classification; from "An Effective Deployment of Contrastive Learning in Multi-label Text Classification" by Lin et al., 2023
    Z1 is the core instance; Z2 is the positive anchor; Z3 is the negative anchor.
    """
    # Calculate similarity between instance and positive anchor
    pos_cos_sim = cls.sim(z1, z2)

    # Calculate similarity between instance and negative anchor
    neg_cos_sim = cls.sim(z1, z3)

    # Calculate loss
    numerator = torch.exp(pos_cos_sim)
    denominator = torch.sum(torch.exp(neg_cos_sim))
    loss = torch.sum(-torch.log(numerator / denominator)) / cls.config.batch_size
    
    return loss


def jaccard_similarity_contrastive_loss(cls, z1, z2, z3, batch):
    """
    Jaccard similarity contrastive loss (JSCL) for multi-label classification; from "An Effective Deployment of Contrastive Learning in Multi-label Text Classification" by Lin et al., 2023
    Z1 is the core instance; Z2 is the positive anchor; Z3 is the negative anchor. Similarity here may be slightly different than the original implementation, although their code was not made avaiilable.
    """
    def similarity(list1, list2):
        pseudo_intersection = 0
        pseudo_union = 0
        for i, item in enumerate(list1):
            if list1[i] == list2[i]:
                pseudo_intersection += 1
            ## If we want to consider only true positives... not exactly js right now; correlated, but not as extreme
            # if list1[i] == list2[i] and list1[i] == 1:
            #     pseudo_intersection += 1
            # if list1[i] == 1 or list2[i] == 1:
            #     pseudo_union += 1
        # return pseudo_intersection / pseudo_union
        return pseudo_intersection / len(list1)

    def calculate_label_similarity(inst_labels, anchor_labels):
        tns_inst_other = torch.empty(inst_labels.shape[0], 1)
        for i in range(inst_labels.shape[0]):
            tns_tmp_a = inst_labels[i]
            tns_tmp_b = anchor_labels[i]
            tns_inst_other[i] = (similarity(tns_tmp_a, tns_tmp_b))
        return tns_inst_other

    # Calculate similarity between instance and positive anchor
    pos_cos_sim = cls.sim(z1, z2)

    # Hard negative
    neg_cos_sim = cls.sim(z1, z3)

    # Label similarity
    #label_jaccard_similarity = calculate_label_similarity(np.array(batch.labels.cpu()), np.array(batch.labels_pos.cpu()))
    label_jaccard_similarity = calculate_label_similarity(batch.labels, batch.labels_pos)

    # Calculate loss
    numerator = torch.exp(pos_cos_sim.to('cuda')) * torch.flatten(label_jaccard_similarity.to('cuda'))
    denominator = torch.sum(torch.exp(neg_cos_sim.to('cuda')))  
    loss = torch.sum(-torch.log(numerator / denominator)) / cls.config.batch_size
    loss = loss.to('cuda')
    
    return loss
