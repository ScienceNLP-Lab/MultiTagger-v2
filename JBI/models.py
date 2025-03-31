import torch
import torch.nn as nn
import math
from transformers import AutoModel
from adapters import AutoAdapterModel


class BertForMultiLabelClassification(nn.Module):
    def __init__(self, config, num_labels):
        super(BertForMultiLabelClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_model_name)
        self.dropout = nn.Dropout(config.bert_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, batch):
        outputs = self.bert(batch.text_ids, attention_mask=batch.attention_mask_text)
        out = self.dropout(outputs[1])
        out = self.classifier(out)
        return out


class Specter2ForMultiLabelClassification(nn.Module):
    def __init__(self, config, num_labels):
        super(Specter2ForMultiLabelClassification, self).__init__()
        if config.bert_model_name == "allenai/specter2_base" and config.bert_adapter:
            self.bert = AutoAdapterModel.from_pretrained(config.bert_model_name)
            self.bert.load_adapter("allenai/specter2_classification", source="hf", set_active=True)
        else:
            self.bert = AutoModel.from_pretrained(config.bert_model_name, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.bert_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, batch):
        outputs = self.bert(batch.text_ids, attention_mask=batch.attention_mask_text)
        out = self.dropout(outputs.last_hidden_state[:, 0, :])
        out = self.classifier(out)
        return out


# HeroCon Model - adapted from HeroCon code - https://github.com/Leo02016/HeroCon/blob/main/module.py
class Similarity(nn.Module):
    """Cosine similarity"""
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class MLPLayer(nn.Module):
    """Head for getting the representations over model's CLS representation."""
    def __init__(self, bert):
        super().__init__()
        self.dense = nn.Linear(bert.config.hidden_size, bert.config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class BertForMultiLabelClassificationHeroCon(nn.Module):
    def __init__(self, config, num_labels, temp=0.05, clip=0.95):
        super(BertForMultiLabelClassificationHeroCon, self).__init__()
        self.clip = clip
        self.alpha = config.cl_alpha
        self.beta = config.cl_beta
        self.labels_num = num_labels
        self.device = config.device
        self.temp = temp
        self.w1 = config.adnce_w1
        self.w2 = config.adnce_w2
        self.sim = Similarity(temp=self.temp)
        if config.bert_model_name == "allenai/specter2_base" and config.bert_adapter:
            self.bert = AutoAdapterModel.from_pretrained(config.bert_model_name)
            self.bert.load_adapter("allenai/specter2_classification", source="hf", set_active=True)
        else:
            self.bert = AutoModel.from_pretrained(config.bert_model_name, add_pooling_layer=False)
        self.cl_representations = MLPLayer(self.bert)
        self.cos = self.exp_cosine_sim
        self.dropout = nn.Dropout(config.bert_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.weightfunction = nn.Sequential(nn.Linear(2 * num_labels, 1), nn.Sigmoid())

    def forward(self, batch, training=False, sup_contrastive_mode="HeroCon", focusing=0, clipping=0):
        # logits inference
        outputs = self.bert(batch.text_ids, attention_mask=batch.attention_mask_text)
        out = self.dropout(outputs.last_hidden_state[:, 0, :])  # uses [CLS]
        logits = self.classifier(out)

        ## contrastive learning
        l2 = torch.tensor(0, device=self.device, dtype=torch.float64)  # unsupervised loss
        l3 = torch.tensor(0, device=self.device, dtype=torch.float64)  # supervised loss
        if training:
            original_representation = self.cl_representations(outputs.last_hidden_state[:, 0, :])
            temp_labels = batch.labels
            if self.alpha != 0:
                # unsupervised contrastive learning
                noisy_outputs = self.bert(batch.text_ids, attention_mask=batch.attention_mask_text)
                noisy_representations = self.cl_representations(noisy_outputs.last_hidden_state[:, 0, :])  # batch size, hidden state dimensions
                cos_sim = self.sim(original_representation.unsqueeze(1), noisy_representations.unsqueeze(0))
                if self.w1 > 0 and self.w2 > 0: # adnce
                    pos = torch.exp(torch.diag(cos_sim))
                    mu = self.w1
                    sigma = self.w2
                    neg_score = cos_sim * self.temp
                    weight = 1. / (sigma * math.sqrt(2 * math.pi)) * torch.exp( - (neg_score - mu) ** 2 / (2 * math.pow(sigma, 2)))
                    weight = weight / weight.mean(dim=-1, keepdim=True)
                    neg = torch.sum(torch.exp(cos_sim) * weight.detach(), dim=1)
                    l2 = - torch.log(pos / neg).mean()
                else:  # simcse
                    labels = torch.arange(cos_sim.size(0)).long().to(self.device)
                    ce_loss = nn.CrossEntropyLoss()
                    l2 = ce_loss(cos_sim, labels)
            if self.beta != 0:
                if clipping:
                    proba = torch.sigmoid(logits)
                    if clipping == 1:
                        # remove instances in batch with potentially incorrect false negative labels
                        def filter_batches(preds, labs, clip):
                            diff = preds - labs # one-sided to identify false negatives
                            max_diff = torch.max(diff, dim=1).values
                            mask = max_diff <= clip
                            return mask
                        fn_mask = filter_batches(proba, temp_labels, self.clip)
                        temp_labels = temp_labels[fn_mask]
                        original_representation = original_representation[fn_mask]
                        # skip supervised CL for batch if less than 1 instance remaining 
                        # since no negative anchors to contrast
                        if original_representation.shape[0] <= 1:
                            return [self.alpha * l2, self.beta * l3], logits
                    if clipping == 2:
                        # label correction in batch with potentially incorrect false negative labels
                        def correct_labels(preds, labs, clip):
                            diff = preds - labs # one-sided to identify false negatives
                            mask = diff <= clip
                            labs[~mask] += 1
                            return labs
                        temp_labels = correct_labels(proba, temp_labels, self.clip)
                
                # Calculate weighting for asymmetric focusing -> downweights easy and potentially misclassified labels
                if focusing:
                    # gamma (1 and 4) and clip (0.05) terms are hardcoded right now but can be adjusted
                    anti_temp_labels = 1 - temp_labels
                    if clipping == 1:
                        xs_pos = torch.sigmoid(logits[fn_mask])
                    else:
                        xs_pos = torch.sigmoid(logits)
                    xs_neg = 1.0 - xs_pos
                    xs_neg.add_(0.05).clamp_(max=1)
                    xs_pos = xs_pos * temp_labels
                    xs_neg = xs_neg * anti_temp_labels
                    asymmetric_w = torch.pow(1 - xs_pos - xs_neg, 1 * temp_labels + 4 * anti_temp_labels)


                # supervised contrastive learning - adapted from HeroCon code - https://github.com/Leo02016/HeroCon/blob/main/module.py
                for i in range(self.labels_num):
                    pos_idx = torch.where(temp_labels[:, i] == 1)[0]
                    if len(pos_idx) == 0:
                        continue
                    neg_idx = torch.where(temp_labels[:, i] != 1)[0]
                    pos_sample = original_representation[pos_idx, :]
                    neg_sample = original_representation[neg_idx, :]
                    size = neg_sample.shape[0] + 1
                    if sup_contrastive_mode == "WeighCon":
                        # WeighCon Model - https://github.com/ScienceNLP-Lab/LLM-SSC/tree/main
                        n1 = temp_labels.shape[0]
                        sim = self.weightfunction(self.pairwise_concatenate(temp_labels, n1)).reshape(n1, n1)
                        pos_weight = sim[pos_idx, :][:, pos_idx]
                        neg_weight = 1 - sim[pos_idx, :][:, neg_idx]
                    elif sup_contrastive_mode == "HeroCon":
                        # HeroCon Model - https://github.com/Leo02016/HeroCon/blob/main/module.py
                        dist = self.hamming_distance_by_matrix(temp_labels)
                        pos_weight = 1 - dist[pos_idx, :][:, pos_idx] / self.labels_num
                        neg_weight = dist[pos_idx, :][:, neg_idx]
                    pos_dis = self.cos(pos_sample, pos_sample) * pos_weight
                    neg_dis = self.cos(pos_sample, neg_sample) * neg_weight
                    denominator = neg_dis.sum(1) + pos_dis
                    if focusing:
                        pos_asymmetric_w = asymmetric_w[pos_idx, i].unsqueeze(1) # weighting for each pos instance in batch based on focusing for that label
                        pos_dis = pos_dis * pos_asymmetric_w
                        neg_asymmetric_w = asymmetric_w[neg_idx, i].unsqueeze(1).transpose(0, 1)
                        neg_dis = neg_dis * neg_asymmetric_w
                        denominator = neg_dis.sum(1) + pos_dis
                    l3 += torch.mean(torch.log(denominator / (pos_dis * size)))
        return [self.alpha * l2, self.beta * l3], logits

    def pairwise_concatenate(self, matrix1, n1):
        idx = torch.cartesian_prod(torch.arange(n1), torch.arange(n1))
        concatenated_pairs = torch.cat((matrix1[idx[:, 0]], matrix1[idx[:, 1]]), dim=1)
        return concatenated_pairs

    def hamming_distance_by_matrix(self, labels):
        return torch.matmul(labels, (1 - labels).T) + torch.matmul(1 - labels, labels.T)

    def exp_cosine_sim(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.exp(torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature))


# For model optimization with ONNX Runtime
class BertForMultiLabelClassificationForOnnx(nn.Module):
    def __init__(self, config, num_labels):
        super(BertForMultiLabelClassificationForOnnx, self).__init__()
        self.bert = AutoModel.from_pretrained(config.bert_model_name)
        self.dropout = nn.Dropout(config.bert_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.activation = nn.Sigmoid()

    def forward(self, text_ids, attention_mask_text):
        outputs = self.bert(text_ids, attention_mask=attention_mask_text)
        out = self.dropout(outputs[1])  # uses [CLS]
        out = self.classifier(out)
        out = self.activation(out)
        return out
