import numpy as np
import pandas as pd
import torch
from collections import namedtuple, Counter, defaultdict
from transformers import AutoTokenizer
import ast
import pickle
from pathlib import Path
from tqdm import tqdm
import random

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
rng = np.random.default_rng()

batch_fields = ['text_ids', 'attention_mask_text', 'labels', 'text_ids_pos', 'attention_mask_text_pos', 'labels_pos', 'text_ids_neg', 'attention_mask_text_neg', 'labels_neg', 'index', 'PMID']
Batch = namedtuple('Batch', field_names=batch_fields, defaults=[0] * len(batch_fields))

instance_fields = ['text', 'text_ids', 'attention_mask_text', 'labels', 'text_ids_pos', 'attention_mask_text_pos', 'labels_pos', 'text_ids_neg', 'attention_mask_text_neg', 'labels_neg', 'PMID', 'index', 'loss_mask']
Instance = namedtuple('Instance', field_names=instance_fields, defaults=[0] * len(instance_fields))


def process_instance(tokenizer, data, ind, max_length=512):
    text_ids = tokenizer.encode(
        data['contents'][ind],
        add_special_tokens=True,
        truncation=True,
        max_length=max_length
        )

    pad_num = max_length - len(text_ids)
    attn_mask = [1] * len(text_ids) + [0] * pad_num
    text_ids = text_ids + [0] * pad_num
    labels = data['binary_labels'][ind]

    return text_ids, attn_mask, labels


def select_anchors(config, data, ind, cache):
    # select negative and positive anchors for each instance...
    pos_ind, neg_ind = 0, 0
    if config.contrastive_loss == 'scl':
        # negative anchor is anything not an exact match
        labels_str = data['binary_labels_str'][ind]
        if cache[labels_str] == 0:
            filtered = data[data['binary_labels_str'] == labels_str].index.to_numpy()
            unfiltered = data[~data.index.isin(filtered)].index.to_numpy()
            cache[labels_str] = (filtered, unfiltered)
    else:
        # for jaccard similarity cl - positive anchor must contain at least 1 overlapping positive label
        labels = data['positive_labels'][ind]
        labels_str = str(labels)
        if cache[labels_str] == 0:
            filter = data['positive_labels_str'].str.contains('|'.join(str(x) for x in labels))
            filtered = data[filter].index.to_numpy()
            unfiltered = data[~filter].index.to_numpy()
            cache[labels_str] = (filtered, unfiltered)

    pos_ind = rng.choice(cache[labels_str][0])
    neg_ind = rng.choice(cache[labels_str][1])

    return pos_ind, neg_ind, cache


def process_data_for_bert(config, tokenizer, data, max_length=512):
    instances = []
    c = 0
    cache = defaultdict(int)
    data_cache = defaultdict(int)
    pmids = list(data.index)
    for ind in tqdm(pmids, total=len(pmids)):
        if config.contrastive_loss == 'unsup' and c == 0:
            last_ind = data.tail(1).index.item()
            text_ids_prior, attn_mask_prior, labels_prior = process_instance(tokenizer, data, last_ind, max_length)

        if data_cache[ind] == 0:
            text_ids, attn_mask, labels = process_instance(tokenizer, data, ind, max_length)
            data_cache[ind] = (text_ids, attn_mask, labels)
        else:
            text_ids, attn_mask, labels = data_cache[ind]
        
        instance = Instance(
            text_ids=text_ids,
            attention_mask_text=attn_mask,
            labels=labels,
            PMID=ind,
            index=c
        )

        if config.contrastive_loss:
            # select positive and negative anchors
            if config.contrastive_loss == 'unsup':
                text_ids_pos = text_ids
                attn_mask_pos = attn_mask
                labels_pos = labels
                # negatives for this instance
                text_ids_neg = text_ids_prior
                attn_mask_neg = attn_mask_prior
                labels_neg = labels_prior
                # save negatives for next instance
                text_ids_prior = text_ids
                attn_mask_prior = attn_mask
                labels_prior = labels
            elif config.contrastive_loss == 'scl' or config.contrastive_loss == 'jscl':
                pos_ind, neg_ind, cache = select_anchors(config, data, ind, cache)
                if data_cache[pos_ind] == 0:
                    text_ids_pos, attn_mask_pos, labels_pos = process_instance(tokenizer, data, pos_ind, max_length)
                    data_cache[pos_ind] = (text_ids_pos, attn_mask_pos, labels_pos)
                else:
                    text_ids_pos, attn_mask_pos, labels_pos = data_cache[pos_ind]
                if data_cache[neg_ind] == 0:
                    text_ids_neg, attn_mask_neg, labels_neg = process_instance(tokenizer, data, neg_ind, max_length)
                    data_cache[neg_ind] = (text_ids_neg, attn_mask_neg, labels_neg)
                else:
                    text_ids_neg, attn_mask_neg, labels_neg = data_cache[neg_ind]

            instance = Instance(
                text_ids=text_ids,
                attention_mask_text=attn_mask,
                labels=labels,
                text_ids_pos=text_ids_pos,
                attention_mask_text_pos=attn_mask_pos,
                labels_pos=labels_pos,
                text_ids_neg=text_ids_neg,
                attention_mask_text_neg=attn_mask_neg,
                labels_neg=labels_neg,
                PMID=ind,
                index=c
            )
        instances.append(instance)
        c += 1
    return instances


def collate_fn(batch, gpu=True):
    batch_text_idxs = []
    batch_attention_masks_text = []
    batch_labels = []
    batch_text_idxs_pos = []
    batch_attention_masks_text_pos = []
    batch_labels_pos = []
    batch_text_idxs_neg = []
    batch_attention_masks_text_neg = []
    batch_labels_neg = []
    batch_index = []
    batch_sid = []
    for inst in batch:
        # current
        batch_text_idxs.append(inst.text_ids)
        batch_attention_masks_text.append(inst.attention_mask_text)
        batch_labels.append(inst.labels)
        # positive anchor
        batch_text_idxs_pos.append(inst.text_ids_pos)
        batch_attention_masks_text_pos.append(inst.attention_mask_text_pos)
        batch_labels_pos.append(inst.labels_pos)
        # negative anchor
        batch_text_idxs_neg.append(inst.text_ids_neg)
        batch_attention_masks_text_neg.append(inst.attention_mask_text_neg)
        batch_labels_neg.append(inst.labels_neg)

        batch_index.append(inst.index)
        batch_sid.append(inst.PMID)
    if gpu:
        # current
        batch_text_idxs = torch.cuda.LongTensor(batch_text_idxs)
        batch_attention_masks_text = torch.cuda.FloatTensor(batch_attention_masks_text)
        batch_labels = torch.cuda.FloatTensor(batch_labels)
        # positive anchor
        batch_text_idxs_pos = torch.cuda.LongTensor(batch_text_idxs_pos)
        batch_attention_masks_text_pos = torch.cuda.FloatTensor(batch_attention_masks_text_pos)
        batch_labels_pos = torch.cuda.FloatTensor(batch_labels_pos)
        # negative anchor
        batch_text_idxs_neg = torch.cuda.LongTensor(batch_text_idxs_neg)
        batch_attention_masks_text_neg = torch.cuda.FloatTensor(batch_attention_masks_text_neg)
        batch_labels_neg = torch.cuda.FloatTensor(batch_labels_neg)

        batch_index = torch.cuda.LongTensor(batch_index)
        batch_sid = batch_sid
    else:
        # current
        batch_text_idxs = torch.LongTensor(batch_text_idxs)
        batch_attention_masks_text = torch.FloatTensor(batch_attention_masks_text)
        batch_labels = torch.FloatTensor(batch_labels)
        # positive anchor
        batch_text_idxs_pos = torch.LongTensor(batch_text_idxs_pos)
        batch_attention_masks_text_pos = torch.FloatTensor(batch_attention_masks_text_pos)
        batch_labels_pos = torch.FloatTensor(batch_labels_pos)
        # negative anchor
        batch_text_idxs_neg = torch.LongTensor(batch_text_idxs_neg)
        batch_attention_masks_text_neg = torch.FloatTensor(batch_attention_masks_text_neg)
        batch_labels_neg = torch.FloatTensor(batch_labels_neg)

        batch_index = torch.LongTensor(batch_index)
        batch_sid = batch_sid
    return Batch(
        text_ids=batch_text_idxs,
        attention_mask_text=batch_attention_masks_text,
        labels=batch_labels,
        text_ids_pos=batch_text_idxs_pos,
        attention_mask_text_pos=batch_attention_masks_text_pos,
        labels_pos=batch_labels_pos,
        text_ids_neg=batch_text_idxs_neg,
        attention_mask_text_neg=batch_attention_masks_text_neg,
        labels_neg=batch_labels_neg,
        index=batch_index,
        PMID=batch_sid
    )


def undersample(df, proportion, minimum_val):
    undersampled_list = []
    df['binary_labels_str'] = df['binary_labels'].apply(str)
    label_counts_dict = df['binary_labels_str'].value_counts().to_dict()
    for label in label_counts_dict:
        if label_counts_dict[label] <= minimum_val:
            filtered_idx = df[df["binary_labels_str"].values == label].index.to_list()
            undersampled_list.extend(filtered_idx)
        else:
            number_of_samples_w_lower_bound = max(round(label_counts_dict[label] * proportion), minimum_val)
            filtered_idx = df[df["binary_labels_str"].values == label].index.to_list()
            if number_of_samples_w_lower_bound >= len(filtered_idx): # compares desired # of samples w/ number available in the dataset
                undersampled_list.extend(filtered_idx)  # if there are not enough, then just add all
            else:  # if there are enough, then samples
                undersample_ = random.sample(filtered_idx, number_of_samples_w_lower_bound)
                undersampled_list.extend(undersample_)
    df.drop(columns=['binary_labels_str'], inplace=True)
    return undersampled_list


def data_load(config):
    labels = pd.read_csv(config.label_file)
    data = pd.read_csv(config.data_file, low_memory=False)

    # future work: extract pages again, a lot more formats that are uncaptured (e.g., 3005-13); calculate document page length if available
    cols2remove = ['pmcid', 'doi', 'journal_issn', 'page_start', 'page_end', 'mesh', 'pub_type']
    data = data.drop(columns=cols2remove)

    # Check for missing data by column...
    check_nulls = False
    if check_nulls:
        print(data.isna().sum())

    if config.verbalize:
        data['title'] = data['title'].map(lambda s: "This article's title is " + s + '.' if not pd.isnull(s) else np.nan)
        data['journal_title'] = data['journal_title'].map(lambda s: "This article is published in " + s + '.' if not pd.isnull(s) else np.nan)
        data['pub_date'] = data['pub_date'].map(lambda s: "This article was published in " + str(int(s)) + '.' if not pd.isnull(s) else np.nan)
        data['keywords'] = data['keywords'].map(lambda s: "This article's keywords are " + " and ".join(s.split(';')) + '.' if not pd.isnull(s) else np.nan)
        data['no_references'] = data['no_references'].map(lambda s: "This article's cited " + str(int(s)) + ' references.' if not pd.isnull(s) else np.nan)
        data['no_authors'] = data['no_authors'].map(lambda s: "This article was written by " + str(int(s)) + ' authors.' if not pd.isnull(s) else np.nan)
        data['list_of_chemicals'] = data['list_of_chemicals'].map(lambda s: ast.literal_eval(s) if not pd.isnull(s) else np.nan)
        data['list_of_chemicals'] = data['list_of_chemicals'].map(lambda s: "The chemicals mentioned in the article are " + " and ".join(s) + '.' if (isinstance(s, list) or isinstance(s, str)) else np.nan)
        data['no_chemicals'] = data['no_chemicals'].map(lambda s: "This article used " + str(int(s)) + ' chemicals.' if not pd.isnull(s) else np.nan)
        data['nct_identifiers'] = data['nct_identifiers'].map(lambda s: ast.literal_eval(s) if not pd.isnull(s) else np.nan)
        data['nct_identifiers'] = data['nct_identifiers'].map(lambda s: "The article mentions the national clinical trial numbers " + " and ".join(s) + '.' if (isinstance(s, list) or isinstance(s, str)) else np.nan)
        data['all_caps'] = data['all_caps'].map(lambda s: ast.literal_eval(s) if not pd.isnull(s) else np.nan)
        data['all_caps'] = data['all_caps'].map(lambda s: "The article uses the abbreviations " + " and ".join(s) + '.' if (isinstance(s, list) or isinstance(s, str)) else np.nan)
    
    if config.verbalize_missing:
        data['title'] = data['title'].fillna("This article's title is unknown.")
        data['journal_title'] = data['journal_title'].fillna("This article's journal is unknown.")
        data['pub_date'] = data['pub_date'].fillna("This article's publication date is unknown.")
        data['keywords'] = data['keywords'].fillna("This article's keywords are unknown.")
        data['no_references'] = data['no_references'].fillna("The number of references cited in this article is unknown.")
        data['no_authors'] = data['no_authors'].fillna("The number of authors who wrote this article is unknown.")
        data['list_of_chemicals'] = data['list_of_chemicals'].fillna("The chemicals used in this article are unknown.")
        data['no_chemicals'] = data['no_chemicals'].fillna("The number of chemicals in this article is unknown.")
        data['nct_identifiers'] = data['nct_identifiers'].fillna("No national clinical trial numbers were detected in this article.")
        data['all_caps'] = data['all_caps'].fillna("No abbreviations were detected in this article.")
    else:
        data['title'] = data['title'].fillna('')
        data['journal_title'] = data['journal_title'].fillna('')
        data['pub_date'] = data['pub_date'].fillna('')
        data['keywords'] = data['keywords'].fillna('')
        data['no_references'] = data['no_references'].fillna('')
        data['no_authors'] = data['no_authors'].fillna('')
        data['list_of_chemicals'] = data['list_of_chemicals'].fillna('')
        data['no_chemicals'] = data['no_chemicals'].fillna('')
        data['nct_identifiers'] = data['nct_identifiers'].fillna('')
        data['all_caps'] = data['all_caps'].fillna('')

    data.rename(columns={'pmid': 'ids'}, inplace=True)
    data['ids'] = data['ids'].astype(int)
    data.set_index('ids', inplace=True)
    contents_order = ['journal_title', 'pub_date', 'keywords', 'no_references', 'no_authors', 'no_chemicals', 'list_of_chemicals', 'nct_identifiers', 'all_caps', 'title', 'abstract']
    if config.remove_feature:
        contents_order.remove(config.remove_feature)
    data['contents'] = data[contents_order].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    data['contents'] = data['contents'].fillna('')
    data = data[['contents']]

    labels['ids'] = labels['ids'].astype(int)
    labels.set_index('ids', inplace=True)
    label_list = list(labels.columns)
    labels[label_list] = labels[label_list].astype(int)
    
    if config.align_for_comparison_with_v1:
        remove_labels = ['scientific_integrity_review', 'published_erratum', 'clinical_trial_protocol', 'expression_of_concern', 'veterinary_observational_study', 'veterinary_randomized_controlled_trial', 'clinical_conference', 'newspaper_article', 'retraction_of_publication', 'veterinary_clinical_trial']
        temp = [i for i in label_list if i not in remove_labels]
        label_list = temp

    labels['binary_labels'] = labels[label_list].values.tolist()
    labels = labels.loc[:, ['binary_labels']]

    complete = data.merge(labels, how='outer', left_index=True, right_index=True)

    def get_positives(row):
        pos_list = [i for i in range(len(row.binary_labels)) if row.binary_labels[i] > 0]
        if len(pos_list) == 0:
            pos_list = ['none']
        else:
            pos_list = pos_list
        return pos_list

    complete['positive_labels'] = complete.apply(get_positives, axis = 1)
    complete['positive_labels_str'] = complete['positive_labels'].astype(str)
    complete['binary_labels_str'] = complete['binary_labels'].apply(str)

    # for debugging...
    #complete = complete.sample(frac=0.001)
    
    # Check for length and if any articles are missing all data...
    check_length = False
    if check_length:
        complete['length'] = complete['contents'].map(lambda s: len(s.split()) if not pd.isnull(s) else 0)
        print(complete['length'].describe().apply(lambda x: format(x, 'f')))

    with open('data/train_ids.pkl', 'rb') as file:
        train_ids = pickle.load(file)
        # print(len(train_ids))
    with open('data/val_ids.pkl', 'rb') as file:
        val_ids = pickle.load(file)
        # print(len(val_ids))
    with open('data/test_ids.pkl', 'rb') as file:
        test_ids = pickle.load(file)
        # print(len(test_ids))

    # Split
    train_df = complete[complete.index.isin(train_ids)]
    val_df = complete[complete.index.isin(val_ids)]
    test_df = complete[complete.index.isin(test_ids)]

    if config.align_for_comparison_with_v1:
        with open('data/v1_val_ids.pkl', 'rb') as file:
            v1_val_ids = pickle.load(file)
        val_df = complete[complete.index.isin(v1_val_ids)]
        with open('data/v1_test_ids.pkl', 'rb') as file:
            v1_test_ids = pickle.load(file)
        test_df = complete[complete.index.isin(v1_test_ids)]

    # Undersampling
    if config.undersampling < 1.0:
        undersample_path = Path(f'data/train_df_{str(int(config.undersampling * 100))}_percent_{str(config.undersampling_min_thresh)}_threshold.pkl')
        if undersample_path.exists():
            with open(undersample_path, 'rb') as file:
                train_idxs = pickle.load(file)
        else:
            train_idxs = undersample(train_df, config.undersampling, config.undersampling_min_thresh)
            with open(undersample_path, 'wb') as file:
                pickle.dump(train_idxs, file)

        train_df = train_df[train_df.index.isin(train_idxs)]

    tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name, do_lower_case=True, use_fast=True)
    train_dataset, val_dataset, test_dataset = 0, 0, 0
    if config.train_val_test == 'train':
        train_dataset = process_data_for_bert(config, tokenizer, train_df)
        val_dataset = process_data_for_bert(config, tokenizer, val_df)
    elif config.train_val_test == 'val':
        val_dataset = process_data_for_bert(config, tokenizer, val_df)
    elif config.train_val_test == 'test':
        test_dataset = process_data_for_bert(config, tokenizer, test_df)

    return train_dataset, val_dataset, test_dataset, label_list


class TestClass:
    def __init__(self):
        self.data_file = 'data/pubmed/pubmed_data.csv'
        self.label_file = 'data/labels/stratified_data.csv'
        self.train_val_test = 'train'
        self.undersampling = 1.0
        self.verbalize = ''
        self.verbalize_missing = ''
        self.contrastive_loss = ''
        self.remove_feature = ''
        self.align_for_comparison_with_v1 = ''
        self.bert_model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'



if __name__ == '__main__':
    config = TestClass()
    train_dataset, val_dataset, test_dataset, list_name = data_load(config)
