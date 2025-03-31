import numpy as np
import pandas as pd
import torch
import re
from collections import namedtuple
from transformers import AutoTokenizer
import ast
import pickle
from tqdm import tqdm
from num2words import num2words


batch_fields = ['text_ids', 'attention_mask_text', 'labels', 'index', 'PMID']
Batch = namedtuple('Batch', field_names=batch_fields, defaults=[0] * len(batch_fields))

instance_fields = ['text', 'text_ids', 'attention_mask_text', 'labels', 'PMID', 'index', 'loss_mask']
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


def process_data_for_bert(tokenizer, data, max_length=512):
    instances = []
    c = 0
    pmids = list(data.index)
    print("INFO: Processing data...")
    for ind in tqdm(pmids, total=len(pmids)):
        text_ids, attn_mask, labels = process_instance(tokenizer, data, ind, max_length)
        
        instance = Instance(
            text_ids=text_ids,
            attention_mask_text=attn_mask,
            labels=labels,
            PMID=ind,
            index=c
            )

        instances.append(instance)
        c += 1

    return instances


def collate_fn(batch, config):
    batch_text_idxs = []
    batch_attention_masks_text = []
    batch_labels = []
    batch_index = []
    batch_sid = []
    for inst in batch:
        # current
        batch_text_idxs.append(inst.text_ids)
        batch_attention_masks_text.append(inst.attention_mask_text)
        batch_labels.append(inst.labels)

        batch_index.append(inst.index)
        batch_sid.append(inst.PMID)
    
    # Convert data into correct format
    batch_text_idxs = torch.as_tensor(batch_text_idxs, dtype=torch.long, device=config.device)
    batch_attention_masks_text = torch.as_tensor(batch_attention_masks_text, dtype=torch.float, device=config.device)
    batch_labels = torch.as_tensor(batch_labels, dtype=torch.float, device=config.device)
    batch_index = torch.as_tensor(batch_index, dtype=torch.long, device=config.device)
    batch_sid = batch_sid

    return Batch(
        text_ids=batch_text_idxs,
        attention_mask_text=batch_attention_masks_text,
        labels=batch_labels,
        index=batch_index,
        PMID=batch_sid
    )


def data_load(config):
    labels = pd.read_csv(config.label_file)

    train_ids = [int(ids) for ids in labels[labels['split'] == 'train']['ids'].to_list()]
    val_ids = [int(ids) for ids in labels[labels['split'] == 'val']['ids'].to_list()]
    test_ids = [int(ids) for ids in labels[labels['split'] == 'test']['ids'].to_list()]
    labels.drop(columns=['split'], inplace=True)

    labels[labels.columns] = labels[labels.columns].astype(int) # ensure these are integers
    labels.set_index('ids', inplace=True)
    label_list = list(labels.columns)

    data = pd.read_csv(config.data_file, low_memory=False)
    data.rename(columns={'pmid': 'ids'}, inplace=True)
    data['ids'] = data['ids'].astype(int)
    data.set_index('ids', inplace=True)

    cols2remove = ['pmcid', 'doi', 'mesh', 'pub_type']
    if config.align_full_text_only_comparison or config.full_text:
        cols2remove.remove('pmcid')
    data = data.drop(columns=cols2remove)

    if config.verbalize:
        def custom_join(list_):
            if isinstance(list_, list):
                length = len(list_)
                if length == 0:
                    return np.nan
                if length == 1:
                    return list_[0]
                if length == 2:
                    return f'{list_[0]} and {list_[1]}'
                return ', '.join(list_[:-1]) + f' and {list_[-1]}'
            else:
                return list_

        if config.verbalize != 'short':
            data['title'] = data['title'].map(lambda s: "This article's title is " + s.rstrip('.') + '.' if not pd.isnull(s) else np.nan)
            data['journal_title'] = data['journal_title'].map(lambda s: "This article is published in " + s + '.' if not pd.isnull(s) else np.nan)
            if config.verbalize == 'original':
                data['keywords'] = data['keywords'].map(lambda s: "This article's keywords are " + " and ".join(s.split(';')) + '.' if not pd.isnull(s) else np.nan)
                data['pub_date'] = data['pub_date'].map(lambda s: "This article was published in " + str(int(s)) + '.' if not pd.isnull(s) else np.nan)
                data['no_references'] = data['no_references'].map(lambda s: "This article's cited " + str(int(s)) + ' references.' if not pd.isnull(s) else np.nan)
                data['no_authors'] = data['no_authors'].map(lambda s: "This article was written by " + str(int(s)) + ' authors.' if not pd.isnull(s) else np.nan)
                data['nct_identifiers'] = data['nct_identifiers'].map(lambda s: ast.literal_eval(s) if not pd.isnull(s) else np.nan)
                data['nct_identifiers'] = data['nct_identifiers'].map(lambda s: "The article mentions the national clinical trial numbers " + " and ".join(s) + '.' if (isinstance(s, list) or isinstance(s, str)) else np.nan)
                data['all_caps'] = data['all_caps'].map(lambda s: ast.literal_eval(s) if not pd.isnull(s) else np.nan)
                data['all_caps'] = data['all_caps'].map(lambda s: "The article uses the abbreviations " + " and ".join(s) + '.' if (isinstance(s, list) or isinstance(s, str)) else np.nan)
                data['no_chemicals'] = data['no_chemicals'].map(lambda s: "This article used " + str(int(s)) + ' chemicals.' if not pd.isnull(s) else np.nan)
                data['list_of_chemicals'] = data['list_of_chemicals'].map(lambda s: ast.literal_eval(s) if not pd.isnull(s) else np.nan)
                data['list_of_chemicals'] = data['list_of_chemicals'].map(lambda s: " and ".join(s) + '.' if isinstance(s, list) else s)
                data['list_of_chemicals'] = data['list_of_chemicals'].map(lambda s: "The chemicals mentioned in the article are " + s.replace("['", "").replace("']", "") + '.' if isinstance(s, str) else np.nan)
                # features and their order
                contents_order = ['journal_title', 'pub_date', 'keywords', 'no_references', 'no_authors', 'no_chemicals', 'list_of_chemicals', 'nct_identifiers', 'all_caps', 'title', 'abstract']
            else:
                data['keywords'] = data['keywords'].map(lambda s: "This article's keywords are " + custom_join(s.split(';')) + '.' if not pd.isnull(s) else np.nan)
                data['pub_date'] = data['pub_date'].map(lambda s: "This article was published in " + num2words(int(s)) + '.' if not pd.isnull(s) else np.nan)
                data['no_references'] = data['no_references'].map(lambda s: "This article cited " + num2words(int(s)) + ' references.' if not pd.isnull(s) else np.nan)
                data['no_authors'] = data['no_authors'].map(lambda s: "This article was written by " + num2words(int(s)) + ' authors.' if not pd.isnull(s) else np.nan)
                data['no_affiliations'] = data['no_affiliations'].map(lambda s: "Authors are from " + num2words(int(s)) + ' different affiliations.' if not pd.isnull(s) else np.nan)
                data['all_caps_title'] = data['all_caps_title'].map(lambda s: custom_join(ast.literal_eval(s)) if not pd.isnull(s) else np.nan)
                data['all_caps_title'] = data['all_caps_title'].map(lambda s: "The title uses the abbreviations " + s + '.' if isinstance(s, str) else np.nan)
                data['no_chemicals'] = data['no_chemicals'].map(lambda s: "This article used " + num2words(int(s)) + ' chemicals.' if not pd.isnull(s) else np.nan)
                data['list_of_chemicals'] = data['list_of_chemicals'].map(lambda s: custom_join(ast.literal_eval(s)) if not pd.isnull(s) else np.nan)
                data['list_of_chemicals'] = data['list_of_chemicals'].map(lambda s: "The chemicals mentioned in the article are " + s + '.' if isinstance(s, str) else np.nan)
                # features and their order
                contents_order = ['journal_title', 'pub_date', 'keywords', 'no_references', 'no_authors', 'no_affiliations', 'no_chemicals', 'list_of_chemicals', 'all_caps_title', 'title', 'abstract']
        else:
            data['title'] = data['title'].map(lambda s: "This article's title is " + s.rstrip('.') + '.' if not pd.isnull(s) else np.nan)
            
            def date_journal(row):
                if not pd.isnull(row['pub_date']) and not pd.isnull(row['journal_title']):
                    return f"This article was published in {row['journal_title']} in {num2words(int(row['pub_date']))}."
                elif not pd.isnull(row['pub_date']):
                    return f"This article was published on {row['pub_date']}."
                elif not pd.isnull(row['journal_title']):
                    return f"This article was published in {row['journal_title']}."
                else:
                    return np.nan
            data['pub_date_journal_title'] = data.apply(lambda s: date_journal(s), axis=1)
            data['keywords'] = data['keywords'].map(lambda s: "This article's keywords are " + custom_join(s.split(';')) + '.' if not pd.isnull(s) else np.nan)
            data['no_references'] = data['no_references'].map(lambda s: "This article cited " + num2words(int(s)) + ' references.' if not pd.isnull(s) else np.nan)
           
            def author_affiliations(row):
                if not pd.isnull(row['no_authors']) and not pd.isnull(row['no_affiliations']):
                    if int(row['no_authors']) == 1:
                        auth = 'author'
                    else:
                        auth = 'authors'
                    if int(row['no_affiliations']) == 1:
                        affil = 'affiliation'
                    else:
                        affil = 'different affiliations'
                    return f"This article was written by {num2words(int(row['no_authors']))} {auth} from {num2words(int(row['no_affiliations']))} {affil}."
                elif not pd.isnull(row['no_authors']):
                    if int(row['no_authors']) == 1:
                        auth = 'author'
                    else:
                        auth = 'authors'
                    return f"This article was written by {row['no_authors']} {auth}."
                elif not pd.isnull(row['no_affiliations']):
                    if int(row['no_affiliations']) == 1:
                        affil = 'affiliation'
                    else:
                        affil = 'different affiliations'
                    return f"Authors are from {row['no_affiliations']} {affil}."
                else:
                    return np.nan
            data['no_authors_no_affiliations'] = data.apply(lambda s: author_affiliations(s), axis=1)
            data['all_caps_title'] = data['all_caps_title'].map(lambda s: custom_join(ast.literal_eval(s)) if not pd.isnull(s) else np.nan)
            data['all_caps_title'] = data['all_caps_title'].map(lambda s: "The title uses the abbreviations " + s + '.' if isinstance(s, str) else np.nan)
            
            def chemical_feature(row):
                if not pd.isnull(row['no_chemicals']) and not pd.isnull(row['list_of_chemicals']):
                    return f"This article used {num2words(int(row['no_chemicals']))} chemicals: {custom_join(ast.literal_eval(row['list_of_chemicals']))}."
                elif not pd.isnull(row['no_chemicals']):
                    return f"This article used {num2words(int(row['no_chemicals']))} chemicals."
                elif not pd.isnull(row['list_of_chemicals']):
                    return f"The chemicals mentioned in the article are {custom_join(ast.literal_eval(row['list_of_chemicals']))}."
                else:
                    return np.nan
            data['no_chemicals_list_of_chemicals'] = data.apply(lambda s: chemical_feature(s), axis=1)
            # features and their order
            contents_order = ['pub_date_journal_title', 'keywords', 'no_references', 'no_authors_no_affiliations', 'no_chemicals_list_of_chemicals', 'all_caps_title', 'title', 'abstract']

    if config.verbalize_missing:
        if config.verbalize_missing != 'abstract':
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
        elif config.verbalize_missing == 'abstract':
            data['abstract'] = data['abstract'].fillna("No abstract was detected in this artcle.")
    
    if config.full_text or config.align_full_text_only_comparison:
        pmc_data = pd.read_csv("data/pmc/pmc_data.csv") 
        pmc_data['simple_sentences'] = pmc_data['simple_sentences'].apply(ast.literal_eval)
        pmc_data['guidelines'] = pmc_data['guidelines'].apply(ast.literal_eval)
        pmc_data['ethics'] = pmc_data['ethics'].apply(ast.literal_eval)
        pmc_data['table_captions'] = pmc_data['table_captions'].apply(ast.literal_eval)
        pmc_data['figure_captions'] = pmc_data['figure_captions'].apply(ast.literal_eval)

        # Only option is to verblize without verbalizing missing features
        if 'first_sentence' in config.full_text:
            pmc_data['first_methods'] = pmc_data['first_methods'].map(lambda s: s if not pd.isnull(s) or not s else np.nan)
            pmc_data['first_methods'] = pmc_data['first_methods'].fillna("")

            contents_order.extend(['first_methods'])

        if 'simple_sentences' in config.full_text:
            pmc_data['simple_sentences'] = pmc_data['simple_sentences'].map(lambda s: "No sentences containing labels detected." if (isinstance(s, list) and len(s) == 0) else s)
            pmc_data['simple_sentences'] = pmc_data['simple_sentences'].map(lambda s: " ".join(s) if (isinstance(s, list)) else s)
            pmc_data['simple_sentences'] = pmc_data['simple_sentences'].map(lambda s: s if (isinstance(s, str)) else np.nan)
            pmc_data['simple_sentences'] = pmc_data['simple_sentences'].fillna("")

            contents_order.extend(['simple_sentences'])

        if 'guidelines' in config.full_text:
            pmc_data['guidelines'] = pmc_data['guidelines'].map(lambda s: "No reporting guidelines detected." if (isinstance(s, list) and len(s) == 0) else s)
            pmc_data['guidelines'] = pmc_data['guidelines'].map(lambda s: "The following reporting guidelines are mentioned: " + " and ".join(s) + '.' if (isinstance(s, list)) else s)
            pmc_data['guidelines'] = pmc_data['guidelines'].map(lambda s: s if (isinstance(s, str)) else np.nan)
            pmc_data['guidelines'] = pmc_data['guidelines'].fillna("")

            contents_order.extend(['guidelines'])

        if 'nct_in_article' in config.full_text:
            def verbalize_article_nct(row):
                if not pd.isna(row.nct_in_methods) or not pd.isna(row.nct_in_tables):
                    nct_in_methods = f'{row.nct_in_methods} clinical trial identifiers found in the methods section.'
                    nct_in_tables = f'{row.nct_in_tables} clinical trial identifiers found in tables.'
                else:
                    nct_in_methods = ''
                    nct_in_tables = ''
                nct_in_article = nct_in_methods + ' ' + nct_in_tables
                return nct_in_article
            
            pmc_data['nct_in_article'] = pmc_data.apply(verbalize_article_nct, axis=1)
            pmc_data['nct_in_article'] = pmc_data['nct_in_article'].fillna("")

            contents_order.extend(['nct_in_article'])

        if 'ethics' in config.full_text:
            pmc_data['ethics'] = pmc_data['ethics'].map(lambda s: "No ethical approvals detected." if (isinstance(s, list) and len(s) == 0) else s)
            pmc_data['ethics'] = pmc_data['ethics'].map(lambda s: "The following ethical approvals are mentioned: " + " and ".join(s) + '.' if (isinstance(s, list)) else s)
            pmc_data['ethics'] = pmc_data['ethics'].map(lambda s: s if (isinstance(s, str)) else np.nan)
            pmc_data['ethics'] = pmc_data['ethics'].fillna("")

            contents_order.extend(['ethics'])

        if 'non_text' in config.full_text:
            pmc_data['num_tables'] = pmc_data['num_tables'].map(lambda s: "There are " + num2words(int(s)) + " tables." if not pd.isnull(s) else np.nan)
            pmc_data['num_figures'] = pmc_data['num_figures'].map(lambda s: "There are " + num2words(int(s)) + " figures." if not pd.isnull(s) else np.nan)
            pmc_data['full_text_length'] = pmc_data['full_text_length'].map(lambda s: "The article is " + num2words(int(s)) + " words long." if not pd.isnull(s) else np.nan)
            
            pmc_data['num_tables'] = pmc_data['num_tables'].fillna("")
            pmc_data['num_figures'] = pmc_data['num_figures'].fillna("")
            pmc_data['full_text_length'] = pmc_data['full_text_length'].fillna("")  # length of article roughly in words

            contents_order.extend(['num_tables', 'num_figures', 'full_text_length'])

        if 'section_heading' in config.full_text:
            pmc_data['primary_section_heading_list'] = pmc_data['primary_section_heading_list'].map(lambda s: custom_join(ast.literal_eval(s)) if not pd.isnull(s) else np.nan)
            pmc_data['primary_section_heading_list'] = pmc_data['primary_section_heading_list'].map(lambda s: "The section headings are " + s + '.' if isinstance(s, str) else np.nan)
            pmc_data['primary_section_heading_list'] = pmc_data['primary_section_heading_list'].fillna("")

            contents_order.extend(['primary_section_heading_list'])

        if 'textrank' in config.full_text:
            pmc_data['textrank_sentences'] = pmc_data['textrank_sentences'].map(lambda s: s if (isinstance(s, str)) else np.nan)
            pmc_data['textrank_sentences'] = pmc_data['textrank_sentences'].fillna("")

            contents_order.extend(['textrank_sentences'])
        
        if 'captions' in config.full_text:
            pmc_data['table_captions'] = pmc_data['table_captions'].map(lambda s: "No table captions detected." if (isinstance(s, list) and len(s) == 0) else s)
            pmc_data['table_captions'] = pmc_data['table_captions'].map(lambda s: "Table captions are " + " and ".join(s) + '.' if (isinstance(s, list)) else s)
            pmc_data['table_captions'] = pmc_data['table_captions'].map(lambda s: s if (isinstance(s, str)) else np.nan)

            pmc_data['figure_captions'] = pmc_data['figure_captions'].map(lambda s: "No figure captions detected." if (isinstance(s, list) and len(s) == 0) else s)
            pmc_data['figure_captions'] = pmc_data['figure_captions'].map(lambda s: "Figure captions are " + " and ".join(s) + '.' if (isinstance(s, list)) else s)
            pmc_data['figure_captions'] = pmc_data['figure_captions'].map(lambda s: s if (isinstance(s, str)) else np.nan) 
            
            pmc_data['table_captions'] = pmc_data['table_captions'].fillna("")
            pmc_data['figure_captions'] = pmc_data['figure_captions'].fillna("")

            contents_order.extend(['table_captions', 'figure_captions'])

        if 'first_paragraph' in config.full_text:
            pmc_data['first_paragraph'] = pmc_data['first_paragraph'].map(lambda s: str(s).replace('\n', ' ').replace('  ', ' ') if not pd.isnull(s) or not s else np.nan)
            pmc_data['first_paragraph'] = pmc_data['first_paragraph'].fillna("")

            contents_order.extend(['first_paragraph'])

        if 'summaries' in config.full_text:
            if 'v1' in config.full_text:
                summary_data = pd.read_csv("data/pmc/method_intro_extractive_v1.csv")
            if 'v2' in config.full_text:
                summary_data = pd.read_csv("data/pmc/method_intro_extractive_v2.csv")
            if 'v3' in config.full_text:
                summary_data = pd.read_csv("data/pmc/method_intro_extractive_v3.csv")
            if 'v4' in config.full_text:
                summary_data = pd.read_csv("data/pmc/method_intro_llama_v2.csv")
                llama_nonsense = [
                    "Here's a summary of the article, focusing on the study design:",
                    "Here is a summary of the article, focusing on the study design:",
                    "Here's a summary of the article focusing on the study design:",
                    "Here is a summary of the article focusing on the study design:",
                    "To summarize the article's focus on the study design:",
                    "Here's a summary of the study design:",
                    "Here is a summary of the study design:",
                    "Here’s my attempt to summarize the study:",
                    "Here is my attempt to summarize the study:",
                    "To answer your question about summarizing the study design:",
                    "I'm happy to help you with summarizing the article!",
                    "I'm happy to help you with summarizing the article's focus on the study design.",
                ]
                summary_data['summary'] = summary_data['summary'].replace(llama_nonsense, '')
            if 'v5' in config.full_text:
                summary_data = pd.read_csv("data/pmc/method_intro_primer.csv")
            summary_data['pmid'] = summary_data['pmid'].astype(int)
            summary_data['summary'] = summary_data['summary'].fillna("")

            contents_order.extend(['summary'])

    if config.full_text or config.align_full_text_only_comparison:
        # merge full-text data w/ pubmed data
        pmc_data.rename(columns={'PMCID': 'pmcid'}, inplace=True)
        data = data.reset_index()
        data = pd.merge(data, pmc_data, how="inner", on="pmcid")
        if 'summaries' in config.full_text:
            summary_data.drop('split', axis=1, inplace=True)
            data = pd.merge(data, summary_data, how="left", left_on="ids", right_on="pmid")
        data = data.set_index('ids')

    data.fillna("", inplace=True)

    if config.remove_feature:
        contents_order.remove(config.remove_feature)

    # merge contents and format text input nicely
    data['contents'] = data[contents_order].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    data['contents'] = data['contents'].map(lambda s: re.sub(r"\s+", " ", s).strip().replace("..", "."))  # replace white space with single space
    if config.full_text:
        def remove_duplicate_sentences(text):
            sentences = text['contents'].split(". ")
            unique_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()  # Remove leading/trailing whitespaces
                if sentence not in unique_sentences:
                    unique_sentences.append(sentence)
            return ". ".join(unique_sentences).strip()  # Add back the final period
        data['contents'] = data.apply(remove_duplicate_sentences, axis=1)

    data = data[['contents']]

    # Adds new split labels to try to add additional information to the model regarding topics or design subtypes
    if config.label_split:
        def add_label_split(list_of_new_labels, labels_df):
            for li in list_of_new_labels:
                with open(f'data/split/{li}_idx.pkl', 'rb') as file:
                    new_label_idx = pickle.load(file)
                labels_df[li] = np.where(labels_df.index.isin(new_label_idx), 1, 0)
            return labels_df
        if config.label_split == 'combination':
            new_labels = ['cohort_follow_up', 'cohort_longitudinal', 'cohort_prospective', 'cohort_retrospective', 'humans', 'animals', 'veterinary']
        elif config.label_split == 'cohort':
            new_labels = ['cohort_follow_up', 'cohort_longitudinal', 'cohort_prospective', 'cohort_retrospective']
        elif config.label_split == 'generalized_rct':
            new_labels = ['generalized_rct']
        elif config.label_split == 'humans':
            new_labels = ['humans']
        elif config.label_split == 'animals':
            new_labels = ['animals']
        elif config.label_split == 'veterinary':
            new_labels = ['veterinary']
            # new_labels = ['dogs', 'cats', 'cattle', 'horses', 'swine']
        labels = add_label_split(new_labels, labels)
        label_list.extend(new_labels)

    labels['binary_labels'] = labels[label_list].values.tolist()
    labels = labels.loc[:, ['binary_labels']]

    complete = data.merge(labels, how='left', left_index=True, right_index=True)
    
    # for debugging...
    # complete = complete.sample(frac=0.001)
    
    # Split
    train_df = complete[complete.index.isin(train_ids)]
    val_df = complete[complete.index.isin(val_ids)]
    test_df = complete[complete.index.isin(test_ids)]
    
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name, do_lower_case=True, use_fast=True)

    if config.max_length == -1:
        max_token_length_training = train_df['contents'].apply(lambda features: len(tokenizer(features)['input_ids'])).max()
        if 'longformer' in config.bert_model_name.lower() or 'bigbird' in config.bert_model_name.lower():
            model_max_length = 4096
        else:
            model_max_length = 512
        config.max_length = min([max_token_length_training, model_max_length])

    if config.train_val_test == 'val':
        train_dataset, test_dataset = None, None
        val_dataset = process_data_for_bert(tokenizer, val_df, max_length=config.max_length)
    elif config.train_val_test == 'test':
        train_dataset, val_dataset = None, None
        test_dataset = process_data_for_bert(tokenizer, test_df, max_length=config.max_length)
    else:
        train_dataset = process_data_for_bert(tokenizer, train_df, max_length=config.max_length)
        val_dataset = process_data_for_bert(tokenizer, val_df, max_length=config.max_length)
        test_dataset = process_data_for_bert(tokenizer, test_df, max_length=config.max_length)
    
    # return test_df
    return train_dataset, val_dataset, test_dataset, label_list, config


if __name__ == '__main__':
    class TestClass:
        def __init__(self):
            self.data_file = 'data/pubmed/pubmed_data.csv'
            self.label_file = 'data/labels_human/split_stratified_data.csv'
            self.label_split = ''
            self.train_val_test = 'test'
            self.max_length = 512
            self.verbalize = 'short'
            self.verbalize_missing = ''
            self.contrastive_loss = ''
            self.remove_feature = '' # no_affiliations
            self.align_full_text_only_comparison = ''
            self.bert_model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
            self.full_text = ''
    config = TestClass()
    train_dataset, val_dataset, test_dataset, label_list = data_load(config)
    
    # test_df = data_load(config)

    # with open('data/rct_fp_list.pkl', 'rb') as file:
    #     fp_list = pickle.load(file)
    #     fp_data = test_df[test_df.index.isin(fp_list)]

    # print('False positives')
    # for tup in fp_data.head(30).iterrows():
    #     print(tup[0])
    #     print(tup[1]['contents'])
    #     print('-'*75)
    # print()

    # with open('data/rct_fn_list.pkl', 'rb') as file:
    #     fn_list = pickle.load(file)
    #     fn_data = test_df[test_df.index.isin(fn_list)]

    # print('False negatives')
    # for tup in fn_data.head(30).iterrows():
    #     print(tup[0])
    #     print(tup[1]['contents'])
    #     print('-'*75)
    # print()

    # with open('data/multi_tp_list.pkl', 'rb') as file:
    #     tp_list = pickle.load(file)
    #     tp_data = test_df[test_df.index.isin(tp_list)]

    # print('True positives')
    # for tup in tp_data.head(50).iterrows():
    #     print(tup[0])
    #     print(tup[1]['contents'])
    #     print('-'*75)
    # print()

