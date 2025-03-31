import csv
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from Bio import Entrez
import xml.etree.ElementTree as ET
from math import sqrt
from operator import itemgetter
import spacy
import pytextrank
import pubmed_parser as pp
from collections import defaultdict
import importlib
import string
import pubmed_parser as pp

import os
from utils import loop_thru_all_files
from sentence_segmentation import segment


nlp = spacy.load("en_core_sci_lg", exclude=['ner'])
nlp.add_pipe("textrank")


# Full-Text Parsing - Shruthan's work - parse_xml, textrank_summarize
package = importlib.import_module("pubmed_parser")
importlib.reload(package)
pp = package

def parse_xml(path):
    punctuation = set(string.punctuation) - set("+-")
    punctuation = "".join(punctuation)

    def strip_edge_punctuation(text):
        table = str.maketrans("", "", punctuation)
        return text.translate(table).strip()

    def find_nct_identifiers(text):
        nct_pattern = re.compile("(NCT[0-9]+)")
        nct2add = nct_pattern.findall(str(text).upper())
        if len(nct2add) > 0:
            return nct2add
        return []

    features = defaultdict(str)
    try:
        article = pp.parse_pubmed_xml(path)
    except:
        article = ''
    try:
        tables = pp.parse_pubmed_table(path)
    except:
        tables = ''
    try:
        figures = pp.parse_pubmed_caption(path)
    except:
        figures = ''
    try:
        references = pp.parse_pubmed_references(path)
    except:
        references = ''

    ## Full-text
    paras = pp.parse_pubmed_paragraph(path, all_paragraph=True)
    full_text = ""
    prev_section = []
    for para in paras:
        if len(para["section"]) > 0:
            for i in range(len(para["section"])):
                if i >= len(prev_section) or para["section"][i] != prev_section[i]:
                    full_text += (
                        "\n" + (i + 2) * "#" + " " + para["section"][i].strip() + "\n"
                    )
            prev_section = para["section"]
        full_text += para["text"].strip() + "\n"
    # Clean Full-Text
    parsed_ft = full_text.split("\n")
    parsed_ft = [ft for ft in parsed_ft if ft]
    corrected_parsed_ft = []
    temp = ''
    for ft in parsed_ft:
        if ft.startswith('#'):
            clean_ft = ft
            temp += clean_ft.replace('#', '').strip()
        if not ft.startswith('#'):
            if temp:
                corrected_parsed_ft.append(temp)
            else:
                corrected_parsed_ft.append(ft)
            temp = ''
    features["full_text"] = corrected_parsed_ft

    if tables:
        features["num_tables"] = len(tables)
        features["table_captions"] = [
            (x["caption"] + "\n" + x["footer"]).strip() for x in tables
        ]
        features["table_labels"] = [x["label"] for x in tables]
    else:
        features["num_tables"], features["table_captions"], features["table_labels"] = (
            0,
            [],
            [],
        )

    if figures:
        features["num_figures"] = len(figures) if figures else 0
        features["figure_captions"] = [x["fig_caption"].strip() for x in figures]
        features["figure_labels"] = [x["fig_label"].strip() for x in figures]
    else:
        (
            features["num_figures"],
            features["figure_captions"],
            features["figure_labels"],
        ) = (0, [], [])

    # NCT identifier
    nct_identifiers = []
    if tables:
        for table in tables:
            for row in table["table_values"]:
                for value in row:
                    nct_identifiers.extend(find_nct_identifiers(value))

    for caption in features["figure_captions"]:
        nct_identifiers.extend(find_nct_identifiers(caption))

    for caption in features["table_captions"]:
        nct_identifiers.extend(find_nct_identifiers(caption))

    nct_identifiers.extend(find_nct_identifiers(features["full_text"]))

    nct_identifiers_in_ref = []
    if references:
        for reference in references:
            nct_identifiers_in_ref.extend(
                find_nct_identifiers(reference["article_title"])
            )
    features["nct_identifiers_in_references"] = nct_identifiers_in_ref

    return features


def textrank_summarize(text, limit_phrases=2, limit_sentences=40, return_raw=False):
    """
    limit_sentences: number of sentences to return
    return_raw:
        if True,
            returns a list [sent_id, sent_text, rank]
            list is sorted by rank (best sentence first)
            sent_id is the position of sentence in full text
            sent_id can be used to sort the sentences as they appear in full text
        if False,
            best sentences are extracted by textrank
            the top "limit_sentences" number of sentences are selected
            selected sentences are sorted in the order in which they appear in full text
            sentences are concatenated to a form a paragraph and returne
    """
    doc = nlp(text)
    sent_bounds = [[s.start, s.end, set([])] for s in doc.sents]
    phrase_id = 0
    unit_vector = []
    for p in doc._.phrases:
        unit_vector.append(p.rank)
        for chunk in p.chunks:
            for sent_start, sent_end, sent_vector in sent_bounds:
                if chunk.start >= sent_start and chunk.end <= sent_end:
                    # ic(sent_start, chunk.start, chunk.end, sent_end)
                    sent_vector.add(phrase_id)
                    break

        phrase_id += 1

        if phrase_id == limit_phrases:
            break

    sum_ranks = sum(unit_vector)

    unit_vector = [rank / sum_ranks for rank in unit_vector]

    sent_rank = {}
    sent_id = 0

    for sent_start, sent_end, sent_vector in sent_bounds:
        sum_sq = 0.0
        for phrase_id in range(len(unit_vector)):
            if phrase_id not in sent_vector:
                sum_sq += unit_vector[phrase_id] ** 2.0

        sent_rank[sent_id] = sqrt(sum_sq)
        sent_id += 1

    sent_text = {}
    sent_id = 0

    for sent in doc.sents:
        sent_text[sent_id] = sent.text
        sent_id += 1

    num_sent = 0
    summary_sentences = []
    for sent_id, rank in sorted(sent_rank.items(), key=itemgetter(1)):
        summary_sentences.append([sent_id, sent_text[sent_id], rank])
        num_sent += 1

        if num_sent == limit_sentences:
            break

    if return_raw:
        return summary_sentences

    # sort "important" sentences in order as in the original text
    summary_sentences.sort(key=lambda x: x[0])
    summary_sentences = [x[1].strip() for x in summary_sentences]
    summary = " ".join(summary_sentences)

    return summary


def extract_methods(methods):
    try:
        methods_text = ''
        for paragraph in methods.iter('p'):
            methods_text = methods_text + ' ' + paragraph.text
        methods_text = methods_text.replace('  ', ' ')
        methods_sentences = segment(methods_text)
    except TypeError:
        methods_sentences = [{'sentence': ''}]
    return methods_sentences


def extract_first_paragraph(methods):
    try:
        paragraph_text = ''
        for paragraph in methods.iter('p'):
            paragraph_text = paragraph_text + ' ' + paragraph.text
            if len(paragraph_text.split(' ')) > 5:
                break
        paragraph_text = paragraph_text.replace('  ', ' ')
    except:
        paragraph_text = ''
    return paragraph_text


def detect_labels(text_dict):
    # find sentences with labels - input:
    label_names = ['autobiography', 'bibliography', 'biography', 'case-control_studies', 'case_reports', 'clinical_conference', 'clinical_studies_as_topic', 'clinical_study', 'clinical_trial', 'clinical_trial_protocol', 'cohort_studies', 'comment', 'congress', 'consensus_development_conference', 'cross-cultural_comparison', 'cross-over_studies', 'cross-sectional_studies', 'diagnostic_test_accuracy', 'double-blind_method', 'editorial', 'evaluation_studies_as_topic', 'evaluation_study', 'expression_of_concern', 'feasibility_studies', 'focus_groups', 'follow-up_studies', 'genome-wide_association_study', 'historical_article', 'human_experimentation', 'humans', 'interview', 'interviews_as_topic', 'lecture', 'legal_case', 'letter', 'longitudinal_studies', 'matched-pair_analysis', 'meta-analysis', 'multicenter_study', 'news', 'newspaper_article', 'personal_narrative', 'portrait', 'practice_guideline', 'predictive_value_of_tests', 'prospective_studies', 'published_erratum', 'random_allocation', 'randomized_controlled_trial_humans', 'reproducibility_of_results', 'retraction_of_publication', 'retrospective_studies', 'review', 'scientific_integrity_review', 'systematic_review', 'systematic_reviews_as_topic', 'twin_study', 'validation_study', 'veterinary_clinical_trial', 'veterinary_randomized_controlled_trial']
    remove = ['clinical_trial_protocol', 'clinical_studies_as_topic', 'evaluation_studies_as_topic', 'interviews_as_topic', 'systematic_reviews_as_topic', 'humans', 'news', 'review', 'veterinary_clinical_trial', 'veterinary_randomized_controlled_trial']
    # removes ambiguous or redundant labels and changes plural/singular terms to something that could be either - to root
    label_names = [label.replace('_', ' ').replace('reports', 'report').replace('study', 'stud').replace('studies', 'stud').replace('groups', 'group').replace('analysis', 'analy').replace('retraction of publication', 'retraction') for label in label_names if label not in remove]
    # simplifies names for certain categories
    label_names = [label.replace('double-blind method', 'double-blind').replace('retraction of publication', 'retraction').replace('published erratum', 'erratum').replace('randomized controlled trial humans', 'randomized controlled trial').replace('longitudinal stud', 'longitudinal').replace('retrospective stud', 'retrospective').replace('cohort stud', 'cohort').replace('prospective stud', 'prospective').replace('multicenter stud', 'multicenter').replace('cross-sectional stud', 'cross-sectional').replace('cross-over stud', 'cross-over').replace('case-control stud', 'case-control').strip() for label in label_names]
    no_hyphen_variants = ['double blind', 'case control', 'cross over', 'cross sectional', 'meta analy', 'matched pair_analy', 'genome wide association', 'cross cultural comparison', 'follow up stud']
    label_names.extend(no_hyphen_variants)
    matched_sentences = []
    for sent_dict in text_dict:
        for label_name in label_names:
            if label_name in sent_dict['sentence'].lower():
                matched_sentences.append(sent_dict['sentence'])
                break  # we don't want to add a sentence multiple times if multiple labels within
    return matched_sentences


def detect_guidelines(article_full_text):
    """Detect presence of a guideline"""
    guidelines = ['STaRI', 'STARD', 'QUOROM', 'MOOSE', 'CONSORT', 'STROBE', 'PRISMA', 'SPIRIT', 'CARE', 'AGREE', 'SRQR', 'ARRIVE', 'SQUIRE', 'CHEERS', 'MDAR', 'RIVER', 'IMPROVE', 'PREPARE']
    guidelines_mixed = ['Lambeth Conventions'] # lower case match
    secondary = ['guideline', 'statement', 'checklist', 'report', 'recommend', 'extension', 'criteria', 'item', 'figure', 'diagram']
    secondary_cap = [word.capitalize() for word in secondary]
    secondary.extend(secondary_cap)
    parsed_sentences = segment(" ".join(article_full_text))
    guidelines_present = set()
    for sentence_dict in parsed_sentences:
        sentence = sentence_dict['sentence'].replace('(', ' ').replace(')', ' ').replace(',', '').replace(';', '').replace(':', '').replace('.', '').split(' ')
        for word in sentence:
            for guideline in guidelines:
                if guideline == word:
                    for second in secondary:
                        if second in sentence_dict['sentence']:
                            guidelines_present.add(guideline)
        for guideline in guidelines_mixed:
            if guideline.lower() in sentence_dict['sentence'].lower():
                guidelines_present.add(guideline)
    return list(guidelines_present)


def detect_ethics(article_full_text):
    """Detect presence of an ethics review board - IRB or IACUC"""
    patterns_found = []
    irb_pattern=re.compile('institutional\sreview\sboard', re.IGNORECASE)
    patterns_found.extend(irb_pattern.findall(str(article_full_text)))
    iacuc_pattern=re.compile('institutional\sanimal\scare\sand\suse\scommittee', re.IGNORECASE)
    patterns_found.extend(iacuc_pattern.findall(str(article_full_text)))
    if len(patterns_found) > 0:
        return patterns_found
    return []


def detect_identifiers(text):
    patterns_found = []
    pattern_0=re.compile('(U[0-9]+-[0-9]+-[0-9]+)', re.IGNORECASE)
    patterns_found.extend(pattern_0.findall(str(text).upper()))
    pattern_1=re.compile('(NCT[ ;:/-]{0,2}[0-9]{8})', re.IGNORECASE)
    patterns_found.extend(pattern_1.findall(str(text).upper()))
    pattern_2=re.compile('([12][0-9]{3}-[0-9]{6}-[0-9]{2})', re.IGNORECASE)
    patterns_found.extend(pattern_2.findall(str(text).upper()))
    pattern_3=re.compile('(SNCTP[0-9]+)', re.IGNORECASE)
    patterns_found.extend(pattern_3.findall(str(text).upper()))
    pattern_4=re.compile('(ISRCTN\s?[0-9]+)', re.IGNORECASE)
    patterns_found.extend(pattern_4.findall(str(text).upper()))
    pattern_5=re.compile('(ACTRN[0-9]+)', re.IGNORECASE)
    patterns_found.extend(pattern_5.findall(str(text).upper()))
    pattern_6=re.compile('(ChiCTR[0-9]+)', re.IGNORECASE)
    patterns_found.extend(pattern_6.findall(str(text).upper()))
    pattern_7=re.compile('(CTRI\/[0-9]+\/[0-9]+\/[0-9]+)', re.IGNORECASE)
    patterns_found.extend(pattern_7.findall(str(text).upper()))
    pattern_8=re.compile('(IRCT[0-9A-Z]+)', re.IGNORECASE)
    patterns_found.extend(pattern_8.findall(str(text).upper()))
    pattern_9=re.compile('(KCT[0-9]+)', re.IGNORECASE)
    patterns_found.extend(pattern_9.findall(str(text).upper()))
    pattern_10=re.compile('(PHRR[0-9]+-[0-9]+)', re.IGNORECASE)
    patterns_found.extend(pattern_10.findall(str(text).upper()))
    pattern_11=re.compile('(SLCTR\/[0-9]+\/[0-9]+)', re.IGNORECASE)
    patterns_found.extend(pattern_11.findall(str(text).upper()))
    pattern_12=re.compile('(TCTR[0-9]+)', re.IGNORECASE)
    patterns_found.extend(pattern_12.findall(str(text).upper()))
    pattern_13=re.compile('(RPCEC[0-9]+)', re.IGNORECASE)
    patterns_found.extend(pattern_13.findall(str(text).upper()))
    pattern_14=re.compile('(PACTR[0-9]+)', re.IGNORECASE)
    patterns_found.extend(pattern_14.findall(str(text).upper()))
    pattern_15=re.compile('(TFDA[0-9]+)', re.IGNORECASE)
    patterns_found.extend(pattern_15.findall(str(text).upper()))
    if len(patterns_found) > 0:
        return patterns_found
    return []


def extract_all_feature(article, features):
    # tries to extract first sentence from methods; if not, then first sentence from full-text-body
    root = ET.fromstring(article)
    for methods in root.findall('.//sec[@sec-type="methods"]'):
        methods_sentences = extract_methods(methods)
        first_paragraphs = extract_first_paragraph(methods)
        if methods_sentences:
            features['first_methods'] = methods_sentences[0]['sentence']
            features['simple_sentences'] = detect_labels(methods_sentences)
    if not features['first_methods']:
        for methods in root.findall('.//sec[@sec-type="materials|methods"]'):
            methods_sentences = extract_methods(methods)
            first_paragraphs = extract_first_paragraph(methods)
            if methods_sentences:
                features['first_methods'] = methods_sentences[0]['sentence']
                features['simple_sentences'] = detect_labels(methods_sentences)
    if not features['first_methods']:
        with open('structured_abstract_sections.pkl', 'rb') as file:
            normalized_headings = pickle.load(file)  # lower cased; keys: ['methods', 'background', 'results', 'conclusions', 'objective']
        # loop through sections and if title in  normalized section with Methods Mapping - if in set of methods papers
        non_abstract, started, finished = False, False, False
        try:
            first_body  = [sect.text for child in root.findall('.//body/sec') for sect in child][0]
            last_body  = [sect.text for child in root.findall('.//body/sec') for sect in child][-1]
        except IndexError:
            # No body section detected
            first_body = ''
            last_body = ''
            non_abstract = True
        methods_text = ''
        first_paragraphs = ''
        for sections in root.findall('.//sec'):
            if sections:
                for subs in sections:
                    if subs.text == first_body:
                        non_abstract = True
                    if not non_abstract:
                        continue
                    if subs.tag == 'title' and started and subs.text:
                        if subs.text == last_body or subs.text.lower() in normalized_headings['background'] or subs.text.lower() in normalized_headings['results'] or subs.text.lower() in normalized_headings['conclusions']:
                            finished = True
                    if finished:
                        break
                    if subs.tag == 'title' and subs.text:
                        if subs.text.lower() in normalized_headings['methods']:
                            started = True
                    if started and subs.text:
                        if len(first_paragraphs.split(' ')) < 5:
                            first_paragraphs = first_paragraphs + ' ' + subs.text
                        methods_text = methods_text + ' ' + subs.text
                if finished:
                    break
        methods_sentences = segment(methods_text)
        features['simple_sentences'] = detect_labels(methods_sentences)
        first_sentence = ''
        for meth in methods_sentences:
            first_sentence = first_sentence + ' ' + meth['sentence']
            if '.' in meth['sentence']:
                break
        features['first_methods'] = first_sentence
    # if no methods detected, extract first sentence of body
    if not features['first_methods']:
        first_paragraphs = ''
        for child in root.findall('.//body/p'):
            if child.text:
                if len(first_paragraphs.split(' ')) < 5:
                    first_paragraphs = first_paragraphs + ' ' + child.text
                sentences = segment(child.text)
                if sentences:
                    features['first_methods'] = sentences[0]['sentence']
                    if features['first_methods']:
                        break
    if not features['first_methods']:
        first_paragraphs = ''
        for child in root.findall('.//body/*/p'):
            if child.text:
                if len(first_paragraphs.split(' ')) < 5:
                    first_paragraphs = first_paragraphs + ' ' + child.text
                sentences = segment(child.text)
                if sentences:
                    features['first_methods'] = sentences[0]['sentence']
                    if features['first_methods']:
                        break
    if not features['first_methods']:
        first_paragraphs = ''
        # correction to published manuscript
        matches = re.findall(r"\>(.*?)\:\<\/", article)
        for match in matches:
            if 'correction' in match:
                if match.endswith('.'):
                    features['first_methods'] = match
                else:
                    features['first_methods'] = match + '.'
    features['first_paragraph'] = first_paragraphs
    return features


if __name__ == '__main__':
    directory = "pmc/full-text-articles"

    try:
        pmids_df = pd.read_csv('pmc/pmc_data.csv')
        pmids = pmids_df['PMCID'].to_list()
        new = False
    except FileNotFoundError:
        pmids = []
        new = True
    files = loop_thru_all_files(directory, filter_list=pmids)

    data = []

    with open('pmc/pmc_data.csv', 'a') as csvfile:
        fieldnames = ['num_tables', 'table_captions', 'table_labels', 'num_figures', 'figure_captions', 'figure_labels', 'nct_identifiers_in_references', 'PMCID', 'nct_in_tables', 'first_methods', 'simple_sentences', 'first_paragraph', 'full_text_length', 'nct_in_methods', 'guidelines', 'ethics', 'textrank_sentences']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if new:
            writer.writeheader()
        for pmcid_path in tqdm(files, total=len(files)):
            pmcid = pmcid_path.split('/')[-1].split('.')[0]
            try:
                features = parse_xml(pmcid_path)
                features['PMCID'] = pmcid
                with open(pmcid_path, 'r') as file:
                    file_content = file.read().replace('\n', ' ').replace('\r', '')
                    table_num_pattern = re.compile("Table\s*([0-9]+)")
                    # If table extraction fails, regex is used as a backup...
                    # Get number of tables
                    number_of_tables = [int(num) for num in table_num_pattern.findall(file_content)]
                    try:
                        number_of_tables = max(number_of_tables)
                    except ValueError:
                        number_of_tables = 0
                    if features['num_tables'] < number_of_tables:
                        features['num_tables'] = number_of_tables
                        # Get table captions
                        table_caption_pattern = re.compile("Table\s*[0-9]+\s*\<\/label\>\s*\<caption>\s*(?:\<p\>)?\s*(.*?)(?:\<\/p\>)?\<\/caption\>")
                        table_captions = table_caption_pattern.findall(file_content)
                        table_captions = [tc.replace('</p>', '').strip() for tc in table_captions]
                        features['table_captions'] = table_captions
                    if features['num_tables'] > 0:
                        table_content_pattern = re.compile("\<\/?table\-wrap(.*?)\>")
                        table_contents = "".join(table_content_pattern.findall(file_content))
                        features['nct_in_tables'] = int(len(set(detect_identifiers(table_contents))))
                    else:
                        features['nct_in_tables'] = 0

                # Get first sentence in methods section
                with open(pmcid_path, 'r') as file:
                    raw_article = file.read()
                    # clean article
                    article = re.sub(r'\<\/?ext\-link(.*?)\>', '', raw_article)
                    article = re.sub(r'\<\/?xref(.*?)\>', '', article)
                    article = re.sub(r'\<\/?fig(.*?)\>', '', article)
                    article = re.sub(r'\<\/?graphic(.*?)\>', '', article)
                    article = re.sub(r'\<\/?sup(.*?)\>', '', article)
                    article = re.sub(r'\<\/?sub(.*?)\>', '', article)
                    article = re.sub(r'\<\/?italic(.*?)\>', '', article)
                    article = re.sub(r'\<\/?bold(.*?)\>', '', article)
                    article = re.sub(r'\<\/?label(.*?)\>', '', article)
                    article = re.sub(r'\<\/?media(.*?)\>', '', article)
                    article = re.sub(r'\<\/?related\-article(.*?)\>', '', article)
                    article = re.sub(r'\<\/?(\!\-\-)?supplementary\-material(.*?)\>', '', article)
                    article = re.sub(r'\<\!\-\-(.*?)\-\-\>', '', article)
                    article = re.sub("\<p\>[^]]*\<\/p>", lambda x:x.group(0).replace('\n',' '), article)
                    article = article.replace('  ', ' ')
                    article = article.replace('&#8211;', '-')
                    features = extract_all_feature(article, features)
                    features['full_text_length'] = len(''.join(article).split(' '))  # get rough full-text length
                    # Detect clinical trial number (NCT) in body of article...
                    features['nct_in_methods'] = len(set(detect_identifiers(article)))
                # Detect guideline presence...
                features['guidelines'] = detect_guidelines(features['full_text'])
                # Detect guideline presence...
                features['ethics'] = detect_ethics(raw_article)

                # TextRank features...
                try:
                    article = re.sub(r'\<floats\-group[^>]*\>(?:[\S\n\t\v ]*?)\<\/floats\-group\>', '', article)
                    article = re.sub(r'\<table\-wrap\-group[^>]*\>(?:[\S\n\t\v ]*?)\<\/table\-wrap\-group\>', '', article)
                    article = re.sub(r'\<table\-wrap[^>]*\>(?:[\S\n\t\v ]*?)\<\/table\-wrap\>', '', article)
                    root = ET.fromstring(article)
                    try:
                        body = "".join(root.find('body').itertext()).replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
                    except AttributeError:
                        body = " ".join(features['full_text'])
                    textrank_features = textrank_summarize(body, limit_sentences=5)
                    features['textrank_sentences'] = textrank_features

                except:
                    features['textrank_sentences'] = np.nan
                    
                del features['full_text']

                # Post-Extraction Cleaning
                for feats in features:
                    if isinstance(features[feats], str):
                        features[feats] = re.sub(r'\<\/?(.*?)\>', '', features[feats])
                    if isinstance(features[feats], list):
                        for i, item in enumerate(features[feats]):
                            features[feats][i] = re.sub(r'\<\/?(.*?)\>', '', features[feats][i])
                writer.writerow(features)

            except ET.ParseError:
                try:
                    features = parse_xml(pmcid_path)
                    features['PMCID'] = pmcid
                    with open(pmcid_path, 'r') as file:
                        file_content = file.read().replace('\n', ' ').replace('\r', '')
                        table_num_pattern = re.compile("Table\s*([0-9]+)")
                        # If table extraction fails, regex is used as a backup...
                        # Get number of tables
                        number_of_tables = [int(num) for num in table_num_pattern.findall(file_content)]
                        try:
                            number_of_tables = max(number_of_tables)
                        except ValueError:
                            number_of_tables = 0
                        if features['num_tables'] < number_of_tables:
                            features['num_tables'] = number_of_tables
                            # Get table captions
                            table_caption_pattern = re.compile("Table\s*[0-9]+\s*\<\/label\>\s*\<caption>\s*(?:\<p\>)?\s*(.*?)(?:\<\/p\>)?\<\/caption\>")
                            table_captions = table_caption_pattern.findall(file_content)
                            table_captions = [tc.replace('</p>', '').strip() for tc in table_captions]
                            features['table_captions'] = table_captions
                        if features['num_tables'] > 0:
                            table_content_pattern = re.compile("\<\/?table\-wrap(.*?)\>")
                            table_contents = "".join(table_content_pattern.findall(file_content))
                            features['nct_in_tables'] = int(len(set(detect_identifiers(table_contents))))
                        else:
                            features['nct_in_tables'] = 0

                    # Get first sentence in methods section
                    with open(pmcid_path, 'r') as file:
                        raw_article = file.read()
                        # clean article
                        article = re.sub(r'\<\/?ext\-link(.*?)\>', '', raw_article)
                        article = re.sub(r'\<\/?xref(.*?)\>', '', article)
                        article = re.sub(r'\<\/?fig(.*?)\>', '', article)
                        article = re.sub(r'\<\/?graphic(.*?)\>', '', article)
                        article = re.sub(r'\<\/?sup(.*?)\>', '', article)
                        article = re.sub(r'\<\/?sub(.*?)\>', '', article)
                        article = re.sub(r'\<\/?italic(.*?)\>', '', article)
                        article = re.sub(r'\<\/?bold(.*?)\>', '', article)
                        article = re.sub(r'\<\/?label(.*?)\>', '', article)
                        article = re.sub(r'\<\/?media(.*?)\>', '', article)
                        article = re.sub(r'\<\/?related\-article(.*?)\>', '', article)
                        article = re.sub(r'\<\/?(\!\-\-)?supplementary\-material(.*?)\>', '', article)
                        article = re.sub(r'\<\!\-\-(.*?)\-\-\>', '', article)
                        article = re.sub("\<p\>[^]]*\<\/p>", lambda x:x.group(0).replace('\n',' '), article)
                        article = article.replace('  ', ' ')
                        article = article.replace('&#8211;', '-')
                        features = extract_all_feature(article, features)
                        features['full_text_length'] = len(''.join(article).split(' '))  # get rough full-text length
                        # Detect clinical trial number (NCT) in body of article...
                        features['nct_in_methods'] = len(set(detect_identifiers(article)))
                    # Detect guideline presence...
                    features['guidelines'] = detect_guidelines(features['full_text'])
                    # Detect guideline presence...
                    features['ethics'] = detect_ethics(raw_article)

                    # TextRank features... no cleaning article
                    try:
                        root = ET.fromstring(article)
                        try:
                            body = "".join(root.find('body').itertext()).replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
                        except AttributeError:
                            body = " ".join(features['full_text'])
                        textrank_features = textrank_summarize(body, limit_sentences=5)
                        features['textrank_sentences'] = textrank_features
                    except:
                        features['textrank_sentences'] = np.nan

                    del features['full_text']

                    # Post-Extraction Cleaning
                    for feats in features:
                        if isinstance(features[feats], str):
                            features[feats] = re.sub(r'\<\/?(.*?)\>', '', features[feats])
                        if isinstance(features[feats], list):
                            for i, item in enumerate(features[feats]):
                                features[feats][i] = re.sub(r'\<\/?(.*?)\>', '', features[feats][i])

                    writer.writerow(features)
                except ET.ParseError:
                    print(pmcid)
