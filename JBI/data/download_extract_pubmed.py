import os
import numpy as np
import pandas as pd
from Bio import Entrez
import re
import math
import csv
from tqdm import tqdm
from random import randint
from time import sleep
from utils import batch
import secret_variables # .py file containing variable 'email'


def extract_article_data_from_pubmed(list_of_pmids, email):
    Entrez.email = email
    handle = Entrez.efetch(db="pubmed", id=','.join(map(str, list_of_pmids)), retmode="xml")
    records = Entrez.read(handle, validate=False)
    articles = []
    for pubmed_article in records['PubmedArticle']:
        # All extracted features
        article = {
            'pmid': np.nan,
            'pmcid': np.nan,
            'doi': np.nan,
            'title': np.nan,
            'pub_date': np.nan,
            'journal_title': np.nan,
            'mesh': np.nan,
            'pub_type': np.nan,
            'keywords': np.nan,
            'abstract': np.nan,
            'no_references': np.nan,
            'no_authors': np.nan,
            'no_affiliations': np.nan,
            'no_chemicals': np.nan,
            'list_of_chemicals': np.nan,
            'nct_identifiers': np.nan,
            'all_caps': np.nan,
            'all_caps_title': np.nan,
        }
        
        # Article Identifiers (PMID, PMCID, DOI)
        try:
            for id in pubmed_article['PubmedData']['ArticleIdList']:
                if id.attributes['IdType'] == 'pubmed':
                    article['pmid'] = str(id)
                elif id.attributes['IdType'] == 'pmc':
                    article['pmcid'] = str(id)
                elif id.attributes['IdType'] == 'doi':
                    article['doi'] = str(id)
        except:
            pass
        
        # Article Title 
        try:
            article['title'] = pubmed_article['MedlineCitation']['Article']['ArticleTitle']
        except:
            pass
        
        # Publication Date - Year
        try:
            article['pub_date'] = str(pubmed_article['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']['Year'])
        except:
            pass
        
        # Journal Information (ISSN and Title)
        try:
            article['journal_title'] = str(pubmed_article['MedlineCitation']['Article']['Journal']['Title'])
        except:
            pass

        # MeSH Terms
        try:
            article['mesh'] = '; '.join(
                [str(mesh['DescriptorName']) for mesh in pubmed_article['MedlineCitation']['MeshHeadingList']])
        except:
            pass
        
        # Publication Type
        try:
            article['pub_type'] = '; '.join(
                [str(p) for p in pubmed_article['MedlineCitation']['Article']['PublicationTypeList']])
        except:
            pass

        # Keywords - Author Specified Keywords List
        try:
            keywords2add = '; '.join(
                [str(k) for keyw in pubmed_article['MedlineCitation']['KeywordList'] for k in keyw])
            if len(keywords2add) > 0:
                article['keywords'] = keywords2add
        except:
            pass

        # Abstract
        try:
            if len(pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText']) > 1:
                abstract = ''
                for abst in pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText']:
                    abstract = abstract + abst.attributes['Label'] + '\n' + str(abst) + '\n'
                article['abstract'] = abstract
            else:
                article['abstract'] = pubmed_article['MedlineCitation']['Article']['Abstract']['AbstractText'][0]
        except:
            pass

        # Number of References
        try:
            article['no_references'] = len(pubmed_article['PubmedData']['ReferenceList'][0]['Reference'])
        except:
            pass
        
        # Number of Authors
        try:
            article['no_authors'] = len(pubmed_article['MedlineCitation']['Article']['AuthorList'])
        except:
            pass

        # Number of Affiliations
        try:
            affiliations = set()
            for authors in pubmed_article['MedlineCitation']['Article']['AuthorList']:
                try:
                    affiliations.add(authors['AffiliationInfo'][0]['Affiliation'])
                except:
                    pass
            if len(affiliations) > 0:
                article['no_affiliations'] = len(affiliations)
        except:
            pass

        # Chemical Information - Number of chemicals and list of chemical names 
        try:
            article['no_chemicals'] = len(pubmed_article['MedlineCitation']['ChemicalList'])
        except:
            pass
        try:
            article['list_of_chemicals'] = [str(chem['NameOfSubstance']) for chem in pubmed_article['MedlineCitation']['ChemicalList']]
        except:
            pass

        # Regex Extraction - NCT Identifiers and All Caps
        try:
            nct_pattern = re.compile('(NCT[0-9]+)')
            nct2add = nct_pattern.findall(str(pubmed_article).upper())
            if len(nct2add) > 0:
                article['nct_identifiers'] = nct2add
        except:
            pass
        try:
            ignore_list = list(line.strip() for line in open('structured_headings.txt'))
            ignore_list.append('AND')
            ignore_list.append('III')
            ignore_list.append('MEDLINE')
            caps_pattern = re.compile(r'\b([A-Z]{3,})\b')
            content2search = str(article['title']) + ' ' + str(article['abstract'])
            list2add = [i for i in set(caps_pattern.findall(content2search)) if i not in ignore_list]
            if len(list2add) > 0:
                article['all_caps'] = list2add
        except:
            pass
        try:
            ignore_list = list(line.strip() for line in open('structured_headings.txt'))
            ignore_list.append('AND')
            ignore_list.append('III')
            ignore_list.append('MEDLINE')
            caps_pattern = re.compile(r'\b([A-Z]{3,})\b')
            content2search = str(article['title'])
            list2add = [i for i in set(caps_pattern.findall(content2search)) if i not in ignore_list]
            if len(list2add) > 0:
                article['all_caps_title'] = list2add
        except:
            pass

        articles.append(article)
    
    # Books - sometimes tagged as reviews
    for pubmed_book_article in records['PubmedBookArticle']:
        article = {
            'pmid': np.nan,
            'pmcid': np.nan,
            'doi': np.nan,
            'title': np.nan,
            'pub_date': np.nan,
            'journal_title': np.nan,
            'mesh': np.nan,
            'pub_type': np.nan,
            'keywords': np.nan,
            'abstract': np.nan,
            'no_references': np.nan,
            'no_authors': np.nan,
            'no_chemicals': np.nan,
            'list_of_chemicals': np.nan,
            'nct_identifiers': np.nan,
            'all_caps': np.nan,
            'all_caps_title': np.nan,
        }
        
        # Article Identifiers (PMID, PMCID, DOI)
        try:
            for id in pubmed_book_article['PubmedBookData']['ArticleIdList']:
                if id.attributes['IdType'] == 'pubmed':
                    article['pmid'] = str(id)
                elif id.attributes['IdType'] == 'pmc':
                    article['pmcid'] = str(id)
                elif id.attributes['IdType'] == 'doi':
                    article['doi'] = str(id)
        except:
            pass
        
        # Book Title 
        try:
            article['title'] = pubmed_book_article['BookDocument']['Book']['BookTitle']
        except:
            pass
        
        # Publication Date - Year
        try:
            article['pub_date'] = str(pubmed_book_article['BookDocument']['Book']['PubDate']['Year'])
        except:
            pass
        
        # Publisher Information (Name)
        try:
            article['journal_title'] = str(pubmed_book_article['BookDocument']['Book']['Publisher']['PublisherName'])
        except:
            pass

        # Abstract
        try:
            if len(pubmed_book_article['BookDocument']['Abstract']['AbstractText']) > 1:
                abstract = ''
                for abst in pubmed_book_article['BookDocument']['Abstract']['AbstractText']:
                    abstract = abstract + abst.attributes['Label'] + '\n' + str(abst) + '\n'
                article['abstract'] = abstract
            else:
                article['abstract'] = pubmed_book_article['BookDocument']['Abstract']['AbstractText'][0]
        except:
            pass
        
        # Number of Authors
        try:
            article['no_authors'] = len(pubmed_book_article['BookDocument']['Book']['AuthorList'])
        except:
            pass

        # Number of Affiliations
        try:
            affiliations = set()
            for authors in pubmed_book_article['BookDocument']['Book']['AuthorList']:
                try:
                    affiliations.add(authors['AffiliationInfo'][0]['Affiliation'])
                except:
                    pass
            if len(affiliations) > 0:
                article['no_affiliations'] = len(affiliations)
        except:
            pass

        # All caps in title
                # Regex Extraction - NCT Identifiers and All Caps
        try:
            nct_pattern = re.compile('(NCT[0-9]+)')
            nct2add = nct_pattern.findall(str(pubmed_article).upper())
            if len(nct2add) > 0:
                article['nct_identifiers'] = nct2add
        except:
            pass
        try:
            # change ignore list to capitalized version of structured headings list
            ignore_list = list(line.strip() for line in open('structured_headings.txt'))
            ignore_list.append('AND')
            ignore_list.append('III')
            ignore_list.append('MEDLINE')
            caps_pattern = re.compile(r'\b([A-Z]{3,})\b')
            content2search = str(article['title']) + ' ' + str(article['abstract'])
            list2add = [i for i in set(caps_pattern.findall(content2search)) if i not in ignore_list]
            if len(list2add) > 0:
                article['all_caps'] = list2add
            
            content2search = str(article['title'])
            list2add = [i for i in set(caps_pattern.findall(content2search)) if i not in ignore_list]
            if len(list2add) > 0:
                article['all_caps_title'] = list2add
            
        except:
            pass

        articles.append(article)

    return articles


def run_pass(batch_obj, out_file_loc, field_names, batch_total, email):
    try:
        os.mkdir(out_file_loc.split('/')[0])  # creates dir if does not exist; fails if more than 1 level deep
    except FileExistsError:
        pass
    with open(out_file_loc, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        # create a list of article data (batch_data = [dictionaries]) for all articles in batch
        for id_batch in tqdm(batch_obj, total=batch_total):
            sleep(randint(1,3))
            batch_data = extract_article_data_from_pubmed(id_batch, email)
            for article_data in batch_data:
                writer.writerow(article_data)


if __name__ == '__main__':
    first_pass = True
    second_pass = True
    field_names = ['pmid', 'pmcid', 'doi', 'title', 'pub_date', 'journal_title', 'mesh', 'pub_type', 'keywords', 'abstract', 'no_references', 'no_authors', 'no_affiliations', 'no_chemicals', 'list_of_chemicals', 'nct_identifiers', 'all_caps', 'all_caps_title']

    # First pass, downloading all data
    if first_pass:
        in_file = 'labels_human/stratified_data.csv'
        out_file = 'pubmed/pubmed_data.csv'
        all_ids = pd.read_csv(in_file)
        batch_length = math.ceil(len(list(all_ids['ids'])) / 9900)
        batched_ids = batch(list(map(str, list(all_ids['ids']))), 9900)
        
        run_pass(batched_ids, out_file, field_names, batch_length, secret_variables.email)

    # Second pass, re-trying downloads for data not initially extracted due to errors with connectivity to PubMed, etc.
    if second_pass:
        if first_pass:
            sleep(10)

        out_file_2 = 'pubmed/pubmed_data_errors.csv'
        labels = pd.read_csv('labels_human/stratified_data.csv')
        data = pd.read_csv('pubmed/pubmed_data.csv')

        data.rename(columns={'pmid': 'ids'}, inplace=True)
        data['ids'] = data['ids'].astype(str)
        data.set_index('ids', inplace=True)
        labels['ids'] = labels['ids'].astype(str)
        labels.set_index('ids', inplace=True)

        complete = data.merge(labels, how='outer', left_index=True, right_index=True)
        features = ['pmcid', 'doi', 'title', 'pub_date', 'journal_title', 'mesh', 'pub_type', 'keywords', 'abstract', 'no_references', 'no_authors', 'no_chemicals', 'list_of_chemicals', 'nct_identifiers', 'all_caps', 'all_caps_title']
        complete_features = complete[features]
        missing = complete_features[features].isna().all(axis=1)
        missing_ids = list(complete_features[missing].index)

        if missing_ids:
            print(f'INFO: Re-downloading errors; {len(missing_ids)} failed articles...')
            print(missing_ids)

            batch_length = math.ceil(len(missing_ids) / 9900)
            batched_ids = batch(missing_ids, 9900)
            run_pass(batched_ids, out_file_2, field_names, batch_length, secret_variables.email)
        
            # merge data together
            data2merge = pd.read_csv(out_file_2)

            data2merge.rename(columns={'pmid': 'ids'}, inplace=True)
            data2merge['ids'] = data2merge['ids'].astype(str)
            data2merge.set_index('ids', inplace=True)
            field_names.remove('pmid')
            complete_features.loc[complete_features.index.isin(data2merge.index), field_names] = data2merge[field_names]
            
            complete_features = complete_features.reset_index()
            complete_features.rename(columns={'ids': 'pmid'}, inplace=True)

            complete_features.to_csv('pubmed/pubmed_data.csv')
        else:
            print('INFO: No errors. All articles downloaded.')
