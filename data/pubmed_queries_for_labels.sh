#!/bin/sh

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated)" | efetch -format uid > labels/all_possible_ids.txt

esearch -db pubmed -query "((1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND humans[MeSH Terms]) AND clinical trial[pt] NOT case-control studies[mh] NOT cohort studies[mh] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT review[pt]" | efetch -format uid > labels/cluster_ids/clinical_trial_ids.txt

esearch -db pubmed -query "((1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND humans[MeSH Terms]) AND randomized controlled trial [pt] NOT case-control studies[MeSH Terms] NOT cohort studies[MeSH Terms] NOT review[pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT review[pt]" | efetch -format uid > labels/cluster_ids/randomized_controlled_trial_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND humans[MeSH Terms] AND clinical trial protocol[pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT case-control studies[mh] NOT cohort studies[mh] NOT review[pt]" | efetch -format uid > labels/cluster_ids/clinical_trial_protocol_ids.txt

esearch -db pubmed -query "((1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND humans[MeSH Terms]) AND Clinical Study [Publication Type] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT review[pt]" | efetch -format uid > labels/cluster_ids/clinical_study_ids.txt

esearch -db pubmed -query "((1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND humans[MeSH Terms]) AND Clinical Studies as Topic [mh]" | efetch -format uid > labels/cluster_ids/clinical_studies_as_topic_ids.txt

esearch -db pubmed -query "((1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND humans[MeSH Terms]) AND Case Reports [Publication Type] NOT editorial[Publication Type]" | efetch -format uid > labels/cluster_ids/case_reports_ids.txt

esearch -db pubmed -query "((1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND humans[MeSH Terms]) AND Human Experimentation[mh:noexp]" | efetch -format uid > labels/cluster_ids/human_experimentation_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND cross-over study[mh] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT review[pt]" | efetch -format uid > labels/cluster_ids/cross_over_study_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND humans[MeSH Terms] AND cohort studies[mh:noexp] NOT case-control studies[MeSH Terms] NOT clinical trial [pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT review[pt]" | efetch -format uid > labels/cluster_ids/cohort_study_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND humans[MeSH Terms] AND case-control studies[MeSH Terms:noexp] NOT cohort studies[MeSH Terms] NOT clinical trial [pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT review[pt]" | efetch -format uid > labels/cluster_ids/case_control_study_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Prospective studies[mh:noexp] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT review[pt]" | efetch -format uid > labels/cluster_ids/prospective_studies_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Retrospective studies[mh:noexp] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT review[pt]" | efetch -format uid > labels/cluster_ids/retrospective_studies_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Longitudinal Studies[mh:noexp] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT review[pt]" | efetch -format uid > labels/cluster_ids/longitudinal_studies_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Follow-Up Studies[mh:noexp] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT review[pt]" | efetch -format uid > labels/cluster_ids/follow_up_studies_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND humans[MeSH Terms] AND cross-sectional studies [mh] NOT Longitudinal Studies[mh:noexp] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT review[pt]" | efetch -format uid > labels/cluster_ids/cross_sectional_study_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Double-Blind Method[mh:noexp]" | efetch -format uid > labels/cluster_ids/double_blind_method_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Random Allocation[mh:noexp]" | efetch -format uid > labels/cluster_ids/random_allocation_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Matched-Pair Analysis[mh:noexp]" | efetch -format uid > labels/cluster_ids/matched_pair_analysis_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Twin Study [Publication Type]" | efetch -format uid > labels/cluster_ids/twin_study_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND genome-wide association study[mh]" | efetch -format uid > labels/cluster_ids/genome_wide_association_study_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Clinical Conference [Publication Type] NOT Case Reports [Publication Type]" | efetch -format uid > labels/cluster_ids/clinical_conference_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Multicenter Study [Publication Type]" | efetch -format uid > labels/cluster_ids/multicenter_study_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND practice guideline[Publication Type] NOT practice guidelines as topic[MeSH Terms]" | efetch -format uid > labels/cluster_ids/practice_guideline_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Meta-Analysis [Publication Type] NOT Meta-Analysis as Topic[mh]" | efetch -format uid > labels/cluster_ids/meta_analysis_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Review [Publication Type]" | efetch -format uid > labels/cluster_ids/review_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND systematic review[pt]" | efetch -format uid > labels/cluster_ids/systematic_review_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND systematic review[all fields] or systematic reviews[all fields] AND Review Literature as Topic[mh]" | efetch -format uid > labels/cluster_ids/systematic_review_as_topic_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Reproducibility of Results[mh] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT review[pt]" | efetch -format uid > labels/cluster_ids/reproducibility_of_results_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Published Erratum [Publication Type]" | efetch -format uid > labels/cluster_ids/published_erratum_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Retraction of Publication [Publication Type]" | efetch -format uid > labels/cluster_ids/retraction_of_publication_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Expression of Concern [Publication Type]" | efetch -format uid > labels/cluster_ids/expression_of_concern_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Scientific Integrity Review [Publication Type]" | efetch -format uid > labels/cluster_ids/scientific_integrity_review_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Focus Groups[mh:noexp]" | efetch -format uid > labels/cluster_ids/focus_groups_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Letter [Publication Type]" | efetch -format uid > labels/cluster_ids/letter_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Editorial [Publication Type]" | efetch -format uid > labels/cluster_ids/editorial_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Comment [Publication Type]" | efetch -format uid > labels/cluster_ids/comment_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Autobiography [Publication Type]" | efetch -format uid > labels/cluster_ids/autobiography_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Bibliography [Publication Type]" | efetch -format uid > labels/cluster_ids/bibliography_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Biography [Publication Type]" | efetch -format uid > labels/cluster_ids/biography_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Personal narrative [Publication Type]" | efetch -format uid > labels/cluster_ids/personal_narratives_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Portrait [Publication Type]" | efetch -format uid > labels/cluster_ids/portaits_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Congress [Publication Type]" | efetch -format uid > labels/cluster_ids/congress_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Historical article [Publication Type]" | efetch -format uid > labels/cluster_ids/historical_article_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND News [Publication Type]" | efetch -format uid > labels/cluster_ids/news_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Newspaper Article [Publication Type]" | efetch -format uid > labels/cluster_ids/newspaper_article_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Consensus Development Conference[Publication Type]" | efetch -format uid > labels/cluster_ids/consensus_development_conference_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Lecture [Publication Type]" | efetch -format uid > labels/cluster_ids/lectures_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Legal Case [Publication Type]" | efetch -format uid > labels/cluster_ids/legal_cases_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Interview [Publication Type]" | efetch -format uid > labels/cluster_ids/interview_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Interviews as Topic [mh:noexp]" | efetch -format uid > labels/cluster_ids/interviews_as_topic_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Cross-Cultural Comparison[mh]" | efetch -format uid > labels/cluster_ids/cross_cultural_comparison_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Feasibility Studies[mh:noexp]" | efetch -format uid > labels/cluster_ids/feasibility_studies_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Evaluation Study [Publication Type]" | efetch -format uid > labels/cluster_ids/evaluation_studies_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Evaluation Studies as Topic [mh:noexp]" | efetch -format uid > labels/cluster_ids/evaluation_studies_as_topic_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Validation Study [Publication Type]" | efetch -format uid > labels/cluster_ids/validation_studies_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Predictive Value of Tests[mh:noexp]" | efetch -format uid > labels/cluster_ids/predictive_value_of_tests_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND Veterinary clinical trial [PT] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT case-control studies[mh] NOT cohort studies[mh] NOT review[pt]" | efetch -format uid > labels/cluster_ids/veterinary_clinical_trial_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND veterinary randomized controlled trial [pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT case-control studies[MeSH Terms] NOT cohort studies[MeSH Terms] NOT review[pt]" | efetch -format uid > labels/cluster_ids/veterinary_randomized_controlled_trial_ids.txt

esearch -db pubmed -query "(1987:2023[dp] AND (english[Language] OR english abstract[pt]) NOT indexingmethod_automated) AND observational study, veterinary [pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT practice guideline[pt] NOT review[pt]" | efetch -format uid > labels/cluster_ids/veterinary_observational_study_ids.txt

