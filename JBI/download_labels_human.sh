#!/bin/sh

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND Autobiography[pt]" | efetch -format uid > labels_human/cluster_ids/autobiography_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND Bibliography[pt]" | efetch -format uid > labels_human/cluster_ids/bibliography_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND Biography[pt]" | efetch -format uid > labels_human/cluster_ids/biography_ids.txt

sleep 30

esearch -db pubmed -query $"((1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated))) AND \"Case Reports\"[pt] NOT editorial[pt]" | efetch -format uid > labels_human/cluster_ids/case_reports_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"case-control studies\"[mh:noexp] NOT \"cohort studies\"[mh:noexp] NOT \"clinical trial\"[pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT review[pt]" | efetch -format uid > labels_human/cluster_ids/case-control_studies_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Clinical Conference\"[pt] NOT \"Case Reports\"[pt]" | efetch -format uid > labels_human/cluster_ids/clinical_conference_ids.txt

sleep 30

esearch -db pubmed -query $"((1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated))) AND \"Clinical Studies as Topic\"[mh]" | efetch -format uid > labels_human/cluster_ids/clinical_studies_as_topic_ids.txt

sleep 30

esearch -db pubmed -query $"((1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated))) AND \"Clinical Study\"[pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT review[pt]" | efetch -format uid > labels_human/cluster_ids/clinical_study_ids.txt

sleep 30

esearch -db pubmed -query $"((1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated))) AND \"clinical trial\" [pt] NOT \"case-control studies\"[mh:noexp] NOT \"cohort studies\"[mh:noexp] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT review[pt]" | efetch -format uid > labels_human/cluster_ids/clinical_trial_ids.txt

sleep 30

esearch -db pubmed -query $"((1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated))) AND \"Clinical Trials as Topic\"[mh]" | efetch -format uid > labels_human/cluster_ids/clinical_trials_as_topic_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"clinical trial protocol\"[pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT \"case-control studies\"[mh:noexp] NOT \"cohort studies\"[mh:noexp] NOT review[pt]" | efetch -format uid > labels_human/cluster_ids/clinical_trial_protocol_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"cohort studies\"[mh:noexp] NOT \"case-control studies\"[mh:noexp] NOT \"clinical trial\"[pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT review[pt]" | efetch -format uid > labels_human/cluster_ids/cohort_studies_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND Comment[pt]" | efetch -format uid > labels_human/cluster_ids/comment_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND Congress[pt]" | efetch -format uid > labels_human/cluster_ids/congress_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Consensus Development Conference\"[pt]" | efetch -format uid > labels_human/cluster_ids/consensus_development_conference_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Cross-Cultural Comparison\"[mh]" | efetch -format uid > labels_human/cluster_ids/cross-cultural_comparison_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"cross-over studies\"[mh] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT review[pt]" | efetch -format uid > labels_human/cluster_ids/cross-over_studies_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"cross-sectional studies\"[mh] NOT \"Longitudinal Studies\"[mh:noexp] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT review[pt]" | efetch -format uid > labels_human/cluster_ids/cross-sectional_studies_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt])) AND ((((\"diagnostic accuracy\"[ti] OR \"diagnostic test accuracy\"[ti]) NOT (letter[pt] OR editorial[pt] OR review[pt] OR \"practice guideline\"[pt]))))" | efetch -format uid > labels_human/cluster_ids/diagnostic_test_accuracy_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Double-Blind Method\"[mh:noexp]" | efetch -format uid > labels_human/cluster_ids/double-blind_method_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND Editorial[pt]" | efetch -format uid > labels_human/cluster_ids/editorial_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Evaluation Studies as Topic\"[mh:noexp]" | efetch -format uid > labels_human/cluster_ids/evaluation_studies_as_topic_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Evaluation Study\"[pt]" | efetch -format uid > labels_human/cluster_ids/evaluation_study_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Expression of Concern\"[pt]" | efetch -format uid > labels_human/cluster_ids/expression_of_concern_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Feasibility Studies\"[mh:noexp]" | efetch -format uid > labels_human/cluster_ids/feasibility_studies_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Focus Groups\"[mh:noexp]" | efetch -format uid > labels_human/cluster_ids/focus_groups_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Follow-Up Studies\"[mh:noexp] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT review[pt]" | efetch -format uid > labels_human/cluster_ids/follow-up_studies_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"genome-wide association study\"[mh]" | efetch -format uid > labels_human/cluster_ids/genome-wide_association_study_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Historical article\"[pt]" | efetch -format uid > labels_human/cluster_ids/historical_article_ids.txt

sleep 30

esearch -db pubmed -query $"((1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated))) AND \"Human Experimentation\"[mh:noexp]" | efetch -format uid > labels_human/cluster_ids/human_experimentation_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND Interview[pt]" | efetch -format uid > labels_human/cluster_ids/interview_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Interviews as Topic\"[mh:noexp]" | efetch -format uid > labels_human/cluster_ids/interviews_as_topic_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND Lecture[pt]" | efetch -format uid > labels_human/cluster_ids/lecture_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Legal Case\"[pt]" | efetch -format uid > labels_human/cluster_ids/legal_case_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND Letter[pt]" | efetch -format uid > labels_human/cluster_ids/letter_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Longitudinal Studies\"[mh:noexp] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT review[pt]" | efetch -format uid > labels_human/cluster_ids/longitudinal_studies_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Matched-Pair Analysis\"[mh:noexp]" | efetch -format uid > labels_human/cluster_ids/matched-pair_analysis_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Meta-Analysis\"[pt]" | efetch -format uid > labels_human/cluster_ids/meta-analysis_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Meta-Analysis as Topic\"[mh]" | efetch -format uid > labels_human/cluster_ids/meta-analysis_as_topic_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Multicenter Study\"[pt] NOT review[pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt]" | efetch -format uid > labels_human/cluster_ids/multicenter_study_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND News[pt]" | efetch -format uid > labels_human/cluster_ids/news_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Newspaper Article\"[pt]" | efetch -format uid > labels_human/cluster_ids/newspaper_article_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Personal narrative\"[pt]" | efetch -format uid > labels_human/cluster_ids/personal_narrative_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND Portrait[pt]" | efetch -format uid > labels_human/cluster_ids/portrait_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"practice guideline\"[pt]" | efetch -format uid > labels_human/cluster_ids/practice_guideline_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"practice guidelines as topic\"[mh]" | efetch -format uid > labels_human/cluster_ids/practice_guidelines_as_topic_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Predictive Value of Tests\"[mh:noexp]" | efetch -format uid > labels_human/cluster_ids/predictive_value_of_tests_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Prospective studies\"[mh:noexp] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT review[pt]" | efetch -format uid > labels_human/cluster_ids/prospective_studies_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Published Erratum\"[pt]" | efetch -format uid > labels_human/cluster_ids/published_erratum_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Random Allocation\"[mh:noexp] NOT review[pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt]" | efetch -format uid > labels_human/cluster_ids/random_allocation_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"randomized controlled trial\" [pt] AND humans[mh] NOT \"case-control studies\"[mh:noexp] NOT \"cohort studies\"[mh:noexp] NOT review[pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT \"Randomized Controlled Trial, Veterinary\"[pt]" | efetch -format uid > labels_human/cluster_ids/randomized_controlled_trial_humans_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Reproducibility of Results\"[mh:noexp] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT review[pt]" | efetch -format uid > labels_human/cluster_ids/reproducibility_of_results_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2025[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Retraction of Publication\"[pt]" | efetch -format uid > labels_human/cluster_ids/retraction_of_publication_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Retrospective studies\"[mh:noexp] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT review[pt]" | efetch -format uid > labels_human/cluster_ids/retrospective_studies_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND Review[pt]" | efetch -format uid > labels_human/cluster_ids/review_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt])) AND \"Scientific Integrity Review\"[pt]" | efetch -format uid > labels_human/cluster_ids/scientific_integrity_review_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) AND ((((\"systematic literature review\"[tiab] OR \"systematic review\"[tiab] OR \"search strategy\"[tiab] OR \"cochrane database syst rev\"[ta]) AND (review[pt] OR \"meta-analysis\"[pt]) NOT (letter[pt] OR \"newspaper article\"[pt] OR comment[pt])))))" | efetch -format uid > labels_human/cluster_ids/systematic_review_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND ((\"systematic review\"[all fields] or \"systematic reviews\"[all fields]) AND \"Review Literature as Topic\"[mh])" | efetch -format uid > labels_human/cluster_ids/systematic_reviews_as_topic_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Twin Study\"[pt] NOT review[pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt]" | efetch -format uid > labels_human/cluster_ids/twin_study_ids.txt

sleep 30

esearch -db pubmed -query $"(1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)) AND \"Validation Study\"[pt] NOT review[pt] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt]" | efetch -format uid > labels_human/cluster_ids/validation_study_ids.txt

sleep 30

esearch -db pubmed -query $"1987:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) AND (\"Clinical Trial, Veterinary\"[pt] OR (\"clinical trial\"[pt] AND (dogs[mh] OR cats[mh] OR cattle[mh] OR horses[mh] OR swine[mh] OR chickens[mh] OR rabbits[mh] OR rats[mh] OR sheep[mh] OR mice[mh] OR goats[mh] OR deer[mh] OR geese[mh] OR lizards[mh] OR turtles[mh] OR \"animals, wild\"[mh] OR birds[mh]))) NOT humans[mh] NOT \"infants\"[tiab] NOT \"humans\"[ti] NOT \"patients\"[tiab] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT \"case-control studies\"[mh:noexp] NOT \"cohort studies\"[mh:noexp] NOT review[pt] NOT (indexingmethod_curated OR indexingmethod_automated)" | efetch -format uid > labels_human/cluster_ids/veterinary_clinical_trial_ids.txt

sleep 30

esearch -db pubmed -query $"1987:2023[dp] AND (english[Language] OR english abstract[pt]) AND (\"Randomized Controlled Trial, Veterinary\"[pt] OR  (\"randomized controlled trial\"[pt] AND (dogs[mh] OR cats[mh] OR cattle[mh] OR horses[mh] OR swine[mh] OR chickens[mh] OR rabbits[mh] OR rats[mh] OR sheep[mh] OR mice[mh] OR goats[mh] OR deer[mh] OR geese[mh] OR lizards[mh] OR turtles[mh] OR \"animals, wild\"[mh] OR birds[mh]))) NOT humans[mh] NOT \"infants\"[tiab] NOT \"humans\"[ti] NOT \"patients\"[tiab] NOT editorial[pt] NOT letter[pt] NOT comment[pt] NOT \"practice guideline\"[pt] NOT \"case-control studies\"[mh:noexp] NOT \"cohort studies\"[mh:noexp] NOT review[pt] NOT (indexingmethod_curated OR indexingmethod_automated)" | efetch -format uid > labels_human/cluster_ids/veterinary_randomized_controlled_trial_ids.txt

