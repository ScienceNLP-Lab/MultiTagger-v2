#!/bin/sh

echo "1987-1992"

esearch -db pubmed -query $"((1987:1992[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)))" | efetch -format uid > labels_human/all_possible_ids/all_possible_ids_1987_1992.txt

sleep 10

echo "1992-1997"

esearch -db pubmed -query $"((1992:1997[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)))" | efetch -format uid > labels_human/all_possible_ids/all_possible_ids_1992_1997.txt

sleep 10

echo "1997-2002"

esearch -db pubmed -query $"((1997:2002[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)))" | efetch -format uid > labels_human/all_possible_ids/all_possible_ids_1997_2002.txt

sleep 10

echo "2002-2007"

esearch -db pubmed -query $"((2002:2007[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)))" | efetch -format uid > labels_human/all_possible_ids/all_possible_ids_2002_2007.txt

sleep 10

echo "2007-2012"

esearch -db pubmed -query $"((2007:2012[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)))" | efetch -format uid > labels_human/all_possible_ids/all_possible_ids_2007_2012.txt

sleep 10

echo "2012-2017"

esearch -db pubmed -query $"((2012:2017[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)))" | efetch -format uid > labels_human/all_possible_ids/all_possible_ids_2012_2017.txt

sleep 10

echo "2017-2023"

esearch -db pubmed -query $"((2017:2023[dp] AND (english[Language] OR \"english abstract\"[pt]) NOT (indexingmethod_curated OR indexingmethod_automated)))" | efetch -format uid > labels_human/all_possible_ids/all_possible_ids_2017_2023.txt
