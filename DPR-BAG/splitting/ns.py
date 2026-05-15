# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from common import LABELS, DATASET_REGISTRY, load_hf_dataset, clean_for_json
import re
import json

def distribute_paragraphs_to_six_facets(text):
    facets = [""] * 6
    if not isinstance(text, str) or not text.strip():
        return facets
    
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
    # print(f"Total paragraphs: {len(paragraphs)}")
    if not paragraphs:
        return facets
    
    total_len = sum(len(p) for p in paragraphs)
    cutoffs = [(total_len * i) // 6 for i in range(1, 6)]
    
    current_facet_idx = 0
    cumulative_len = 0
    facet_contents = [[] for _ in range(6)]
    
    for p in paragraphs:
        p_len = len(p.strip())
        if current_facet_idx < 5 and cumulative_len >= cutoffs[current_facet_idx] and facet_contents[current_facet_idx]:
            current_facet_idx += 1
            
        facet_contents[current_facet_idx].append(p.strip())
        cumulative_len += p_len
    
    for i in range(6):
        facets[i] = "\n\n".join(facet_contents[i])
        
    return facets

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()),
                        help="Dataset name. See DATASET_REGISTRY for available options.")
    parser.add_argument("--output", required=True,
                        help="Output JSONL with 6 facet columns appended.")
    parser.add_argument("--split", default="test",
                        help="Dataset split to process (default: test).")
    parser.add_argument("--cache-dir", default=None,
                        help="Optional HuggingFace datasets cache directory.")
    args = parser.parse_args()

    output_file = args.output

    full_test_ds = load_hf_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    total_rows = len(full_test_ds)

    print(f"Total rows to process: {total_rows}")


    for i, row in enumerate(full_test_ds):
        # print(len(row['sections']))
        six_facets = distribute_paragraphs_to_six_facets('\n\n'.join(row['sections']))
        dict_row = {
            "first_facet": six_facets[0].strip() if six_facets[0].strip() else "",
            "second_facet": six_facets[1].strip() if six_facets[1].strip() else "",
            "third_facet": six_facets[2].strip() if six_facets[2].strip() else "",
            "fourth_facet": six_facets[3].strip() if six_facets[3].strip() else "",
            "fifth_facet": six_facets[4].strip() if six_facets[4].strip() else "",
            "sixth_facet": six_facets[5].strip() if six_facets[5].strip() else "",
        }
        
        save_data = dict(row)
        save_data.pop('sections', None)
        save_data.pop('section_names', None)
        save_data.update(dict_row)
        clean_data = clean_for_json(save_data)
        with open(output_file, 'a', encoding='utf-8') as f_out:
            json.dump(clean_data, f_out, ensure_ascii=False)
            f_out.write('\n') 
            f_out.flush() 
        
        
        current_idx = i + 1
        if current_idx % 10 == 0:
            print(f"Processed {current_idx}/{total_rows} rows...")
    
    

