
import json
import spacy
import argparse
from scispacy.linking import EntityLinker

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="outputs/pairs.jsonl",
                        help="Path to the JSONL file containing prediction and reference pairs")
    parser.add_argument("--ref", default="ref", help="JSONL field name for reference abstract")
    parser.add_argument("--id", default="article_id", help="JSONL field name for article ID")
    parser.add_argument("--pred", default="pred", help="JSONL field name for predicted abstract")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Number of top UMLS candidates to consider per entity")
    args = parser.parse_args()

    print("Loading scispacy model (en_core_sci_md)...")
    try:
        nlp = spacy.load("en_core_sci_md")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})


    def get_umls_cuis(text, top_k=1):
        doc = nlp(text)
        cuis = set()
        for ent in doc.ents:
            if ent._.kb_ents:
                candidates = [cui for cui, score in ent._.kb_ents[:top_k]]
                cuis.update(candidates)
        return cuis

    input_file = args.pairs
    print(f"Reading data from: {input_file}")
    
    data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data = json.load(f)
            else:
                data = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        return

    print(f"Total entries loaded: {len(data)}")
    
    if len(data) == 0:
        print("File is empty!")
        return


    valid_samples = 0
    total_recall = 0

    ids = []
    raw_recall = []
    raw_ref_count = []
    raw_intersection_count = []
    
    for i, entry in enumerate(data): 
        ref_text = entry.get(args.ref, "")
        gen_text = entry.get(args.pred, "")
        
        if not ref_text: continue
        
        ents_ref = get_umls_cuis(ref_text, top_k=args.top_k)
        ents_gen = get_umls_cuis(gen_text, top_k=args.top_k)
        
        if len(ents_ref) > 0:
            intersection = ents_ref.intersection(ents_gen)
            recall = len(intersection) / len(ents_ref)
            total_recall += recall
            valid_samples += 1
            ids.append(str(entry.get(args.id)))
            raw_recall.append(float(recall))
            raw_ref_count.append(len(ents_ref))
            raw_intersection_count.append(len(intersection))
            
    print(f"Valid Samples: {valid_samples}")
    if valid_samples > 0:
        print(f"Avg UMLS Recall: {total_recall / valid_samples:.4f}")
    
    out_path = args.pairs.replace(".jsonl", "_umls_recall_results.json")
    
    with open(out_path, "w") as f:
        json.dump({
            "ids": ids,
            "raw_umls_recall": raw_recall,
            "raw_ref_cui_count": raw_ref_count,
            "raw_intersection_count": raw_intersection_count,
        }, f, indent=4)
    print(f"Detailed scores saved to {out_path}")

if __name__ == "__main__":
    main()