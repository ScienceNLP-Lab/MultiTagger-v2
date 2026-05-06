import pandas as pd
import torch
import torch.nn as nn
from common import LABELS, DATASET_REGISTRY, load_hf_dataset, clean_for_json
from sentence_transformers import SentenceTransformer, InputExample, losses
import json
import argparse

class Classifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  

torch.serialization.add_safe_globals([Classifier])

label2id = {
     "methods" : 0,
     "background" : 1,
     "results" : 2,
     "conclusions" : 3, 
     "objective": 4,
    "none": -1
}
id2label = {v: k for k, v in label2id.items()}

threshold = 0.4


def predict(model, new_embeddings, device):
    model.to(device)
    new_embeddings = torch.tensor(new_embeddings, dtype=torch.float32).to(device) 

    with torch.no_grad():  
        outputs = model(new_embeddings)  
        probabilities = torch.softmax(outputs, dim=1)  
        
        predictions = torch.argmax(outputs, dim=1)  
        max_probs, max_indices = torch.max(probabilities, dim=1)  
        predictions = torch.where(max_probs >= threshold, max_indices, torch.tensor(-1, device=outputs.device))

    return predictions.cpu().numpy()



def predict_section_labels(section_names, bert_model, classifier, device):
    """Given a list of raw section header strings, return a list of BOMRC labels."""
    clean_texts = [text.lower() if text is not None else "" for text in section_names]
    embeddings = bert_model.encode(clean_texts)
    pred_ids = predict(classifier, embeddings, device)

    labels = []
    for raw_text, pred in zip(section_names, pred_ids):
        if raw_text is None:
            labels.append("none")
        else:
            labels.append(id2label[pred])
    return labels


def combine_sections_by_label(sections, predicted_labels):
    """Concatenate sections sharing the same predicted BOMRC label."""
    facets = {label: "" for label in LABELS}
    for text, sec in zip(sections, predicted_labels):
        if sec in facets:
            facets[sec] += text
        else:
            facets["none"] += text
    return facets

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY.keys()),
                        help="Dataset name. See DATASET_REGISTRY for available options.")
    parser.add_argument("--output", required=True,
                        help="Output JSONL with 6 facet columns appended.")
    parser.add_argument("--bert-model-path", required=True,
                        help="Path to fine-tuned SentenceTransformer model.")
    parser.add_argument("--classifier-model-path", required=True,
                        help="Path to classifier checkpoint (.pth).")
    parser.add_argument("--split", default="test",
                        help="Dataset split to process (default: test).")
    parser.add_argument("--cache-dir", default=None,
                        help="Optional HuggingFace datasets cache directory.")
    args = parser.parse_args()

    output_file = args.output

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading SentenceTransformer from {args.bert_model_path}...")
    bert_model = SentenceTransformer(args.bert_model_path)
    print(f"Loading classifier from {args.classifier_model_path}...")
    classifier = torch.load(args.classifier_model_path, weights_only=False)
    classifier.eval()

    print("\n=== Models loaded. Starting SH classification ===\n")

    full_test_ds = load_hf_dataset(args.dataset, split=args.split, cache_dir=args.cache_dir)
    total_rows = len(full_test_ds)

    print(f"Total rows to process: {total_rows}")

    for i, row in enumerate(full_test_ds):
        predicted_labels = predict_section_labels(
            row["section_names"], bert_model, classifier, device
        )
        facets = combine_sections_by_label(row["sections"], predicted_labels)

        save_data = dict(row)
        save_data.pop("sections", None)
        save_data.pop("section_names", None)
        save_data.update(facets)
        clean_data = clean_for_json(save_data)

        with open(output_file, "a", encoding="utf-8") as f_out:
            json.dump(clean_data, f_out, ensure_ascii=False)
            f_out.write("\n")
            f_out.flush()

        current_idx = i + 1
        if current_idx % 10 == 0:
            print(f"Processed {current_idx}/{total_rows} rows...")
