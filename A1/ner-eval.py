import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import ast  


manual_ner_path = "annotated-ner-manual.json"
with open(manual_ner_path, "r", encoding="utf-8") as f:
    manual_annotations = json.load(f)

true_entities = defaultdict(list)
text_data = {}

for annotation in manual_annotations:
    article_id = annotation["id"]
    text_data[article_id] = annotation["data"]["text"]  
    
    for result in annotation["annotations"][0]["result"]:
        entity_text = result["value"]["text"]
        entity_label = result["value"]["labels"][0]
        true_entities[article_id].append((entity_text, entity_label))

#loads ner preds from csv
ner_csv_path = "ozempic_news_ner.csv"
df_ner = pd.read_csv(ner_csv_path)

#Extracts predicted entities from Named_Entities column
predicted_entities = {}

for index, row in df_ner.iterrows():
    article_id = index  # Assuming row index matches manual annotation ID
    if isinstance(row["Named_Entities"], str):
        try:
            pred_ents = ast.literal_eval(row["Named_Entities"])  
            predicted_entities[article_id] = pred_ents if isinstance(pred_ents, list) else [pred_ents]
        except (SyntaxError, ValueError):
            predicted_entities[article_id] = []
    else:
        predicted_entities[article_id] = []

# Flatten lists for evaluation
true_labels = []
pred_labels = []

for article_id in true_entities:
    true_set = set(true_entities[article_id])
    pred_set = set(predicted_entities.get(article_id, []))
    
    # True Positives
    for entity in true_set & pred_set:
        true_labels.append(entity[1])
        pred_labels.append(entity[1])
    
    # False Negatives (missed entities)
    for entity in true_set - pred_set:
        true_labels.append(entity[1])
        pred_labels.append("O")  # No entity detected
    
    # False Positives (incorrect predictions)
    for entity in pred_set - true_set:
        true_labels.append("O")
        pred_labels.append(entity[1])

precision = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
recall = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
