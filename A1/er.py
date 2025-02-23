import pandas as pd
import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

nlp = spacy.load("en_core_web_sm")

#fuzzy matching er
def entity_resolution(entities):
    resolved = {}

    for entity in entities:
        if not resolved:  # First entity, add it directly
            resolved[entity] = entity
            continue
        
        match = process.extractOne(entity, resolved.keys(), scorer=fuzz.token_sort_ratio)

        if match is None:  #edge case
            resolved[entity] = entity
            continue

        best_match, score = match

        if score > 90:  # Adjust threshold as needed
            resolved[entity] = best_match
        else:
            resolved[entity] = entity  # No close match, keep as new entry

    return resolved

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

df = pd.read_csv("ozempic_news_ner.csv")  

#15 articles for er
df_sample = df.head(15).copy()
df_sample["entities"] = df_sample["Text"].apply(extract_entities)

# Flatten list of all entities and apply er
all_entities = [entity for sublist in df_sample["entities"] for entity in sublist]
resolved_entities = entity_resolution(all_entities)

df_sample["resolved_entities"] = df_sample["entities"].apply(lambda ents: [resolved_entities.get(ent, ent) for ent in ents])
df_sample.to_csv("resolved_entities_sample.csv", index=False)

print("Entity Resolution completed on 15 articles and saved to 'resolved_entities_sample.csv'.")
