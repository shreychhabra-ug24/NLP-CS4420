import pandas as pd
import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Load spaCy's NER model
nlp = spacy.load("en_core_web_sm")

# Sample function for entity resolution using fuzzy matching
def entity_resolution(entities):
    resolved = {}

    for entity in entities:
        if not resolved:  # First entity, add it directly
            resolved[entity] = entity
            continue
        
        match = process.extractOne(entity, resolved.keys(), scorer=fuzz.token_sort_ratio)

        if match is None:  # Handle case where no match is found
            resolved[entity] = entity
            continue

        best_match, score = match

        if score > 80:  # Adjust threshold as needed
            resolved[entity] = best_match
        else:
            resolved[entity] = entity  # No close match, keep as new entry

    return resolved

# Function to extract named entities
def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

df = pd.read_csv("ozempic_news_ner.csv")  

# Select only 15 articles for ER
df_sample = df.head(15).copy()

# Extract entities from each article
df_sample["entities"] = df_sample["Text"].apply(extract_entities)

# Flatten list of all entities and apply entity resolution
all_entities = [entity for sublist in df_sample["entities"] for entity in sublist]
resolved_entities = entity_resolution(all_entities)

# Store the resolved entities back into the DataFrame
df_sample["resolved_entities"] = df_sample["entities"].apply(lambda ents: [resolved_entities.get(ent, ent) for ent in ents])

# Save results
df_sample.to_csv("resolved_entities_sample.csv", index=False)

print("Entity Resolution completed on 15 articles and saved to 'resolved_entities_sample.csv'.")
