import pandas as pd
from stanza.server import CoreNLPClient

#using a neural model instead of the statistical standard model for better accuracy
client = CoreNLPClient(annotators=["tokenize", "ssplit", "pos", "lemma", "ner", "parse", "coref"], properties={'coref.algorithm': 'neural'}, timeout=600000, memory="8G")

def resolve_coreferences(text):
    if not isinstance(text, str) or text.strip() == "":
        return text  # Skip empty or non-string values

    ann = client.annotate(text)
    
    #extracts coref chains
    coref_chains = {}
    for chain in ann.corefChain:
        representative = chain.mention[0]
        rep_text = text.split()[representative.beginIndex:representative.endIndex]
        rep_text = " ".join(rep_text)

        for mention in chain.mention[1:]:  # Skip the representative
            mention_text = text.split()[mention.beginIndex:mention.endIndex]
            mention_text = " ".join(mention_text)
            coref_chains[mention_text] = rep_text

    #replaces mentions with representative names
    resolved_text = text
    for mention, rep in coref_chains.items():
        resolved_text = resolved_text.replace(mention, rep)

    return resolved_text


df = pd.read_csv("ozempic_news_150.csv")  

# Apply coreference resolution
df["resolved_text"] = df["Text"].apply(resolve_coreferences)

# Save the updated DataFrame
df.to_csv("ozempic_neural_coref.csv", index=False)

print("Coreference resolution completed and stored in 'resolved_text' column.")
