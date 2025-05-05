from bert_tokenizer import get_impressions_from_csv, tokenize
from transformers import BertTokenizer
import json

csv_path = "../../clean_dataset.csv"
output_path = "encoded_impressions.json"

# Load impressions and tokenizer
impressions = get_impressions_from_csv(csv_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize (truncates to 512 tokens)
encoded = tokenize(impressions, tokenizer)

# Save to JSON
with open(output_path, "w") as f:
    json.dump(encoded, f)
