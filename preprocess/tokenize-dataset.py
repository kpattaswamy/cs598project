from bert_tokenizer import get_impressions_from_csv, tokenize
from transformers import BertTokenizer
import json

csv_path = "../constructed-data/clean_dataset.csv"
output_path = "../constructed-data/encoded_impressions.json"

impressions = get_impressions_from_csv(csv_path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

encoded = tokenize(impressions, tokenizer)

with open(output_path, "w") as f:
    json.dump(encoded, f)
