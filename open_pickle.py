import pickle

# For config.pkl
with open("headline_model/config.pkl", "rb") as f:
    config = pickle.load(f)
print("Config:", config)

# For tokenizers
with open("headline_model/headline_tokenizer.pkl", "rb") as f:
    headline_tokenizer = pickle.load(f)
print("Headline tokenizer word index:", headline_tokenizer.word_index)

with open("headline_model/stats_tokenizer.pkl", "rb") as f:
    stats_tokenizer = pickle.load(f)
print("Stats tokenizer word index:", stats_tokenizer.word_index)
