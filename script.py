import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd

s3_path = 's3://hugging-face-text-multiclass-text-classification-bucket/training_data/newsCorpora.csv'
df = pd.read_csv(s3_path, sep='\t', names=[
                 'ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

# remember we want to classify the title based on the category

df = df[['TITLE', 'CATEGORY']]


my_dict = {
    'e': 'Entertainment',
    'b': 'Business',
    't': 'Science',
    'm': 'Health'
}

# Change the 'CATEGORY' columns' cryptic symbols to accord with more readable dictionary. x in the function represents symbols


def update_cat(x):
    return my_dict[x]


# Update the CATEGORY column
df_work['CATEGORY'] = df_work['CATEGORY'].apply(lambda x: update_cat(x))

print(df)

# Training the model with 5% of the subset just to make sure the model can train

df = df.sample(frac=0.05, random_state=1)

df = df.reset_index(drop=True)

print(df)

encode_dict = {}


def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]


# Apply the encoding to the 'CATEGORY' column
df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        title = str(self.data.TITLE.[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_token=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncations=True

        )
        ids = input['input_ids']
        mask = input['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)

        }

    def __len__(self):
        return self.len
