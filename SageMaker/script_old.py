import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset  # Fixed spelling (Dataloader -> DataLoader)
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

# Change the 'CATEGORY' column's cryptic symbols to accord with more readable dictionary. 'x' in the function represents symbols

# # Redundant section (for demonstration purposes only):
# def update_cat(x):
#     return my_dict[x]

# # Update the CATEGORY column
# df['CATEGORY'] = df['CATEGORY'].apply(lambda x: update_cat(x))  

# print(df)

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
        title = str(self.data.TITLE[index])  # Removed the incorrect dot after TITLE
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,  # Fixed typo (add_special_token -> add_special_tokens)
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True  # Fixed typo (truncations -> truncation)
        )
        ids = inputs['input_ids']  # Changed input to inputs
        mask = inputs['attention_mask']  # Changed input to inputs

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)
        }

    def __len__(self):
        return self.len

train_size = 0.8
train_dataset = df.sample(frac=train_size, random_state=200)  # we are choosing 80% of the data at random
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)  # we are only using the indices NOT in the training indices

print(f'Full dataset: {df.shape}')
print(f'Train dataset: {train_dataset.shape}')
print(f'Test dataset: {test_dataset.shape}')

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2

training_set = NewsDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = NewsDataset(test_dataset, tokenizer, MAX_LEN)  # Added MAX_LEN as a parameter

# we will not be doing any parallel data loading to keep things cheap, thus num_workers = 0
train_parameters = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0,
}

test_parameters = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0,
}

training_loader = DataLoader(training_set, **train_parameters)  # Fixed Dataloader to DataLoader
testing_loader = DataLoader(testing_set, **test_parameters)  # Fixed Dataloader to DataLoader


