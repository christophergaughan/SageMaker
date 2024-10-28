import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd

s3_path = 's3://hugging-face-text-multiclass-text-classification-bucket/training_data/newsCorpora.csv'
df = pd.read_csv(s3_path, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

# remember we want to classify the title based on the category

df = df[['TITLE', 'CATEGORY']]


my_dict = {
    'e':'Entertainment',
    'b':'Business',
    't':'Science',
    'm':'Health'
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
