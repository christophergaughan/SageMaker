import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset  # Fixed spelling (Dataloader -> DataLoader)
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd
from encodings import big5
from pyspark.examples.src.main.python.ml.aft_survival_regression import training

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

# Adding custom layers (like torch.nn.Linear) gives the flexibility to fine-tune the model for specific tasks. We are trying to parse text into 4 categories
class DistilBERTClass(torch.nn.Module):
    
    def __init__(self):
        # call the torch.nn.Module
        super(DistilBERTClass, self).__init__()
        # l1 is a layer and we are building the model on- here loading the distilbert-base-uncased model- case insensitive, backbone of neural network
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # fully connected linear layer, transform the 768 features, will result in almost 600000 trainable weights and biases
        self.pre_classifier = torch.nn.Linear(768, 768)  # Fixed typo (pre_classifier -> pre_classifier)
        # randomly drop 30% of the neurons, regularization method to prevent overfitting, i.e. the model doesn't over-rely on a few neurons
        self.dropout = torch.nn.Dropout(0.3)
        # Final classifier layer, maps feature vector into our 4 output classes above- output logits (raw-output)
        # NOTE: Don't get confused by the Linear term below, it's just a It’s a building block of neural networks, not limited to regression tasks.  It's a fully connected layer in PyTorch
        self.classifier = torch.nn.Linear(768, 4)  # Fixed typo (classifier -> classifier)
        
    # Create the forward method on the DistilBERTClass class, every PyTorch model needs a forward method, we will call methods above
    def forward(self, input_ids, attention_mask):
        # output of the distilbert model, we are not using the hidden states, only the output, attention mask tells model what to focus on
        # remember padding allows some homogeneity in terms of our attention mask size
        # we are calling self.l1 NOT the distilbert model, this would disrupt our forward flow of weight and biases
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        # This represents the final output of the transformer layer of distilbert, output of self.l1 above and saving it as the hidden state
        hidden_state = output_1[0]
        # pool the hidden state to get a single vector, slicing gets hidden state associated with first token CLS
        # CLS token is specifically used in classification tasks as it encapsulates what follows in sentence via attention mechanism
        pooler = hidden_state[:, 0]
        # pass the pooled output to the pre_classifier layer, calling the self.classifier above
        pooler = self.pre_classifier(pooler)
        # calling the ReLu activation function, introduces non-linearity into model, helps with learning complex patterns, mitigates vanishing gradient
        pooler = torch.nn.ReLU()(pooler)
        # we're just passing the previous output to the dropout layer
        pooler = self.dropout(pooler)
        # pass the output to the classifier layer- we're just getting probabilities here- high numbers/higher probability
        output = self.classifier(pooler)
        return output
    
# Define the function to calculate the accuracy of the model
# big_idx is like the guesses of the model, targets are the actual values
def calculate_accu(big_idx, targets):
    # below we get the max logit value- which will be represented as a 1 and the rest as zeros eg [1, 0, 0, 0]
    n_correct = (big_idx==targets).sum().item()
    '''
    here is an example of output logits to various categories and print statements to help visualize the process
    [0.88, 0.1, 0.33, 0.7] I love The Office 1 0 0 0 target 1 0 0 0
    [0.99, 0.04, 0.5, 0.77] Mash is a great show 1 0 0 0 target 1 0 0 0
    [0.38, 0.12, 0.10, 0.88] Musk lands on Mars 0 0 0 1 target 0 0 0 1
    [0.12, 0.10, 0.71, 0.52] Breakthrough in cancer vaccine 0 0 1 0 target 0 0 1 0
    we could put in a print statement like
    print(big_idx == targets) and we'd get a tensor like ([True, True, True, True])
    print(big_idx == targets).sum() would give us tensor(4)
    print(big_idx == targets).sum().item() would give us the integer 4
    '''
    return n_correct

def train(epoch, model, device, training_loader, optimizer, loss_function)):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        # remember in the forward function we had input_ids and attention_mask
        outputs = model(ids, mask)
        # the outputs are going to be the raw logits from the outputs returned above
        loss = loss_function(outputs, targets)
        # add up each instance of loss from each epoch
        tr_loss += loss.item()
        # finds predicted class labels based on the highest output score, dim=1 means across columns, remember above about the max value in logits
        big_val, big_idx = torch.max(outputs.data, dim=1)
        # we call the calculate_accu function above, an integer between 0 and 4
        n_correct += calculate_accu(big_idx, targets)

        nb_tr_steps += 1
        # getting the zeroth element of the shape of the targets tensor
        nb_tr_examples += targets.size(0)
        # every time the steps is divisible by 5000
        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()