#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import transformers as t
import datasets
import torch
import pandas as pd


# In[ ]:


def create_pairs_list(csv_file):
    # Load the DataFrame from the CSV file
    df = pd.read_csv(csv_file)

    # Extract the 'Other Character Lines' and 'Donkey Lines' columns
    others_lines = df['Other Character Lines'].dropna().tolist()
    donkey_lines = df['Donkey Lines'].dropna().tolist()

    # Create a list of tuples from the columns
    pairs_list = list(zip(others_lines, donkey_lines))

    return pairs_list

# Example usage:
# Assuming df is your DataFrame
pairs_list = create_pairs_list('donkey_data.csv')

# Display the list
print(pairs_list[100])


# In[ ]:


class DonkeyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
        self.ds = create_pairs_list('donkey_data.csv')
        

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx): 
        #'prompt'
        TEMPLATE = "Below is something that a person has said to you. Write a response to that person.\n\n### Line:\n{line}\n\n### Response:\n"
        pair = self.ds[idx]
        prompt = TEMPLATE.format(line = pair[0])
        prompt = prompt + pair[1]
        #tokenize
        res = self.tokenizer(prompt)
        res["input_ids"].append(self.tokenizer.eos_token_id)
        res["attention_mask"].append(1)
        res["labels"] = res["input_ids"].copy()
        return res

    def max_sequence_length(self):
        return max(len(pair[0] + pair[1]) for pair in self.ds)


# In[ ]:


if __name__ == '__main__':
    
    traindataset = DonkeyDataset()
    i = torch.randint(0, 195, (1,)).item()
    print('max_seq_len:', traindataset.max_sequence_length())
    print('ith input:', traindataset[i])

