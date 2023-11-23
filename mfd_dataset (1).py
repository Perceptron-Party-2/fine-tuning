#!/usr/bin/env python
# coding: utf-8

# In[40]:


import torch
from io import open
import itertools
import json
import transformers as t
import datasets


# In[81]:


def loadAndExtractSentencePairs(fileName):
    lines = {}
    conversations = {}
    qa_pairs = []

    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            lineJson = json.loads(line)

            # Extract fields for line object
            lineObj = {}
            lineObj["lineID"] = lineJson["id"]
            lineObj["characterID"] = lineJson["speaker"]
            lineObj["text"] = lineJson["text"]
            lines[lineObj['lineID']] = lineObj

            # Extract fields for conversation object
            if lineJson["conversation_id"] not in conversations:
                convObj = {}
                convObj["conversationID"] = lineJson["conversation_id"]
                convObj["movieID"] = lineJson["meta"]["movie_id"]
                convObj["lines"] = [lineObj]
            else:
                convObj = conversations[lineJson["conversation_id"]]
                convObj["lines"].insert(0, lineObj)
            conversations[convObj["conversationID"]] = convObj

            # Extract pairs of sentences
            if len(convObj["lines"]) > 1:
                for i in range(len(convObj["lines"]) - 1):
                    inputLine = convObj["lines"][i]["text"].strip()
                    targetLine = convObj["lines"][i+1]["text"].strip()
                    if inputLine and targetLine:
                        qa_pairs.append([inputLine, targetLine])

    return qa_pairs


# In[86]:


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.tokenizer = t.AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
        self.ds = loadAndExtractSentencePairs("utterances.jsonl")
        

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


# In[87]:


if __name__ == '__main__':
    
    traindataset = TrainDataset()
    i = torch.randint(0, 221282, (1,)).item()
    print('max_seq_len:', traindataset.max_sequence_length())
    print('ith input:', traindataset[i])


# In[ ]:




