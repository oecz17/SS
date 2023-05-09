# Databricks notebook source
# MAGIC %md
# MAGIC ## Install and import libraries

# COMMAND ----------

# MAGIC %pip install --quiet datasets
# MAGIC %pip install --quiet nltk
# MAGIC %pip install --quiet rouge_score
# MAGIC %pip install --quiet evaluate
# MAGIC 
# MAGIC %pip install --quiet accelerate
# MAGIC %pip install --quiet bitsandbytes
# MAGIC %pip install --quiet git+https://github.com/huggingface/transformers.git

# COMMAND ----------

import time
import torch
import evaluate
import numpy as np
import pandas as pd

from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorWithPadding

import nltk
from datasets import load_metric
nltk.download('punkt')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data and model

# COMMAND ----------

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("HHousen/distil-led-large-cnn-16384")
model = AutoModelForSeq2SeqLM.from_pretrained("/dbfs/FileStore/descriptions_gold/").to(device)

# COMMAND ----------

model_name = 'descriptions'
gold_set = spark.sql(f'select * from {model_name}.gold_set').toPandas()
gold_set.shape

# COMMAND ----------

def split_dataset(dataset):
  train_df, validation_df = train_test_split(dataset, test_size=0.2, random_state=42)
  new_test_samples_df, eval_df = train_test_split(validation_df, test_size=0.5, random_state=42)

  print("Training set split: {:.0%}".format(train_df.shape[0]/dataset.shape[0]))
  print("Validation set split: {:.0%}".format(eval_df.shape[0]/dataset.shape[0]))
  print("Test set split: {:.0%}".format(new_test_samples_df.shape[0]/dataset.shape[0]))

  return train_df, eval_df, new_test_samples_df

# COMMAND ----------

train_df, eval_df, test_df = split_dataset(gold_set)

test_companyid = test_df.sort_values(by=['paraphrase_model'], ascending=False)['companyid'][:250].tolist()
test_df = test_df[test_df['companyid'].isin(test_companyid)]
print(train_df.shape, eval_df.shape, test_df.shape)

test_df = test_df.reset_index(drop=True).drop(columns=['name', 'companyid', 'paraphrase_model'])
test_df = Dataset.from_pandas(test_df).select(range(50))
test_df.shape

# COMMAND ----------

def tokenize_function(example):
  tokenized=tokenizer(example["fulltext"], truncation=True, max_length=4096)
  return tokenized

tokenized_datasets = test_df.map(tokenize_function, batched=True)
tokenized_datasets

# COMMAND ----------

data_collator =  DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

tokenized_datasets = tokenized_datasets.remove_columns(["fulltext", "description"])
#tokenized_datasets = tokenized_datasets.rename_column("description", "labels")
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
tokenized_datasets.column_names

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test predictions with dataloader, batch = 1

# COMMAND ----------

test_dataloader = DataLoader(
    tokenized_datasets, batch_size=1, collate_fn=data_collator
)

# COMMAND ----------

for batch in test_dataloader:
    break
{k: v.shape for k, v in batch.items()}

# COMMAND ----------

metric = evaluate.load("rouge")
model.eval()
preds = []
curr2 = time.time()
for batch in test_dataloader:
    curr = time.time()
    batch = {k: v.to(device) for k, v in batch.items()}
    
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    global_attention_mask = torch.zeros_like(batch['attention_mask'])
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask.to(device))
    predicted_description = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(time.time()-curr)
    preds.extend(predicted_description)
    torch.cuda.empty_cache() 

print('Full time: ',time.time()-curr2)
print(metric.compute(predictions=preds,
                  references=test_df['description']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test predictions with .map, batch = 1

# COMMAND ----------

def generate_answer(batch):
  curr = time.time()
  inputs_dict = tokenizer(batch["fulltext"], max_length=4096,  return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to(device)
  attention_mask = inputs_dict.attention_mask.to(device)
  global_attention_mask = torch.zeros_like(attention_mask)
  # put global attention on <s> token
  global_attention_mask[:, 0] = 1

  predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
  batch["predicted_des"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  print(time.time()-curr)
  torch.cuda.empty_cache() 
  return batch

curr2 = time.time()
result = test_df.map(generate_answer, batched=True, batch_size=1)
print('Full time: ',time.time()-curr2)
# load rouge

print(metric.compute(predictions=result["predicted_des"],
                  references=test_df['description']))

# COMMAND ----------

print(metric.compute(predictions=result["predicted_des"],
                  references=test_df['description']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test predictions with dataloader, batch = 1, cache=True

# COMMAND ----------

model.config.use_cache=True
#model.config.task_specific_params.length_penalty=-1
#model.config.task_specific_params.num_beams=3

# COMMAND ----------

test_dataloader = DataLoader(
    tokenized_datasets, batch_size=1, collate_fn=data_collator
)

# COMMAND ----------

metric = evaluate.load("rouge")
model.eval()
preds = []
curr2 = time.time()
for batch in test_dataloader:
    curr = time.time()
    batch = {k: v.to(device) for k, v in batch.items()}
    
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    global_attention_mask = torch.zeros_like(batch['attention_mask'])
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask.to(device))
    predicted_description = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(time.time()-curr)
    preds.extend(predicted_description)
    torch.cuda.empty_cache() 
print('Full time: ',time.time()-curr2)
print(metric.compute(predictions=preds,
                  references=test_df['description']))



# COMMAND ----------

# MAGIC %md
# MAGIC ## Test predictions with dataloader, batch = 4, cache=True

# COMMAND ----------

test_dataloader = DataLoader(
    tokenized_datasets, batch_size=4, collate_fn=data_collator
)

# COMMAND ----------

metric = evaluate.load("rouge")
model.eval()
preds = []
curr2 = time.time()
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    global_attention_mask = torch.zeros_like(batch['attention_mask'])
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask.to(device))
    predicted_description = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    preds.extend(predicted_description)
    torch.cuda.empty_cache() 
print('Full time: ',time.time()-curr2)

print(metric.compute(predictions=preds,
                  references=test_df['description']))



# COMMAND ----------

# MAGIC %md
# MAGIC ## Test predictions with .map, batch = 2, cache=True

# COMMAND ----------

curr2 = time.time()
result = test_df.map(generate_answer, batched=True, batch_size=2)
print('Full time: ',time.time()-curr2)
# load rouge

print(metric.compute(predictions=result["predicted_des"],
                  references=test_df['description']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test predictions with DataLoader, batch = 1, cache=True, quantization

# COMMAND ----------

import torch.nn as nn

import bitsandbytes as bnb
from bitsandbytes.nn import Linear8bitLt

# COMMAND ----------

model_8bit = AutoModelForSeq2SeqLM.from_pretrained("/dbfs/FileStore/descriptions_gold/", device_map="auto", load_in_8bit=True)

# COMMAND ----------

model_8bit.get_memory_footprint()

# COMMAND ----------

model_8bit.config.use_cache=True
#model.config.task_specific_params.length_penalty=-1
#model.config.task_specific_params.num_beams=3

test_dataloader = DataLoader(
    tokenized_datasets, batch_size=1, collate_fn=data_collator
)

# COMMAND ----------

metric = evaluate.load("rouge")
model_8bit.eval()
preds = []
curr2 = time.time()
for batch in test_dataloader:
    curr = time.time()
    batch = {k: v.to(device) for k, v in batch.items()}
    
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    global_attention_mask = torch.zeros_like(batch['attention_mask'])
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    
    with torch.no_grad():
        outputs = model_8bit.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask.to(device))
    predicted_description = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(time.time()-curr)
    preds.extend(predicted_description)
    torch.cuda.empty_cache() 
print('Full time: ',time.time()-curr2)
print(metric.compute(predictions=preds,
                  references=test_df['description']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test predictions with DataLoader, batch = 4, cache=True, ordered dataset

# COMMAND ----------

model.config.use_cache=True

# COMMAND ----------

def tokenize_function(example):
  tokenized=tokenizer(example["fulltext"], truncation=True, max_length=4096)
  return tokenized

tokenized_datasets = test_df.map(tokenize_function, batched=True)
tokenized_datasets

# COMMAND ----------

len_input = [len(t) for t in tokenized_datasets['input_ids']]
tokenized_datasets = tokenized_datasets.add_column('length',len_input)

data_collator =  DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

tokenized_datasets = tokenized_datasets.remove_columns(["fulltext", "description"])
#tokenized_datasets = tokenized_datasets.rename_column("description", "labels")
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
sorted_ds = tokenized_datasets.sort('length')
sorted_ds.column_names


# COMMAND ----------

sorted_ds['input_ids'][6].shape

# COMMAND ----------

test_dataloader = DataLoader(
    sorted_ds, batch_size=4, collate_fn=data_collator
)

# COMMAND ----------

metric = evaluate.load("rouge")
model.eval()
preds = []
curr2 = time.time()
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    global_attention_mask = torch.zeros_like(batch['attention_mask'])
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask.to(device))
    predicted_description = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    preds.extend(predicted_description)
    torch.cuda.empty_cache() 
print('Full time: ',time.time()-curr2)

# COMMAND ----------

l_input = [len(input) for input in sorted_ds['input_ids']]
import matplotlib.pyplot as plt 
    
plt.plot(l_input) 
plt.title('Ordered length of test set') 
plt.draw() 
plt.show() 

# COMMAND ----------


