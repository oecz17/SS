# Databricks notebook source
# MAGIC %md
# MAGIC ## Install and import libraries

# COMMAND ----------

# MAGIC %pip install h5py 
# MAGIC %pip install typing-extensions 
# MAGIC %pip install wheel
# MAGIC %pip install datasets

# COMMAND ----------

import os
import glob
import json
import time
import torch
import pathlib
import datetime
import numpy as np
import pandas as pd

from datetime import timezone

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorWithPadding

# COMMAND ----------

!pip install nltk
!pip install rouge_score

# COMMAND ----------

import nltk
from datasets import load_metric
nltk.download('punkt')
import time

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load data and model

# COMMAND ----------

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# COMMAND ----------

curr = time.time()
tokenizer = AutoTokenizer.from_pretrained("HHousen/distil-led-large-cnn-16384")

model = AutoModelForSeq2SeqLM.from_pretrained("/dbfs/FileStore/descriptions_gold/")#.to("cuda")#.half()
#model = AutoModelForSeq2SeqLM.from_pretrained("HHousen/distil-led-large-cnn-16384")
#model_finetuned = AutoModelForSeq2SeqLM.from_pretrained("/dbfs/FileStore/descriptions_gold/v2/")
model.config.max_length = 350
model.config.min_length = 100

print(time.time()-curr)

# COMMAND ----------

model_name = 'descriptions'

# COMMAND ----------

gold_set = spark.sql(f'select * from {model_name}.gold_set').toPandas()
gold_set.shape

# COMMAND ----------

#df = gold_set[['fulltext', 'description']]
#dataset = Dataset.from_pandas(df)#.select(range(100))
#dataset

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

test_companyid = test_df.sort_values(by=['paraphrase_model'], ascending=False)['companyid'][:500].tolist()

test_df = test_df[test_df['companyid'].isin(test_companyid)]

print(train_df.shape, eval_df.shape, test_df.shape)

test_df = test_df.reset_index(drop=True).drop(columns=['name', 'companyid', 'paraphrase_model'])

test_df = Dataset.from_pandas(test_df).select(range(200))
test_df.shape

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
  batch["predicted_abstract"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  print(time.time()-curr)
  torch.cuda.empty_cache() 
  return batch


result = test_df.map(generate_answer, batched=True, batch_size=1)

# load rouge
rouge = load_metric("rouge")

print("Result:", rouge.compute(predictions=result["predicted_abstract"], references=result["description"], rouge_types=["rouge2"])["rouge2"].mid)


# COMMAND ----------

model_finetuned = AutoModelForSeq2SeqLM.from_pretrained("/dbfs/FileStore/descriptions_gold/v2/").to(device)
tokenizer = AutoTokenizer.from_pretrained("HHousen/distil-led-large-cnn-16384")

def generate_answer(batch):
  curr = time.time()
  inputs_dict = tokenizer(batch["fulltext"], max_length=4096,  return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to(device)
  attention_mask = inputs_dict.attention_mask.to(device)
  global_attention_mask = torch.zeros_like(attention_mask)
  # put global attention on <s> token
  global_attention_mask[:, 0] = 1

  predicted_abstract_ids = model_finetuned.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
  batch["predicted_abstract"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  print(time.time()-curr)
  torch.cuda.empty_cache() 
  return batch


result = test_df.map(generate_answer, batched=True, batch_size=1)

# load rouge
rouge = load_metric("rouge")

print("Result:", rouge.compute(predictions=result["predicted_abstract"], references=result["description"], rouge_types=["rouge2"])["rouge2"].mid)


# COMMAND ----------

torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quantization with pytorch

# COMMAND ----------

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
print(quantized_model)

# COMMAND ----------

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print_size_of_model(model)
print_size_of_model(quantized_model)

# COMMAND ----------

def generate_answer(batch):
  curr = time.time()
  inputs_dict = tokenizer(batch["fulltext"], padding="max_length", max_length=int(8192/2), return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to(device)
  attention_mask = inputs_dict.attention_mask.to(device)
  global_attention_mask = torch.zeros_like(attention_mask)
  # put global attention on <s> token
  global_attention_mask[:, 0] = 1

  predicted_abstract_ids = quantized_model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
  batch["predicted_abstract"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  print(time.time()-curr)
  torch.cuda.empty_cache() 
  return batch


result = test_df.map(generate_answer, batched=True, batch_size=1)

# load rouge
rouge = load_metric("rouge")

print("Result:", rouge.compute(predictions=result["predicted_abstract"], references=result["description"], rouge_types=["rouge2"])["rouge2"].mid)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimum ONNX Quantization

# COMMAND ----------

from optimum.onnxruntime import ORTModelForSeq2SeqLM

model_ort = ORTModelForSeq2SeqLM.from_pretrained("HHousen/distil-led-large-cnn-16384", from_transformers=True)
  
def generate_answer(batch):
  curr = time.time()
  inputs_dict = tokenizer(batch["fulltext"], padding="max_length", max_length=int(8192/2), return_tensors="pt", truncation=True)
  input_ids = inputs_dict.input_ids.to(device)
  attention_mask = inputs_dict.attention_mask.to(device)
  global_attention_mask = torch.zeros_like(attention_mask)
  # put global attention on <s> token
  global_attention_mask[:, 0] = 1

  predicted_abstract_ids = model_ort.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
  batch["predicted_abstract"] = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
  print(time.time()-curr)
  torch.cuda.empty_cache() 
  return batch


result = dataset.map(generate_answer, batched=True, batch_size=1)

# load rouge
rouge = load_metric("rouge")

print("Result:", rouge.compute(predictions=result["predicted_abstract"], references=result["description"], rouge_types=["rouge2"])["rouge2"].mid)

print(time.time()-curr)
