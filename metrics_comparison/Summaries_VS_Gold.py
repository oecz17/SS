# Databricks notebook source
# MAGIC %pip install evaluate
# MAGIC %pip install bert_score

# COMMAND ----------

# MAGIC %md
# MAGIC # imports

# COMMAND ----------

import pandas as pd
import evaluate

# COMMAND ----------

# MAGIC %md
# MAGIC # load data

# COMMAND ----------

df = spark.sql('''select * from scraped_content.sample_company_summaries''').toPandas()
df = df.melt(id_vars=['Company', 'GoldSummary'], var_name='Model', value_name='ObtainedSummary')
df

# COMMAND ----------

# MAGIC %md
# MAGIC # Metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rouge-Score

# COMMAND ----------

rouge_score = evaluate.load('rouge')

def compute_rouge(x):
  score = rouge_score.compute(predictions=[x.ObtainedSummary], references=[x.GoldSummary])
  return pd.Series(score)

# COMMAND ----------

result = df.apply(compute_rouge, axis=1)
df = df.merge(result, left_index=True, right_index=True)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bleu

# COMMAND ----------

bleu_score = evaluate.load('bleu')

def compute_bleu(x):
  score = bleu_score.compute(predictions=[x.ObtainedSummary], references=[[x.GoldSummary]])
  return pd.Series(score)

# COMMAND ----------

result = df.apply(compute_bleu, axis=1)
df = df.merge(result, left_index=True, right_index=True)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bert Score

# COMMAND ----------

bertscore = evaluate.load('bertscore')

def compute_bertscore(x):
  score = bertscore.compute(predictions=[x.ObtainedSummary], references=[x.GoldSummary], lang='en')
  return pd.Series(score)

# COMMAND ----------

result = df.apply(compute_bertscore, axis=1)
df = df.merge(result, left_index=True, right_index=True)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cosine similary. Paraphrase model

# COMMAND ----------

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# COMMAND ----------

lst = df.ObtainedSummary.to_list()
lst2 = df.GoldSummary.to_list()

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2')

def normalize(x):
    m = torch.sum(x * x, axis=-1, keepdims=True)
    return x / torch.sqrt(m)
  
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
  
encoded_obtained_desc = tokenizer(lst, padding=True, truncation=True, return_tensors='pt')
encoded_gold_desc = tokenizer(lst2, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
  v1 = model(**encoded_obtained_desc)
  v2 = model(**encoded_gold_desc)

v1 = mean_pooling(v1, encoded_obtained_desc['attention_mask'])
v2 = mean_pooling(v2, encoded_gold_desc['attention_mask'])

v1 = normalize(v1)
v2 = normalize(v2)
scores = pd.DataFrame(torch.mm(v1, v2.T).numpy())

# COMMAND ----------

result = pd.Series(np.diag(scores), index=scores.index)
result.name = 'paraphrase_model'
df = df.merge(result, left_index=True, right_index=True)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Save

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

# COMMAND ----------

df = df.reset_index()

# COMMAND ----------

fs = FeatureStoreClient()

fs.create_table(
  name='scraped_content.metrics_sample_company_summaries',
  df=spark.createDataFrame(df),
  primary_keys=['index'],
  description='Sample of company summaries with gold descriptions with all calculated metrics'
)
