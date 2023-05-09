# Databricks notebook source
# MAGIC %pip install mlflow
# MAGIC %pip install datasets
# MAGIC %pip install rouge_score
# MAGIC %pip install nltk

# COMMAND ----------

import nltk
import torch
import mlflow
import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nltk.download('punkt')

# COMMAND ----------

class TextGeneration(mlflow.pyfunc.PythonModel):
  def __init__(self, saved_model, checkpoint):
    self.saved_model = saved_model
    self.checkpoint = checkpoint

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
    self.seq2seq = AutoModelForSeq2SeqLM.from_pretrained(self.saved_model)
    self.seq2seq.config.use_cache=True
    
  def batch_generator(self, texts, batch_size):
    L = len(texts)
    counter = 0
    samples = []
    text_lengths = [len(t) for t in texts]
    sorted_tuple = sorted(zip(texts, text_lengths), key=lambda x:x[1])
    sorted_texts = [text for text,length in sorted_tuple]    
    while counter < L:
      text = sorted_texts[counter]
      samples.append(text)
      if len(samples) == batch_size:
        encoded_samples = self.tokenizer(samples, max_length=4096, return_tensors="pt", truncation=True, padding=True)
        input_ids = encoded_samples.input_ids
        attention_mask = encoded_samples.attention_mask
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1
        yield input_ids, attention_mask, global_attention_mask
        samples = []
      counter += 1
    
    if samples:
      encoded_samples = self.tokenizer(samples, max_length=4096, return_tensors="pt", truncation=True, padding=True)
      input_ids = encoded_samples.input_ids
      attention_mask = encoded_samples.attention_mask
      global_attention_mask = torch.zeros_like(attention_mask)
      global_attention_mask[:, 0] = 1
      yield input_ids, attention_mask, global_attention_mask
        
  def predict(self, context, model_input):
    batch_size = 4
    columns = model_input.columns
    text = model_input['fulltext'].values.tolist()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    self.seq2seq.to(device)

    batch_loader = self.batch_generator(text, batch_size)

    predictions = []
    self.seq2seq.eval()
    for batch in batch_loader:
      print(batch)
      input_ids, attention_mask, global_attention_mask = batch
      with torch.no_grad():
        outputs = self.seq2seq.generate(input_ids.to(device), attention_mask=attention_mask.to(device), global_attention_mask=global_attention_mask.to(device))
      predictions.extend(outputs.detach().cpu())
      torch.cuda.empty_cache() 
    return np.array(predictions)

# COMMAND ----------

model_name = 'descriptions'
transformer = 'HHousen/distil-led-large-cnn-16384'

test_df = spark.sql(f'select * from {model_name}.test').toPandas()[:20]
print("Test set size: {}".format(test_df.shape[0]))

# COMMAND ----------

#Example function of compute_metrics to adapt once the jira task of select metric is completed
metric = load_metric("rouge")
tokenizer = AutoTokenizer.from_pretrained(transformer)
def compute_metrics(eval_pred):
    predictions, decoded_labels = eval_pred['preds'], eval_pred['labels']
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    dictionary = {k: round(v, 4) for k, v in result.items()}
    #print(dictionary)
    return dictionary

# COMMAND ----------

path = "/dbfs/FileStore/descriptions_gold/"
prev_wrappedModel = TextGeneration(path, transformer)

preds = prev_wrappedModel.predict(None, test_df)

text_lengths = [len(t) for t in test_df['fulltext']]
sorted_tuple = sorted(zip(test_df['description'], text_lengths), key=lambda x:x[1])
sorted_des = [text for text,length in sorted_tuple]    

pred = {'preds':preds,'labels':sorted_des}

prev_model_metrics = compute_metrics(pred)
prev_model_metrics

# COMMAND ----------

from mlflow.models.signature import infer_signature

# COMMAND ----------

wrappedModel = TextGeneration(transformer, transformer)
signature = infer_signature(test_df[:2], wrappedModel.predict(None, test_df[:2]))

# COMMAND ----------

test_df[:2]

# COMMAND ----------


