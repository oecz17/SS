# Databricks notebook source
# MAGIC %md
# MAGIC ## Install and import libraries

# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC %pip install datasets
# MAGIC %pip install "ray[tune]"
# MAGIC %pip install datadog
# MAGIC %pip install azure-storage-file-share
# MAGIC %pip install pydantic
# MAGIC %pip install nltk
# MAGIC %pip install rouge_score
# MAGIC %pip install spacy
# MAGIC !python -m spacy download en_core_web_sm

# COMMAND ----------

#!pip install git+https://github.com/NVIDIA/apex

# COMMAND ----------

import os
import glob
import json
import time
import torch
import mlflow
import pathlib
import datetime
import pydantic
import numpy as np
import pandas as pd

from datetime import timezone

from datasets import Dataset
from datadog import initialize, statsd
from torch.utils.data import DataLoader
from mlflow.models.signature import infer_signature
from azure.storage.fileshare import ShareFileClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSequenceClassification, pipeline

import sys
sys.stdout.fileno = lambda: False

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.schedulers import PopulationBasedTraining

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

from mlflow.tracking import MlflowClient

client = MlflowClient()

import nltk
from datasets import load_metric
nltk.download('punkt')

import spacy
nlp = spacy.load('en_core_web_sm')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set experiment

# COMMAND ----------

dbutils.widgets.dropdown(name="debugging", defaultValue="True", choices=["True", "False"])
debugging = pydantic.parse_obj_as(bool, dbutils.widgets.get("debugging"))
print('Debugging ', debugging)

model_name = 'descriptions'
print(model_name)

#transformer = 'HHousen/distil-led-large-cnn-16384'
dbutils.widgets.text("transformer", defaultValue='HHousen/distil-led-large-cnn-16384')
transformer = getArgument("transformer")
print(transformer)

tables_names = spark.sql(f"SHOW TABLES FROM {model_name}").toPandas().tableName.values.tolist()
tables_names

metric_to_compare = 'eval_rouge1'

if ('registered_models_info' in tables_names):
  is_first_model_run = False

else:
  is_first_model_run = True

# COMMAND ----------

mlflow.set_experiment(f"/Users/ocontreras@sourcescrub.com/Re-training__{model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Select cpu or gpu device

# COMMAND ----------

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define helper functions or information

# COMMAND ----------

#workspace = dev
dbutils.widgets.text("workspace", defaultValue="dev")
workspace = getArgument("workspace")
print(workspace)

# COMMAND ----------

def split_dataset(dataset):
  train_df, eval_df = train_test_split(dataset, test_size=0.1, random_state=42)

  print("Training set split: {:.0%}".format(train_df.shape[0]/dataset.shape[0]))
  print("Validation set split: {:.0%}".format(eval_df.shape[0]/dataset.shape[0]))

  return train_df, eval_df

# COMMAND ----------

def preprocess_function(examples):
  # tokenize the inputs and labels
  inputs = tokenizer(examples["fulltext"], padding=True, truncation=True, max_length=4096)
  outputs = tokenizer(examples["description"],padding=True,truncation=True, max_length=512)

  examples["input_ids"] = inputs.input_ids
  examples["attention_mask"] = inputs.attention_mask

  # create 0 global_attention_mask lists
  examples["global_attention_mask"] = len(examples["input_ids"]) * [[0 for _ in range(len(examples["input_ids"][0]))]]

  # since above lists are references, the following line changes the 0 index for all samples
  examples["global_attention_mask"][0][0] = 1
  examples["labels"] = outputs.input_ids

  # We have to make sure that the PAD token is ignored
  examples["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in examples["labels"]]

  return examples

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

#model.save_pretrained('/dbfs/FileStore/descriptions_gold/bias/')

# COMMAND ----------

#Example function of compute_metrics to adapt once the jira task of select metric is completed
metric = load_metric("rouge")

tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model = AutoModelForSequenceClassification.from_pretrained('/dbfs/FileStore/descriptions_gold/bias/')

classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device = 0)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bias = classifier(decoded_preds)
    bias_score = [1-b['score'] if b['label'] == 'Non-biased' else b['score'] for b in bias]
    # Extract a few results
    result = {key: value.mid.fmeasure * 80 + 20 - max(bs-0.45,0)*20 for (key, value), bs in zip(result.items(), bias_score)}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    dictionary = {k: round(v, 4) for k, v in result.items()}
    print(dictionary)
    return dictionary

# COMMAND ----------

def tune_transformer(train_dataset, eval_dataset, num_samples=7, gpus_per_trial=0):

  def get_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(transformer, use_cache=False)
    model.gradient_checkpointing_enable()
    # set generate hyperparameters
    model.config.num_beams = 4
    model.config.max_length = 500
    model.config.min_length = 0
    model.config.length_penalty = 0
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    return model

  tokenized_train = train_dataset.map(preprocess_function, batched=True, batch_size=4)
  tokenized_eval = eval_dataset.map(preprocess_function, batched=True, batch_size=4)

  tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "global_attention_mask", "labels"])
  tokenized_eval.set_format(type="torch",columns=["input_ids", "attention_mask", "global_attention_mask", "labels"])

  training_args = Seq2SeqTrainingArguments(
    output_dir=".",
    learning_rate=1e-5,  # config
    do_train=True,
    do_eval=True,
    no_cuda=gpus_per_trial <= 0,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,#True
    metric_for_best_model=metric_to_compare,
    greater_is_better=True,
    save_total_limit=1,
    num_train_epochs=2,  # config
    max_steps=-1,
    per_device_train_batch_size=1,  # config
    per_device_eval_batch_size=1,  # config
    gradient_accumulation_steps=8,
    warmup_steps=0,  # config
    weight_decay=0.01,  # config
    logging_dir="./logs",
    skip_memory_metrics=True,
    predict_with_generate=True,
    fp16=True,
    #half_precision_backend="cuda_amp",
    report_to="none")

  trainer = Seq2SeqTrainer(
    model_init=get_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    #data_collator=data_collator,
    compute_metrics=compute_metrics)

  tune_config = {
    "num_train_epochs": tune.randint(2, 4),
    #"generation_num_beams":tune.choice([2,3,4]),
    "gradient_accumulation_steps": tune.choice([2, 4, 8, 16])}

  scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric=metric_to_compare,
    mode="max",
    perturbation_interval=1,
    hyperparam_mutations={
      "weight_decay": tune.uniform(0.0, 0.3),
      "learning_rate": tune.uniform(9e-6, 1e-4),
      })

  reporter = CLIReporter(
    parameter_columns={
      "weight_decay": "w_decay",
      "learning_rate": "lr",
      "gradient_accumulation_steps": "gradient_acc",
      "num_train_epochs": "num_epochs"
      #"generation_num_beams":'num_beams'
    },
      metric_columns=[metric_to_compare, "eval_loss", "epoch", "training_iteration"])

  trainer.hyperparameter_search(
    #verbose=0,
    hp_space=lambda _: tune_config,
    backend="ray",
    n_trials=num_samples,
    resources_per_trial={"cpu": 16, "gpu": gpus_per_trial},
    scheduler=scheduler,
    keep_checkpoints_num=1,
    checkpoint_score_attr="training_iteration",
    stop=TrialPlateauStopper(metric= metric_to_compare, grace_period= 1),
    progress_reporter=reporter,
    local_dir=f"~/ray_results_{model_name}/",
    name="tune_transformer_pbt",
    log_to_file=True)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create train, eval and test datasets

# COMMAND ----------

#Load current dataset
table_name = 'temporary_gold_set'
if is_first_model_run:
  table_name = 'gold_set'
ps_dataframe = spark.sql(f'select * from {model_name}.{table_name}').toPandas()
test_df = spark.sql(f'select * from {model_name}.test_v2').toPandas()
print("Dataset size: {}".format(ps_dataframe.shape[0]))
print("Tes set size: {}".format(test_df.shape[0]))
ps_dataframe.head()

# COMMAND ----------

if is_first_model_run:
  #Divide into train, eval and test dataset with stratify
  train_df, eval_df = split_dataset(ps_dataframe)

# COMMAND ----------

train_companyid = train_df.sort_values(by=['paraphrase_model'], ascending=False)['companyid'][:(400)].tolist()
eval_companyid = eval_df.sort_values(by=['paraphrase_model'], ascending=False)['companyid'][:(50)].tolist()
test_companyid = test_df['companyid'].tolist()
#test_companyid = test_df['companyid'][:500].tolist()
#test_companyid = test_df.sort_values(by=['paraphrase_model'], ascending=False)['companyid'][:50].tolist()

# COMMAND ----------

train_df = train_df[train_df['companyid'].isin(train_companyid)]
eval_df = eval_df[eval_df['companyid'].isin(eval_companyid)]
test_df = test_df[test_df['CompanyId'].isin(test_companyid)]

# COMMAND ----------

test_scraped_content = test_df['fulltext'].values.tolist()
test_scraped_content = [t[:900000] for t in test_scraped_content]
filtered_scraped_content = []
nlp.add_pipe('sentencizer')
for doc in nlp.pipe(test_scraped_content, disable=["tok2vec", "tagger","parser", "attribute_ruler", "lemmatizer", "ner"]):
  filtered_sentences = []
  for sent in doc.sents:
    if sent.text not in filtered_sentences:
      filtered_sentences.append(sent.text)
  filtered_text = ' '.join(filtered_sentences)
  filtered_scraped_content.append(filtered_text)

test_df['fulltext'] = filtered_scraped_content

# COMMAND ----------

print(train_df.shape, eval_df.shape, test_df.shape)

train_df = train_df.reset_index(drop=True).drop(columns=['name', 'companyid', 'paraphrase_model'])
eval_df = eval_df.reset_index(drop=True).drop(columns=['name', 'companyid', 'paraphrase_model'])
#test_df = test_df.reset_index(drop=True).drop(columns=['name', 'companyid'])

train_dataset = Dataset.from_pandas(train_df)#.select(range(400))
eval_dataset = Dataset.from_pandas(eval_df)#.select(range(50))
test_df = Dataset.from_pandas(test_df)#.select(range(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperparameter optimization and parameters tracking

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained('HHousen/distil-led-large-cnn-16384')

# COMMAND ----------

with mlflow.start_run() as mlflow_run:

  tune_transformer(train_dataset=train_dataset, eval_dataset=eval_dataset, num_samples=4, gpus_per_trial=1)

  dirpath = f"../../../../../../root/ray_results_{model_name}/tune_transformer_pbt"
  runs_paths = glob.glob(dirpath+'/_objective*')

  best_metric = 0
  best_run_path = ''
  best_model_checkpoint_path = ''
  for run_path in runs_paths:
    with mlflow.start_run(nested=True):

      #Log metrics
      run_result_path = run_path + '/result.json'

      trial=[]
      keys_to_remove = ['timesteps_total', 'episodes_total', 'node_ip', 'hostname', 'trial_id', 'experiment_id', 'date', 'config','should_checkpoint','done']

      for line in open(run_result_path,'r'):
        metrics = json.loads(line)
        trial.append(metrics)

        for k, v in metrics['config'].items():
          metrics['config.'+k] = v
        for k in keys_to_remove:
          del metrics[k]
        mlflow.log_metrics(metrics, step=int(metrics['epoch']))
        
      sorted_metrics = sorted(trial, key=lambda k: k[metric_to_compare], reverse=True)

      #Log parameters
      run_params_path = run_path + '/params.json'

      with open(run_params_path, 'r') as file:
        mlflow.log_params(json.load(file))
      mlflow.log_param(key='_name_or_path', value = transformer)

      #Log latest model
      models_checkpoints = glob.glob(run_path+'/checkpoint*')
      models_checkpoints.sort(key=os.path.getmtime)
      run_checkpoint_path = glob.glob(models_checkpoints[-1]+'/*')[0]

      wrappedModel = TextGeneration(run_checkpoint_path, transformer)
      signature = infer_signature(train_df[:2], wrappedModel.predict(None, train_df[:2]))
      #log_onnx(run_checkpoint_path, num_labels, train_dataset, run_checkpoint_path)
      artifacts = {pathlib.Path(file).stem: os.path.join(run_checkpoint_path, file)
                   for file in os.listdir(run_checkpoint_path)
                   if not os.path.basename(file).startswith('.')}

      mlflow.pyfunc.log_model(model_name, loader_module=None, data_path=None, code_path=None,
                            conda_env=None, python_model=wrappedModel,
                            artifacts=artifacts, signature=signature)

      #Save path of checkpoint of best model
      if best_metric < sorted_metrics[0][metric_to_compare]:
        best_metric = sorted_metrics[0][metric_to_compare]
        best_model_checkpoint_path = run_checkpoint_path
        best_run_path = run_path

  #Select best model
  best_run_id = mlflow_run.info.run_id
  wrappedModel = TextGeneration(best_model_checkpoint_path, transformer)
  signature = infer_signature(train_df[:2], wrappedModel.predict(None, train_df[:2]))
  #log_onnx(run_checkpoint_path, num_labels, train_dataset, run_checkpoint_path)
  artifacts = {pathlib.Path(file).stem: os.path.join(best_model_checkpoint_path, file)
               for file in os.listdir(best_model_checkpoint_path)
               if not os.path.basename(file).startswith('.')}
  mlflow.pyfunc.log_model(model_name, loader_module=None, data_path=None, code_path=None,
                          conda_env=None, python_model=wrappedModel,
                          artifacts=artifacts, signature=signature)

# COMMAND ----------

best_model_checkpoint_path

# COMMAND ----------

model_finetuned = AutoModelForSeq2SeqLM.from_pretrained(best_model_checkpoint_path)
model_finetuned.config.use_cache=True

#dbutils.widgets.text("version", defaultValue='6')
#version = getArgument("version")
version = 11
print(version)

# COMMAND ----------

model_finetuned.config.version = int(version)
model_finetuned.save_pretrained(f"/dbfs/FileStore/descriptions_gold/v{version}/")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate model

# COMMAND ----------

version = 11
print(version)

# COMMAND ----------

model_finetuned = AutoModelForSeq2SeqLM.from_pretrained(f"/dbfs/FileStore/descriptions_gold/v{version}/", return_dict_in_generate=True).to(device)
#model_finetuned.config.use_cache=True
#model_finetuned.config.num_beams = 4
#model_finetuned.config.max_length = 500
#model_finetuned.config.min_length = 0
model_finetuned.config.length_penalty = 0
#model_finetuned.config.early_stopping = True
#model_finetuned.config.no_repeat_ngram_size = 3
tokenizer = AutoTokenizer.from_pretrained('HHousen/distil-led-large-cnn-16384')

# COMMAND ----------

def compute_metrics(eval_pred):
  predictions, labels = eval_pred.predictions, eval_pred.label_ids
  decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  # Replace -100 in the labels as we can't decode them.
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

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
  print(dictionary)
  return dictionary

# COMMAND ----------

def tokenize_function(example):
  tokenized=tokenizer(example["fulltext"], truncation=True, max_length=8192)
  return tokenized

tokenized_datasets = test_df.map(tokenize_function, batched=True)

len_input = [len(t) for t in tokenized_datasets['input_ids']]
tokenized_datasets = tokenized_datasets.add_column('length',len_input)

data_collator =  DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_finetuned)

#tokenized_datasets = tokenized_datasets.rename_column("description", "labels")
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])
tokenized_datasets = tokenized_datasets.sort('length')
sorted_ds = tokenized_datasets.sort('length')
#sorted_ds = sorted_ds.remove_columns(["fulltext", "description"])
sorted_ds = sorted_ds.remove_columns(["fulltext", "name", "website", "companyid"])
sorted_ds.column_names


# COMMAND ----------

test_dataloader = DataLoader(
    sorted_ds, batch_size=1, collate_fn=data_collator
)

model_finetuned.eval()
preds = []
info_scores = []
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    global_attention_mask = torch.zeros_like(batch['attention_mask'])
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1

    with torch.no_grad():
        outputs = model_finetuned.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask.to(device), output_scores=True)
    predicted_description = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    info_scores.extend(outputs.sequences_scores)
    preds.extend(predicted_description)
    torch.cuda.empty_cache()

# COMMAND ----------

 print("Result:", metric.compute(predictions=preds, references=tokenized_datasets["description"], rouge_types=["rouge1"])["rouge1"].mid)

# COMMAND ----------

preds

# COMMAND ----------

s = test_df['fulltext'].str.len().sort_values().index
s

# COMMAND ----------

test_df = test_df.reindex(s)
test_df['length'] = [len(f) for f in test_df.fulltext.values]
test_df.head()

# COMMAND ----------

test_df['Predicted_description'] = preds
test_df.head(10)

# COMMAND ----------

test_df = test_df[['companyid','name','website','Predicted_description','length']]
sparkDF=spark.createDataFrame(test_df)
sparkDF.printSchema()
sparkDF.show()

# COMMAND ----------

results_df = pd.DataFrame(tokenized_datasets["companyid"], index=np.arange(len(tokenized_datasets["companyid"])), columns=['companyid'])
results_df['Predicted_description'] = preds
results_df.head()

# COMMAND ----------

test_df = test_df[['companyid', 'fulltext', 'name', 'website', 'length']]
results_df = pd.merge(results_df, test_df, on='companyid')
results_df.head()

# COMMAND ----------

results_df.head(20)

# COMMAND ----------

sparkDF=spark.createDataFrame(results_df)
sparkDF.printSchema()
sparkDF.show()

# COMMAND ----------

fs = FeatureStoreClient()
table_name = "descriptions.v2_results_for_sultan"


fs.write_table(
  name=table_name,
  df=sparkDF,
  #mode='merge'
  mode='overwrite'
)#"""

print('Created successfully')

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select * from descriptions.v2_results_for_sultan

# COMMAND ----------

with pd.option_context('display.max_colwidth', None):
  display(results_df)

# COMMAND ----------
