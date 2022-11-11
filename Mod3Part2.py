# -*- coding: utf-8 -*-
"""2nd_Mod3_A01652370.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1e3kXlrWiTEtGEpt8tvLaG2l8cLpHsecG

# Functions
"""

from datasets import load_dataset

#Get the values for input_ids, token_type_ids, attention_mask
def tokenize_adjust_labels(all_samples_per_split):
  tokenized_samples = tokenizer.batch_encode_plus(all_samples_per_split["tokens"], is_split_into_words=True, padding="max_length")
  #tokenized_samples is not a datasets object so this alone won't work with Trainer API, hence map is used 
  #so the new keys [input_ids, labels (after adjustment)]
  #can be added to the datasets dict for each train test validation split
  total_adjusted_labels = []
  print(len(tokenized_samples["input_ids"]))
  for k in range(0, len(tokenized_samples["input_ids"])):
    prev_wid = -1
    word_ids_list = tokenized_samples.word_ids(batch_index=k)
    existing_label_ids = all_samples_per_split["ner_tags"][k]
    i = -1
    adjusted_label_ids = []
   
    for wid in word_ids_list:
      if(wid is None):
        adjusted_label_ids.append(-100)
      elif(wid!=prev_wid):
        i = i + 1
        adjusted_label_ids.append(existing_label_ids[i])
        prev_wid = wid
      else:
        label_name = label_names[existing_label_ids[i]]
        adjusted_label_ids.append(existing_label_ids[i])
        
    total_adjusted_labels.append(adjusted_label_ids)
  tokenized_samples["labels"] = total_adjusted_labels
  return tokenized_samples

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_names[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    flattened_results = {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }
    for k in results.keys():
      if(k not in flattened_results.keys()):
        flattened_results[k+"_f1"]=results[k]["f1"]

    return flattened_results

"""# Load and preprocess the dataset"""

from datasets import load_dataset

dataset = load_dataset("wikiann", "en")

label_names = dataset["train"].features["ner_tags"].feature.names

pre_model = "bert-base-multilingual-cased"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(pre_model)

tokenized_dataset = dataset.map(tokenize_adjust_labels, batched=True)

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)

"""# Wandb"""

import os
import wandb
os.environ["WANDB_API_KEY"]=input("WANDB API KEY GOES HERE: ")
os.environ["WANDB_ENTITY"]="henrymach"
os.environ["WANDB_PROJECT"]="finetune_bert_ner"

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
from datasets import load_metric
metric = load_metric("seqeval")

# Commented out IPython magic to ensure Python compatibility.
# %%wandb
# 
# model = AutoModelForTokenClassification.from_pretrained(pre_model, num_labels=len(label_names))
# training_args = TrainingArguments(
#     output_dir="./fine_tune_bert_output",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=7,
#     weight_decay=0.01,
#     logging_steps = 1000,
#     report_to='wandb',    
#     run_name = "2nd_Mod3_A01652370",
#     save_strategy='epoch'
# )
# 
# N_EXAMPLES_TO_TRAIN = int(int(input("PERCENT_OF_DATASET_TO_TRAIN (1% = 200 samples):"))*(tokenized_dataset["train"]).shape[0]/100)
# #N_EXAMPLES_TO_TRAIN = int(1*(tokenized_dataset["train"]).shape[0]/100)
# 
# 
# small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(N_EXAMPLES_TO_TRAIN))
# small_eval_dataset = tokenized_dataset["validation"].shuffle(seed=42).select(range(N_EXAMPLES_TO_TRAIN))
# 
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     data_collator=data_collator,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )
# 
# trainer.train()
# wandb.finish()