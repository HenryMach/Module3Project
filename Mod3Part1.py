from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

model = AutoModelForSequenceClassification.from_pretrained(MODEL)

with open('tiny_movie_reviews_dataset.txt') as f:
    for text in f.readlines():
      encoded_input = tokenizer(text, return_tensors='pt')
      output = model(**encoded_input)
      scores = output[0][0].detach().numpy()
      scores = softmax(scores)
      ranking = np.argsort(scores)
      ranking = ranking[::-1]
      l = labels[ranking[0]]
      if l == 'neutral':
        print(labels[ranking[1]])
      else:
        print(labels[ranking[0]])