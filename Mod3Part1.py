from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

# import ordering: https://stackoverflow.com/questions/20762662/whats-the-correct-way-to-sort-python-import-x-and-from-x-import-y-statement


_TASK='sentiment'
_MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
# Naming: https://google.github.io/styleguide/pyguide.html#316-naming and https://www.ceos3c.com/python/python-constants/



# It’s best practice to have your code in a descriptive method or small class, if possible. 
# makes it easier for other modules to import the functionality later if needed! 

# Classes: https://www.dataquest.io/blog/using-classes-in-python/


tokenizer = AutoTokenizer.from_pretrained(_MODEL)

labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{TASK}/mapping.txt"
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
      l = labels[ranking[0]] # nit: discouraged to have one-letter variables
      if l == 'neutral':
        print(labels[ranking[1]])
      else:
        print(labels[ranking[0]])
