import numpy as np
from googletrans import Translator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

training_size = int(input("Training size (<= 100): "))
with open('en_corpus.txt', encoding='utf-8', errors='ignore') as f:
    en_lines = f.readlines()
    en_lines = en_lines[:training_size]

with open('es_corpus.txt', encoding='utf-8', errors='ignore') as f:
    es_lines = f.readlines()
    es_lines = es_lines[:training_size]

# Google Translator
translator = Translator()

scores = []
for i in range(len(es_lines)):
    translation = translator.translate(es_lines[i], dest='en', src="es")
    ref = [(translation.text).split()]
    test = (en_lines[i]).split()
    smoothie = SmoothingFunction().method7
    score = (sentence_bleu(ref, test, smoothing_function=smoothie))
    scores.append(score)
scores = np.array(scores)

print('GoogleTrans:', round(scores.mean(), 2), )

# Hugging Face

tokenizer = AutoTokenizer.from_pretrained("mrm8488/mbart-large-finetuned-opus-es-en-translation")
model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/mbart-large-finetuned-opus-es-en-translation")

scores = []
for i in range(len(es_lines)):
    snippet = es_lines[i]
    inputs = tokenizer.encode(snippet, return_tensors="pt", padding=True, max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=128, num_beams=None, early_stopping=True)
    translated = tokenizer.decode(outputs[0]).replace('<pad>', "").strip()
    translated = translated[3:-4]

    ref = [translated.split()]
    test = (en_lines[i]).split()
    score = (sentence_bleu(ref, test, smoothing_function=smoothie))
    scores.append(score)
    #print(("|" * len(scores)), len(scores), "%")
scores = np.array(scores)

print('HuggingFace:', round(scores.mean(), 2), )
