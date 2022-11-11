# Module3Project

Mod3Part1.py solves "Warm up: Out of the Box Sentiment Analysis"
 It doesn´t use any APIs and the runtime is friendly
 
 
Mod3Part2.py solves "NER: Take a basic, pretrained NER model, and train further on a task-specific dataset"
 It uses a WANDB API. It´s been used to save the eval. values through the training process and create the visual interpretation of these as graphs.
 Sign up at https://wandb.ai/site to get the necessary API KEY.
 
 For a better runtime when the input for declaring "PERCENT_OF_DATASET_TO_TRAIN" appears, type 1 to train on a 200 rows dataset.
 Graphs will still be good enough.


Mod3Part3.py solves "Warm up: Out of the Box Sentiment Analysis"
 It doesn´t use any APIs.
 
 The runtime is definitely NOT friendly.
 When the input for declaring "training_size" appears, I´d reccomend to type any positive number smaller than 5.
 This will only iterate the translators a few times and print the BLEU scores in a more comfortable time.
 GoogleTrans isn´t the issue here as it runs fast enough for the 100 lines in the txt file. However, the second model (HuggingFace) is incredibly slow to compute.
 
