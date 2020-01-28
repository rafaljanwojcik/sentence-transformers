import os
#os.chdir('./sentence-transformers')
print(os.getcwd())

import sys
# insert at 1, 0 is the script path (or '' in REPL)
#google cloud/local
sys.path.insert(1, os.getcwd()+'/sentence-transformers')
#google colab
#sys.path.insert(1, '/content/sentence-transformers')

import pandas as pd
import numpy as np
from re import sub
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSDataReader
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logging.info('Reading questions.csv dataset')
temp = pd.read_csv("questions.csv", header=None)
train, dev = train_test_split(temp, test_size=0.3, random_state=42)
dev, test = train_test_split(dev, test_size=0.3, random_state=42)
train.to_csv('train.csv', header=False, index=False)
dev.to_csv('dev.csv', header=False, index=False)
test.to_csv('test.csv', header=False, index=False)
c = pd.read_csv('dev.csv', header=None)
c[~(c[5]=='is_duplicate')].to_csv('dev.csv', index=False, header=False)


train_batch_size = 64
model_name = 'bert-base-nli-mean-tokens'
num_epochs = 4
model_save_path = 'output/training_stsbenchmark_continue_training-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model = SentenceTransformer(model_name)

dataset = STSDataReader('', s1_col_idx = 3, s2_col_idx = 4, score_col_idx = 5, delimiter=',', normalize_scores=False, quoting=1)
logging.info("Read QuoraQuestions train dataset")
train_data = SentencesDataset(dataset.get_examples('train.csv'), model, show_progress_bar=True)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

logging.info("Read QuoraQuestions dev dataset")
dev_data = SentencesDataset(dataset.get_examples('dev.csv'), model, show_progress_bar=True)
dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

warmup_steps = math.ceil(len(train_data)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
