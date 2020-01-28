import os
os.chdir('./sentence-transformers')
print(os.getcwd())
# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/jupyter/sentence-transformers')

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
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator
from sentence_transformers.readers import STSDataReader
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

logging.info('Reading questions.csv dataset')
temp = pd.read_csv("../questions.csv", header=None)
train, dev = train_test_split(temp, test_size=0.3, random_state=42)
dev.to_csv('../dev_testing.csv', header=False, index=False)
c = pd.read_csv('../dev_testing.csv', header=None)
c[~(c[5]=='is_duplicate')].to_csv('../dev_testing.csv', index=False, header=False)


dataset = STSDataReader('', s1_col_idx = 3, s2_col_idx = 4, score_col_idx = 5, delimiter=',', normalize_scores=False, quoting=1)

train_batch_size = 64
model_name = 'bert-base-nli-mean-tokens'
num_epochs = 4
model_save_path = 'output/training_stsbenchmark_continue_training-bert-base-nli-mean-tokens-2020-01-27_09-55-45'
model = SentenceTransformer(model_save_path)

logging.info('reading test dataset')
test_data = SentencesDataset(examples=dataset.get_examples("../dev_testing.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
evaluator = LabelAccuracyEvaluator(test_dataloader)

model.evaluate(evaluator, output_path = '/home/jupyter/')