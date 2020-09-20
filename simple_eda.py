import numpy as np
import pandas as pd


import conf
from utils import PathProcess

train_user_path = '../../data/tb/sample_train/05_user.csv'
train_user_df = pd.read_csv(train_user_path)

test_user_df = pd.read_csv('../../data/tb/sample_test/05_user.csv')

print('x')