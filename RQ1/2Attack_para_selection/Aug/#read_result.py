import sys
sys.path.append("..")
from Mutation.xss_attack import XssFuzzer
from Mutation.sql_attack import SqlFuzzer
import pandas as pd
import os
import numpy as np
from Tools.Resultsaver import Results
import time
import pickle
from sklearn.model_selection import train_test_split
from XLnet_Adapter import *
from tqdm import tqdm
from MCMCattacker import MHM

with open("./Augmentation/augdata/advrecord_0.pickle","rb") as file:
    data=pickle.load(file)

print(data.xsscount)
print(data.sqlcount)
print(data.xss_succ)
print(data.sql_succ)
print(data.xss_iter)
print(data.sql_iter)