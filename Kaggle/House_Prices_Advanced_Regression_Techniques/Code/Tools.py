import pandas as pd
import numpy as np
import matplotlib as plt

def LoadData(datalink, sep=','):
    return pd.read_csv(open(datalink, 'r'), sep=',')