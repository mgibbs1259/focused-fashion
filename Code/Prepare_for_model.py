import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

validation = pd.read_csv("val_ann.csv")
train = pd.read_csv("train_ann.csv")
mlb = MultiLabelBinarizer()

