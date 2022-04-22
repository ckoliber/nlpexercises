import sys as sys
import numpy as np
import pandas as pd
import hmm as hmm
import sklearn.preprocessing as skprocessing

n_iter = int(sys.argv[1])
init_path = sys.argv[2]
train_path = sys.argv[3]
output_path = sys.argv[4]

data_set = pd.read_csv(train_path, header=None, names=["Text"]).applymap(lambda x: list(x)).to_numpy().flatten()
train_features = np.concatenate(data_set)
train_lengths = np.array([len(x) for x in data_set])

train_features = skprocessing.LabelEncoder().fit(train_features).transform(train_features)

model = hmm.load_model(init_path)
model.n_iter = n_iter

model.fit(train_features, train_lengths)
hmm.save_model(output_path, model)
