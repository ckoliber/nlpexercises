import sys as sys
import numpy as np
import pandas as pd
import hmm as hmm
import sklearn.preprocessing as skprocessing

models = "".join(open(sys.argv[1]).readlines()).split("\n")[:-1]
test_path = sys.argv[2]
output_path = sys.argv[3]

data_set = pd.read_csv(test_path, header=None, names=["Text"]).applymap(lambda x: list(x)).to_numpy().flatten()
test_features = np.concatenate(data_set)
test_lengths = np.array([len(x) for x in data_set])

test_features = skprocessing.LabelEncoder().fit(test_features).transform(test_features)

scores = []
for model_name in models:
    model = hmm.load_model(f"{'/'.join(output_path.split('/')[:-1])}/{model_name}")
    scores.append(model.score(test_features, test_lengths))

f = open(output_path, "w+")
for index, model_index in enumerate(np.argmax(np.array(scores), axis=0)):
    f.write(f"{models[model_index]} {scores[model_index][index]}\n")
