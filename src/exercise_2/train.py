import sys as sys
import numpy as np
import pandas as pd
import hmm as hmm


def main() -> int:
    n_iter = int(sys.argv[1])
    init_path = sys.argv[2]
    train_path = sys.argv[3]
    output_path = sys.argv[4]

    train_features = pd.read_csv(train_path, header=None, names=[
                                 "Text"]).head(10).to_numpy().flatten()
    train_features = np.array(list(map(lambda x: np.array(
        list(map(lambda t: ord(t) - ord('A'), x))), train_features)))

    model = hmm.load_model(init_path)
    model.n_iter = n_iter

    model.fit(train_features)
    hmm.save_model(output_path, model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
