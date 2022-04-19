import sys as sys
import numpy as np
import pandas as pd
import hmm as hmm


def main() -> int:
    models = "".join(open(sys.argv[1]).readlines()).split("\n")[:-1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]

    test_features = (
        pd.read_csv(test_path, header=None, names=[
                    "Text"]).to_numpy().flatten()
    )
    test_features = list(map(lambda x: np.array(
        list(map(lambda t: ord(t) - ord('A'), x))), test_features))

    scores = []
    for model_name in models:
        model = hmm.load_model(
            f"{'/'.join(output_path.split('/')[:-1])}/{model_name}")
        scores.append(model.score(test_features))

    f = open(output_path, "w+")
    for index, model_index in enumerate(np.argmax(np.array(scores), axis=0)):
        f.write(f"{models[model_index]}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
