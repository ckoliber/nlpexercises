import sys as sys
import numpy as np
import pandas as pd
import hmm as hmm


def main() -> int:
    lines1 = open("../../lib/hmm_data/testing_result1.txt").readlines()
    lines2 = open("../../lib/hmm_data/testing_answer.txt").readlines()

    succ = 0
    for i in range(len(lines1)):
        if lines1[i] == lines2[i]:
            succ += 1

    print(succ)


if __name__ == "__main__":
    sys.exit(main())
