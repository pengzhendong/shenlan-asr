import os
import sys

if __name__ == '__main__':
    output_dir = sys.argv[1]
    os.system('cd {} && python gmm_estimator.py'.format(output_dir))
    acc = open('{}/acc.txt'.format(output_dir)).readlines()[0]
    assert float(acc) > 0.95
