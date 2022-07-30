import os
import sys

if __name__ == '__main__':
    output_dir = sys.argv[1]
    os.system('cd {} && python dnn.py > acc.txt'.format(output_dir))
    log = open('{}/acc.txt'.format(output_dir)).readlines()
    acc = log[-1].strip().split(': ')[-1]
    assert float(acc) > 0.95
