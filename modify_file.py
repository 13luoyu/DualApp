import fileinput
import sys

file_path = 'original.py'
file_path = sys.executable[:-10] + 'lib/python3.7/site-packages/tensorflow/python/keras/layers/core.py'
file_add = True

for line in fileinput.input(file_path, backup='.bak', inplace=1):
    if line.startswith('import') :
        if file_add :
            print(line.rstrip() + '\nimport tensorflow as tf')
            file_add = False
        else:
            print(line.rstrip())
    else:
        print(line.rstrip())