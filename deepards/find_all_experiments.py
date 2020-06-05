"""
Because I keep losing experiment names
"""
import glob
import os
from pprint import pprint

files = glob.glob(os.path.join(os.path.dirname(__file__), 'results/*.pth'))
experiments = set()
for file in files:
    prefix = "_".join(file.split('_')[:-1])
    idx = prefix.find('results')
    prefix = prefix[idx+8:]
    experiments.add(prefix)
pprint(experiments)
