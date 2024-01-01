import json
import os
from tqdm import tqdm   

with open('datasets/EgoExo4d/takes.json', 'r') as f:
    all_takes = json.load(f)

cooking_takes = []
for take in tqdm(all_takes):
    if 'cook' in take['take_name']:
        cooking_takes.append(take['take_uid'])

print('Cooking has {} takes'.format(len(cooking_takes)))
        
