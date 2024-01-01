import json
import os
from tqdm import tqdm   

os.system('egoexo -o datasets/EgoExo4d/ -y --part metadata --splits train val test')

with open('datasets/EgoExo4d/takes.json', 'r') as f:
    all_takes = json.load(f)

cooking_takes = []
for take in tqdm(all_takes):
    if 'cook' in take['take_name']:
        cooking_takes.append(take['take_uid'])

print('Cooking has {} takes'.format(len(cooking_takes)))

cooking_takes_str = ''
for take in cooking_takes:
    cooking_takes_str += take + ' '

# print(cooking_takes_str)

command = f'egoexo -o datasets/EgoExo4d/ -y --parts takes annotations metadata trajectory --uids {cooking_takes_str}'
os.system(command)