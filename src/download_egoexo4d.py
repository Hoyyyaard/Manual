import json
import os
from tqdm import tqdm   
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='egoexo4d', choices=['egoexo4d', 'ego4d'])
argparser.add_argument('--output_dir', required=True, type=str)
args = argparser.parse_args()

opd = args.output_dir

if args.dataset == 'egoexo4d':

    os.system(f'egoexo -o {opd} -y --part metadata --splits train val test')

    with open(f'{opd}/takes.json', 'r') as f:
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

    command = f'egoexo -o {opd} -y \
                --parts takes annotations metadata trajectory \
                --uids {cooking_takes_str}'
    os.system(command)

else:
    os.system(f'ego4d -o {opd} -y --dataset annotations')

    with open(f'{opd}/ego4d.json', 'r') as f:
        all_videos = json.load(f)['videos']

    cooking_takes = []
    for take in tqdm(all_videos):
        if len(take['scenarios']) == 0 :
            continue
        for sc in take['scenarios']:
            if 'Cooking' == sc and len(take['scenarios']) == 1:
                cooking_takes.append(take['video_uid'])
                break

    print('Cooking has {} takes'.format(len(cooking_takes)))

    cooking_takes_str = ''
    for take in cooking_takes:
        cooking_takes_str += take + ' '

    # print(cooking_takes_str)

    command = f'ego4d -o {opd} -y --datasets full_scale \
                --video_uids {cooking_takes_str}'
    os.system(command)