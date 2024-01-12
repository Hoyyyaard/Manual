import os,json
from tqdm import tqdm

##########################################EgoExo4d#########################################
print('#####################################EgoExo4d data analysis#########################################')
with open("datasets/EgoExo4d/annotations/splits.json") as f:
    splits_info = json.load(f)
split_to_take_uids = splits_info['split_to_take_uids']
take_uid_to_split = splits_info['take_uid_to_split']
for k, v in split_to_take_uids.items():
    print(f"{k} set has {len(v)} takes")
    
anno_count = 0      
take_uid_to_benchmark = splits_info['take_uid_to_benchmark']
for k, v in take_uid_to_benchmark.items():
    if 'atomic_action_descriptions' in v :
        anno_count += 1
print(f"Total annotations: {anno_count}/{len(take_uid_to_benchmark)}")

cooking_tasks = []
with open("datasets/EgoExo4d/takes.json") as f:
    takes_info = json.load(f)
for take in takes_info:
    if 'cook' in take['take_name']:
        cooking_tasks.append(take['take_uid'])
print(f'cooking tasks: {len(cooking_tasks)}')

cooking_tasks_w_narration = []
train = 0
val = 0
cooking_tasks_w_narration_text_image_pairs_train = 0
cooking_tasks_w_narration_text_image_pairs_val = 0
with open("datasets/EgoExo4d/annotations/atomic_descriptions_train.json") as f:
    annos_info_train = json.load(f)['annotations']
with open("datasets/EgoExo4d/annotations/atomic_descriptions_val.json") as f:
    annos_info_val = json.load(f)['annotations']
for take_uid in cooking_tasks:
    if take_uid in annos_info_train.keys() :
        cooking_tasks_w_narration.append(take_uid)
        train += 1
        cooking_tasks_w_narration_text_image_pairs_train += len(annos_info_train[take_uid][0]['descriptions'])
    elif take_uid in annos_info_val.keys():
        val += 1
        cooking_tasks_w_narration.append(take_uid)
        cooking_tasks_w_narration_text_image_pairs_val += len(annos_info_val[take_uid][0]['descriptions'])
print(f'datasets/EgoExo4d/cooking tasks w narration: {len(cooking_tasks_w_narration)} train: {train} val: {val}')
print(f'datasets/EgoExo4d/cooking tasks w narration text image pairs: train: {cooking_tasks_w_narration_text_image_pairs_train} val: {cooking_tasks_w_narration_text_image_pairs_val} ')

cooking_tasks_w_keystep_train = []
cooking_tasks_w_keystep_val = []
with open("datasets/EgoExo4d/annotations/keystep_train.json") as f:
    keystep_info_train = json.load(f)['annotations']
with open("datasets/EgoExo4d/annotations/keystep_val.json") as f:
    keystep_info_val = json.load(f)['annotations']
for take_uid in cooking_tasks:
    if take_uid in keystep_info_train.keys() :
        cooking_tasks_w_keystep_train.append(take_uid)
    elif take_uid in keystep_info_val.keys():
        cooking_tasks_w_keystep_val.append(take_uid)
print(f'cooking tasks w keystep: train {len(cooking_tasks_w_keystep_train)} val {len(cooking_tasks_w_keystep_val)}')



##########################################Ego4d#########################################
print('#####################################Ego4d data analysis#########################################')
with open(f'datasets/Ego4d/ego4d.json', 'r') as f:
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

goalstep_train_uids = []
with open(f'datasets/Ego4d/v2/annotations/goalstep_train.json', 'r') as f:
    goalstep_train = json.load(f)['videos']
for gs in goalstep_train:
    goalstep_train_uids.append(gs['video_uid'])
    
goalstep_val_uids = []
with open(f'datasets/Ego4d/v2/annotations/goalstep_val.json', 'r') as f:
    goalstep_val = json.load(f)['videos']
for gs in goalstep_val:
    goalstep_val_uids.append(gs['video_uid'])

goalstep_train_uids_cooking = 0
goalstep_val_uids_cooking = 0
for take_uid in cooking_takes:
    if take_uid in goalstep_train_uids:
        goalstep_train_uids_cooking+=1
    elif take_uid in goalstep_val_uids:
        goalstep_val_uids_cooking+=1
print('Cooking has {} takes in goalstep_train'.format(goalstep_train_uids_cooking))
print('Cooking has {} takes in goalstep_val'.format(goalstep_val_uids_cooking))


narrations_uids = []
with open(f'datasets/Ego4d/v2/annotations/all_narrations_redacted.json', 'r') as f:
    narrations_uids = json.load(f)['videos'].keys()

narrations_uids_cooking = 0
for take_uid in cooking_takes:
    if take_uid in narrations_uids:
        narrations_uids_cooking+=1

print('Cooking has {} takes in narrations'.format(narrations_uids_cooking))