# prepare ckpts dir from aliyunpan

# preprocess epic-kitchen as image-text pairs
'''

1. put your epic-kitchen raw video to datasets dir as followed:

2. clone annotations 
git clone https://github.com/epic-kitchens/epic-kitchens-100-annotations.git

data tree after 1.2.
- Manual
    - datasets
        -epic-kitchen
            -EK100_256p
            -epic-kitchens-100-annotations

3. preprocess to image-text pairs (output text_image_pairs dir in epic-kitchen)
python src/dataset.py --preprocess --filter_frame

'''

# finetune t2i model
'''python
bash scripts/text2image_finetune.sh
'''