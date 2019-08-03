import os
import random

path = "C:/Users/myson/gitface/mytrain/images/source"

name_list = os.listdir(path)
train =name_list[:700]
test = name_list [700:]

with open('C:/Users/myson/gitface/mytrain/images/age_data/train_data/train_age_group_0.txt', "a+", encoding='utf8') as f:
    for fname in train:
        f.write("%s\n" % fname)

with open('C:/Users/myson/gitface/mytrain/images/age_data/test_data/test_age_group_0.txt', "a+", encoding='utf8') as f:
    for fname in test:
        f.write("%s\n" % fname)

path = "C:/Users/myson/gitface/mytrain/images/real"

name_list = os.listdir(path)
train =name_list[:6000]
test = name_list [6000:]
with open('C:/Users/myson/gitface/mytrain/images/age_data/train_data/train_age_group_1.txt', "a+", encoding='utf8') as f:
    for fname in train:
        f.write("%s\n" % fname)

with open('C:/Users/myson/gitface/mytrain/images/age_data/test_data/test_age_group_1.txt', "a+", encoding='utf8') as f:
    for fname in test:
        f.write("%s\n" % fname)