import os
import shutil
f = open("C:/Users/myson/gitface/mytrain/images/age_data/test_data/test_age_group_0.txt", 'r', encoding='utf-8-sig')
lines = f.read().split()
dst="C:/Users/myson/gitface/mytrain/images/test/"

for l in lines:
    src=os.path.join("C:/Users/myson/gitface/mytrain/images/source/",l)
    d = os.path.join(dst, l)
    shutil.copyfile(src, d)