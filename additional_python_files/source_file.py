import os
import random
path="C:/Users/myson/gitface/mytrain/images/source"

name_list=os.listdir(path)

with open('C:/Users/myson/gitface/mytrain/images/source.txt', "a+",encoding='utf8') as f:
    for fname in name_list:

        f.write("%s\n" % (fname+" 0"))

        