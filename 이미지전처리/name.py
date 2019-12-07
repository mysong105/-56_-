#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
from os import rename, listdir

def rename_file():
    # 현재 위치(.)의 파일을 모두 가져온다. 
    for filename in os.listdir("C:\\Users\\jiki\\Documents\\deeplearning\\전처리\\name"):

        # 파일 확장자가 (properties)인 것만 처리 
        #if filename.endswith("properties"):
        # 파일명에서 AA를 BB로 변경하고 파일명 수정 
        new_filename = filename.replace("", "7_1_2_22")
        filename = os.path.join("C:\\Users\\jiki\\Documents\\deeplearning\\전처리\\name",filename)
        new_filename = os.path.join("C:\\Users\\jiki\\Documents\\deeplearning\\전처리\\name",new_filename)
        os.rename(filename, new_filename) 

if __name__ == "__main__":

    rename_file()


# 현재 위치의 파일 목록
files = listdir('C:\\Users\\jiki\\Documents\\deeplearning\\전처리\\name')

# 파일명에 번호 추가하기 
count = 0
for filename in files:
    # 파이썬 실행파일명은 변경하지 않음
    if sys.argv[0].split("\\")[-1] == name:
        continue
    
    rename("C:\\Users\\jiki\\Documents\\deeplearning\\전처리\\name\\{0}".format(filename), "C:\\Users\\jiki\\Documents\\deeplearning\\전처리\\name\\19_1_2_{0}.jpg".format(count))
    count += 1