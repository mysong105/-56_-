import os
dir ="C:/Users/myson/gitface/mydata/1"
num=0
name_list= os.listdir(dir)
for name in name_list :
    new= "1_"+str(num)+".jpg"
    os.rename(os.path.join(dir,name),os.path.join(dir,new))
    num=num+1


dir ="C:/Users/myson/gitface/mydata/0"
num=0
name_list= os.listdir(dir)
for name in name_list :
    new = "0_" + str(num)+".jpg"
    os.rename(os.path.join(dir,name),os.path.join(dir,new))
    num=num+1