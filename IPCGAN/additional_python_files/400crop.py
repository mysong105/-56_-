from PIL import Image
import os
dir ="C:/Users/myson/gitface/mytrain/images/real/"
name= os.listdir(dir)
for i in name:
    image= Image.open(dir+i)
    image= image.resize((400,400))
    image.save("C:/Users/myson/gitface/mytrain/images/cropped/"+i)


dir ="C:/Users/myson/gitface/mytrain/images/source/"
name= os.listdir(dir)
for i in name:
    image= Image.open(dir+i)
    image= image.resize((400,400))
    image.save("C:/Users/myson/gitface/mytrain/images/cropped/"+i)

