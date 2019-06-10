from imutils import paths
import argparse
import requests
import cv2
import os

# argument parse 생성하고 arguments parse
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True,
                help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory of images")
args = vars(ap.parse_args())

ptext = "urls.txt"

o = open(ptext, "r")
url0 = o.read()
o.close()

urls = url0.split()
print("The number of urls: {}".format(len(urls)))
print("=========================")

total = 0

for url in urls:
    try:
        # url로부터 image 다운로드
        r = requests.get(url, timeout=60)

        # image 저장
        p = os.path.sep.join([args["output"], "{}.jpg".format(
            str(total).zfill(8))])

        f = open(p, "wb")
        f.write(r.content)
        f.close()

        # update the counter
        print("[INFO] downloaded: {}".format(p))
        total += 1

        # handle if any exceptions are thrown (download 중)
    except Exception as e:
        print("[INFO] error downloading {}...skipping".format(p))
        print("\n{} {}".format(e, url))
        pass


# image paths 반복
for imagePath in paths.list_images(args["output"]):
    # initialize if the image should be deleted or not
    delete = False

    # image load
    try:
        image = cv2.imread(imagePath)

        # if image is `None` -> can't load it -> delete it
        if image is None:
            delete = True

    # if OpenCV can't load the image -> delete it
    except:
        print("Except")
        delete = True

    # check to see if the image should be deleted
    if delete:
        print("[INFO] deleting {}".format(imagePath))
        os.remove(imagePath)
