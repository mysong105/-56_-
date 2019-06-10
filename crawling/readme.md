
<h3>crawling은 Google 이미지에서 진행하였고 특정 키워드별로 검색하여 결과를 얻었습니다.</h3>
<BR><BR>

1차적으로 Google 이미지 개발자 도구 console 창에서 JavaScript 코드를 통해 키워드에 따른 이미지 url을 수집하였고,

2차적으로 python 코드를 통해 수집된 url들을 이미지로 다운로드 했습니다. 
<BR><BR>
  
### * JavaScript : using jQuery library
Google 이미지에서 키워드 검색 -> 개발자 도구 console 창 -> 전체 Scroll down 
-> 위의 JavaScript 코드 수행 -> URLs 수집
<BR><BR>
![Alt text](https://github.com/mysong105/team56/blob/master/crawling/readme_images/javascript.JPG)
<BR><BR>

### * Python : using requests library
urls.txt 파일에서 urls 로드 -> loop 돌면서 각 url에 따른 이미지 다운로드

-> 00000000.jpg 부터 1씩 증가하면서 디스크에 저장

-> 전체 파일들 다시 loop 돌면서 OpenCV로 열리는지 확인, 안 열릴 시 delete

-> cmd창에서 '$ python [코드파일이름].py --urls urls.txt --output [저장디렋토리]' 명령 수행

![Alt text](https://github.com/mysong105/team56/blob/master/crawling/readme_images/cmd.JPG)
<BR><BR>
