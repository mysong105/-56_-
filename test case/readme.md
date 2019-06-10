## IPCGAN의 TEST 결과입니다.

### <경우에 따른 test 결과>
서양인/동양인, 아동/10대 후반/성인으로 나누어 경우에 따른 결과를 확인해보았습니다.

#### ● 아시아 성인의 test 결과
![Asian_adult_before](https://user-images.githubusercontent.com/47961925/59163075-c15b6d00-8b36-11e9-986d-c141ba3f8ca2.jpg)
![Asian_adult_after](https://user-images.githubusercontent.com/47961925/59163077-cfa98900-8b36-11e9-8af8-f5aa2e02cf70.jpg)  
(원본 -> 나이 든 모습)

#### ● 서양인 성인의 test 결과
![Westerner_adult_before](https://user-images.githubusercontent.com/47961925/59163085-d46e3d00-8b36-11e9-8692-b88fbfde9206.jpg)
![Westerner_adult_after](https://user-images.githubusercontent.com/47961925/59163084-d46e3d00-8b36-11e9-9ca6-f6681a4864b9.jpg)  
(원본 -> 나이 든 모습)

#### ● 아시아 10대 후반의 test 결과
![Asian_teen_before](https://user-images.githubusercontent.com/47961925/59163083-d46e3d00-8b36-11e9-95eb-09469dae6e5d.jpg)
![Asian_teen_after](https://user-images.githubusercontent.com/47961925/59163082-d46e3d00-8b36-11e9-87ae-6a907e2948de.jpg)  
(원본 -> 나이 든 모습)

#### ● 서양인 10대 후반의 test 결과
![Westerner_teen_before](https://user-images.githubusercontent.com/47961925/59163079-d33d1000-8b36-11e9-8614-7c6b6ee3572f.jpg)
![Westerner_teen_after](https://user-images.githubusercontent.com/47961925/59163088-d506d380-8b36-11e9-9967-1e543dec5d27.jpg)  
(원본 -> 나이 든 모습)

#### ● 아시아 아이의 test 결과
![Asian_kid_before](https://user-images.githubusercontent.com/47961925/59163081-d3d5a680-8b36-11e9-871e-860dc06aafb9.jpg)
![Asian_kid_after](https://user-images.githubusercontent.com/47961925/59163080-d3d5a680-8b36-11e9-93a7-e1fb91dc361e.jpg)  
(원본 -> 나이 든 모습)

#### ● 서양인 아이의 test 결과
![Westerner_kid_before](https://user-images.githubusercontent.com/47961925/59163087-d506d380-8b36-11e9-8a1f-473a6f3da491.jpg)
![Westerner_kid_after](https://user-images.githubusercontent.com/47961925/59163086-d506d380-8b36-11e9-922b-68cafcbe8531.jpg)  
(원본 -> 나이 든 모습)

### <test 결론>
학습데이터가 서양인 중 10대 후반부터 구성되어 있어서 서양인의 10대 후반과 성인의 경우가 가장 자연스러운 결과를 얻을 수 있었습니다. 동양인과 아동의 경우는 부자연스러운 결과를 얻었고 특히 아동의 경우 동서양의 구분 없이 나이 든 모습이 부자연스럽게 표현되었습니다. 따라서 저희의 목표를 기준으로 동양인과 아동 위주의 데이터를 모아 추가로 학습시킬 계획입니다. 학습은 원래의 학습데이터에 새로운 데이터를 추가하거나 새로 확보한 데이터만으로 학습을 시켜 성능이 좋은 쪽으로 결정할 예정입니다. 또한 전체적인 성능을 향상시키기 위해 perceptual loss에 쓰이는 AlexNet을 다른 cnn네트워크로 바꾸거나 눈, 코, 입을 따로 학습시키는 등 모듈 내부의 네트워크에 변화를 줄 계획입니다.
