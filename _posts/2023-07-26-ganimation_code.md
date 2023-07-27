---
layout: post
title: GANimation(2) - 논문 구현
# subtitle:
categories: gan
tags: [gan, ganimation, 생성 모델, 논문 구현]
# sidebar: []
use_math: true
---

:wave::wave: <a href="https://solee328.github.io/gan/2023/07/04/ganimation_paper.html" target="_blank">GANimation(1) - 논문 리뷰</a>에 이은 GANimation 논문 구현 글입니다!

공식 코드는 <a href="https://github.com/albertpumarola/GANimation/tree/master" target="_blank">github</a>에서 제공되고 있습니다.
<br><br>

---

## 1. 데이터셋
논문 구현 글의 시작을 알리는 데이터셋입니다ㅎㅎㅎ<br>
논문의 경우 EmotioNet을 사용했다하는데 찾아보니 <a href="http://cbcsl.ece.ohio-state.edu/EmotionNetChallenge/" target="_blank">EmotioNet Challenge</a>이 있었습니다. <a href="http://cbcsl.ece.ohio-state.edu/dbform_emotionet.html" target="_blank">EmotioNet Database Access Form</a>에 데이터를 신청할 수 있지만 연구자도 뭣도 아닌 저는 만인의 데이터셋인 <a href="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" target="_blank">CelebA</a>를 사용하는 것으로 결정했습니다:joy:

GANimation은 action unit이라는 condition을 사용하는 condition GAN입니다. 조건을 사용하기 위해 CelebA를 다운받은 후에는 AU 라벨링 과정을 거쳐야 하며 <a href="https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units" target="_blank">OpenFace</a>를 사용해 AU 라벨링이 가능합니다.





<br><br>

---

## 2. 모델



### Generator



### Discriminator

<br><br>

---

## 3. Loss


### Adversarial Loss



### Attention Loss


### Conditional Expression Loss


### Identity Loss


###  Full Loss



---

## 4. 학습

### scheduler

### 학습
<br><br>

---

## 5. 결과


### 시도_1

### 시도_2


<br><br>

---

<br>
