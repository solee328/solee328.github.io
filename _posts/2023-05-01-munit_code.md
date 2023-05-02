---
layout: post
title: MUNIT(2) - 논문 구현
# subtitle:
categories: gan
tags: [gan, munit, multimodal, unsupervised, 생성 모델, 논문 구현]
# sidebar: []
use_math: true
---

저번 글인 <a href="https://solee328.github.io/gan/2023/04/19/munit_paper.html" target="_blank">MUNIT(1) - 논문 리뷰</a>에 이은 MUNIT 코드 구현입니다! 논문의 공식 코드는 <a href="https://github.com/NVlabs/MUNIT" target="_blank">github</a>에서 제공되고 있습니다.
<br><br>

---

## 1. 데이터셋
논문에서 사용된 데이터 셋 중에서 Animal Translation dataset을 사용해 동물 간의 변환을 구현하는 것으로 목표를 잡았습니다. 이유는 하나입니다. 귀여우니까요 :see_no_evil: :dog: :cat: :tiger:

하지만 MUNIT의 <a href="https://github.com/NVlabs/MUNIT/issues/22" target="_blank">issue</a>에서 ImageNet 데이터셋의 저작권 때문에 사용한 데이터는 공개되지 않는다고 합니다. 유사한 데이터셋을 찾아보다 <a href="https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq" target="_blank">Animal-Faces-HQ(AFHQ)</a>를 찾게 되어 AFHQ 데이터셋을 사용했습니다.

### 논문
UNIT의 <a href="https://github.com/mingyuliutw/UNIT/issues/27" target="_blank">issue</a>에서 데이터 처리 방법을 알 수 있었습니다. VGG를 이용해 Template matching을 통해 개와 고양이 품종의 머리 부분의 이미지를 찾았으며 각 카테고리 별로 이미지는 1000장 ~ 10000장이였다 합니다. 또한 종이 다양하게 섞여있도록 데이터셋을 만들었는데 예시로 고양이 카테고리 안에는 이집트 고양이, 페르시안 고양에, 범무늬 고양이 등 다양한 종이 섞어 사용합니다.

### AFHQ
AFHQ에는


---

## 2. 모델
<a href="https://arxiv.org/abs/1804.04732" target="_blank">MUNIT 논문</a>의 **B. Training Details**에서 간략한 모델 구조를 확인할 수 있습니다.

- Generator architecture
  - Content encoder: $\bm{\mathsf{c7s1-64, d128, d256, R256, R256, R256, R256}}$
  - Style encoder: $\bm{\mathsf{c7s1-64, d128, d256, d256, d256, GAP, fc8}}$
- Discriminator architecture : $\bm{\mathsf{d64, d128, d256, d512}}$


#### Content Encoder

#### Style Encoder
Style Encoder에는 Global Average Pooling가 구조로 포함되어 있습니다.
JINSOL KIM님의 <a href="https://gaussian37.github.io/dl-concept-global_average_pooling/" target="_blank">Global Average Pooling</a>을 참고했습니다.

### Generator


#### Decoder

AdaIN


### Discriminator
Discriminator는 Pix2PixHD의 Multi-scale discriminator를 사용합니다. Pix2PixHD에서는 고해상도의 이미지를 처리하기 위해 더 깊은 네트워크 또는 더 큰 convolution kernel을 사용해하나 두 가지 방법 모두 네트워크 용량을 증가시키고 잠재적으로 overfitting을 유발할 수 있으며 더 큰 메모리 공간을 필요로 한다는 단점을 언급하며 이를 해결하기 위해 multi-scale discriminator를 제안합니다.

multi scale discriminator는 말그대로 여러 개의 판별 모델을 사용하는 방법입니다. 네트워크 구조(PatchGAN)은 동일하지만 서로 다른 크기의 이미지에서 동작하는 판별 모델을 사용합니다. 원본 이미지 크기에서 동작하는 D_1

<br><br>

---

## 3. Loss

### Adversarial Loss
LSGAN?


### Bidirectional Reconstruction Loss

#### Image reconstruction

#### Latent reconstruction

### Total?

---

## 4. 학습

### scheduler


### 학습

---

### 5. 결과
