---
layout: post
title: MUNIT(2) - 논문 구현
# subtitle:
categories: gan
tags: [gan, munit, multimodal, unsupervised, 생성 모델, 논문 구현]
# sidebar: []
use_math: true
---

dd
<br><br>

---

## 1. 데이터셋


### 논문


### 대체


---

## 2. 모델

### Generator
#### Content Encoder

#### Style Encoder
Style Encoder에는 Global Average Pooling가 구조로 포함되어 있습니다.
JINSOL KIM님의 <a href="https://gaussian37.github.io/dl-concept-global_average_pooling/" target="_blank">Global Average Pooling</a>을 참고했습니다.

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
