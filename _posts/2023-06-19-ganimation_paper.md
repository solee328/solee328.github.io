---
layout: post
title: GANimation(1) - 논문 리뷰
# subtitle:
categories: gan
tags: [gan, ganimation, unsupervised, face animation, action-unit, AU, 생성 모델, 논문 리뷰]
# sidebar: []
use_math: true
---

안녕하세요!
이번 글은 이미지의 표정을 연속적으로 변화시켜 표정 애니메이션을 만들 수 있는 GANimation을 살펴보겠습니다.
<br><br>

---

## 소개

generative adversarial network가 발전함에 따라 StarGAN[4]과 같은 구조로 많은 발전이 이루어졌습니다. 나이, 머리색, 성별과 같이 여러 얼굴 속성을 변경할 수 있는 StarGAN은 데이터셋에 정의된 속성만 변경할 수 있으며 불연속적입니다. RaFD[16] 데이터셋을 학습했다면 얼굴 표정에 대한 8개의 이진 라벨, 즉 슬픔, 중립, 분노, 경멸, 혐오, 놀람, 두려움, 행복함에 대해 얼굴 표정을 변경할 수 있습니다.

### AU
하지만 얼굴 표정은 얼굴 근육으로 결합되고 조절된 행동의 결과로 하나의 라벨로 정의될 수 없습니다. Pal Ekman과 Wallace Friesen은 얼굴 근육을 기반으로 얼굴 표정을 분석하기 위해 Action Units(AUs)의 관점에서 얼굴 표정을 표현하기 위한 Facial Action Coding System(FACS)를 개발했습니다. action units의 수는 많지 않지만 7,000개 이상의 다른 AU 조합이 가능하다고 합니다. 각각 AU에 대한 정보는 <a href="https://www.cs.cmu.edu/~face/facs.htm" target="_blank">FACS-Facial Action Coding System</a>에서 확인할 수 있습니다. 공포에 대한 얼굴 표정은 일반적으로 Inner Brow Raiser(AU1), Outer Brow Raiser(AU2), Brow Lowerer(AU4), Upper Lid Raiser(AU5), Lid Tighttener(AU7), Lip Stretcher(AU20), Jaw Drop(AU26)에 의해 생성되며 각 AU의 크기에 따라 표현하는 공포의 감정 크기가 달라집니다.

StarGAN[4]와 같이 특정 도메인에 해당하는 이미지를 조건화하는 대신 각 action unit의 존재 유무와 크리를 나타내는 1차원 벡터에 조건화되는 GAN 구조를 구축합니다.


### Attention Layer

---

## 사용 데이터셋
AU 라벨링이된 100만개의 얼굴 표정 이미지를 가진 EmotionNet Dataset[3]을 사용했으며 그 중 200,0000개를 사용했습니다.

### AU 라벨링
입력 RGB 이미지를 $\mathrm{I _{y_r}} \in \mathbb{R} ^{H \times W \times 3}$이고 임의의 얼굴 표정이 표현되어 있다 정의합니다. 모든 표정 표현은 $N$개의 action unit으로 이루어져 있으며 $\mathrm{y_r} = (y_1, \dots, y_N)^{\mathsf{T}}$에 인코딩됩니다. 이때 각 $y_n$은 n번째 action unit의 크기를 0과 1 사이의 정규화된 값을 나타냅니다. 0부터 1 사이의 값으로 표현 덕분에 continuous한 표현이 가능하며 여러 표정 사이 자연스러운 보간이 가능해 사실적이고 부드러운 얼굴 표정을 표현할 수 있습니다.

GANimation의 목표는 action unit $\mathrm{y_r}$에 해당하는 입력 이미지 $\mathrm{I _{y_r}}$을 목표 action unit $\mathrm{y_g}$에 해당하는 결과 이미지 $\mathrm{I _{y_g}}$로 변환할 수 있는 매핑 $\mathcal{M}$을 학습하는 것입니다. 매핑 $\mathcal{M} : (\mathrm{I _{y_r}}, y_g) \rightarrow \mathrm{I _{y_g}} $을 추정하기 위해 목표 action unit $\mathrm{y}^m_g$를 랜덤하게 생성해 사용합니다. 입력 이미지의 표정을 변환하기 위해 목표 표현을 가진 페어 이미지 $\mathrm{I _{y_g}}$가 필요하지 않은 비지도 방식을 사용합니다.

<br><br>
---

## 모델

동일한 사람에 대해 서로 다른 표현의 학습 이미지 페어가 필요하지 않도록 2단계로 문제를 구성합니다.
1. 입력 이미지를 원하는 표정으로 렌더링하는 AU 조건의 bidirectional adversarial architecture
2. 변화하는 배경과 조명 조건을 처리할 수 있도록 표정과 관련된 이미지의 영역만 처리하는 것에 초점을 맞추는 attention layer 사용


### Generator

### Condition Critic

---

## Loss
[1, 9]에서 향상된 성능을 보여준 Earth Mover Distance metric 사용

CycleGAN[38], DiscoGAN[13], StarGAN[4]와 같이 입력 이미지와 변환 이미지 사이 주요 속성을 보존하기 위한 cycle consistency 사용

### Image Adversarial Loss

### Attention Loss

### Conditional Expression Loss

### Identity Loss

### Full Loss

---

## 평가 지표


---

## 결과


<br><br>

---

MUNIT 논문 리뷰는 여기서 끝입니다! 다음 글은 MUNIT 코드 구현 글이 되겠네요.

긴 글 끝까지 봐주셔서 감사합니다. 코드 구현에서 뵙겠습니다 :lemon:
