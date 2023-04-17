---
layout: post
title: MUNIT(1) - 논문 리뷰
# subtitle:
categories: gan
tags: [gan, munit, multimodal, unsupervised, 생성 모델, 논문 리뷰]
# sidebar: []
use_math: true
---

이번 논문은 <a href="https://arxiv.org/abs/1804.04732" target="_blank">MUNIT</a>(Multimodal Unsupervised Image-to-Image Translation) 입니다.


---

## 소개
<div>
  <img src="/assets/images/posts/munit/paper/fig1.png" width="700" height="200">
</div>
> **Figure 1.** MUNIT 그림 예시.<br>
(a)각 도메인 $\mathcal{X}_i$의 이미지는 공유 콘텐츠 공간 $\mathcal{C}$와 도메인 별 스타일 공간 $\mathcal{S}_i$로 인코딩됩니다.<br>
(b) $\mathcal{X}_1$의 이미지(표범)을 $\mathcal{X}_2$(집 고양이)로 변환하려면 표범 이미지의 콘텐츠 코드를 고양이 도메인 스타일 공간의 임의의 스타일 코드와 재조합합니다. 스타일 코드가 달라지면 출력이 달라져 다양한 고양이 출력이 가능합니다.


비지도 이미지 변환(Unsupervised image-to-image translation)에 다양한 출력을 생성할 수 있는 Multimodal이 가능한 프레임워크인 Multimodal Unsupervised Image-to-image Translation(MUNIT)을 소개합니다. 기존 모델들은 one-to-one 매핑으로 결정론적 출력만이 가능했고 하나의 입력 이미지에는 이미지 변환이 된 하나의 출력 이미지가 나왔었습니다. 결정론적(deterministic)이 아닌 확률론적(stochastic)인 모델을 위해 노이즈를 함께 넣더라도 네트워크는 노이즈를 무시하도록 학습했다고 합니다[6, 26].

MUNIT은 이미지의 잠재 공간(latent space)을 콘텐츠 공간(content space), 스타일 공간(style space)로 분해될 수 있다 가정합니다. 이때 서로 다른 도메인의 이미지가 콘텐츠 공간은 공유하나(shared content space), 스타일 공간은 공유하지 않는다 가정합니다. 이미지에서 보존해야 하는 정보는 콘텐츠 코드(content code)에 인코딩되고, 도메인 별 속성 정보는 스타일 코드(style code)에 인코딩됩니다. 스타일 공간에서 다양한 스타일 코드는 샘플링할 수 있으며 따라서 MUNIT은 다양한 멀티모달(multi-modal) 출력을 생성할 수 있습니다.

멀티모달 모델이 StarGAN처럼 여러 도메인을 다룰 수 있는 모델이라 생각했었는데 한 도메인에서 여러 출력을 이끌어 낼 수 있는 것 또한 멀티 모달을 의미하는지 처음 알게 되었습니다.
<br><br>

---

## 모델

### Assumption
MUNIT은 도메인 $\mathcal{X}_1$에서 온 이미지 $x_1$ ($x_1 \in \mathcal{X}_1$)과 도메인 $\mathcal{X}_2$에서 온 이미지 $x_2$ ($x_2 \in \mathcal{X}_2$)를 사용합니다. 이미지 $x_1$을 도메인 $\mathcal{X}_1$에서 도메인 $\mathcal{X}_2$로 변환하는 모델 $p(x _{1 \rightarrow 2} | x_1)$과 이미지 $x_2$을 도메인 $\mathcal{X}_2$에서 도메인 $\mathcal{X}_1$로 변환하는 모델 $p(x _{2 \rightarrow 1} | x_2)$을 사용해 $p(x_1 | x_2)$와 $p(x_2 | x_1)$을 추정하는 것이 MUNIT의 목표입니다.

$p(x_1 \| x_2)$, $p(x_2 \| x_1)$는 복잡한 multimodal distribution으로 이를 해결하기 위해 부분적으로 공유되는 잠재 공간을 가정합니다. 이미지 $x_i$는 두 도메인에서 공유되는 잠재 콘텐츠 코드 $c \in \mathcal{C}$와 각각 도메인에 고유한 잠재 스타일 코드 $s_i \in \mathcal{S}_i$에서 생성된다 가정합니다. 두 도메

### Auto-Encoder

### GAN

---

## Loss

### Bidirectional Reconstruction Loss

#### Image Reconstruction

#### Latent Reconstruction

### Adversarial Loss

### Total Loss

### Domain-invariant perceptual loss
고해상도 이미지일 경우 사용

---

## 이론 가정

---

## 사용 데이터셋
총 4가지 데이터셋을 사용합니다.

- Edges $\leftrightarrow$ shoes / hangbags<br>
Isola[6], Yu et al[80],Zhu et al[81]이 제공한 데이터셋을 사용합니다. HED[82]에서 생성한 신발과 핸드백의 edge map 이미지가 포함되어 있습니다. 페어 데이터를 사용하지 않고 edges $\leftrightarrow$ shoes와 edges $\leftrightarrow$ handbag에 대해 학습합니다.

- Animal image Translation<br>
ImageNet에 house cats, big cats, dogs 3가지 도메인의 이미지를 수집합니다. 각 도메인에는 세분화된 4개의 모드가 포함되며 변환 모델을 학습하는 동안에는 이미지의 모드를 알 수 없습니다. 각 페어 도메인에 대해 별도의 모델을 학습했습니다.

- Street scene images<br>
SYNTHIA[83] 데이터셋의 이미지와 Cityscape[84] 데이터셋 간 변환과 Liu et al[15]의 거리 이미지에서 여름과 겨울 거리 이미지 변환, 2가지 변환에 대해 실험했습니다. SYNTHIA 데이터셋의 경우 계절, 날씨, 조명의 조건이 다른 이미지들을 포함하는 SYNTHIA-Seqs 하위 데이터셋을 사용했습니다.

- Yosemite summer $\leftrightarrow$ winter (HD)<br>
Liu et al[15] 데이터셋을 사용했으며 실제 주행 영상에서 추출한 여름 및 겨울의 거리 이미지들이 포함되어 있습니다.
<br><br>

---

## 평가 지표

### Human preference
이전 논문들에서도 등장했었던 Amazon Mechanical Turk(AMT) 연구를 통해 지각 연구를 수행합니다. Wang et al[20]의 연구와 유사하게 작업자에게 입력 이미지와 서로 다른 방법으로 변환한 결과 이미지 2개가 주어지며 어떤 결과 이미지가 더 정확해 보이는지 선택할 시간이 무제한으로 주어집니다. 비교를 위해 500개의 무작위 질문을 생성했으며 각 질문에는 5명의 다른 작업자가 답변합니다.

### LPIPS distance
변환의 다양성을 측정하기 위해 Zhu et al[11]과 같이 동일한 입력에서 무작위로 샘플링된 2개의 변환 결과 간의 평균 LPIPS Distance[77]을 계산합니다.

LPIPS는 이미지 간의 차이를 측정하는 방법으로 인간의 시각적 인식과 상관관계가 있는 것으로 입증되었습니다[77]. VGG, AlexNet, SqueezeNet을 이용해 두 이미지의 feature map을 이용해 2개의 feature가 유사한지 측정하는 방식으로 MUNIT 논문에서는 ImageNet으로 학습된 AlexNet[78]을 사용합니다.

$$
d(x, x_0) = \Sigma \frac{1}{H_l W_l} \Sigma \| w_l \odot (\hat{y}^l_{hw} - \hat{y}^l_{0hw}) \| _2
$$

위의 수식이 LPIPS를 계산시 사용하는 수식으로 이미지가 deep feature를 추출하는 모델의 레이어 별 feature map 간의 차이를 L2 distance로 계산함을 의미합니다. $l$은 layer, $H, W$는 각각 Height, Width를 의미하며 $\hat{y}^l_{hw}, \hat{y}^l_{0hw}$은 AlexNet의 $l$ 레이어를 통과했을 때의 feature map을 의미합니다. $w_l$은 scale factor로 두 이미지의 크기가 다른 경우에도 LPIPS Distance를 계산하기 위해 사용합니다. 이미지의 크기가 같다면 1로 두기 때문에 따로 계산할 필요가 없습니다.

Zhu et al[11]에 따라 100개의 입력 이미지를 사용하고 입력 당 19개의 출력 페어 이미지를 만들어 총 1900 페어 이미지를 가지고 계산하게 됩니다.


### (Conditional) Inception Score
Inception Score(IS)[34]는 이미지 생성 작업에서 널리 사용되는 지표입니다. 논문은 멀티모달 이미지 변환을 평가하는데 더 적합한 Conditional Inception Score(CIS)를 제안합니다.


---

## 결과


### Baseline 모델


### edges $\leftrightarrow$ shoes/handbags

### animal translation

### Cityscape $\leftrightarrow$ SYNTHIA

### Yosemite summer $\leftrightarrow$ winter
