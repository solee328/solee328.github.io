---
layout: post
title: StarGAN(1) - 논문 리뷰
# subtitle:
categories: gan
tags: [gan, stargan, multi domain, 생성 모델, 논문 리뷰]
# sidebar: []
use_math: true
---

## 소개
<div>
  <img src="/assets/images/posts/stargan/paper/fig1.png" width="600" height="300">
</div>
> Figure 1. RaFD 데이터셋에서 학습한 지식을 CelebA 데이터셋에 적용한 멀티 도메인 이미지 간 변환 결과. 첫 번째와 여섯 번째 열은 입력 이미지를 표시하고 나머지 열은 StarGAN에서 생성한 이미지입니다. 이미지들은 단일 생성 네트워크에 의해 생성되었으며 화남, 기쁨, 두려움 과 같은 표정 라벨은 CelebA가 아닌 RaFD에서 가져온 것입니다.

Multi-domain image-to-image 변환 모델인 StarGAN입니다. 이전까지 포스팅했던 생성 모델 논문들은 2개의 도메인을 사용했었고 그 사이만 변환이 가능했었습니다. 따라서 또 다른 도메인으로 이미지를 변환하기 위해서는 다시 모델을 학습해야 했습니다. 이런 한계를 극복하기 위해 StarGAN은 여러 데이터셋을 사용해 멀티 도메인 변환이 가능한 것이 특징으로 위의 Fig1 과 같이 RaFD에 있던 표정에 대한 라벨을 CelebA 데이터셋에 적용한 것이 가능합니다.

<div>
  <img src="/assets/images/posts/stargan/paper/fig2.png" width="500" height="280">
</div>
> Figure 2. Cross-domain 모델과 우리가 제안한 starGAN의 비교. (a)여러 도메인을 처리하기 위해 모든 이미지 도메인 페어에 대해 Cross-domain 모델을 구축해야 합니다. (b) StarGAN은 단일 생성 모델을 사용해 여러 도메인 간의 매핑을 학습할 수 있습니다. 그림은 여러 도메인을 연결하는 별 모댱의 토폴로지를 나타냅니다.

기존 모델들은 $k$ 도메인 간의 변환을 위해서는 $k(k-1)$ 개의 생성 모델을 학습해야 합니다. 얼굴에 대한 도메인만을 사용한다면 얼굴 모양과 같은 모든 도메인의 이미지에서 학습할 수 있는 global features가 존재함에도 불구하고 각 생성 모델은 해당하는 2개의 도메인 데이터에 대해서만 학습해야 하므로 전체 학습 데이터를 완전히 활용할 수 없으므로 효율적이지 않고 효과적이지 못합니다.

StarGAN은 Fig2에서 볼 수 있듯이 단일 생성 모델만으로 여러 도메인 간의 매핑을 학습할 수 있습니다. 생성 모델은 원본 이미지와 도메인 정보를 활용하는데 도메인 정보를 모델에 넘겨주기 위해 라벨를 사용합니다. 라벨은 binary 또는 one-hot vector 형태로 모델이 입력 이미지를 원하는 목표 도메인으로 유연하게 변하는 것을 학습하도록 도와줍니다.

데이터 셋이 하나가 아닌 여러 개를 사용할 경우 라벨에 `mask vector(마스크 벡터)`가 추가해 사용합니다. 사용하는 데이터셋의 라벨만 확인해 명시적으로 제공하는 라벨에만 집중할 수 있도록 어떤 데이터셋을 사용하는지 모델에 알려주는 역할을 합니다.


---

## 사용 데이터셋
StarGAN에서 사용한 데이터셋은 2개로 CelebA와 RaFD가 있습니다. 위의 Figure 1에서 좌측에 해당하는 머리색, 성별, 나이, 피부색은 CelebA 데이터셋에서 나온 라벨이고 우측의 표정에 관한 라벨은 RaFD에 속해있습니다.

### CelebA
<a href="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" target="_blank">The CelebFaces Attributes dataset(CelebA)</a>에는 40개의 이진 속성으로 라벨리 달인 유명인의 202,599장 얼굴 이미지가 포함되어 있습니다. 논문에서는 178x218 크기의 원본 이미지를 178x178 크기로 crop한 후 128x128 크기로 resize합니다. 무작위로 2,000개의 이미지를 테스트 데이터으로 선택하고 나머지 모든 이미지를 학습 데이터로 사용합니다. 40개의 속성 중 머리색(검정색, 금색, 갈색), 성별(남, 여), 나이(젊음, 늚음) 속성을 사용해 7개의 도메인으로 구성합니다.

### RaFD
<a href="https://rafd.socsci.ru.nl/RaFD2/RaFD?p=main" target="_blank">The Radboud Faces Database(RaFD)</a>에는 67명의 참가자로부터 수집한 4,824개의 이미지로 구성되어 있습니다. 각 참가자는 세 가지 다른 방향에서 8가지 표정을 짓고 5개의 카메라 각도에서 촬영합니다. 8가지 표정은 분노, 혐호, 공포, 행복, 슬픔, 놀람, 경멸, 중립입니다. 논문에서는 얼굴이 중앙에 위치하도록 256x256 크기로 이미지를 crop한 후 128x128 크기로 이미지를 resize 합니다.


---

## 단일 데이터셋 학습
CelebA에서 검은색 머리, 금색 머리, 갈색 머리, 성별 남, 성별 여, 나이 젊음, 나이 늙음 속성으로 총 7개의 도메인을 사용합니다. 우선 단일 데이터셋에서 다중 도메인을 사용하는 경우를 살펴봅시다!

<div>
  <img src="/assets/images/posts/stargan/paper/fig3.png" width="600" height="250">
</div>
> Figure 3. 판별 모델 $D$와 생성 모델 $G$ 두 모듈로 구성된 StarGAN 개요.

(a)는 판별 모델 학습 과정을 보여줍니다. 진짜 이미지던 가짜 이미지던 이미지를 입력받으면 해당 이미지가 진짜인지 가짜인지 판별합니다. 여기까지는 이전의 모델들과 큰 차이가 없지만 여기서 domain classification이 추가됩니다. 판별 모델은 입력 받은 이미지가 어떤 도메인을 가지고 있는지 예측하며 이를 domain classification이라 합니다. 실제 데이터셋에는 라벨 값이 함께 있으므로 실제 데이터셋을 학습할 때 판별 모델이 예측한 도메인과 실제 도메인 간의 차이를 classification loss로 계산해 사용합니다.

(b)는 생성 모델이 이미지를 생성하는 과정입니다. 이미지와 함께 condition으로 Target domain에 해당하는 라벨을 입력 값으로 받습니다. 예시로 검은 머리를 가진 젊은 여성의 사진을 나이 든 여성의 사진으로 바꾸고 싶은 경우 사진과 함께 라벨로 [1, 0, 0, 0, 0] (순서대로 검은색 머리, 금색 머리, 갈색 머리, 성별 남/여, 나이 젊음/늙음)을 넣어주면 됩니다. 생성 모델은 이미지를 입력으로 주어진 라벨에 맞는 이미지로 변환한 가짜 이미지를 생성하게 됩니다.

(c)는 CycleGAN에서 소개했던 <a href="https://solee328.github.io/gan/2023/02/09/cyclegan_paper.html#h-32-cycle-consistency-loss" target="_blank">cycle consistency loss</a>와 같은 의미를 가지고 있습니다. (b)에서 생성한 가짜 이미지에 원본 이미지에 해당 하는 검은 머리, 성별 여, 나이 젊음에 대한 라벨([1, 0, 0, 0, 1])을 주어 Reconstructed 이미지를 만듭니다. Reconstructed 이미지와 원본 이미지 간의 차이를 계산해 Reconstruced Loss로 사용합니다. CycleGAN에서는 두 도메인으로 변환하기 위해 생성 모델 2개를 사용해 Cycle Consistency Loss를 계산했지만 StarGAN에서는 하나의 생성모델로 Reconstructed Loss를 계산하는 것이 차이점입니다.

(d)는 (b)에서 생성한 가짜 이미지로 판별 모델을 속이는 모습을 표현한 것으로 생성 모델은 판별 모델을 속이기 위한 가짜 이미지를 생성하고 판별 모델은 진짜와 가짜 이미지를 구별하는 adversarial loss를 나타냅니다.

---

## Loss function
loss는 Adversarial Loss, Domain Classfication Loss, Reconstruction Loss 총 3가지 loss로 나뉩니다.

### Adversarial Loss
Adversarial Loss의 수식은 아래와 같습니다.

$$
\mathcal{L} _{adv} = \mathbb{E} _x [logD _{src}(x)] + \mathbb{E} _{x, c}[log(1-D _{src}(G(x, c)))]
$$

$G$는 이미지 $x$와 원하는 도메인인 타겟 도메인에 해당하는 라벨 $c$를 입력으로 받아 이미지 $G(x, c)$를 생성하고 $D$는 실제 이미지와 가짜 이미지를 구별하고자 합니다. ㅇ


학습 프로세스를 안정화하고 더 높은 품질의 이미지를 생성하기 위해 Eq(1)을 아래와 같이 정의된 <a href="https://arxiv.org/abs/1704.00028" target="_blank">gradient penalty</a>가 있는 <a href="https://arxiv.org/abs/1701.07875" target="_blank">Wasserstein GAN</a>으로 대체합니다.

WGAN에서는 Wasserstein distance를 사용한 새로운 방법을 제시했으나 전제로 1-Lipschitz function에 공간에 위치해야해 제약 조건으로 Weight Clipping을 사용했습니다. 1-Lipschitz는 ~~~~이며 weight clipping으로 ~~~~하게 해주었습니다. Weight Clipping의 방법으로도 기존의 DCGAN과 같은 모델보다 좋은 성능이 나왔으며 CNN, MLP 환경, Batch normalization이 없는 환경에서도 학습이 이루어진 모습을 보여주어 mode collapse 현상 없이 기존의 방법보다 안정적임을 보여주었습니다.

하지만 Weight Clipping은 WGAN 논문에서도 좋지 않은 방법이라 표현하는데, 단순하게 weight를 제한하는 방법일 뿐더러 모델의 성능이 하이퍼 파라미터에 민감하기 때문입니다. 모델의 weight를 $[-c, c]$로 제한한다면 $c$의 값이 하이퍼 파라미터가 됩니다. 이때 $c$의 값이 크면 $[-c, c]$까지 도달하는 시간이 오래 걸려 optimal까지 학습하는 시간이 오래 걸리게 되고 $c$ 값이 작으면 gradient vanishing 문제가 발생하게 됩니다.

이를 해결하기 위해 이후 논문인 WGAN-GP에서는 weight clipping 대신 gradient penalty를 사용합니다. Clipping을 사용하는 대신 gradient norm이 목표 norm 값인 1보다 커지면 모델에 불이익을 주는 방식입니다.

<img src="/assets/images/posts/stargan/paper/wgan.png" width="500" height="70">

$\hat{x}$는 x와 $\sim{x}$ 사이에서 샘플링된 값으로 실제 데이터와 생성된 가짜 데이터의 모든 점이 될 수 있는 값을 의미합니다. 미분 가능한 함수 $f$에서 어떤 점을 뽑아도 $[\nabla f(\hat{x})] = 1$으로 만들어 1-Lipschitz를 만족하게 만들기 때문에 gradient norm 값이 1과 차이가 난다면 오차를 주도록 차이 값을 제곱해 gradient penalty로 사용합니다.

$$
\mathcal{L} _{adv} = \mathbb{E} _x[D _{src}(x)] - \mathbb{E} _{x, c}[D _{src}(G(x, c))] - \lambda _{gp}\mathbb{E} _{\hat{x}}[(\| \nabla _{\hat{x}}D _{src}(\hat{x}) \|_2 - 1)^2]
$$


### Domain classification
판별 모델이 이미지를 입력받아 해당 이미지에 대한 도메인이 무엇인지 예측해 라벨을 출력하고 그 라벨이 실제 라벨과의 차이를 Domain classification loss로 사용합니다. Domain classification loss는 2가지로 나뉘는데 $D$에게 실제 데이터를 준 경우와 $G$가 생성한 가짜 데이터를 준 경우로 나눌 수 있습니다.

우선 $D$에게 실제 데이터를 주어 Domain classification loss를 학습할 때 수식은 아래와 같습니다.

$$
\mathcal{L}^{\mathcal{r}} _{cls} = \mathbb{E} _{x, c'}[-logD _{cls}(c' \mid x)]
$$

수식에서 $c'$는 원본 이미지(실제 데이터 $x$)의 도메인 라벨을 의미하므로 $D _{cls}(c'\mid x)$는 데이터 $x$가 주어졌을 때 $D$가 예측한 $x$의 도메인 라벨을 의미합니다. 위의 $D$를 훈련할 때 위의 목적함수를 최소화함으로써 $D$는 실제 이미지 $x$를 도메인 $c'$로 분류하는 방법을 학습합니다.


다음 수식은 $D$에게 $G$가 생성한 가짜 데이터를 준 경우입니다.

$$
\mathcal{L}^{\mathcal{f}}_{cls} = \mathbb{E}_{x, c}[-logD_{cls}(c | G(x, c))]
$$

생성 모델 $G$는 원본 이미지 $x$와 만들고자 하는 목표인 타겟 도메인 라벨 $c$을 입력으로 받아 이미지($G(x, c)$)를 생성합니다. 생성된 이미지를 $D$에게 입력으로 주어지고 $D$는 이 이미지의 라벨이 무엇인지($D _{cls}(c \mid G(x, c))$) 판별합니다. $G$를 훈련할 때 위의 목적함수를 최소화함으로써 $G$는 $D$가 타겟 도메인 라벨인 $c$로 분류할 수 있는 이미지를 생성하는 방법을 학습합니다.

### Reconstruction Loss
Adversarial loss와 Classification Loss를 최소화함으로써 $G$는 사실적이고 타겟 도메인 라벨 $c$로 분류되는 이미지를 생성하도록 학습합니다. 그러나 위의 2가지 Loss를 최소화해도 $G$가 입력 이미지의 도메인 관련된 부분들은 변경하도록 이미지를 변환하지만 변환된 이미지가 입력 이미지의 내용을 보존한다고 보장할 수는 없습니다. 이 문제를 완화하기 위해 Cycle Consistency Loss를 생성모델에 적용합니다.

$$
\mathcal{L} _{rec} = \mathbb{E} _{x, c, c'}[\| x-G(G(x, c), c') \|_1]
$$

---

## 다중 데이터셋 학습

### Mask Vector


---

## 모델
### 모델 구조
### baseline 비교 모델


---

## 결과
### 단일 데이터셋
#### CelebA
#### RaFD

### 멀티 데이터셋
#### joint training
#### mask vector

---
자세한 모델 구조와 학습에 대한 내용은 다음 모델 구현 글에서 소개하겠습니다.
이번 글도 봐주셔서 감사합니다:)
