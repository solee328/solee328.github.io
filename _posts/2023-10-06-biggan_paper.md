---
layout: post
title: BigGAN - 논문 리뷰
# subtitle:
categories: gan
tags: [gan, BigGAN, 생성 모델, 논문 리뷰]
# sidebar: []
use_math: true
---

이번 논문은 <a href="https://arxiv.org/abs/1809.11096" target="_blank">Large Scale GAN Training for High Fidelity Natural Image Synthesis</a>로 BigGAN이라 불리는 논문입니다.

BigGAN이란 이름에서도 나타내는 것처럼 BigGAN은 기존 GAN의 파라미터의 2~4배의 파라미터를 가지고 있으며

ImageNet의 128x128 해상도에서 Inception Score(IS) Fréchet Inception Distance(FID)를 각각 166.5와 7.4로 이전 글인 SAGAN의 IS인 52.52와 FID 18.65를 넘어서는 class-conditional 이미지 합성 state of the art 모델입니다. BIGGAN에 대해서 지금부터 살펴보겠습니다.:lemon:
<br><br>

---
## 소개
conditional GAN은 많은 발전을 해왔지만 SOTA 모델(SAGAN)조차 아직 실제 이미지(ImageNet)와 차이가 크며, 실제 이미지의 Inception Score는 233에 비교해 SAGAN은 52.5의 Inception Score를 달성했다 소개합니다.

BigGAN은 GAN에서 생성된 이미지들과 실제 이미지인 ImageNet 간의 fidelity(재현성), variety(다양성) 격차를 줄인다는 목표를 가지고 다음과 같은 3가지를 따릅니다.

- GANs가 규모를 키움으로써 큰 이득을 얻는 것을 증명하고 기존에 비해 2~4배 많은 수의 파라미터를 사용하고 8배 큰 batch size 모델을 학습합니다. BigGAN은 일반적인 구조에서 규모 확장성을 개선한 것과 regularization scheme를 수정해 conditioning을 개선한 예시를 소개해 성능을 끌어올린 것을 증명합니다.
- BigGAN은 간단한 sampling 기술로 결과의 fidelity와 variety 사이 trade-off를 명시적으로 find-grained control이 가능한 "truncation trick"에 따를 수 있게 합니다.
- 특정 대규모 GANs가 불안정한 것을 발견했고 이를 묘사하며 분석을 통해 새로운 기술과 기존 기술을 결합한 것이 이런 불안정을 줄일 수 있지만 완벽한 학습의 안정성은 성능을 위해 굉장한 cost를 지불해야만 달성할 수 있다는 것을 증명합니다.

이런 과정을 통해 BigGAN은 class-conditional GANs를 개선해 IS, FID 모두에서 점수를 갱신합니다. ImageNet의 128x128 해상도의 경우 SOTA의 IS, FID인 52.52와 18.65를 BigGAN은 IS, FID를 166.5와 7.4로 향상시킵니다.

그리고 ImageNet보다 훨씬 더 큰 데이터셋 JFT-300M에서도 BigGAN을 학습하며 ImageNet의  128x128, 256x256, 512x512에서 학습된 모델의 가중치 값을 <a href="https://tfhub.dev/s?q=biggan" target="_blank">TF HUB</a>에서 제공합니다.
<br><br>


---

### 개인 정리
SAGAN처럼 spectral normalization(Miyato et al., 2018) 사용
첫번째 단일 값의 실행 추정치(running estimates)로 파라미터들을 정규화(normalization)해 Lipschitz 연속성을 D에 적용하고 top singular direction을 adaptively regularize 해 역 역학(backwards dynamics)를 유도함.
<br><br>

---

## Scaling Up GANs
BigGAN의 첫번째 특징인 규모에 대해서 살펴보겠습니다.

BigGan은 baseline으로 hinge loss를 GAN의 목적함수로 사용한 SAGAN 구조를 사용합니다. class 정보를 class-conditional BatchNorm으로 G에게 제공하고 projection으로 D에게 제공합니다.

최적화 설정은 SAGAN을 따르는데 sepctral Norm을 G에게 적용하며 learning rate를 절반으로 줄여 D step per G step을 2로 사용하도록 수정해 사용합니다.

평가를 위해서 BigGAN은 decay를 0.999로 설정한 ~~를 G의 가중치 평균 이동을 사용합니다.

이전 연구들은 $\mathcal{N}$(0, 0.02$I$)(Radford et al., 2016) 또는 Xavier initialization(Glorot & Bengio, 2010)을 사용했으나 BigGAN은 Orthogonal Initialization을 사용합니다.

- hinge loss를 adversarial loss로 사용한 SAGAN 구조를 사용합니다.
- class 정보는 class-conditional BatchNorm으로 G에게 제공하고 projection으로 D에게 제공합니다
- Spectral Norm을 G에 적용하며, learning rate는 절반으로 줄여 D step per G step을 2로 사용하도록 수정해 사용합니다.


<div>
  <img src="/assets/images/posts/biggan/paper/table1.png" width="600" height="200">
</div>
> Table 1.


<br><br>

---

## Stabilize

<br><br>

---

BIGGAN 논문 리뷰글의 끝입니다. 끝까지 봐주셔서 감사합니다:)
