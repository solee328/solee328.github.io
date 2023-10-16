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

## Setting
우선 BigGAN이 사용한 구조와 설정에 대해서 살펴보겠습니다.

BigGAN이 사용한 구조와 설정은 다음과 같습니다.
- hinge loss를 adversarial loss로 사용한 SAGAN 구조를 사용합니다.
- class 정보는 class-conditional BatchNorm으로 G에게 제공하고 projection으로 D에게 제공합니다.
- Spectral Norm을 G에 적용하며, learning rate는 절반으로 줄여 D step per G step을 2로 사용하도록 수정해 사용합니다.
- decay 0.9999인 moving averages of G's weight
- Orthogonal Initialization을 사용합니다.


### SAGAN
SAGAN은 저번 글이고 hinge loss로 다뤘다

Spectral Norm은 SAGAN을 따라함. 첫번째 단일 값의 실행 추정치(running estimates)로 파라미터들을 정규화(normalization)해 Lipschitz 연속성을 D에 적용하고 top singular direction을 adaptively regularize 해 역 역학(backwards dynamics)를 유도함.
SAGAN에서는 1:1을 했지만 1:2로 수정한 것을 사용


### Conditioning

class 정보를 G와 D에 다른 방식으로 제공합니다.



class-conditional BatchNorm(Dumoulin et al., 2017; de Vreis et al., 2017)

G에는 Conditional Batch Nromalization(CBN) 방식을 사용합니다.
<a href="https://arxiv.org/abs/1707.00683" target="_blank">Modulating early visual processing by language</a>에서 소개되었으며, 기존의 Batch Normalization(BN)으로 클래스 정보가 batch normalization의 learnable parameter인 $\gamma$, $\beta$에 영향을 미칠 수 있도록 해 conditional 정보를 BN에 주는 방법입니다. 주고자 하는 정보 $e_q$를 MLP layer에 통과시켜 channel 수 마다 2개의 값 $\Delta \beta$와 $\Delta \gamma$를 계산합니다.

$$
\Delta \beta = MLP(e_q) \quad \quad \Delta \gamma = MLP(e_q)
$$

이후 Batch Normalization의 $\beta$, $\gamma$에 계산된 값을 더해 Conditional Batch Normalization으로 사용합니다.

$$
\hat{\beta_c} = \beta_c + \Delta \beta_c \quad \quad \hat{\gamma_c} = \gamma_c + \Delta \gamma_c
$$






projection(Miyato & Koyama, 2018)


### weights moving
G의 가중치 이동 평균(Karras et al.(2018); Mescheder et al. (2018); Yazc et al. (2018))


### Initialization
orthogonal Initialization(Saxe et al., 2014)


## Truncation trick
대부분의 이전 연구들은 $z$를 $N(0, I)$ 또는 $U[-1, 1]$에서 선택해 사용했습니다. BigGAN 저자들은 이것에 의문을 가지고 Appendix E에서 대안을 탐구했습니다.


놀랍게도, 가장 좋은 결과는 학습에서 사용된 것과 다른 잠재 분포에서 샘플링한 것이였습니다. $z \sim N(0, I)$으로 학습된 모델과 normal 분포에서 truncated(범위 밖의 값이 해당 범위에 속하도록 다시 샘플링됨)된 $z$를 사용하는 것은 즉시 IS와 FID 점수를 향상시킵니다. 이것을 Truncation Trick이라 부릅니다. 임계값 이상의 크기의 값을 다시 샘플링한 truncated $z$를 사용하면 전체 샘


## Scaling Up GANs
<div>
  <img src="/assets/images/posts/biggan/paper/table1.png" width="600" height="200">
</div>
> Table 1.

SAGAN이 규모를 키워 어떤 성능을 냈는지 Table 1을 통해 확인할 수 있습니다.

Table 1의 1~4행은 단순히 batch size를 최대 8배까지 증가시키는 것으로 IS 점수는 sota에서 46% 향상됨을 보여줍니다. 이런 scale up으로 인한 주목할 만한 부작용은 BigGAN이 더 적은 반복으로 최종 성능에 도달하지만, 불안정해지거나 완전한 training collapse를 겪는다는 것입니다. 이 실험의 경우 collapse 직전에 저장된 checkpoint의 점수를 보고한 것이라 합니다.

이후 channel 수를 50% 증가시켜 파라미터 수를 약 2배로 늘려 IS가 21% 더 개선되었습니다. 깊이를 2배로 늘리는 것은 처음엔 개선으로 이어지지 않았지만 residual block 구조를 사용하는 BigGAN-deep 모델에서 해결되었다고 합니다.


<br><br>

---

## Collapse

<br><br>

---

BIGGAN 논문 리뷰글의 끝입니다. 끝까지 봐주셔서 감사합니다:)
