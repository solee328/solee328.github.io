---
layout: post
title: SAGAN - 논문 리뷰
# subtitle:
categories: gan
tags: [gan, SAGAN, attention, spectral, TTUR, 생성 모델, 논문 리뷰]
# sidebar: []
use_math: true
---

이번 논문은 self-attention을 Generative model에 적용한 SAGAN(<a href="https://arxiv.org/abs/1805.08318" target="_blank">Self-Attention Generative Adversarial Network</a>)입니다. 사실 BIGGAN 논문을 보다 해당 논문이 SAGAN 모델을 바탕으로 한 걸 알게 되어서 SAGAN를 먼저 하게 되었습니다ㅎㅅㅎ

Self-Attention Generative Adversarial Network(SAGAN)은 convolution을 self-attention으로 대체해 long-range dependency 모델링이 가능하도록 제안된 모델입니다. 또한 spectral normalization과 TTUR을 사용해 Inception Score, Fréchet Inception Distance 모두에서 기존 State-of-the-art를 능가하는 성능을 보여주었습니다. 이런 SAGAN에 대해서 지금부터 살펴보겠습니다.:lemon:
<br><br>

---
## 소개
GANs는 ImageNet과 같은 multi-class를 모델링 시 문제는 가지고 있었습니다. 당시 class conditional image generation task에서 SOTA인 <a href="https://arxiv.org/abs/1802.05637" target="_blank">CGANS with Projection Discriminator</a>는 간헐천, 계곡과 같이 객체의 구조적 제약이 거의 없는 이미지(텍스처(질감)으로 구별 가능한 바다, 하늘과 같은 풍경) 생성은 탁월하지만, 하프, 크로스워드 등 일부 클래스(개와 같은 클래스의 이미지는 개의 털 텍스처(질감)은 성공적으로 생성되지만 일부 발이 생성되지 않는 경우가 발생)에서 기하학적/구조적 패턴을 파악하지 못합니다 실제로 위 논문의 Figure 7에서 퓽경의 FID는 낮지만 객체에 대한 FID는 높은 것을 확인할 수 있습니다.

<details>
<summary>FID(Fréchet Inception Distance)</summary>
<span style="color:gray">
  <a href="https://arxiv.org/abs/1706.08500" target="_blank">Fréchet Inception Distance(FID)</a>는 생성 모델에서 생성된 이미지의 품질을 평가하는 데 사용되는 metric으로 Inception Score(IS)를 개선하기 위해 제안되었습니다. 두 분포의 거리를 계산하는 metric으로 값이 낮을 수록 분포가 가까워 생성 이미지가 실제 이미지와 유사함을 의미해 좋습니다.<br><br>

  FID를 계산하기 위해서 우선 pretrain된 Inception V3를 사용해 실제 이미지와 생성된 이미지의 (2048, ) 크기의 feature map을 계산합니다. 계산된 이 feature map들의 분포 차이를 계산하기 위해 정규분포(Gaussian distribution)를 사용합니다. 정규분포는 평균과 분산이 주어져 있을 때 엔트로피를 최대화하는 분포이므로 다차원 정규분포를 따른다고 가정해 두 feature map의 평균(mean)과 공분산(covariance) 차이를 이용해 두 분포의 차이를 계산합니다. 이때 차이는  Wasserstein-2 distance라고도 불리는 Fréchet distance로 계산합니다. 아래 수식이 Fréchet distance를 활용한 FID의 수식입니다.<br><br>

  $
  d^2((m,C),(m_w, C_w)) = \| m-m_w \|^2_2 + Tr(C + C_w - 2(CC_w)^{1/2})
  $

  <br><br>
  $m$, $C$는 실제 이미지의 feature map의 평균과 공분산이고 $m_w$, $C_w$는 생성된 이미지의 feature map의 평균과 공분산이며, Tr은 행렬의 대각합(trace)를 의미합니다.<br><br>

  참고<br>
  - 페이오스님의 <a href="https://m.blog.naver.com/chrhdhkd/222013835684" target="_blank">GAN 평가지표</a><br>
  - viriditass.log님의 <a href="https://velog.io/@viriditass/GAN%EC%9D%80-%EC%95%8C%EA%B2%A0%EB%8A%94%EB%8D%B0-%EA%B7%B8%EB%9E%98%EC%84%9C-%EC%96%B4%EB%96%A4-GAN%EC%9D%B4-%EB%8D%94-%EC%A2%8B%EC%9D%80%EA%B1%B4%EB%8D%B0-How-to-evaluate-GAN" target="_blank">GAN은 알겠는데, 그래서 어떤 GAN이 더 좋은건데?</a>
  <br>
</span>
</details>
<br>


<div>
  <img src="/assets/images/posts/sagan/paper/conv.png" width="600" height="500">
</div>
> convolution이 깊어지면 longdependency가 가능

이에 대한 가능한 설명은 모델이 서로 다른 이미지 영역에 걸쳐 종속성(dependency)를 모델링하기 위해 convolution에 크게 의존한다는 것입니다. convolution은 receptive field가 local에 해당하므로 long range dependency를 위해서는 여러 convolution 레이어를 통과한 후에만 처리할 수 있습니다. 따라서 작은 모델들은 layer 수가 작아 long range dependency 표현 자체가 어렵습니다. 모델이 아닌 최적화 알고리즘을 사용해서도 long range dependency를 포착하도록 할 수 있겠지만 이를 위한 파라미터 값을 발견하기 어려울 뿐만 아니라 이 경우 새로운 입력에 대해 실패하기도 쉬워집니다. convolution의 local을 키우기 위해 convolution kernel의 크기를 증가시키면 receptive field가 커지니 네트워크의 표현 용량을 증가시킬 수 있지만 local convolutional 구조를 사용해 얻은 계산 및 통계의 효율성이 손실됩니다.

따라서 논문에서는 Self-Attention을 도입한 Self-Attention Generative Adversarial Networks(SAGANs)를 제안합니다. self-attention module은 모든 위치에서 feature의 가중치 합으로 위치 반응을 계산하며, attention vector는 적은 비용으로 계산이 가능하기에 long range dependency와 통계 효율성 사이의 더 나은 균형을 보여줍니다.

self-attention 외에도, conditioning(조건)에 대한 기술을 추가합니다. <a href="https://arxiv.org/abs/1802.08768" target="_blank">Is Generator Conditioning Causally Related to GAN Performance?</a>는 well-condition 생성 모델이 더 나은 성능을 보이는 경향을 보여주었는데, 더 좋은 conditioning을 위해 <a href="https://arxiv.org/abs/1802.05957" target="_blank">Spectral Normalization for Generative Adversarial Networks</a>에서 판별 모델에만 적용되었던 spectral normalization 기술을 생성 모델에 적용합니다.

제안된 방법들로 Inception score의 최고점을 36.8에서 52.52로 높였고 Fréchet Inception Distance를 27.62에서 18.65로 줄임으로써 이전 SOTA를 능가함을 보여주었다 합니다. 코드는 <a href="https://github.com/brain-research/self-attention-gan/tree/master" target="_blank">self-attention-gan(TensorFlow)</a>와 <a href="https://github.com/heykeetae/Self-Attention-GAN/tree/master" target="_blank">Self-Attention-GAN(PyTorch)</a>를 참고했습니다.
<br><br>

---

## Self-Attention

<div>
  <img src="/assets/images/posts/sagan/paper/fig1.png" width="600" height="200">
</div>
> Fig.1.


$$
\beta _{j, i} = \frac{exp(s _{ij})}{\sum ^N _{i=1}exp(s _{ij})}, \text{ where } s _{ij} = \mathbf{f}(\mathbf{x_i})^T \mathbf{g}(\mathbf{x_j})
$$

attention map은 픽셀 간의 관계를 나타내는데, 수식의 $\beta _{j, i}$는 $j$번째 영역을 합성 할 때 모델이 $i$번째 위치에 어느 정도 관심을 기울이는지를 나타냅니다.
attention 모듈은 생성 모델과 판별 모델 모두에 적용되었습니다.




attention map visualize


$$
\mathbf{o_j} = \mathbf{v} \left ( \sum^N_{i=1}\beta_{j, i}\mathbf{h}(\mathbf{x_i}) \right ), \mathbf{h}(\mathbf{x_i}) = \mathbf{W_h x_i}, \mathbf{v}(\mathbf{x_i}) = \mathbf{W_v x_i}.
$$

self-attention feature maps는 각 픽셀과 전체 feature map 간의 관계를 나타냅니다.



$$
\mathbf{y_i} = \gamma \mathbf{o_i} + \mathbf{x_i}
$$

최종 출력은 위와 같은데,  $\gamma$는 학습 가능한 스칼라 값으로 0으로 초기화된 상태에서 학습을 시작합니다.


<br><br>

---

## Loss

$$
\begin{align} L_D & = -\mathbb{E} _{(x, y) \sim p _{data}}[min(0, -1 + D(x, y))] \\ & = - \mathbb{E} _{z \sim p_z, y \sim p _{data}}[min(0, -1-D(G(z), y))], \\ L_G &= -\mathbb{E} _{z \sim p_z, y \sim p _{data}} D(G(z), y) \end{align}
$$

Adversarial loss는 hinge를 최소화하는 방법으로 $D$와 $G$를 번갈아가며 학습합니다.
<br><br>

---

## Stabilize
GANs 학습을 안정화하기 위해 Spectral normalization과 Two Time Scale Update Rule(TTUR)을 사용합니다.


### Spectral normalization
Spectral Normalizationd은 <a href="https://arxiv.org/abs/1802.05957" target="_blank">Spectral Normalization for Generative Adversarial Networks</a>에서 GANs 학습 안정화를 위해 판별 모델에 적용되었습니다. 각 layer의 spectral norm을 제한해 판별 모델의 Lipschitz 상수를 제한하는 방법으로 모든 가중치 레이어의 spectral norm은 1로 설정하는 것이 지속적으로 잘 수행되기 때문에 다른 normalization 방법과 비교해 추가적인 hyperparameter 튜닝을 필요로 하지 않는다 합니다. 또한 계산 비용이 적은 것이 장점입니다.

SAGAN은 생성 모델에도 Spectral normalization을 적용하는 것으로 생성 모델의 파라미터 크기의 상승을 방지하고 비정상적인 gradient를 피할 수 있어 Spectral normalization을 생성 모델과 판별 모델 모두에 적용합니다. 이후 생성 모델과 판별 모델 모두의 Spectral normalization이 안정적인 학습을 보여줄 뿐만 아니라 생성 모델 업데이트 당 판별 모델 업데이트 수를 더 적게 만드는 것이 가능해 학습에 대한 계산 비용을 크게 감소시킨다는 것을 발견했다 합니다.


### TTUR
Two Time-Scale Update Rule(TTUR)은 <a href="https://arxiv.org/abs/1706.08500" target="_blank">GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium</a>에서 제안한 방법으로 생성 모델과 판별 모델에 별도의 learning rate를 사용하는 방법입니다. 판별 모델 학습 : 생성 모델 학습 = 5 : 1과 같이 판별 모델의 느린 학습 문제를 보완하기 위해 SAGAN에서는 TTUR을 사용해 판별 모델의 학습 step 수를 더 적게 사용해 동일한 시간에서 더 나은 결과를 얻고자 했습니다.
<br><br>

---

## 결과
정략적 평가를 위해 논문에서는 conditional class 분포와 marginal class 분포 사이의 KL divergence를 계산하는 Inception score와 생성된 이미지와 실제 이미지의 Inception-v3의 feature space의 Wasserstein-2 distance를 계산하는 FID, 각 class 내의 생성된 이미지와 실제 이미지 사이를 비교하는 Intra FID를 계산합니다.

SAGAN 모델은 128x128 이미지를 생성하도록 설계되었으며, Spectral normalization은 생성 모델과 판별 모델에 모두 사용되었습니다. TTUR을 사용하기 때문에 판별 모델의 learning rate는 0.0004이고 생성 모델의 learning rate는 0.0001로 설정되었으며 Adam optimizer(β1 = 0 and β2 = 0.9)를 사용했다 합니다.

### Spectral $ TTUR

<div>
  <img src="/assets/images/posts/sagan/paper/fig3.png" width="600" height="400">
</div>
> Fig.3.

<div>
  <img src="/assets/images/posts/sagan/paper/fig4.png" width="600" height="400">
</div>
> Fig.4.


생성 모델과 판별 모델에 Spectral normalizatiokn(SN)을 적용하고 불균형한 learning rate(TTUR)을 적용해 제안된 안정화 기술의 효과를 평가하기 위한 실험을 진행합니다. `SN on G/D`와 `SN on G/D + TTUR`은 당시 SOTA로 baseline으로 사용된 <a href="https://arxiv.org/abs/1802.05637" target="_blank">CGANS with Projection Discriminator</a>와 비교됩니다. 이 baseline 모델은 SN이 판별 모델에만 사용되었기에 `baseline: SN on D`로 표시됩니다.

판별 모델 $D$과 생성 모델 $G$에 대해 1:1, 즉 한번씩 번갈아가면서 학습을 하면 Figure 3의 가장 왼쪽 하단의 그림과 같이 학습이 매우 불안정해지며 학습 초기에 mode collapse를 보여줍니다. Figure 4의 가장 왼쪽 상단의 그림은 10k iteration에서 baseline 모델에 의해 무작위로 생성된 일부 이미지를 보여줍니다. 원본 논문에서 이런 불안정한 학습 동작들은 $D$와 $G$의 학습 비율을 5:1로 설정해 크게 완화할 수 있지만, 그만큼 학습 시간이 길어지게 됩니다. 제안된 기술은 1:1 학습을 하기 때문에 동인한 시간이 주어졌을 때 모델이 더 나은 결과를 얻을 수 있습니다.

Figure 3의 중간 하단 그림에서 볼 수 있듯이, 생성 모델과 판별 모델에 SN을 적용해 1:1 학습을 진행했을 때 `SN on G/D`가 크게 안정화되었습니다. 그러나 학습 중 결과의 품질이 계속해서 좋아지지만은 않습니다. 260k iteration에서 FID와 IS가 떨어지기 시작합니다. Figure 4의 `SN on G/D`의 160k의 FID 값은 33.39였지만 `SN on G/D`의 260k의 FID값은 72.41로 오히려 결과 이미지의 품질이 떨어짐을 볼 수 있습니다.

하지만 Spectral normalization 뿐만 아니라 생성 모델과 판별 모델에 다른 learning rate(TTUR)을 사용해 학습시킨 `SN on G/D + TTUR`에 의해 생성된 이미지의 품질은 전체 학습 과정 동안 계속해서 향상되어 1M iteration 동안 결과 품질이나 FID 또는 Inception score의 현저한 감소를 관찰되지 않아 정량적, 정성적 결과 모두 GAN에 Spectral normalization과 TTUR을 적용해 1:1 업데이트로 학습하는 것이 안정화에 효과적이라 할 수 있습니다. 또한 두 기법 모두 어떤 모델이던 부가적으로 적용할 수 있는 기술이므로 논문의 이후 실험들에는 생성 모델과 판별 모델에 모두 Spectral normalization과 TTUR을 사용해 1:1 업데이트로 학습했습니다.


### Self-attention

<div>
  <img src="/assets/images/posts/sagan/paper/table1.png" width="400" height="80">
</div>
> Table.1.

Self-attention mechanism의 효과를 확인하기 위해 feature map 크기에 따라 Self-attention을 적용해 비교합니다. 8x8 크기와 같이 feature map이 작은 경우 feature map이 작기 때문에 self-attention이 local convolution과 유사한 역할만을 수행하게 되고 feature map이 커질 경우 더 많은 condition을 선택할 수 있기 때문에 long-range dependency 모델링이 가능해지게 되며 FID 값도 향상됨을 볼 수 있습니다.

동일한 파라미터의 상태에서 self-attention block은 residual block과 비교해 더 나은 결과를 얻을 수 있음을 볼 수 있습니다. 8x8 feature map에서 self-attention을 residual block으로 교체하면 학습이 안정적이지 않아 성능이 크게 저하되고 FID 값이 22.98에서 42.13으로 증가합니다. 학습이 순조롭게 진행되는 32x32 feature map에서도 self-attention block을 residual block으로 교체하면 FID와 Inception score가 더 나쁜 결과를 보여줌을 통해 SAGAN을 사용해 성능이 향상되는 것이 단순히 모델 깊이와 용량의 증가 때문이 아님을 알 수 있습니다.


<div>
  <img src="/assets/images/posts/sagan/paper/fig5.png" width="600" height="400">
</div>
> Fig.5.

SAGAN에서 다양한 이미지에 생성 모델의 attention weight를 시각한 결과를 Figure 5에서 확인할 수 있습니다.


### SOTA와 비교

<div>
  <img src="/assets/images/posts/sagan/paper/table2.png" width="400" height="80">
</div>
> Table.2.

당시 ImageNet의 class conditional 이미지 생성의 state-of-the-art GAN 모델인 <a href="https://arxiv.org/abs/1802.05637" target="_blank">CGANS with Projection Discriminator(SNGAN-projection)</a>와 Conditional GAN 모델인 <a href="https://arxiv.org/abs/1610.09585" target="_blank">AC-GAN</a>과 Inception Score, Intra FID, FID를 비교한 결과를 Table 2에서 확인하실 수 있습니다.
SAGAN은 3 종류의 metric에서 모두 최고를 달성했습니다.


<div>
  <img src="/assets/images/posts/sagan/paper/fig6.png" width="600" height="400">
</div>
> Fig.6.

Figure 6은 ImageNet의 대표적인 클래스에 대한 생성된 이미지를 보여줍니다. SAGAN은 금붕어(goldfish), 세인트버나(saint bernard)와 같이 복잡한 기하학적, 구조적 패턴을 가진 클래스를 합성하기 위해 <a href="https://arxiv.org/abs/1802.05637" target="_blank">CGANS with Projection Discriminator(SNGAN-projection)</a>보다 Intra FID 점수가 낮아 더 나은 성능을 보임을 확인할 수 있습니다. 하지만 반대로 질감으로 구별될 수 있어 기하학적, 구조적 패턴이 거의 없는 돌담(stone wall), 산호 곰팡이(coral fungus)의 경우 오히려 성능이 낮다는 것 또한 확인할 수 있습니다.

SAGAN은 기하학적, 구조적 패턴에 강해 self-attention mechanism을 long-range global dependency를 포착하기 위해 convolution과 같이 사용해 상호 보완적으로 작동해 좋은 결과를 이끌어 내지만, 단순 텍스터에 대한 dependency의 경우 local convolution과 유사한 역할을 해 좋은 결과를 이끌어 내지 못한다고 합니다.



<br><br>

---
ㅇㅇㅇ
