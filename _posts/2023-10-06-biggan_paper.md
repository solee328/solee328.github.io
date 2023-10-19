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

#### CBN(G)
class-conditional BatchNorm(Dumoulin et al., 2017; de Vreis et al., 2017)

<div>
  <img src="/assets/images/posts/biggan/paper/CBN.png" width="500" height="200">
</div>
> Modulating early visual processing by language의 Figure. 2<br>
왼쪽이 일반적인 batch normalization이고 오른쪽이 conditional batch normalization의 개요를 나타냅니다.

G에는 Conditional Batch Nromalization(CBN) 방식을 사용합니다.<br>
<a href="https://arxiv.org/abs/1707.00683" target="_blank">Modulating early visual processing by language</a>에서 소개되었으며, 기존의 Batch Normalization(BN)으로 클래스 정보가 batch normalization의 learnable parameter인 $\gamma$, $\beta$에 영향을 미칠 수 있도록 해 conditional 정보를 BN에 주는 방법입니다. 주고자 하는 condition에 해당하는 $e_q$를 MLP layer에 통과시켜 channel 수 마다 2개의 값 $\Delta \beta$와 $\Delta \gamma$를 계산합니다.

$$
\Delta \beta = MLP(e_q) \quad \quad \Delta \gamma = MLP(e_q)
$$

이후 Batch Normalization의 $\beta$, $\gamma$에 계산된 값을 더해 Conditional Batch Normalization으로 사용합니다.

$$
\hat{\beta_c} = \beta_c + \Delta \beta_c \quad \quad \hat{\gamma_c} = \gamma_c + \Delta \gamma_c
$$

<a href="https://arxiv.org/abs/1707.00683" target="_blank">Modulating early visual processing by language</a>의 경우 VQA(Visual Question Answering) 논문으로 자연어 처리를 위해 위의 Figure 2에서 question이 LSTM과 MLP를 거쳐 $\Delta \beta$와 $\Delta \gamma$를 계산해 CBN에 사용했지만 BigGAN에서는 LSTM 없이 feature map을 MLP의 입력으로 넣어 $\Delta \beta$와 $\Delta \gamma$를 계산하게 됩니다.

또한 <a href="https://arxiv.org/abs/1707.00683" target="_blank">Modulating early visual processing by language</a>에서는 $\beta$, $\gamma$는 pretrained된 ResNet에서 학습된 pretrained $\beta$, $\gamma$로 고정됩니다. BigGAN에서는 pretrained된 값을 사용하거나 임의의 값으로 초기화된 값을 사용해 학습하는 것 모두 가능할 거 같습니다.



#### Projection(D)

<div>
  <img src="/assets/images/posts/biggan/paper/projection.png" width="500" height="250">
</div>
> CGANS with Projection Discriminator의 Figure 1.<br>
conditional GANs의 Discriminator 모델을 보여줍니다. 가장 우측의 모델이 BigGAN에서도 사용한 Projection Discriminator의 모양입니다.

D에는 Projection Discriminator를 사용합니다.<br>
<a href="https://arxiv.org/abs/1802.05637" target="_blank">cGANs with Projection Discriminator</a>에서 제안한 모델로 GAN의 Discriminator에서 conditional 정보를 다루는 방법들이 발전해온 그림과 함께 Projection Discriminator를 제안합니다.


$$
f(x, y; \theta) := f_1(x, y:\theta) + f_2(x;\theta) = y^T V \phi(x; \theta_{\Phi}) + \psi(\phi(x; \theta_{\Phi}); \theta_{\Psi})
$$


$\phi$는 BigGAN에서 Residual network를 사용하며 $\psi$은 $\phi$와 연결되는 scalar function으로 BigGAN에서는 Linear를 사용합니다. $V$는 $y$의 embedding matrix로 $y^T V$가 Figure.1 (d)에서 $y$가 입력되는 부분을 의미합니다.

$\phi(x; \theta_{\Phi})$는 Residual network $\phi$에 $x$가 입력되었을 때의 feature map으로 이 feature map이 두 갈래로 나뉘게 됩니다. 하나는 Linear인 $\psi$로 입력되고 다른 한 갈래는 condition에 해당하는 $y$의 embedding과 합쳐져 $\psi$ 결과 값과 합쳐져 결과로 출력됩니다.

learned parameters는 $\theta = \{ V, \theta_{\Phi}, \theta_{\Psi} \}$로 adversarial loss에 의해 학습된다고 합니다.

대략적인 흐름을 이해하는 데 도움이 될 것 같아 <a href="https://github.com/pfnet-research/sngan_projection/tree/master" target="_blank">github</a>에 있는 코드를 아래에 가져왔습니다.

```python
class SNResNetProjectionDiscriminator(chainer.Chain):
    def __init__(self, ch=64, n_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.activation = activation
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.block1 = OptimizedBlock(3, ch)
            self.block2 = Block(ch, ch * 2, activation=activation, downsample=True)
            self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True)
            self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True)
            self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True)
            self.block6 = Block(ch * 16, ch * 16, activation=activation, downsample=False)
            self.l7 = SNLinear(ch * 16, 1, initialW=initializer)
            if n_classes > 0:
                self.l_y = SNEmbedID(n_classes, ch * 16, initialW=initializer)

    def __call__(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.activation(h)
        h = F.sum(h, axis=(2, 3))  # Global pooling
        output = self.l7(h)
        if y is not None:
            w_y = self.l_y(y)
            output += F.sum(w_y * h, axis=1, keepdims=True)
        return output
```

논문에 씌여있는 github 코드를 가져온건데 $y^T V \phi(x)$ 부분에서 embedding된 $y$가 $\phi(x)$와 합쳐지지 않고 그냥 $\psi$ 값과 합쳐지는 것 같습니다. ~~$\phi$는 어디로 갔을까요..?~~



### EMA(EWA)
$G$의 weight에 moving average를 사용한다고 해 인용을 확인해보니 PGGAN, ProGAN이라 불리는 <a href="https://arxiv.org/abs/1710.10196" target="_blank">Progressive Growing of GANs for Improved Quality, Stability, and Variation</a>에서 사용한 것으로 learning rate를 decay하도록 따로 설정하지는 않지만 $G$의 출력을 시각화하기 위해 exponential weight average를 사용한다고 합니다. exponential weight average는 가장 최신의 weight의 가중치를 더 크게 반영하고자 이전의 가중치들은 iteration이 반복될 때마다 decay 값인 0.999가 곱해져 축적되어 average 값이 가중치로 사용됩니다.


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
