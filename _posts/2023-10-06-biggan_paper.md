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
BigGAN은 지난 글인 <a href="https://solee328.github.io/gan/2023/09/27/sagan_paper.html" target="_blank">SAGAN - 논문리뷰</a>에서 다뤘던 SAGAN이 모델 기반으로 사용됩니다.

SAGAN과 마찬가지로 adversarial loss로 hinge loss를 사용하며 G와 D 모두에 Spectral Normalization을 사용합니다.

Spectral Norm은 첫번째 단일 값의 실행 추정치(running estimates)로 파라미터들을 정규화(normalization)해 Lipschitz 연속성을 D에 적용하고 top singular direction을 adaptively regularize 해 역 역학(backwards dynamics)를 유도함.

SAGAN에서는 G와 D의 학습 step 수를 1:1로 설정해 동일한 시간에서 더 나은 결과를 얻고자 한 것이 특징이지만 BigGAN에서는 G와 D 학습 step 수를 1:2로 수정한 것을 사용합니다.
<br><br>

---


### Conditioning

<div>
  <img src="/assets/images/posts/biggan/paper/fig15.png" width="600" height="300">
</div>
> figure 15. (a) BigGAN의 $G$의 대표적인 구조<br>
(b) BigGAN의 $G$에 사용되는 Residual Block($ResBlock up$)<br>
(c) BigGAN의 $D$에 사용되는 Residual Block($ResBlock down$)


class 정보를 G와 D에 다른 방식으로 제공합니다.

G에는 single shared class embedding으로 Conditional Batch Normliazation(CBN)과 skip connection(skip-z)를 사용합니다. $z$는 모델 입력에서 한번만 쓰이는게 일반적이지만 BigGAN은 Residual Block마다 class 정보와 함께 입력되는 걸 볼 수 있습니다.

latent vector $z$가 channel 차원에 따라 동일한 크기로 분할되고 각 분할된 $z$는 shared class embedding인 CBN과 연결되어 residual block에 conditioning vector로 전달됩니다. 이 $z$를 hierarchical latent space라 하고 skip connection처럼 여러 계층에 전달되는 $z$를 skip-z라 합니다. skip-z는 약 4% 성능 향상과 함께 학습 속도 또한 18% 향상시켰다고 합니다.

D는 Projection Discriminator 방식을 사용합니다. Residual Block과 Scalar function을 사용해 class 정보를 사용하는 것이 특징입니다.

G와 D에 어떤 방식으로 Condition 정보를 주었는지 하나하나 살펴보겠습니다.


#### Shared embedding / CBN(G)
class-conditional BatchNorm(Dumoulin et al., 2017; de Vreis et al., 2017)

<div>
  <img src="/assets/images/posts/biggan/paper/CBN.png" width="500" height="200">
</div>
> Modulating early visual processing by language의 Figure. 2<br>
왼쪽이 일반적인 batch normalization이고 오른쪽이 conditional batch normalization의 개요를 나타냅니다.

G에는 Conditional Batch Normalization(CBN) 방식을 사용합니다.<br>
<a href="https://arxiv.org/abs/1707.00683" target="_blank">Modulating early visual processing by language</a>에서 소개되었으며, 기존의 Batch Normalization(BN)으로 클래스 정보가 batch normalization의 learnable parameter인 $\gamma$, $\beta$에 영향을 미칠 수 있도록 해 conditional 정보를 BN에 주는 방법입니다. 주고자 하는 condition에 해당하는 $e_q$를 MLP layer에 통과시켜 channel 수 마다 2개의 값 $\Delta \beta$와 $\Delta \gamma$를 계산합니다.

$$
\Delta \beta = MLP(e_q) \quad \quad \Delta \gamma = MLP(e_q)
$$

이후 Batch Normalization의 $\beta$, $\gamma$에 계산된 값을 더해 Conditional Batch Normalization으로 사용합니다.

$$
\hat{\beta_c} = \beta_c + \Delta \beta_c \quad \quad \hat{\gamma_c} = \gamma_c + \Delta \gamma_c
$$


<div>
  <img src="/assets/images/posts/biggan/paper/fig15_b.png" width="600" height="300">
</div>
> Figure 15. (b) BigGAN의 $G$에 사용되는 Residual Block($ResBlock up$)


<a href="https://arxiv.org/abs/1707.00683" target="_blank">Modulating early visual processing by language</a>의 경우 VQA(Visual Question Answering) 논문으로 자연어 처리를 위해 위의 Figure 2에서 question이 LSTM과 MLP를 거쳐 $\Delta \beta$와 $\Delta \gamma$를 계산해 CBN에 사용했지만 BigGAN에서는 LSTM 없이 feature map을 Linear의 입력으로 넣어 $\Delta \beta$와 $\Delta \gamma$를 계산해 CBN을 사용합니다.

각 block의 conditioning은 각 block의 BatchNorm layer에 대한 sample 당 gain과 bias를 생성하도록 선형적으로 projection됩니다. bias projection은 zero-centered되며 gain projection은 1을 중심으로 합니다. residual block 수는 영상 해상도에 따라 달라지므로 z의 전체 차원 128x128일 경우 120, 256x256일 경우 140, 512x512일 경우 160입니다.


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
<br><br>

---


### EMA(EWA)
$G$의 weight에 moving average를 사용한다고 해 인용을 확인해보니 PGGAN, ProGAN이라 불리는 <a href="https://arxiv.org/abs/1710.10196" target="_blank">Progressive Growing of GANs for Improved Quality, Stability, and Variation</a>에서 사용한 것으로 learning rate를 decay하도록 따로 설정하지는 않지만 $G$의 출력을 시각화하기 위해 exponential weight average를 사용한다고 합니다. exponential weight average는 가장 최신의 weight의 가중치를 더 크게 반영하고자 이전의 가중치들은 iteration이 반복될 때마다 decay 값인 0.999가 곱해져 축적되어 average 값이 가중치로 사용됩니다.
<br><br>

---

### Orthogonal

#### Initialization
orthogonal Initialization(Saxe et al., 2014)

논문에서는 $N(0, 0.2I)$ 또는 Xavier initialization이 아닌 orthogonal initialization을 사용했다고 합니다. Orthogonal initialization은 <a href="https://arxiv.org/abs/1312.6120" target="_blank">Exact solutions to the nonlinear dynamics of learning in deep linear neural networks</a>에서 사용된 방법으로 random weight를 svd를 통해 얻은 orthogonal matrix를 initial weight로 사용하는 방법입니다.


#### Regularization
대부분의 이전 연구들은 $z$를 $N(0, I)$ 또는 $U[-1, 1]$에서 선택해 사용했습니다. BigGAN 저자들은 이것에 의문을 가지고 대안을 탐구했습니다.

놀랍게도, 가장 좋은 결과는 학습에서 사용된 것과 다른 잠재 분포에서 샘플링한 것이였습니다. $z \sim N(0, I)$으로 학습된 모델과 normal 분포에서 truncated(범위 밖의 값이 해당 범위에 속하도록 다시 샘플링됨)된 $z$를 사용하는 것은 즉시 IS와 FID 점수를 향상시킵니다. 이것을 Truncation Trick이라 부릅니다. threshold 이상의 크기의 값을 다시 샘플링한 truncated $z$를 사용하면 전체 샘

truncation 없이 가장 잘 작동하는 2개의 latent는 Bernoulli $\{ 0, 1 \}$와 Censored Normal max($N(0, I), 0$)로 두 가지 모두 학습 속도를 향상시키고 최종 성능을 약간 향상시켰지만 truncation에는 덜 적합하다고 합니다.


<div>
  <img src="/assets/images/posts/biggan/paper/fig2.png" width="750" height="230">
</div>
> Figure 2. (a) truncation 증가에 대한 효과. 왼쪽에서 오른쪽으로 threshold 2, 1, 0.5, 0.04로 설정되었습니다.<br>
(b)truncation을 적용해 상태가 좋지 않은 모델의 Saturation artifacts

학습에서는 $N(0, I)$을 사용, threshold 밖으로 나가는 경우 다시 샘플링하는 방법을 사용하는 것

figure 2의 a에서 threshold 값에 따라서 좌측으로 갈수록 다양한 샘플이 나오지만 우측으로 갈수록 하나의 결과가 나오는 것을 볼 수 있습니다. figure 2의 b의 경우는 truncation trick이 통하지 않는 경우로 이런 경우는 해결하기 위해 orthogonal regularization을 도입합니다.


$$
R_{\beta}(W) = \beta \| W^{\top} W -I \|^2_F
$$

weight를 orthogonal하게 제한시키는 경우 문제가 발생
다른 방식을 사용

$$
R_{\beta}(W) = \beta \|W^{\top}W \odot(1-I) \|^2_F
$$

orthogonal regularization이 없는 경우 16%의 경우에만 truncation trick을 적용할 수 있는 반면, orthogonal regularization을 사용해 학습한 경우 60%로 늘어났다고 합니다.
<br><br>

---


## Scaling Up GANs
<div>
  <img src="/assets/images/posts/biggan/paper/table1.png" width="600" height="220">
</div>
> Table 1. ablation 실험을 위한 Fréchet Inception Distance(FID, 낮은 점수가 더 좋음)와 Inception Score(IS, 높은 점수가 더 좋음).<br>
Batch는 batch size, Param은 parameter의 수, CH는 각 계층의 단위 수를 나타내는 channel의 수, Shared는 shared embedding의 사용 여부, Skip-z는 잠재 계층에서 다른 계층으로의 skip connection 사용 여부, Ortho.는 Orthogonal Regularization, Itr는 $10^6$ iteration에서 안정적인지 또는 주어진 iteration에서 붕괴(collapse)되는 나타냅니다.<br>
1 ~ 4행 외에는 8번의 random initialization에 걸쳐 결과가 계산됩니다.

SAGAN이 규모를 키워 어떤 성능을 냈는지 Table 1을 통해 확인할 수 있습니다.

Table 1의 1~4행은 단순히 batch size를 최대 8배까지 증가시키는 것으로 IS 점수는 sota에서 46% 향상됨을 보여줍니다. 이런 scale up으로 인한 주목할 만한 부작용은 BigGAN이 더 적은 반복으로 최종 성능에 도달하지만, 불안정해지거나 완전한 training collapse를 겪는다는 것입니다. 이 실험의 경우 collapse 직전에 저장된 checkpoint의 점수를 보고한 것이라 합니다.

이후 channel 수를 50% 증가시켜 파라미터 수를 약 2배로 늘려 IS가 21% 더 개선되었습니다. 깊이를 2배로 늘리는 것은 처음엔 개선으로 이어지지 않았지만 residual block 구조를 사용하는 BigGAN-deep 모델에서 해결되었다고 합니다.


<br><br>

---

## Collapse
위에서 언급된 모델과 기술들의 사용으로 BigGAN은 대규모 모델 사용과 대규모 batch 학습으로 기존 state of the art를 개선했습니다. 하지만 BigGAN은 training collapse를 겪기 때문에 조기 중단(early stopping) 실행을 필요로 합니다.

BigGAN의 저자들은 이전 논문들에서는 안정적이였던 설정들이 BigGAN과 같은 대규모 모델에 적용될 때 불안정해지는 이유를 G와 D의 weight spectra를 분석해 모델의 collapse 원인에 대해 분석합니다.


### Collapse 원인

#### G
<div>
  <img src="/assets/images/posts/biggan/paper/fig3_a.png" width="400" height="300">
</div>
> Figure 3. (a) Spectral Normalization 전 $G$ layer들의 첫번째 single value $\sigma_0$의 plot<br>
$G$의 대부분의 layer들은 잘 작동하는 spectra를 가지고 있지만, 제약이 없는 경우 일부가 학습 내내 값이 상승하고 collapse 시 값이 폭발합니다.



#### D
<div>
  <img src="/assets/images/posts/biggan/paper/fig3_b.png" width="400" height="300">
</div>
> Figure 3. (b) Spectral Normalization 전 $G$ layer들의 첫번째 single value $\sigma_0$의 plot<br>
$D$의 spectra는 noise가 있지만 더 잘 작동합니다.



R1 zero-centered gradient penalty
$$
R_1 := \frac{\gamma}{2} \mathbb{E} _{p \mathcal{D}(x)}[\| \nabla D(x) \|^2_F]
$$





### Collapse 방지
collapse의 증상은 갑작스럽게 발생해 수 백번 iteration 내에 최고치의 결과 품질이 최저치로 떨어진다고 합니다. 저자들은 G의 Singular value가 갑작스럽게 커질 때 collapse를 감지할 수 있지만 singular value가 특정 값(threshold) 이상이 될 때 collapse하는 것은 아니라 합니다. 모델이 collapse되기 1~2만 iteration의 모델 checkpoint에서 일부 hyperparameter를 수정해 collapse를 방지하거나 지연시킬 수 있는지에 대한 실험들을 진행했습니다.

- G, D, G&D의 learning rate를 초기 값에 비해 높이면 즉각적인 collapse가 발생. 초기 learning rate로 설정했을 때 문제 없던 값으로 변경했을 때도 collapse가 발생.
- G의 learning rate를 줄이되 D의 learning rate를 그대로 유지하면 경우에 따라 10만 iteration 이상 collapse를 지연시킬 수 있으나 학습이 실패할 가능성이 있다. 반대로 G의 learning rate를 유지하면서 D의 learning rate를 낮추면 즉각적인 collapse로 이어짐. 저자들은 D의 learning rate를 줄이는 것이 D가 G를 따라잡을 수 없어(keep up) 학습이 collapse 된다고 생각해 G와 D 학습 step 수를 조절했지만 영향을 미치지 않거나 collapse를 지연시키며 학습이 실패함.

위의

Appendix G 참고해 더 적을 것



<br><br>

---

BIGGAN 논문 리뷰글의 끝입니다. 끝까지 봐주셔서 감사합니다:)

지금까지 논문 리뷰의 논문 중 appendix가 가장 긴 논문이였고 정말 다양하고 많은 실험들을 진행했다는 것을 알 수 있는 논문이었습니다.

글에는 적지 않았지만 논문의 Appendix H에서는 실험을 진행했지만 성능을 저하시키거나 영향을 미치지 못한 작업들이 나열되어 있으며 Appendix I에는 실험에 사용한 learning rate, R1 gradient penalty, Dropout rate, Adam $\beta_1$, Orthogonal Regularization penalty 수치들이 정리되어 있습니다. 직접 모델을 실험해 성능 또는 안정성을 개선하고자 하시는 분들에게는 많은 도움이 될 것 같으니 참고하시면 좋겠습니다.
