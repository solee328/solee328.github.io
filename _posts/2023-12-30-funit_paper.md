---
layout: post
title: FUNIT - 논문 리뷰
# subtitle:
categories: gan
tags: [gan, FUNIT, 생성 모델, 논문 리뷰]
# sidebar: []
use_math: true
---

이번 논문은 FUNIT이라 불리는 <a href="https://arxiv.org/abs/1905.01723" target="_blank">Few-shot Unsupervised Image-to-Image Translation</a>입니다 :lemon:

Unsupervised image-to-image translation 모델들의 단점인 특정 클래스에 대한 입력을 수행하기 위해서는 해당 클래스에 대한 수많은 데이터셋을 학습하는 것을 해결하고자, FUNIT은 few-shot을 적용해 새로운 클래스의 이미지 단 몇장으로도 가능한 이미지 변환을 제안합니다. 또한 Few-shot unsupervised image-to-image translation task뿐만 아니라 few-shot classification task와 기존의 unsupervised image-to-image translation task 설정에서도 기존의 state-of-the-art 모델들을 능가하며 성능을 입증합니다.

지금부터 FUNIT에 대해 살펴보겠습니다 :eyes:
<br><br>

---

## 소개
<a href="https://arxiv.org/abs/1606.07536" target="_blank">CoGANs</a>, <a href="https://arxiv.org/abs/1703.00848" target="_blank">UNIT</a>, <a href="https://arxiv.org/abs/1703.05192" target="_blank">DiscoGAN</a>, <a href="https://arxiv.org/abs/1703.10593" target="_blank">CycleGAN</a>과 같은 연구들은 Unsupervised Image-to-Image Translation 설정에서 이미지 클래스를 변환하는 것에 성공했지만, 새로운 클래스에 대한 소수의 이미지들에서 일반화(generalization)은 불가능합니다. 이 모델들이 이미지 변환을 수행하기 위해서는 모든 클래스의 대규모 학습 셋이 필요합니다.

Unsupervised Image-to-Image Translation task의 모델들과는 달리 인간은 일반화(generalization) 능력은 뛰어납니다. 이전에 보지 못했던 이국적인 동물의 사진이 주어져있을 때, 우리는 그 동물이 다른 자세로 있는 상상이 가능합니다. 예를 들어, 서 있는 호랑이를 처음 본 사람이더라도 일생 동안 본 다른 동물들에 대한 정보를 통해 호랑이가 누워있는 모습을 상상하는 것에 어려움이 없습니다.

FUNIT은 이런 격차를 줄이기 위한 시도로, test time에 학습 중에 모델이 보지 못한 클래스의 이미지인 target class의 이미지들을 학습 데이터셋에 포함된 클래스 이미지들인 source class의 이미지와 유사한 이미지로 매핑하는 것을 목표로 하는 Few-shot UNsupervised Image-to-Image Translation(FUNIT) 프레임워크를 제안합니다.
<br><br>

---

## 모델
FUNIT은 <a href="https://papers.nips.cc/paper_files/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html" target="_blank">Generative Adversarial Network(GAN)</a>을 기반으로 하며, conditional image generator $G$와 multi-task adversarial discriminator $D$로 구성됩니다.

하나의 이미지를 입력으로 받는 기존의 unsupervised image translation framework인 <a href="https://arxiv.org/abs/1703.00848" target="_blank">UNIT</a>, <a href="https://arxiv.org/abs/1703.10593" target="_blank">CycleGAN</a>의 conditional image generator들과는 달리 FUNIT의 $G$는 클래스는 $c_x$에 속하는 content 이미지 $\mathrm{x}$와 클래스 $c_y$에 속하는 $K$개의 이미지 집합 $\{y_1, ..., y_K\}$을 동시에 입력으로 가지며 출력 이미지 $\bar{\mathrm{x}}$를 생성합니다.

$$
\bar{\mathrm{x}} = G(\mathrm{x}, \{ y_1, ..., y_K \})
$$


<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/89815e02-3724-4a8e-b528-3e739900b117" width="800" height="200">
</div>
> Figure 1: <b>Training.</b> 학습 셋은 여러 객체 클래스(source class)로 이루어져 있습니다. FUNIT 모델이 source 클래스 사이 이미지를 변환하도록 학습시킵니다.<br>
<b>Deployment.</b> 학습된 모델이 학습 중에 target class의 이미지를 본 적이 없음에도 불구하고 source class의 이미지를 target class와 유사한 이미지로 변환합니다. FUNIT 생성 모델은 1) content 이미지와 2) target class 이미지 셋, 2가지 입력을 사용합니다. target class 이미지와 유사한 입력 이미지의 변환을 생성하는 것을 목표로 합니다.

Figure 1의 Training에서 볼 수 있듯, $G$는 content 이미지 $\mathrm{x}$와 target class($c_y$)의 $K$개의 이미지를 입력으로 받습니다. 입력 받은 Content 이미지 $\mathrm{x}$가 클래스 $c_y$의 style을 가지며 $\mathrm{x}$의 구조를 가질 수 있도록 결과를 생성합니다.

$\mathbb{S}$가 학습 데이터 셋(source class), $\mathbb{T}$가 target class 셋을 나타낼 때, 학습 단계(training time)에서 $G$는 $c_x, c_y \in \mathbb{S}$이고 $c_s \not= c_y$인 2개의 랜덤 추출된 source class 사이를 변환하는 방법을 학습합니다.

Test 단계(testing time)에서, $G$는 학습하지 않은 target class $c \in \mathbb{T}$로부터 $K$ 개의 이미지를 가져가 source class 중 임의의 클래스에서 샘플링된 이미지를 target class와 유사한 이미지로 매핑합니다.

모델의 목표를 확인했으니, 모델을 구성하는 $G$와 $D$의 구조를 살펴보겠습니다.


### Generator

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/5ec156d3-b711-4935-a912-453332779b24" width="800" height="500">
</div>
> Figure 6. 생성 모델 구조 시각화. 변환 결과  $\bar{\mathrm{x}}$를 생성하기 위해, 생성 모델은 클래스 이미지들 $y_1, ..., y_k$(target class)에서 추출된 class latent code $\mathrm{z}_y$와 입력 content 이미지에서 추출된 content latent code $\mathrm{z}_x$를 결합합니다. 비선형성(nonlinearity)와 정규화(normalization)연산은 시각화에 포함되지 않습니다.

few-shot 이미지 변환 모델인 FUNIT의 생성 모델은 Figure 6의 시각화에서 표현된 것처럼 Content Encoder $E_x$, Class Encoder $E_y$, Decoder $F_x$ 3개의 subnetwork로 이루어져 있습니다. 각 block에 표시된 숫자는 해당 layer의 filter수를 내타냅니다. Figure 6의 설명에 씌여있는 것처럼 네트워크에 포함된 activation과 normalization은 Figure 6에 표현되지 않았습니다.

$G$의 입력으로 받은 content 이미지 $\mathrm{x}$는 Content Encoder $E_x$로,  target class($c_y$)의 $K$개의 이미지는 Class Encoder $E_y$에 입력합니다. $E_x$는 객체의 자세와 같은 class invariant latent representation인 Content Code를 추출하고 $E_y$는 객체의 외형과 같은 class specific latent representation인 Class Code을 추출하는 것을 목표로 합니다.

2개의 Encoder가 추출한 code들을 Decoder $F_x$에 입력으로 넣으며,$F_x$는 결과 $\bar{\mathrm{x}}$를 생성합니다. 이때 AdaIN layer에 content code, class code 모두가 사용되며, class code가 객체 외형과 같은 global look을 결정하고 content code가 눈, 코, 입 위치와 같은 local struction을 결정합니다.

Content Encoder $E_x$, Class Encoder $E_y$, Decoder $F_x$를 사용해 결과 이미지 $\bar{\mathrm{x}}$를 아래와 같이 표현할 수 있습니다.

$$
\begin{align*}
\bar{\mathrm{x}} &= G(\mathrm{x}, \{ y_1, ..., y_K \})
\\ &= F_x(\mathrm{z}_x, \mathrm{z}_y)
\\ &= F_x(E_x(\mathrm{x}), E_y(\{ y_1, ..., y_K\}))
\end{align*}
$$


#### Content Encoder
<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/0e842da9-7369-4a94-972d-8180a327e342" width="600" height="250">
</div>
> Figure 6의 Content Encoder 부분

Content Encoder는 여러 개의 2D convolutional layer와 residual block 으로 구성됩니다. Content Encoder는 입력 content 이미지 $x$를 content latent code $z_x$에 매핑하는 것이 목적입니다. feature map인 content code는 3번의 stride=2인 down sampling convolution을 거치며 width, height가 입력의 1/8을 가집니다. 각 layer에서 instance normlization과 ReLU activation이 사용됩니다.

content code는 $\mathrm{x}$의 클래스와는 관계없는 $\mathrm{x}$의 content 정보(class-invariant content information)를 encode 하도록 설계되었습니다. content code는 위치에 대한 정보는 encode 해야하지만, 클래스 별 외형은 encode하지 않아야 합니다. 예시로 위 그림의 강아지의 귀, 눈, 코의 위치는 content code에 정보가 있어야 하지만 귀의 모양이나 색깔은 정보가 포함되지 않도록 해야합니다.


#### Class Encoder
<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/ac135258-af34-4a8d-8e9d-22eb641eda48" width="600" height="450">
</div>
> Figure 6의 Class Encoder 부분


Class Encoder는 VGG와 같은 네트워크를 사용해 $K$개 이미지를 개별적인 intermediate latent code로 만든 후, latent code들의 element-wise mean인 평균 값을 결과 값으로 출력해 최종 class latent code $z_y$를 만듭니다. 각 layer에서 ReLU activation이 사용됩니다.

class code는 $K$개의 클래스 이미지 집합을 클래스 별 정보(class-specific)를 encode하도록 설계되었습니다. 예시로 위 그림의 class code는 털의 질감, 몸의 색깔, 눈의 모양과 같이 사자의 외형에 대한 정보를 가지고 있어야 합니다.



#### Decoder
<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/32ce2c07-9b61-4eb1-ab24-8d5b0fca05d2" width="600" height="500">
</div>
> Figure 6의 Decoder 부분


Decoder는 content code와 class code를 입력으로 받아 $G$의 결과인 $\bar{\mathrm{x}}$를 생성합니다. AdaIN residual block과 nearest neighbor upscale convolution layer로 구성되며 AdaIN residual block을 제외한 각 layer에서 instance normalization과 ReLU activation을 사용합니다.

AdaIN(Adaptive Instance Normalization) residual blocks는 <a href="https://arxiv.org/abs/1804.04732" target="_blank">MUNIT</a>에서 사용한 것과 유사하며 <a href="https://solee328.github.io/gan/2023/06/09/munit_code.html#h-adain-origin" target="_blank">MUNIT(2) - 논문 구현</a>에서 자세한 설명을 볼 수 있습니다!

MUNIT의 AdaIN에서는 style code를 MLP에 입력으로 넣어 style mean, std를 계산해 AdaIN residual block의 affine transform parameter로 계산에 사용했습니다. 이와 비슷하게 FUNIT에서는 class code를 MLP로 계산해 AdaIN residual block의 affine transform parameter로 사용하며 $i=1, 2$인 평균 분산 벡터($\mu_i, \sigma^2_i$)로 decode합니다.

그 다음 content code를 AdaIN residual block으로 계산하는데, AdaIN은 normalization layer로 Affine transform을 사용하는 residual block으로 affine transform parameter로 계산한 $\mu, \sigma$를 대입하면 아래의 수식이 됩니다.
<br><br>

$$
AdaIN(x, y) = \sigma(y) (\frac{x-\mu(x)}{\sigma(x)}) + \mu(y)
$$

우선 content code($x$)의 channel을 zero mean, unit variance를 갖도록 normalize합니다. 이후 class code($y$)로 계산한 $\mu_i$(bias), $\sigma^2_i$(scale)를 사용해 global appearance information을 얻을 수 있도록 사용합니다. 이 과정으로 content의 class 정보를 없애고 class code의 class 정보를 표현하도록 합니다.

AdaIN 이후에는 upscale convolution으로 feature map을 결과 이미지 $\bar{\mathrm{x}}$을 계산합니다.

<br>

### Discriminator
FUNIT의 판별 모델 $D$는 Pix2Pix, CycleGAN, StarGAN, MUNIT 등 다양한 GAN 모델의 $D$ 구조로 사용되는 PatchGAN discriminator을 사용합니다. Leaky ReLU activation를 활용하고 normalization은 사용하지 않습니다.


<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/4e37d701-afdd-4793-8b3e-5f3bd590fd57" width="700" height="320">
</div>
> <a href="https://arxiv.org/pdf/1603.05027.pdf" target="_blank">Identity Mappings in Deep Residual Networks</a>의 Figure 1.

residual block으로 preactivation ResNet-blocks이라고도 불리는 activation first residual blocks을 사용하는 것이 특징입니다. Residual block에서 convolution과 같은 weight 연산 이후 사용되던 activation을 weight 연산 전에 사용해 성능을 올렸다고 합니다.

판별 모델은 시작과 끝은 Convolutional layer를 사용하며 중간에 10개의 activation first residual blocks로 이루어져 아래와 같은 구조를 갖습니다.

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/1b3cf091-ec40-48b0-bc7a-275d4c31ea49" width="400" height="130">
</div>

<br>

마지막은 $\vert\mathbb{S}\vert$로 feature map의 channel을 조절하는데, $\vert\mathbb{S}\vert$는 source 클래스의 수입니다.

FUNIT은 $D$는 하나의 이미지를 $\vert\mathbb{S}\vert$개의 클래스에 대해 해당 클래스에 속하는 이미지인지 아닌지 판단합니다. StarGAN처럼 이미지가 진짜인지 가짜인지 판별하는 layer, 이미지의 클래스를 판별하는 layer를 따로 두는 것이 아니라 하나의 layer로 처리하는  $\vert\mathbb{S}\vert$-class classification 문제를 다룹니다.

source class $c_x$에 해당하는 이미지를 판별할 때, 다른 클래스에 대한 판단 결과는 사용하지 않습니다. 예시로, source class $c_x$의 실제 이미지에 대해 $D$를 업데이트할 때, $c_x$번째 출력이 False로 $G$에서 생성한 이미지로 판단했다면 $D$에게 불이익(penalize)를 줍니다. 반대로 source class $c_x$에 대한 변환된 출력인 가짜 이미지에 대해 $c_x$번째 출력이 True 일때도, $D$에게 불이익을 줍니다.

즉, 다른 클래스의 이미지에 대해 False를 예측하지 못한 경우에는 $D$에게 불이익을 주지 않습니다. $G$를 업데이트할 때 $D$의 $c_x$번째 출력이 False인 경우에만 $G$에게 불이익을 줍니다.
<br><br>

---


## Loss
FUNIT의 loss는 GAN loss $\mathcal{L}_{GAN}$, content image reconstruction loss $\mathcal{L}_R$, feature matching loss  $\mathcal{L}_F$로 이루어져 있습니다. 논문에서는 $\lambda_R=0.1$, $\lambda_F=1$로 설정해 실험했다고 합니다.

$$
\min_D \max_G \mathcal{L} _{GAN}(D, G) + \lambda_R \mathcal{L} _R(G) + \lambda _{F}\mathcal{L} _{FM}(G)
$$


### adversarial
GAN loss는 conditional loss로 $D$의 위첨자 $c_x, c_y$는 클래스를 나타냅니다. loss는 해당 클래스의 binary prediction score를 사용해 계산됩니다. GAN loss로는 SAGAN, BigGAN에서도 사용했던 hinge version을 사용합니다. hinge loss에 대한 설명은 <a href="https://solee328.github.io/gan/2023/09/27/sagan_paper.html#h-loss" target="_blank">SAGAN - 논문리뷰</a>를 확인해주세요.

$$
\mathcal{L} _{GAN}(G, D) = E _{\mathrm{x}}[-\log D^{c_x}(x)] + E _{\mathrm{x}, \{\mathrm{y}_1, ..., \mathrm{y}_K\}}[\log(1-D^{c_y}(\bar{\mathrm{x}}))]
$$


### reconstruction
Content Reconstruction loss는 입력 content 이미지와 입력 class 이미지 모두에 동일한 이미지를 사용하는 경우($K=1$), $G$가 입력과 동일한 출력 이미지를 생성하도록 유도합니다.

대부분의 reconstruction loss는 cycle-consistency loss처럼 $G$에 입력 이미지 $x$와 condition $a$를 준 후, 다시 기존$x$의 condition인 condition $b$를 주어 생성한 이미지와 $x$를 비교했었는데($G(G(x, a), b)$ = x), 같은 condition을 주었을 때 입력과 출력이 동일한 이미지를 생성하는 것이 새롭다고 느껴졌습니다.

$$
\mathcal{L} _R(G) = E _{\mathrm{x}}[\|\mathrm{x} - G(\mathrm{x}, \{\mathrm{x}\})\|^1_1]
$$


### feature matching
$D$의 마지막 layer(prediction layer)를 제거한 $D_f$라는 feature extractor를 구성해 변환 결과 $\bar{x}$와 클래스 이미지 $\{y_1, ..., y_k\}$의 feature를 계산해 차이를 최소화합니다. $D_f$에 입력으로 클래스 이미지와 변환 결과 이미지를 주었을 때 feature map 간의 차이가 없어 변환 결과가 클래스 이미지와 같은 클래스로 인식되도록 합니다.

$$
\mathcal{L} _F(G) = E _{\mathrm{x}, \{\mathrm{y}_1, ...,\mathrm{y}_K\}}[\|D_f(\bar{\mathrm{x}}) - \sum_k \frac{D_f(\mathrm{y}_k)}{K}\|^1_1]
$$


<br><br>

---


## 실험 설정

### Dataset
FUNIT은 실험을 위해 4가지 데이터셋을 사용했으며 내용은 아래와 같습니다.

- **Animal Faces**<br>
  ImageNet의 149개 육식 동물 클래스의 이미지를 사용합니다. 우선 10000장의 육식 동물 얼굴의 bounding box를 수동으로 라벨을 지정한 뒤 이를 사용해 <a href="https://arxiv.org/abs/1504.08083" target="_blank">Faster RCNN</a>을 학습해 detection score 점수가 높은 bounding box만을 사용해 117574장의 동물 얼굴 데이터셋을 구축합니다. 119개의 source class, 30개의 target class로 나누어 사용합니다.

- **Birds**<br>
  <a href="https://ieeexplore.ieee.org/document/7298658" target="_blank">Bird Recognition large scale dataset</a>을 사용하며 북아메리카 새 555종에 대한 48527장의 이미지를 사용합니다. 444개의 source class, 111개의 target class로 나누어 사용합니다.

- **Flowers**<br>
  <a href="https://ieeexplore.ieee.org/document/4756141" target="_blank">Oxfold Flowers</a>를 사용하며 총 102종 8189장의 꽃 이미지를 사용합니다. 85개의 source class, 17개의 target class로 나누어 사용합니다.

- **Foods**<br>
  <a href="https://link.springer.com/chapter/10.1007/978-3-319-16199-0_1" target="_blank">Food Image Dataset</a>을 사용하며 256 종 31395장의 이미지들을 사용합니다. 224개의 source class, 32개의 target class로 나누어 사용합니다.


### Baseline
StarGAN, UNIT, MUNIT, CycleGAN을 baseline 모델로 설정해 FUNIT과 성능을 비교합니다. 이때, 학습 중 target class의 이미지를 사용할 수 있는지 여부에 따라 fair(사용 불가능), unfair(사용 가능)으로 나눠 정의합니다.

- **Fair**<br>
  Fair는 FUNIT의 설정에 해당합니다. 하지만 이전의 Unsupervised Image-to-Image translation 모델 중 같은 방법으로 설계된 모델이 없기 때문에 multi-class unsupervised image-to-image translation의 SOTA에 해당하는 <a href="https://arxiv.org/abs/1711.09020" target="_blank">StarGAN</a>을 확장해 baseline으로 사용합니다.

  Fair StarGAN은 source class 이미지만을 사용해 학습되며, testing에서 target class의 $K$개의 이미지가 주어집니다. $K$개의 이미지에 대해 VGG Conv5 features의 평균을 계산해 각각의 source class 이미지에 대한 평균 VGG Conv5 feature와의 cosine distance를 계산합니다. 이후 cosine distance에 softmax를 적용해 class association vector를 계산합니다.

  학습에 사용되지 않았던 target 클래스의 이미지를 생성하기 위해 one-hot class association vector를 StarGAN 모델의 입력으로 사용해 StarGAN의 class association score가 target class를 few-shot generation에 사용할 수 있는 각각의 source clas가 어떻게 관련되어 있는지 encoding할 수 있다는 가정으로 설계되었습니다. 이 StarGAN을 "StarGAN-Fair-K"로 표시합니다.


- **Unfair**<br>
  Unfair는 target class 이미지들이 학습에 포함됩니다. target class 당 사용 가능한 이미지 수($K$)를 1에서 20까지 다양하게 변경하며 모델들을 학습합니다. <a href="https://arxiv.org/abs/1703.10593" target="_blank">CycleGAN</a>, <a href="https://arxiv.org/abs/1703.00848" target="_blank">UNIT</a>, <a href="https://arxiv.org/abs/1804.04732" target="_blank">MUNIT</a>의 경우 2개의 도메인 간의 변환 모델이기 때문에 source class 이미지를 첫번재 도메인, target class 이미지를 두번째 도메인으로 설정해 학습합니다. $K$개 이미지로 학습한 모델을 "ModelName-Unfair-K"로 표시합니다.



### Metric
모델의 성능 비교를 위해 translation accuracy, content preservation, photorealism, distribution matching 4가지 metric을 사용하며 각자 자세한 내용은 아래와 같습니다.

- **Translation accuracy**<br>
  Translation accuracy는 모델의 변환 정확도를 측정하기 위한 것으로 변환 결과 이미지가 target class에 속하는지를 판별합니다. 이를 위해 2개의 Inception-V3 classifier를 사용합니다.

  'all'로 표시되는 첫번째 classifier는 source class와 target class를 모두를 사용해 ImageNet으로 pretrain된 Inception-V3 모델을 finetuning해 얻습니다. 'test'로 표시되는 다른 classifier는 target class를 사용해 ImageNet으로 pretrain된 Inception-V3 모델을 finetuning해 얻습니다.

  변환 결과 이미지를 두 판별 모델을 사용해 target class의 이미지로 인식할 수 있는지 확인합니다. Top1과 Top5 정확도를 모두 사용해 Top1-all, Top5-all, Top1-test, Top5-test 평가 지표를 사용해 baseline 모델과의 성능을 비교합니다.

- **Content preservation**<br>
  <a href="https://arxiv.org/abs/1603.08155" target="_blank">Perceptual distance</a>를 변형한 domain-invarian perceptual distance(DIPD)를 사용해 content preservation, 콘텐츠 보존 성능을 정량화합니다.

  DIPD를 계산하기 위해 content 이미지와 변환 결과 이미지에서 VGG conv5 feature를 추출한 후, <a href="https://arxiv.org/abs/1701.02096" target="_blank">instance normalization</a>를 적용해 feature map의 평균과 분산을 제거합니다. 이 방식으로 feature의 클래스 별 정보를 제거하고 클래스와 무관한 similarity에 초점을 맞출 수 있습니다. DIPD는 instance normalized feature 사이 L2 distance로 계산됩니다.


- **Photorealism**<br>
  실제와 같은 이미지를 생성했는지 성능을 정량화하기 위해 널리 사용되는 Inception Score(IS)를 사용합니다. $p(t\vert\mathrm{y})$를 변환 결과 이미지 $\mathrm{y}$에 대한 클래스 라벨 $t$의 분포라고 가정했을 때 Inception Score는 다음에 의해 계산됩니다.

  $$
  \mathrm{IS} _C=\exp(\mathrm{E} _{\mathrm{y}\sim p(\mathrm{y})}[\mathrm{KL}(p(t|\mathrm{y})|p(t))])
  $$

  all, test 두가지 지표가 있기 때문에 2개의 학습된 inception 판별 모델을 사용해 IS를 계산합니다.

- **Distribution matching**<br>
  두 이미지 간의 유사성을 측정하도록 설계된 <a href="https://arxiv.org/abs/1706.08500" target="_blank">Fréchet Inception Distance</a>를 사용해 distribution matching를 계산합니다.

  ImageNet pretrain된 Inception-V3 모델의 마지막 average pooling layer의 activation을 FID를 계산하기 위한 featrue vector로 사용합니다. 학습 때 사용하지 않은 $\vert\mathbb{T}\vert$ class와 source 이미지를 $\vert\mathbb{T}\vert$ 클래스로 변환한 결과 셋 간의 FID를 계산해 $\vert\mathbb{T}\vert$ FID score를 얻을 수 있습니다. $\vert\mathbb{T}\vert$ FID score의 평균은 mean FID(mFID)라고 하는 최종 distribution matching performance metric으로 사용됩니다.


## 결과

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/0b354d0b-6396-49b3-b882-0bdc83ee4574" width="800" height="580">
</div>
> Table 1. fair, unfair baseline들과 성능 비교.  $\uparrow$ 는 큰 수가, $\downarrow$는 작은 수가 좋다는 것을 의미합니다.

Table 1에서 볼 수 있듯이, FUNIT 프레임워크는 Animal Face와 North American Birds 데이터셋 모두에 대해 모든 성능 metric에서 few-shot unsupervised image-to-image translation에 대한 baseline들을 능가합니다. 표를 통해 FUNIT 모델의 성능이 $K$와 양의 상관 관계가 있음을 보여줍니다. $K$가 클수록 모든 metric의 개선하며, 가장 큰 성능 향상은 $K=1$에서 $K=5$일 때 발생합니다.

Unfair 모델들은 2개의 도메인 변환만 가능하기 때문에 클래스 수에 따른 여러 개의 모델을 학습하지만, FUNIT은 단 1개의 모델만으로 가능하며 마찬가지로 단 1개의 모델을 사용하는 StarGAN-Fair-K가 있지만 성능의 차가 큼을 볼 수 있습니다. MUNIT이 Unfair 모델이지만 그나마 유사한 성능을 보여주네요.

<br><br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/fd4659c8-679c-49fa-a246-3329d07288e9" width="700" height="750">
</div>
> Figure 2. few-shot unsupervised image-to-image translation 결과 시각화. 결과는 FUNIT-5 모델을 사용해 계산되었습니다. 위에서부터 순서대로 animal face, bird, flower, food 데이터셋에 대한 결과가 있습니다. 각 데이터셋에 대해 하나의 모델을 학습합니다. 각 예시에 대해, 우리는 무작위로 샘플링된 클래스 이미지 5개 중 2개인 $\mathrm{y}_1, \mathrm{y}_2$, 입력 content 이미지 $\mathrm{x}$, 변환 결과 $\bar{\mathrm{x}}$를 시각화합니다. 결과는 FUNIT 이 학습 중 보지 못했던 target class의 이미지를 난이도가 높은 few-shot 설정에서 그럴듯한 변환 결과를 생성한다는 것을 보여줍니다. 우리는 출력 이미지의 객체가 입력과 유사한 포즈를 취한다는 것에 주목합니다.

Figure 2에서 FUNIT-5에 의해 계산된 few-shot 변환 결과를 볼 수 있습니다. 입력 content 이미지 $\mathrm{x}$과 해당 출력 이미지 $\bar{\mathrm{x}}$에 있는 객체의 포즈는 대체로 동일하게 유지됩니다. 출력 이미지는 photorealistic하며 target 클래스의 이미지와 유사합니다.

<br><br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/74775fd3-d7b5-4bc9-a5b0-23e95ac22ba6" width="600" height="600">
</div>
> Figure 3. few-shot image-to-image translation 성능에 대한 시각적 비교. 왼쪽에서 오른쪽 방향으로, 열은 입력 이미지 $\mathrm{x}$, 2개의 target class 이미지 $\mathrm{y}_1, \mathrm{y}_2$, Unfair StarGAN baseline의 변환 결과, fair StarGAN baseline의 변환 결과, FUNIT의 결과입니다.

Figure 3에서 fair가 가능한 모델인 StarGAN과 FUNIT의 시각적 비교를 볼 수 있습니다. 같은 입력에서 StarGAN은 많은 양의 artifact를 가진 이미지를 생성하거나 target class가 아닌 입력된 content 이미지와 유사한 이미지를 출력합니다. 반면, FUNIT은 이미지 변환 결과를 성공적으로 생성합니다.


### number of source class
<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/c43f6c61-48f5-46e3-8171-8a0e9f4d4bce" width="800" height="200">
</div>
> Figure 4. few-shot image translation 성능 vs Animal Faces 데이터 셋 학습 중 사용한 객체 클래스 수. 성능은 학습 중 사용한 객체 source class의 수와 양의 상관관계가 있습니다.

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/f7b3da7a-d734-4b33-a6a3-f4b9a3f22771" width="800" height="200">
</div>
> Figure 7. Few-shot image translation performance vs North American Birds 데이터셋 학습 중 객체 클래스 사용 숫자

Figure 4에서는 animal dataset을 사용해 one-shot 설정(FUNIT-1)에서 학습 셋의 source class 수를 변화시키는 것에 대한 성능을 분석하며 69개에서 119개 클래스까지 10개 간격으로 변화시켜 곡선을 표현합니다. 표시된 것과 같이, 성능은 변환 정확도, 이미지 품질 및 분포 일지 측면에서 객체 클래스 수와 양의 상관 관계가 있습니다. domain-invariant perceptual distance는 평평하게 유지됩니다. 이는 학습 중 더 많은 객체 클래스(다양성 증가)를 보는 FUNIT 모델이 테스트 중에 더 나은 성능을 보임을 보여줍니다.

Figure 7의 bird 변환 작업에서도 동일한 경향임을 보여줍니다. North American Birds 데이터셋을 사용해 학습 셋에서 사용 가능한 source 클래스 수 대비 제안된 모델의 성능을 보고하며 source class 수를 189, 222, 289, 333, 389에서 444로 변경해 성능 점수를 표시합니다. animal 데이터셋과 마찬가지로 모델이 학습 중에 더 많은 수 의 source 클래스를 볼 때 test 중에 더 나은 성능을 발휘합니다.


### ablation study
<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/118c4e14-3052-49da-bf62-f044006ea74c" width="700" height="100">
</div>
> Table 4. content image rconstruction loss weight, $\lambda_R$에 대한 파라미터 민감도 문석. $\uparrow$ 는 숫자가 클 수록 좋고, $\downarrow$는 숫자가 작을 수록 좋다는 것을 의미합니다. 0.1의 값은 content preservation(콘텐츠 보존)과 translation accuracy(변환 정확도)의 좋은 균형(trade-off)를 제공하며, 이는 논문 전체에서 기본값으로 사용됩니다. 본 실험에서 FUNIT-1 모델을 사용합니다.

Table 4에서 content image reconstruction loss의 weight가 Animal Face 데이터셋을 학습한 모델의 성능에 미치는 영향을 분석한 결과를 볼 수 있습니다. 저자들은 $\lambda_R$값이 클수록 translation accuracy(변환 정확도)가 낮아지지만, domain invariant perceptual distance가 작다는 것을 발견했으며,  $\lambda_R=0.1$이 좋은 절충안을 제공한다는 것을 보여주어 논문 전체 실험에서 default로 사용합니다.

흥미롭게도, $\lambda_R=0.01$로 매우 작으면 content preservation 및 translation accuracy 모두에서 성능이 저하됩니다. 이는 content reconstruction loss가 학습을 regularize 하는 데 도움이 된다는 것을 나타낸다고 합니다.

<br>


<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/0173bccf-390d-4d81-8896-aca94de3198f" width="700" height="90">
</div>
> Table 5. object term에 대한 Ablation study. $\uparrow$ 는 숫자가 클 수록 좋고, $\downarrow$는 숫자가 작을 수록 좋다는 것을 의미합니다. FM은 feature matching loss term이 제거된 제안된 프레임워크의 설정을 나타내고 GP는 gradient panalty loss가 제거된 제안된 프레임워크 설정을 나타냅니다. 기본 설정은 대부분의 기준에서 더 나은 성능을 제공합니다. 이 실험에서 우리는 FUNIT-1 모델을 사용합니다.

Table 5에서 loss term들이 Animal Face 데이터를 학습한 모델의 성능에 미치는 영향을 분석한 ablation study 결과를 볼 수 있습니다. feature matching loss를 제거할때 성능이 조금 저하되며, zero-centered gradient penalty를 제거할 때는 크게 성능이 저하되는 것을 볼 수 있습니다.

gradient penalty는 <a href="https://arxiv.org/abs/1801.04406" target="_blank">real gradient penalty regularization</a>를 사용하며 공식 코드에서는 `calc_grad2`에서 코드를 확인할 수 있었습니다. $D$ 학습 중 $D$의 loss를 사용해 gradient를 구하는 것까지는 WGAN-GP의 gradient penalty와 동일합니다. WGAN-GP에서는 (gradient - 1) 값을 제곱해 gradient penalty로 사용했다면 zero-centered gradient penalty에서는 pow(2)로 grad 값 자체를 제곱해 사용합니다.

```python
def calc_grad2(self, d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(outputs=d_out.mean(),
                              inputs=x_in,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.sum()/batch_size
    return reg
```


### vs AdaIN style transfer
<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/95cb0c43-8d28-4241-8580-939cd6c41628" width="600" height="400">
</div>
> Figure 9. few-shot image translation을 위한 FUNIT-1 대 AdaIN style transfer[18]

 few-shot animal face 변환 작업을 위한 <a href="https://arxiv.org/abs/1703.06868" target="_blank">AdaIN transfer network</a>를 학습하고 성능을 비교합니다. AdaIN은 입력 동물의 texture(질감)을 변경할 수 있지만, 모양을 변경하지 않아 결과적으로, 변환 결과의 외형은 입력과 여전히 유사합니다. 그에 반해 FUNIT은 외형과 질감 모두를 변환하고 있는 것을 볼 수 있습니다.


### Failuer case
<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/0a145531-08be-4ba8-95cb-67545868b75c" width="600" height="400">
</div>
> Figure 10. 실패 사례들. 제안된 FUNIT 모델의 전형적인 실패 사례에는 hybrid objects generation(예. column 1, 2, 3, 4), 입력 content 이미지 무시(예. column 5, 6), 입력 클래스 이미지 무시(예. column 7)이 포함됩니다.

Figure 10은 제안된 알고리즘의 몇 가지 실패 사례를 보여줍니다. 사례는 hybrid class generation, 입력 content 이미지 무시, 입력 클래스 이미지 무시 등을 포함하고 있습니다.


### few-shot classification
논문에서는 학습한 FUNIT의 classification을 few-shot classification로서의 성능을 실험합니다.

Animal Faces와 North American Birds 데이터셋을 사용해 few-shot 클래스에 대한 $N$(1, 50, 100) 이미지를 생성하고 생성된 이미지를 사용해 판별 모델을 학습합니다. 학습에 대한 설정은 비교 모델인 <a href="https://arxiv.org/abs/1606.02819" target="_blank">Shrink and Hallucinate (S&H)</a>의 설정을 따라 train, validation, test set으로 나눠 사용합니다. train set은 $\vert\mathbb{T}\vert$개의 class로 구성되며 class마다 생성된 $N$개의 이미지를 가지고 있습니다.

새로운 클래스에 해당하는 final layer feature를 생성하는 방법을 학습하는 <a href="https://arxiv.org/abs/1606.02819" target="_blank">Shrink and Hallucinate (S&H)</a> 방법을 비교 모델로 사용합니다. S&H 방법은 source class 이미지만을 사용해 사전학습된 10-layer ResNet 네트워크를 feature extractor로 사용해 target class에 대해 linear classifier를 학습합니다.

공정한 비교를 위해서 FUNIT과 S&H 모두에 대해서 validation 셋을 사용해 weight 값과 weight decay에 대한 철저한 grid search를 수행하고 test 셋에 대한 성능을 확인했다고 합니다. 실험에 사용한 hyperparameter와 search 알고리즘은 Appendix H를 참고해주세요.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/d04a8ddb-d580-486e-877e-78663db67599" width=470 height="180">
</div>
> Table 3. 분할에 걸쳐 평균을 나타낸 few-shot classification 정확도

위 실험을 기반으로 FUNIT으로 학습된 판별 모델이 feature hallucination(특징 환각)을 기반으로 샘플 갯수 N에 대한 제어 가능한 변수를 가지고 있는 S&H few-shot classification 접근 방식보다 지속적으로 더 나은 성능을 달성한다는 것을 확인할 수 있습니다.

Table 3에서, classification task에 대해 생성된 샘플 수(즉, FUNIT 이미지, S&H의 features)에 따라 FUNIT과 <a href="https://arxiv.org/abs/1606.02819" target="_blank">S&H</a> 방법의 성능을 보고합니다. 두 방법 모두 새로운 클래스 당 하나의 실제 이미지만 사용하는 기존 classifier보다 성능이 뛰어나지만, 생성된 이미지를 사용한 FUNIT이 생성된 feature를 사용하는 S&H method보다 2% 뛰어납니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/ed4d9ba9-b544-4792-bdaf-78ff39f248e5" width=600 height="180">
</div>
> Table 6. 생성된 이미지와 1개의 실제 이미지를 사용할 때 Animal Faces 데이터셋의 5 split에 대한 One-shot 정확도. split 당 5개의 독립적인 실행에 대한 평균 정확도가 보고됩니다(생성 이미지는 매번 다른 셋에서 샘플링됨).

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/a2d1a5ea-d947-4a2e-b9b4-9cbb338aa590" width=600 height="180">
</div>
> Table 7. 생성된 이미지와 1개의 실제 이미지를 사용할 때 North American Birds 데이터셋의 5 split에 대한 One-shot 정확도. split 당 5개의 독립적인 실행에 대한 평균 정확도가 보고됩니다(생성 이미지는 매번 다른 셋에서 샘플링됨).

Table 6, 7에서, Animal Faces와 North American Birds 데이터셋의 5 one-shot split 모두에 대한 one-shot learning의 test 정확도와 관련 차이를 보고합니다. 모든 실험에서, 이미지 생성 모델을 학습하는데 사용된 클래스 셋에 대해 학습된 네트워크의 feature extractor를 사용해 새로운 classifier layer만 학습합니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/a5ff90d4-f893-440e-88e6-1f8a8afa2e78" width=400 height="130">
</div>
> Table 8. prototypical networks method로 생성된 이미지를 사용할 때 Animal Face와 North American birds 데이터셋에 대한 5 splits의 평균 1-shot 정확도

또한 FUNIT은 기존의 few-shot classification 접근법과 함께 사용할 수 있습니다. Table  8에서, 우리는 주어진 few 학습 샘플에서 얻은 가장 가까운 prototype(cluster center)의 라벨을 테스트 샘플에 할당하는 <a href="https://arxiv.org/abs/1703.05175" target="_blank">Prototypical Networks method</a>를 사용해 얻은 1-shot classification 결과를 보여줍니다. FUNIT으로 생성한 샘플을 test time에 클래스당 제공된 샘플 1개와 함께 사용해 class prototype representation을 계산하는 것은 두 데이터 셋 모두에서 5.5% 이상의 정확도를 향상시키는데 도움이 됩니다.
<br><br>

---

## 정리
FUNIT은 최초의 few-shot unsupervised image-to-image translation 프레임워크입니다. 우리는 few shot 생성 기능이 학습 중에 보이는 객체 클래스의 수와 양의 상관관계가 있으며  test time 동안 제공되는 target class의 장(shot) 수와도 양의 상관관계가 있음을 보여주었습니다.

FUNIT은 test time에 사용할 수 있는 모델에게 보여주지 않은 클래스의 몇몇 이미지를 이용해 source class의 이미지를 본 적 없는 객체 클래스의 유사한 이미지로 변환하는 방법을 배울 수 있다는 경험적 증거를 제공했습니다. FUNIT은 새로운 기능을 달성하기는 하지만 다음 몇가지 조건에 따라 작동합니다.
- content encoder $E_x$가 class-invariant latent code $z_x$를 학습할 수 있는지 여부
- class encode $E_y$가 class-specific latent code $z_y$를 학습할 수 있는지 여부
- class encoder $E_y$가 보이지 않는 객체 클래스의 이미지로 일반화할 수 있는지 여부

새로운 클래스가 source 클래스와 기각적으로 관련되어 있을 때 이런 조건을 충족하기 쉬우나, 새로운 객체 클래스의 외관이 source class의 와괸과 극적으로 다를 때 FUNIT은 Figure 5와 같이 변환을 실패합니다. 이 경우 FUNIT은 입력 content 이미지의 색상이 변경된 버전으로 생성하는 경향이 있습니다. 이는 바람직하지 않지만 외형 분포가 극적으로 변경되었기 때문에 이해할 수 있습니다. 이 한계를 해결하는 것이 Future Work입니다.
<br><br>

---

<br>
FUNIT 논문에 대해 살펴봤습니다. 끝까지 봐주셔서 감사합니다:)

Few shot과 관련된 GAN으로는 처음 읽게 된 논문이였습니다. target class의 이미지 수 $N$이 클 수록 좋다진다고는 하지만 $N=1$일 때도 작동하는 것이 인상이 깊었습니다. 학습까지 진행해보고 싶은데 NVIDIA DGX1의 8개의 V100 GPU로 돌렸다는 걸 보고 학습 실험은 무리라는 걸 깨달아서 슬퍼졌습니다:confused:

하지만 아직 남은 논문들은 많고 많으니까요! 열심히 최신 논문들까지 논문 리뷰를 진행해보겠습니다:muscle:
