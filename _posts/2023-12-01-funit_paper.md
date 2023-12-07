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

Unsupervised image-to-image translation 모델들의 단점으로 특정 클래스에 대한 입력을 수행하기 위해서는 해당 클래스에 대한 수많은 데이터셋을 학습하는 것을 꼽았습니다. FUNIT은 few-shot을 적용해 새로운 클래스의 이미지 단 몇장으로도 가능한 이미지 변환을 제안합니다. 또한 이 과정에서 few-shot classification 모델의 성능과 기존의 unsupervised image-to-image translation 설정에서도 기존의 state-of-the-art 모델들을 능가하며 성능을 입증합니다.

지금부터 FUNIT에 대해 살펴보겠습니다 :eyes:

정리한 Appendix : A, B, C, D, E, F
추가 정리 : Related Work

---

## 소개
[30, 46, 29, 25, 55, 52]와 같은 연구들은 Unsupervised Image-to-Image Translation 설정에서 이미지 클래스를 변환하는 것에 성공했지만, 새로운 클래스에 대한 소수의 이미지들에서 일반화(generalization)은 불가능합니다. 이 모델들이 이미지 변환을 수행하기 위해서는 모든 클래스의 대규모 학습 셋이 필요하며 few-shot은 지원하지 않습니다.

그에 반해, 인간은 일반화(generalization)이 뛰어납니다. 이전에 보지 못했던 이국적인 동물의 사진이 주어져있을 때, 우리는 그 동물이 다른 자세로 있는 상상이 가능합니다. 예를 들어, 서 있는 호랑이를 처음 본 사람이더라도 일생 동안 본 다른 동물들에 대한 정보를 통해 호랑이가 누워있는 모습을 상상하는 것에 어려움이 없습니다.

FUNIT은 이런 격차를 줄이기 위한 시도로, test time에 학습(train time) 중 보지 못한 몇 장의 target class의 이미지들을 source class(학습 데이터셋에 포함된 특정 클래스)의 이미지와 유사한 이미지로 매핑하는 것을 목표로 하는 Few-shot UNsupervised Image-to-Image Translation(FUNIT) 프레임워크를 제안합니다.
<br><br>

---

## 모델
FUNIT은 Generative Adversarial Network(GAN)[14]를 기반으로 하며, conditional image generator $G$와 multi-task adversarial discriminator $D$로 구성됩니다.

하나의 이미지를 입력으로 받는 기존의 unsupervised image translation framework[55, 29]의 conditional image generator들과는 달리 FUNIT의 $G$는 content 이미지 $\mathrm{x}$와 $K$개의 이미지 집합 $\{y_1, ..., y_K\}을 동시에 입력으로 가지며 출력 이미지 $\bar{\mathrm{x}}$를 생성합니다.

$$
\bar{\mathrm{x}} = G(\mathrm{x}, \{ y_1, ..., y_K \})
$$

content 이미지의 클래스는 $c_x$이고 $K$개의 이미지 집합 $\{y_1, ..., y_K\}의 클래스를 $c_y$라 할 때, $c_x$는 $c_y$ 다른 클래스를 의미합니다.


```
Figure 1 삽입
```
> Figure 1: <b>Training.</b>. 학습 셋은 여러 객체 클래스(source class)로 이루어져 있습니다. FUNIT 모델이 source 클래스 사이 이미지를 변환하도록 학습시킵니다.<br>
<b>Deployment.</b> 학습된 모델이 학습 중에 target class의 이미지를 본 적이 없음에도 불구하고 source class의 이미지를 target class와 유사한 이미지로 변환합니다. FUNIT 생성 모델은 1) content 이미지와 2) target class 이미지 셋, 2가지 입력을 사용합니다. target class 이미지와 유사한 입력 이미지의 변환을 생성하는 것을 목표로 합니다.

Figure 1에서 보여주듯이, $G$는 입력 content 이미지 $\mathrm{x}$를 출력 이미지 $\bar{\mathrm{x}}$에 매핑해 $\bar{\mathrm{x}}$가 클래스 $c_y$를 유지한 채 $\mathrm{x}$와 구조적 유사성을 공유하도록 생성합니다.

\mathbb{S}가 학습 데이터 셋(source class), $\mathbb{T}$가 target class 셋을 나타낸다 가정해봅시다. 학습 단계(training time)에서, $G$는 $c_x, c_y \in \mathbb{S}$이고 $c_s \not= c_y$인 2개의 랜덤 추출된 source class 사이를 변환하는 방법을 학습합니다.

Test 단계(testing time)에서, $G$는 학습하지 않은 target class $c \in \mathbb{T}$로부터 몇 개의 이미지를 가져가 source class 중 임의의 클래스에서 샘플링된 이미지를 target class $c$와 유사한 이미지로 매핑합니다.


### Generator

```
Figure 6 삽입
```
> Figure 6. 생성 모델 구조 시각화. 변환 결과  $\bar{\mathrm{x}}$를 생성하기 위해, 생성 모델은 클래스 이미지들 $y_1, ..., y_k$에서 추출된 class latent code $\mathrm{z}_y$와 입력 content 이미지에서 추출된 content latent code $\mathrm{z}_x$를 결합합니다. 비선형성(nonlinearity)와 정규화(normalization)연산은 시각화에 포함되지 않습니다.

few-shot 이미지 변환 모델인 FUNIT의 생성 모델은 Figure 6의 시각화에서 표현된 것처럼 Content Encoder $E_x$, Class Encoder $E_y$, Decoder $F_x$ 3개의 subnetwork로 이루어져 있습니다. 각 block에 표시된 숫자는 해당 layer의 filter수를 내타냅니다. 네트워크에 포함된 nonlineaity(activation)과 normalization 연산은 시각화되지 않았습니다.

Content Encoder $E_x$, Class Encoder $E_y$, Decoder $F_x$를 사용해 식을 아래와 같이 표현할 수 있습니다.

$$
\begin{align*}
\bar{\mathrm{x}} &= G(\mathrm{x}, \{ y_1, ..., y_K \}) \\
&= F_x(\mathrm{z}_x, \mathrm{z}_y) = F_x(E_x(\mathrm{x}), E_y(\{ y_1, ..., y_K\}))
\end{align*}
$$

이와 같은 생성 모델 설계를 통해, FUNIT의 $G$는 Content Encoder를 사용해 class invariant latent representation(예. object의 자세)를 추출하고 Class Encoder를 사용해 class specific latent representation(예. object의 외형)을 추출하는 것을 목표로 합니다. AdaIN layer를 통해 Decoder에 class latent code를 공급해 클래스 이미지가 global look(예. object의 외형)을 제외하고 content image가 local struction(예. 눈, 코, 입의 위치)를 결정합니다.


#### Content Encoder
```
Content Encoder clip 사진 삽입
```
Content Encoder는 여러 개의 2D convolutional layer와 residual blocks[16, 22]로 구성됩니다. Content Encoder는 입력 content 이미지 $x$를 spartial feature map인 content latent code $z_x$에 매핑하는 것이 목적입니다.


Content Encoder는 입력된 content 이미지 $\mathrm{x}$를 feature map인 content latent code로 만듭니다. 이 feature map은 3개의 stride 2 down sampling convolution으로 입력 해상도 * 1/8의 width*height를 가지며, $\mathrm{x}$의 클래스와는 관계없는 $\mathrm{x}$의 content 정보(class-invariant content information) encode 하도록 설계되었습니다. feature map은 위치와 같은 정보를 encode하고 class 별 외형은 encode하지 않아야 합니다(예시로, animal face translation task에서 feature map은 귀의 위치는 encode해야 하지만 귀의 모양이나 색깔은 encode하지 않아야 합니다).

Content Encoder의 경우, 각 layer에서 instance normlization과 ReLU nonlinearity가 사용됩니다.


#### Class Encoder
```
Class Encoder clip 사진 삽입
```
Class Encoder는 $K$개의 클래스 이미지 집합을 클래스 별 정보(class-specific)를 담은 class latent code를 만듭니다. Class Encoder는 VGG와 같은 네트워크를 사용해 각 입력 이미지를 intermediate latent code로 만든 후, 이 latent code를 element-wise mean으로 계산해 최종 class latent code를 생성합니다.

Class Encoder의 경우, 각 layer에서 ReLU nonlinearity가 사용됩니다.

#### Decoder
```
Decoder clip 사진 삽입
```
Decoder는 Adaptive Instance Normalization(AdaIN) residual blocks[19]와 upscale convolutional layer로 구성됩니다. AdaIN residual block은 normalization layer로 AdaIN[18]을 사용하는 residual block입니다. AdaIN은 우선 각 입력의 channel의 activation을 zero mean, unit variance을 갖도록 normliazation합니다. 이후 scalar와 bias로 학습된 affine transformation을 사용해 activation의 scale을 조정합니다. affine transformation은 spatially-inveriant이므로 global appearance information을 얻는데만 사용할 수 있습니다. affine transformation parameter들은 2개의 fully connected network를 통해 $z_y$가 adaptively하게 계산됩니다.

Decoder를 우선 class-specific한 class latent code를 AdaIN Residual block의 affine transform parameter로 사용하기 위해 $i=1, 2$인 평균 분산 벡터($\mu_i, \sigma^2_i$)로 decode합니다.

각각의 residual block에서 affine transformation은 feature map의 모든 spatial location(공간 위치)에 적용됩니다. 이를 통해 content latent code를 decode하고 출력 이미지 생성하는 것을 제어합니다.

Decoder의 경우 AdaIN residual block을 제외하고 각 layer에서 instance normalization과 ReLU nonlinearity를 사용합니다. nearest neighbor upsampling을 사용해 spatial dimension(공간 차원)의 feature map을 2배씩 upscaling합니다.


### Discriminator
FUNIT의 판별 모델 $D$는 PatchGAN discriminator[21]을 사용합니다. Leaky ReLU nonlinearity를 활용하고 normalization을 사용하지 않습니다. 판별 모델은 10개의  Convolutional layer와 activation first residual blocks[32]로 구성됩니다. 구조는 다음과 같이 표현됩니다.

$$
\textmd{Conv-64} \rightarrow \textmd{ResBlk-128} \rightarrow \textmd{ResBlk-128} \rightarrow \textmd{AvePool 2x2} \rightarrow \textmd{ResBlk-256} \rightarrow \textmd{ResBlk-256} \rightarrow \textmd{AvePool 2x2} \rightarrow \textmd{ResBlk-512} \rightarrow \textmd{ResBlk-512} \rightarrow \textmd{AvePool 2x2} \rightarrow \textmd{ResBlk-1024} \rightarrow \textmd{ResBlk-1024} \rightarrow \textmd{AvePool 2x2} \rightarrow \textmd{ResBlk-1024} \rightarrow \textmd{ResBlk-1024} \rightarrow \textmd{Conv-}\|\mathbb{S}\|
$$


#### Multi-task Adversarial Discriminator
FUNIT의 판별 모델 $D$는 입력 이미지들에 대해 각각의 이미지가 source class의 실제 이미지인지 $G$에서 온 변환된 결과물인지를 결정하는 binary classification task를 해결하며 학습됩니다. 이때 source class에는 $|\mathbb{S}|$개의 클래스가 있다고 가정하면, $D$는 $|\mathbb{S}|$개의 출력을 생성합니다.

source class $c_x$의 실제 이미지에 대해 $D$를 업데이트할 때, $c_x$번째 출력이 False로 $G$에서 생성한 이미지로 판단했다 $D$에게 불이익(penalize)를 줍니다. 반대로 source class $c_x$에 대한 변환된 출력인 가짜 이미지에 대해 $c_x$번째 출력이 True 일때도, $D$에게 불이익을 줍니다.

이때 다른 클래스의 이미지에 대해 False를 예측하지 못한 경우에는 $D$에게 불이익을 주지 않습니다. $G$를 업데이트할 때 $D$의 $c_x$번째 출력이 False인 경우에만 $G$에게 불이익을 줍니다.

저자들은 경험적으로 판별 모델이 더 어려운 task인 $|\mathbb{S}|$-class classification 문제를 해결하며 기존 판별 모델보다 더 잘 작동한다는 것을 발견했습니다.

```
기존 판별 모델의 classification task 비교
```


## Loss
FUNIT의 loss는 GAN loss $\mathcal{L}_{GAN}$, content image reconstruction loss $\mathcal{L}_R$, feature matching loss  $\mathcal{L}_F$로 이루어져 있습니다.

$$
\min_D \max_G \mathcal{L} _{GAN}(D, G) + \lambda_R \mathcal{L} _R(G) + \lambda _{F}\mathcal{L} _{FM}(G)
$$

Content Reconstruction loss와 Feature matching loss은 모두 Image-to-Image translation[29, 19, 50, 37]에서도 사용되었던 loss로 FUNIT은 기존의 기술을 사용해 새로운 few-shot unsupervised image-to-image translation 설정으로 사용을 확장합니다.

### adversarial
GAN loss는 conditional loss로 $D$의 위첨자 $c_x, c_y$는 클래스를 나타냅니다. loss는 해당 클래스의 binary prediction score를 사용해 계산됩니다.

$$
\mathcal{L} _{GAN}(G, D) = E _{\mathrm{x}}[-\log D^{c_x}(x)] + E _{\mathrm{x}, \{\mathrm{y}_1, ..., \mathrm{y}_K\}}[\log(1-D^{c_y}(\bar{\mathrm{x}}))]
$$


### reconstruction
Content Reconstruction loss는 입력 content 이미지와 입력 class 이미지 모두에 동일한 이미지를 사용하는 경우($K=1$), $G$가 입력과 동일한 출력 이미지를 생성하도록 유도합니다.

$$
\mathcal{L} _R(G) = E _{\mathrm{x}}[\|\mathrm{x} - G(\mathrm{x}, \{\mathrm{x}\})\|^1_1]
$$


### feature matching
Feature matching loss는 학습을 regularize합니다. $D$의 마지막 layer(prediction layer)를 제거한 $D_f$라는 feature extractor를 구성해 변환 결과 $\bar{x}$와 클래스 이미지 $\{y_1, ..., y_k\}$의 feature를 계산해 차이를 최소화합니다.

$$
\mathcal{L} _F(G) = E _{\mathrm{x}, \{\mathrm{y}_1, ...,\mathrm{y}_K\}}[\|D_f(\bar{\mathrm{x}}) - \sum_k \frac{D_f(\mathrm{y}_k)}{K}\|^1_1]
$$



## 실험

### Dataset
FUNIT은 실험을 위해 4가지 데이터셋을 사용했으며 내용은 아래와 같습니다.

- Animal Faces
  ImageNet[9]의 149개 육식 동물 클래스의 이미지를 사용해 데이터셋을 준비합니다. 우선 10000장의 육식 동물 얼굴의 bounding box를 수동으로 라벨을 지정한 뒤 이를 사용해 Faster RCNN[13]을 학습했습니다. 저자들은 detection score 점수가 높은 bounding box만을 사용해 117574장의 동물 얼굴 데이터셋을 구축합니다. 119개의 source class, 30개의 target class로 나누어 사용합니다.

- Birds
  북아메리카 새 555종에 대한 48527장의 이미지로 이루어져 있습니다. 444개의 source class, 111개의 target class로 나누어 사용합니다.

- Flowers
  102종의 8189장의 이미지로 이루어져 있습니다. 85개의 source class, 17개의 target class로 나누어 사용합니다.

- Foods
  256 종류의 31395장의 이미지들로 이루어져 있습니다. 224개의 source class, 32개의 target class로 나누어 사용합니다.


### Baseline
StarGAN, UNIT, MUNIT, CycleGAN을 baseline 모델로 설정해 FUNIT과 성능을 비교합니다. 이때, 학습 중 target class의 이미지를 사용할 수 있는지 여부에 따라 fair(사용 불가능), unfair(사용 가능)으로 나눠 정의합니다.

- Fair
Fair는 FUNIT의 설정에 해당합니다. 하지만 이전의 Unsupervised Image-to-Image translation 모델 중 같은 방법으로 설계된 모델이 없기 때문에 multi-class unsupervised image-to-image translation의 SOTA에 해당하는 StarGAN[8]을 확장해 baseline으로 사용합니다.

Fair StarGAN은 source class 이미지만을 사용해 학습되며, testing에서 target class의 $K$개의 이미지가 주어집니다. $K$개의 이미지에 대해 VGG[42] Conv5 features의 평균을 계산해 각각의 source class 이미지에 대한 평균 VGG Conv5 feature와의 cosine distance를 계산합니다. 이후 cosine distance에 softmax를 적용해 class association vector를 계산합니다.

학습에 사용되지 않았던 target 클래스의 이미지를 생성하기 위해 one-hot class association vector를 StarGAN 모델의 입력으로 사용해 StarGAN의 class association score가 target class를 few-shot generation에 사용할 수 있는 각각의 source clas가 어떻게 관련되어 있는지 encoding할 수 있다는 가정으로 설계되었습니다. 이 StarGAN을 StarGAN-Fair-K로 표시합니다.


- Unfair
Unfair는 target class 이미지들이 학습에 포함됩니다. target class 당 사용 가능한 이미지 수($K$)를 1에서 20까지 다양하게 변경하며 모델들을 학습합니다. CycleGAN[55], UNIT[29], MUNIT[19]의 경우 2개의 도메인 간의 변환 모델이기 때문에 source class 이미지를 첫번재 도메인, target class 이미지를 두번째 도메인으로 설정해 학습합니다. $K$개 이미지로 학습한 모델을 ModelName-Unfair-K로 표시합니다.



### Metric
모델의 성능 비교를 위해 4가지 metric을 사용하며 내용은 아래와 같습니다.

- Translation accuracy
Translation accuracy는 결과 이미지가 target class에 속하는지 여부를 측정합니다. 이를 위해 2개의 Inception V3[45]의 판별 모델을 사용합니다. 하나의 판별 모델은 source class와 target class 모두를 사용해 학습되고 'all'로 표시됩니다. 다른 하나의 판별 모델은 target class를 사용해 학습되며 'test'로 표시됩니다. 두 판별 모델을 사용해 Top1, Top5 정확도를 보고합니다.


우리는 변환 정확도를 측정하기 위해 2개의 Inception-V3[45] classifier를 사용합니다. all로 표시되는 첫번째 분류모델은 모든 source와 target 객체 클래스를 분류하는 작업에 대해 ImageNet으로 pretrain된 Inception-V3 모델을  finetuning해 얻습니다(예. Animal Face 데이터셋의 경우 149개  클래스 모두, North American Birds 데이터 셋의 경우 555개의 클래스 모두를 사용한다). 두번째 분류모델은(test로 표시됨) target 클래스에 대해 분류하는 작업에 대해 ImageNet으로 pretrain된 Inception-V3 모델을 finetuning해 얻습니다(예. Animal Face 데이터셋의 경우 30개의 target class, North American Bird 데이터셋의 경우 11개의 target 클래스를 사용합니다) 우리는 변환 결과에 분류모델을 적용해 결과를 target 클래스의 이미지로 인식할 수 있는지 확인합니다. 만약 해당한다고 판별된다면, 올바른 변환으로 표시합니다. 우리는 Top1과 Top5 정확도를 모두 사용해 경쟁 모델과의 성능을 비교합니다. 따라서 우리는 변환 정확도 측정을 위한 4가지 평가 지표를 가지고 있습니다: Top1-all, Top5-all, Top1-test, Top5-test. 높은 정확도의 unsupervised image-to-image translation 모델이 좋습니다. 우리는 semantic label의 image translation task[21, 50, 7]와 유사한 평가 프로토콜이 image-to-image translation 모델들에 사용되었음을 주목합니다.


- Content preservation
우리는 domain-invariant perceptual distance(DIPD)[19]를 사용해 content preservation(콘텐츠 보존) 성능을 정량화합니다. DIPD는 perceptual distance[22, 54]의 변형입니다. DIPD를 계산하기 위해, 우리는 우선 입력 content 이미지와 변환 결과 이미지에서 VGG[42] conv5 feature를 추출합니다. 우리는 그 다음 instance normalization[47]을 적용해 feature map의 평균과 분산을 제거합니다. 이 방식으로, 우리는 feature의 class-specific information(클래스 별 정보)를 필터링하고[18, 27], class-invariant similarity(유사성)에 초점을 맞출 수 있습니다. DIPD는 instance normalized feature 사이의 L2 distance로 계산됩니다.

Content preservation(콘텐츠 보존)은 domain-invariant(도메인 불변) perceptual distance(DIPD)[19]라고 불리는 perceptual distance[22, 54]의 변형을 기반으로 합니다. distance는 domain 변화[19]에 대항해 더 invariant한 두 개의 normalized된 VGG[42] Conv5 feature 사이의 L2 distance로 제공합니다.


- Photorealism

우리는 이미지 생성 성능을 정량화하기 위해 널리 사용되는 inception score(IS)를 사용합니다. $p(t|\mathrm{y})$를 변환 결과 이미지 $\mathrm{y}$에 대한 클래스 라벨 $t$의 분포라고 가정합니다. Inception score 는 다음에 의해 계산됩니다.

$$
\mathrm{IS} _C=\exp(\mathrm{E} _{\mathrm{y}\sim p(\mathrm{y})}[\mathrm{KL}(p(t|\mathrm{y})|p(t))])
$$

$p(t) = \int_y(p(t|\mathrm{y})d\mathrm{y}$입니다. Salimans et al.[40]에서는 Inception score가 신경망 생성 이미지의 시각적 품질과 양의 상관관계가 있다고 주장합니다.

inception scores(IS)[40]으로 측정됩니다. 변환 정확 측정을 위해 학습된 2개의 inception 판별 모델을 사용해 inception score를 보고하며, 각각 all과 test로 표시됩니다.


- Distribution matching
Frechet Inception Distance FID[17]는 두 이미지 셋 간의 유사성을 측정하도록 설계되었습니다. 우리는 ImageNet pretrain된 Inception-V3[45] 모델의 마지막 average pooling layer의 activation을 FID를 계산하기 위한 이미지의 feature vector로 사용합니다. 우리는 보지 못한 $|\mathbb{T}|$ class가 있으므로, source 이미지를 각 보지 못한 $|\mathbb{T}|$ 클래스로 변환하고 $|\mathbb{T}|$ 변환 결과 셋을 생성합니다. 각 $|\mathbb{T}|$ 변환 결과 셋에 대해, 우리는 ground truth 이미지들과의 FID를 계산합니다. 이를 통해 $|\mathbb{T}|$ FID score를 얻을 수 있습니다. $|\mathbb{T}|$ FID score의 평균은 mean FID(mFID)라고 하는 최종 distribution matching performance metric으로 사용됩니다.

Distribution matching은 FID(Fréchet Inception Distance)[17]를 기반으로 합니다. 우리는 각 $|\mathbb{T}|$ target class에 대한 FID를 계산하고 평균 FID(mFID)를 보고합니다.

### 결과

```
Table 1 삽입
```
> Table 1. fair, unfair baseline들과 성능 비교.  $\uparrow$ 는 큰 수가, $\downarrow$는 작은 수가 좋다는 것을 의미합니다.

Table 1에서 볼 수 있듯이, FUNIT 프레임워크는 Animal Face와 North American Birds 데이터셋 모두에 대해 모든 성능 metric에서 few-shot unsupervised image-to-image translation에 대한 baseline들을 능가합니다. 표를 통해 FUNIT 모델의 성능이 $K$와 양의 상관 관계가 있음을 보여줍니다. $K$가 클수록 모든 metric의 개선하며, 가장 큰 성능 향상은 $K=1$에서 $K=5$까지 발생합니다.

Unfair 모델들은 2개의 도메인 변환만 가능하기 때문에 클래스 수에 따른 여러 개의 모델을 학습하지만, FUNIT은 단 1개의 모델만으로 가능하며 마찬가지로 단 1개의 모델을 사용하는 StarGAN-Fair-K가 있지만 성능의 차가 큼을 볼 수 있습니다. MUNIT이 Unfair 모델이지만 그나마 유사한 성능을 보여주네요.


```
Figure 2 삽입
```
> Figure 2. few-shot unsupervised image-to-image translation 결과 시각화. 결과는 FUNIT-5 모델을 사용해 계산되었습니다. 위에서부터 순서대로 animal face, bird, flower, food 데이터셋에 대한 결과가 있습니다. 각 데이터셋에 대해 하나의 모델을 학습합니다. 각 예시에 대해, 우리는 무작위로 샘플링된 클래스 이미지 5개 중 2개인 $\mathrm{y}_1, \mathrm{y}_2$, 입력 content 이미지 $\mathrm{x}$, 변환 결과 $\bar{\mathrm{x}}$를 시각화합니다. 결과는 FUNIT 이 학습 중 보지 못했던 target class의 이미지를 난이도가 높은 few-shot 설정에서 그럴듯한 변환 결과를 생성한다는 것을 보여줍니다. 우리는 출력 이미지의 객체가 입력과 유사한 포즈를 취한다는 것에 주목합니다.

Figure 2에서 FUNIT-5에 의해 계산된 few-shot 변환 결과를 볼 수 있습니다. 입력 content 이미지 $\mathrm{x}$과 해당 출력 이미지 $\bar{\mathrm{x}}$에 있는 객체의 포즈는 대체로 동일하게 유지됩니다. 출력 이미지는 photorealistic하며 target 클래스의 이미지와 유사합니다.


```
Figure 3 삽입
```
> Figure 3. few-shot image-to-image translation 성능에 대한 시각적 비교. 왼쪽에서 오른쪽 방향으로, 열은 입력 이미지 $\mathrm{x}$, 2개의 target class 이미지 $\mathrm{y}_1, \mathrm{y}_2$, Unfair StarGAN baseline의 변환 결과, fair StarGAN baseline의 변환 결과, FUNIT의 결과입니다.

Figure 3에서 fair가 가능한 모델인 StarGAN과 FUNIT의 시각적 비교를 볼 수 있습니다. 같은 입력에서 StarGAN은 많은 양의 artifact를 가진 이미지를 생성하거나 target class가 아닌 입력된 content 이미지와 유사한 이미지를 출력합니다. 반면, FUNIT은 이미지 변환 결과를 성공적으로 생성합니다.


### number of source class
```
Figure 4
```
> Figure 4. few-shot image translation 성능 vs Animal Faces 데이터 셋 학습 중 사용한 객체 클래스 수. 성능은 학습 중 사용한 객체 source class의 수와 양의 상관관계가 있습니다.

```
Figure 7
```
> Figure 7. Few-shot image translation performance vs North American Birds 데이터셋 학습 중 객체 클래스 사용 숫자

Figure 4에서, 우리는 animal dataset을 사용해 one-shot 설정(FUNIT-1)에서 학습 셋의 source class 수를 변화시키는 것에 대한 성능을 분석합니다. 우리는 69개에서 119개 클래스까지 10개 간격으로 변화시켜 곡선을 표현합니다. 표시된 것과 같이, 성능은 변환 정확도, 이미지 품질 및 분포 일지 측면에서 객체 클래스 수와 양의 상관 관계가 있습니다. domain-invariant perceptual distance는 평평하게 유지됩니다. 이는 학습 중 더 많은 객체 클래스(다양성 증가)를 보는  FUNIT 모델이 테스트 중에 더 나은 성능을 보임을 보여줍니다. Appendix C에 제공된 bird dataset에서도 유사한 경향이 관찰됩니다.

우리는 few-shot 변환 성능이 animal face 변환 작업을 위한 학습 셋의 source class 수와 양의 상관관계가 있음을 보여줍니다. Figure 7에서 우리는 bird 변환 작업에서도 동일한 경향임을 보여줍니다. 구체적으로, 우리는 North American Birds 데이터 셋을 사용해 학습 셋에서 사용 가능한 source 클래스 수에 따른 제안된 모델의 성능을 보고합니다. 구체적으로, 우리는 North American Birds 데이터셋을 사용해 학습 셋에서 사용 가능한 source 클래스 수 대비 제안된 모델의 성능을 보고합니다. 우리는 source class 수를 189, 222, 289, 333, 389에서 444로 변경해 성능 점수를 표시합니다. 우리는 점수의 곡선이 메인 논문에 표시된 Animal Face 데이터셋의 곡선과 동일한 경향을 가진다는 것을 발견했습니다. 모델을 학습 중에 더 많은 수 의 source 클래스를 볼 때 test 중에 더 나은 성능을 발휘합니다.


### ablation study
```
Table 5 삽입
```
> Table 5. object term에 대한 Ablation study. $\uparrow$ 는 숫자가 클 수록 좋고, $\downarrow$는 숫자가 작을 수록 좋다는 것을 의미합니다. FM은 feature matching loss term이 제거된 제안된 프레임워크의 설정을 나타내고 GP는 gradient panalty loss가 제거된 제안된 프레임워크 설정을 나타냅니다. 기본 설정은 대부분의 기준에서 더 나은 성능을 제공합니다. 이 실험에서 우리는 FUNIT-1 모델을 사용합니다.

Table 5에서 loss term들이 Animal Face 데이터를 학습한 모델의 성능에 미치는 영향을 분석한 ablation study 결과를 나타냅니다. feature matching loss를 제거할때 성능이 조금 저하되며, zero-centered gradient penalty를 제거할 때는

gradient penalty가 본문에서는 설명이 없고 Appendix의 Table 5에서만 나와있어서 WGAN-GP의 gradient penalty인가 했는데, <a href="https://github.com/NVlabs/FUNIT/issues/18" target="_blank">공식 repo issue</a>에서 관련 글을 찾을 수 있었습니다. $D$ 학습 중 $D$의 loss를 사용해 gradient를 구하는 것까지는 WGAN-GP와 동일합니다. WGAN-GP에서는 (gradient - 1) 값을 제곱해 gradient penalty로 사용했다면 여기서는 pow로 exponential

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


```
Table 4 삽입
```
> Table 4. content image rconstruction loss weight, $\lambda_R$에 대한 파라미터 민감도 문석. $\uparrow$ 는 숫자가 클 수록 좋고, $\downarrow$는 숫자가 작을 수록 좋다는 것을 의미합니다. 0.1의 값은 content preservation(콘텐츠 보존)과 translation accuracy(변환 정확도)의 좋은 균형(trade-off)를 제공하며, 이는 논문 전체에서 기본값으로 사용됩니다. 본 실험에서 FUNIT-1 모델을 사용합니다.

Table 4에서 content image reconstruction loss의 weight가 Animal Face 데이터셋을 학습한 모델의 성능에 미치는 영향을 분석한 결과를 볼 수 있습니다. 저자들은 $\lambda_R$값이 클수록 translation accuracy(변환 정확도)가 낮아지지만, domain invariant perceptual distance가 작다는 것을 발견했으며,  $\lambda_R=0.1$이 좋은 절충안을 제공한다는 것을 보여주어 논문 전체 실험에서 default로 사용합니다.

흥미롭게도, $\lambda_R=0.01$로 매우 작으면 content preservation 및 translation accuracy 모두에서 성능이 저하됩니다. 이는 content reconstruction loss가 학습을 regularize 하는 데 도움이 된다는 것을 나타낸다고 합니다.

### vs AdaIN style transfer
```
Figure 9 삽입
```
> Figure 9. few-shot image translation을 위한 FUNIT-1 대 AdaIN style transfer[18]

 few-shot animal face 변환 작업을 위한 AdaIN transfer network[18]를 학습하고 Appendix E에서 성능을 비교합니다. 우리는 style transfer network가 입력 동물의 texture(질감)을 변경할 수 있지만, 모양을 변경하지 않다는 것을 발견했습니다. 결과적으로, 변환 결과는 입력과 여전히 유사합니다.

 Figure 9에서, 우리는 few-shot animal face translation task를 위해 제안된 방법을 AdaIN style transfer[18]과 비교합니다. AdaIN style transfer 방법은 입력 동물의 질감을 변경할 수 있지만 외형은 변경하지 않습니다. 결과적으로 변환 결과는 외형 측면에서 입력과 여전히 유사합니다.


### Failuer case
```
Figure 10 삽입
```
> Figure 10. 실패 사례들. 제안된 FUNIT 모델의 전형적인 실패 사례에는 hybrid objects generation(예. column 1, 2, 3, 4), 입력 content 이미지 무시(예. column 5, 6), 입력 클래스 이미지 무시(예. column 7)이 포함됩니다.

Figure 10은 제안된 알고리즘의 몇 가지 실패 사례를 보여줍니다. 사례는 hybrid class generation, 입력 content 이미지 무시, 입력 클래스 이미지 무시 등을 포함하고 있습니다.


### few-shot classification
```
Table 3 삽입
```
> Table 3. 분할에 걸쳐 평균을 나타낸 few-shot classification 정확도

animal과 bird 데이터셋을 사용해  few-shot classification에 대한 FUNIT을 평가합니다. 특히, 우리는 학습된 FUNIT 모델을 사용해 각 few-shot 클래스에 대한 N(1, 50 100까지 다양) 이미지를 생성하고 생성된 이미지를 사용해 판별 모델을 학습합니다. 우리는 FUNIT으로 학습된 판별 모델이 feature hallucination(특징 환각) 을 기반으로 샘플 갯수 N에 대한 제어 가능한 변수를 가지고 있는 Hariharan et al.[15]에서 제안한 few-shot classification 접근 방식보다 지속적으로 더 나은 성능을 달성한다는 것을 발견했습니다.

Appendix H 정리


## 정리

Appendix G, J 정리

우리는 최초의 few-shot unsupervised image-to-image translation 프레임워크를 소개했습니다. 우리는 few shot 생성 기능이 학습 중에 보이는 객체 클래스의 수와 양의 상관관계가 있으며  test time 동안 제공되는 target class의 장(shot) 수와도 양의 상관관계가 있음을 보여주었습니다.

FUNIT은 test time에 사용할 수 있는 모델에게 보여주지 않은 클래스의 몇몇 이미지를 이용해 source class의 이미지를 본 적 없는 객체 클래스의 유사한 이미지로 변환하는 방법을 배울 수 있다는 경험적 증거를 제공했습니다. FUNIT은 새로운 기능을 달성하기는 하지만 다음 몇가지 조건에 따라 작동합니다. 1) content encoder $E_x$가 class-invariant latent code(클래스 불변 잠재 코드)  $z_x$를 학습할 수 있는지 여부, 2)class encode $E_y$가 class-specific latent code(클래스 별 잠재 코드) $z_y$를 학습할 수 있는지 여부, 그리고 가장 중요한 것은 3) class encoder $E_y$가 보이지 않는 객체 클래스의 이미지로 일반화할 수 있는지 여부 입니다.

우리는 새로운 클래스가 source 클래스와 기각적으로 관련되어 있을 때 이런 조건을 충족하기 쉽다은 것을 관찰했습니다. 그러나 새로운 객체 클래스의 외관이 source class의 와괸과 극적으로 다를 때 FUNIT은 Figure 5와 같이 변환을 실패합니다. 이 경우 FUNIT은 입력 content 이미지의 색상이 변경된 버전으로 생성하는 경향이 있습니다. 이는 바람직하지 않지만 외형 분포가 극적으로 변경되었기 때문에 이해할 수 있습니다. 이 한계를 해결하는 것이 우리의 향후 작업입니다.


---

FUNIT 논문에 대해 살펴봤습니다. 끝까지 봐주셔서 감사합니다:)
