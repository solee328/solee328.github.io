---
layout: post
title: Pix2Pix(1) - 논문 리뷰
# subtitle:
categories: gan
tags: [gan, pix2pix, 생성 모델, 논문 리뷰]
# sidebar: []
use_math: true
---

## 소개

<img src="/assets/images/posts/pix2pix/paper/fig1.png" width="650" height="300">


Pair 데이터를 사용한 생성 모델인 Pix2Pix는 생성 모델을 Conditional GAN, 판별 모델을 PatchGAN으로 구성해 다양한 비전, 그래픽 과제들에 적용할 수 있는 범용적인 모델이라는 큰 장점으로 가지고 있습니다.

이번 글은 논문 리뷰로 기본적으로는 논문의 내용을 이해하기 위해 내용 번역을 논문 내용을 적고 내용 요약과 추가 설명이 필요하다 생각되는 세션에는 <font color='41 69 E1'>[정리]</font>로 설명을 적겠습니다.

---

## Abstract
image-to-image 변환 문제에 대한 범용 솔루션으로 conditional adversarial networks(이후 조건부 GAN으로 대체 사용하겠습니다)를 사용하며 입력 이미지에서 출력 이미지로의 매핑을 학습하기 위한 loss 함수를 사용합니다. 본 논문은 이 접근 방식이 label map, edge map에서 이미지를 합성/재구성하는 것과 이미지를 colorizing에 효과적이라는 것을 보여줍니다.

## 1. Introduction
이미지 처리, 컴퓨터 그래픽스, 컴퓨터 비전의 많은 과제들은 입력 이미지를 출력 이미지로 변환하는 것으로 이미지는 RGB image, gradient field, edge map, semantic label map 등 다양하게 렌더링될 수 있습니다. 자동 언어 번역이 가능한 것처럼 자동 image-to-image 변환 또한 충분한 학습 데이터가 주어진다면 한 장면의 표현을 다른 장면으로 변환하는 작업으로 정의할 수 있습니다. 과거의 이런 과제들은 각각 별도의 특수 목적의 시스템으로 처리되었지만 본 논문에서 우리의 목표는 이러한 모든 문제에 대한 공통된 프레임워크를 개발하는 것입니다.<br>

Convolutional Neural Nets(CNNs)로 다양한 이미지 예측 문제를 해결하는 방향으로 연구들이 진행되고 있고 CNNs는 loss 함수를 최소화하는 방법을 학습합니다. 하지만 효과적인 loss 함수를 만드는 것은 아직 많은 수작업이 필요합니다. 단순한 접근 방식으로 CNN에게 예측(predicted)과 실제(ground truth)의 유클리디안 거리를 최소화하는 것은 흐릿한 결과를 생성하는 경향이 있습니다. 유클리디안 거리를 최소화하는 것은 결과의 모든 값을 평균화하는 것이기 때문이 흐릿한 결과를 유도하기 때문입니다.

'현실과 구별할 수 없는 출력을 만든다'와 같이 고수준의 목표를 명시하는 대신 목표를 달성하기 위한 적절한 loss 함수를 학습할 수 있다면 매우 바람직할 것 입니다. 최근 제안된 Generative Adversarial Networks(GANs)에 의해 이를 수행할 수 있으며 GANs는 출력이 실제인지 가짜인지 구별하는 판별 모델 loss를 학습하는 동시에 이 loss를 최소화하기 위한 생성 모델 또한 학습합니다. 흐릿한 이미지는 명백히 가짜로 보이기 때문에 GANs는 흐릿한 이미지를 가짜로 판별하고 생성 모델은 더 선명한 이미지를 생성하게 될 것입니다.

본 논문에서 우리는 GANs의 조건 설정을 탐구합니다. GAN이 데이터 생성 모델을 학습하는 것처럼 조건부 GANs(cGAN)은 조건부 생성 모델을 학습합니다. cGAN은 조건부 입력 이미지에 따라 출력 이미지를 생성하는 image-to-image 변환에 적합합니다.

조건부 GANs가 다양한 과제들에 대해 합리적인 결과를 생성한다는 것을 입증합니다. 코드는 <a href="https://github.com/phillipi/pix2pix" target="_blank">https://github.com/phillipi/pix2pix</a> 에서 확인할 수 있습니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
기존의 과제 해결 방법은 해당 과제를 해결하기 위한 프레임워크를 개발했다면 본 논문의 프레임워크는 다양한 과제를 해결할 수 있음을 말하고 있습니다.<br><br>
이미지 예측에서 사용되는 CNN에서 L1, L2와 같은 유클리디안 거리를 사용한 loss는 흐릿한 결과를 생성하나 GAN loss를 함께 사용한다면 보다 선명한 이미지를 생성할 수 있게 됩니다.<br><br>
논문은 image-to-image에 적합한 조건부 GAN을 사용하며 위의 링크를 통해 공식 코드와 결과를 github에서 확인하실 수 있습니다.
</font>


## 2. Related work
Structed losses for image modeling, Conditional GANs와 관련된 논문들이 열거되어 있습니다.

Structed losses for image modeling에서 image-to-image 변환 문제는 주로 per-pixel classification이나 regression으로 수식화하고 문제를 해결했으나 조건부 GAN은 loss를 학습한다는 점에서 기존 수식 모델과는 다르고 출력과 정답의 오차가 있는 구조에 대해 loss를 이용해 불이익을 줄 수 있다는 점이 나와있습니다.

Conditional GANs에서는 조건부 Gan이 inpainting, future state prediction, super resolution과 같은 여러 image-to-image 분야에서 인상적인 결과를 달성한 분야들이 나열되어있습니다. 이전의 연구들은 각 방법들은 목적에 맞는 다른 term(예시: L2 regression)들로 loss를 조정했는데 본 논문의 연구는 특정 목적이 없는 것이 특징입니다. 또한 이전 논문들과는 생성 모델과 판별 모델의 구조에 차이가 있는데 생성 모델은 'U-Net'을 사용하고 판별 모델은 'PatchGAN'을 사용하는 것이 특징입니다.

## 3. Method

<div>
  <img src="/assets/images/posts/pix2pix/paper/fig2.png" width="450" height="200">
</div>
> Figure 2 : edges $\rightarrow$ photo 를 학습한 조건부 GAN입니다. 생성 모델 $D$는 생성모델이 합성한 가짜와 진짜 이미지를 구별하도록 학습합니다. 조건이 없는 GAN과는 다르게 생성 모델과 판별 모델 모두 edge map을 확인할 수 있습니다.

GANs는 랜덤 노이즈 벡터 $z$를 출력 이미지 $y$로 매핑하는 매핑 $G : z \rightarrow y$ 을 수행하는 모델입니다. 대조적으로 조건부 GANs는 조건에 해당하는 이미지 $x$와 랜덤 노이즈 벡터 $z$를 출력 이미지 $y$로 매핑하는 매핑 $G : \{ x, z \} \rightarrow y$ 을 학습해 수행합니다.

### 3.1. Objective
조건부 GAN의 목적 함수는 아래의 수식과 같습니다.

$$
\mathcal{L} _{cGAN}(G, D) = \mathbb{E} _{x,y} [logD(x, y)] + \mathbb{E} _{x, z}[log(1-D(x, G(x, z)))] \tag{1}
$$

$G$는 목적 함수를 최소화하려고 하며 목적 함수를 최대화하려는 $D$에 대항합니다. 이를 다시 정리하면 $G^* = arg \min_G \max_D \mathcal{L} _{cGAN}(G, D)$ 로 표현할 수 있습니다.

<a href="https://arxiv.org/abs/1604.07379" target='_blank'>'Context Encoders: Feature Learning by Inpainting'</a>과 같은 연구는 전통적인 loss와 GAN의 목적 함수를 섞는 것이 도움이 된다는 것을 발견했습니다. 판별 모델이 생성 모델로부터 생성된 가짜 이미지와 진짜 이미지를 구별해야 한다는 목적은 언제나 같지만 생성 모델의 경우 판별 모델을 속일 뿐만 아니라 실제 이미지와 최대한 유사해야 하기 때문에 L2 loss가 도움이 됩니다. 저자들은 L1이 흐릿함을 덜 유발한다는 점에서 L1 distance가 L2 distance 보다 사용하기 좋다는 것 또한 연구했다 합니다.

$$
\mathcal{L}_{L1}(G) = \mathbb{E} _{x, y, z}[\Vert y-G(x, z) \Vert_1]
\tag{3}
$$
> L1 loss 수식으로 실제 정답인 $y$와 생성 모델 $G$의 출력 값 사이의 차이값에 절대값을 취해 결과 값과 정답 값의 오차 합을 계산해 최소화하는 방법입니다.


<br><br>
위의 두 식을 합쳐 본 논문에서 사용하는 최종 목적 함수는 다음와 같습니다.

$$
G^* = arg \min\limits_{G} \max\limits_{D} \mathcal{L} _{cGAN}(G, D) + \lambda\mathcal{L} _{L1}(G)
\tag{4}
$$

$z$를 입력으로 같이 넣어주지 않는다해도 $x$를 $y$로 매핑하는 것을 모델이 학습할 수 있지만 확정된 출력 값을 생성하며 delta function의 분포만 표현할 수 있습니다. <a href="https://arxiv.org/abs/1511.05440" target="_blank">Deep multi-scale video prediction beyond mean square error</a> 연구의 경우 노이즈 $z$를 넣어 학습한 것과 $z$를 넣지 않은 채로 학습한 결과에 차이를 발견할 수 없었다 합니다. 따라서 본 연구에서는 노이즈 $z$를 $G$에 입력하지 않으며 대신 dropout을 이용했습니다. dropout 사용으로 노이즈를 사용한 것과 같은 distribution이 나오기를 바랬지만 결과에서는 minor stochasticity만이 발견되었으며 이러한 low stochasticity는 현재 연구에서 해결되지 않는 중요한 문제입니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
과거부터 주로 사용되던 loss를 GAN loss와 섞어 판별 모델을 속일 뿐만 아니라 현실적인 이미지로 만들 수 있도록 할 수 있습니다. L1 loss와 L2 loss 중 L1이 이미지의 blur함이 덜 해 L1 distance가 사용하기 좋습니다.<br><br>

delta function은 $\delta(x)$로 표현할 수 있으며 $\delta(x-a)$의 경우 $a$에서만 $\infty$의 값을 가지면 그 외에는 0의 값을 가집니다. delta function의 분포만 표현한다는 것은 하나의 출력만 생성하는 model collapse 또는 helvetica scenario라 불리는 하나의 결과를 출력하는 현상이 나타남을 의미합니다.<br><br>

하지만 <a href="https://arxiv.org/abs/1511.05440" target="_blank">Deep multi-scale video prediction beyond mean square error</a>의 경우 영상의 다음 장면을 예측하는 연구로 새로운 장면을 생성하기 위해서는 이전 화면만이 입력으로 필요했기에 기존에 사용하던 $z$를 사용할 의미가 사라졌습니다. 이와 같이 노이즈가 필요없는 경우도 있었기에 논문에서는 $z$를 $G$에 입력하지 않는 대신 dropout을 사용합니다. dropout 사용으로 출력의 다양성을 기대했으나 결과에서는 minor stochasticity 만이 발견되었다는 것은 출력의 분포가 다양하지 않았다는 것을 의미합니다.<br><br>

참고 : <a href="https://m.blog.naver.com/laonple/221358887811" target="_blank">라온피플 Pix2Pix</a>
</font>

### 3.2. Network architectures
<a href="https://arxiv.org/abs/1511.06434" target="_blank">dcgan</a>의 생성 모델과 판별 모델 구조를 채택했습니다. 생성 모델과 판별 모델 모두 <a href="https://arxiv.org/abs/1502.03167" target="_blank">batch normalization</a>에 소개된 convolution-BatchNorm-ReLu 구조의 모듈을 사용합니다.

#### 3.2.1 Generator with skips

<div>
  <img src="/assets/images/posts/pix2pix/paper/fig3.png" width="450" height="200">
</div>
> Figure 3: 생성 모델 구조에 사용할 수 있는 2가지 구조입니다.<br>
'U-Net'은 encoder와 decoder의 레이어들 사이의 skip connection을 가진 encoder-decorder 입니다.

image-to-image 변환 문제는 고해상도 입력을 고해상도 출력으로 매핑해야 한다는 것입니다. 입출력 이미지의 텍스쳐 등의 질감을 달라야 하지만 기본 구조는 같도록 만들어야 한다. 이 문제에 대한 이전의 다양한 솔루션들은 <a href="https://www.science.org/doi/10.1126/science.1127647" target="_blank">Reducing the Dimensionality of Data with Neural Networks</a>의 encoder-decoder 네트워크를 사용합니다. 이 네트워크는 입력이 bottleneck 이라 불리는 레이어까지 downsample되는 레이어들을 통과합니다. 이미지 변환 문제의 경우 입력과 출력 사이에 low-level 정보들이 공유되고 이 정보들은 네트워크를 통해 전달되는 것이 바람직합니다. 하지만 bottleneck으로 이런 정보들이 손실되는 것을 피하기 위해 'U-Net'의 모양을 따라 skip connection을 추가합니다. 우리는 각 레이어 $i$와 레이어 $n - i$ 사이에 skip connection을 추가하며 여기서 $n$은 전체 레이어의 수이다.

<font color='41 69 E1'>
<b>[정리]</b><br>
bottleneck은 convolution으로 channel 수를 줄이고 다시 convolution으로 channel 수를 늘릴 때 channel 수가 줄어든 layer를 bottleneck layer라 합니다. 대부분의 경우 1 x 1 convolution으로 channel 수를 증가시키고 연산량을 줄이는데 정보량과 정보손실은 trade off 관계로 channel을 줄이고 늘리는 과정에서 정보가 손실되게 됩니다. encoder에서의 bottlenect으로 인해 이미지의 edge, corner와 같은 low level 정보가 손실되는 것을 줄이기 위해 skip connection을 사용한 U-Net을 사용합니다.<br><br>

U-Net은 encoder-decoder에 skip connection을 추가한 것과 같은 모양으로 Figure 3에서 표현된 것과 같이 $i$ 레이어에서 온 feature map과 $n-i-1$레이어에서 up-conv 된 feature map이 concatenation되어 $n-i$ 레이어가 됩니다.
</font>


#### 3.2.2 Markovian discriminator (PatchGAN)
L2, L1 loss가 이미지 생성 문제에 대해 흐릿함(blur)을 생성한다는 것은 잘 알려져 있습니다. L2, L1 loss는 high frequency를 잘 포착하지 못하지만 low frequency를 정확하게 포착합니다. 따라서 low frequency 정확도를 높이기 위해 L1 term을 사용하며 이 L1이 high frequency 정확도가 낮다는 것을 알기에 판별 모델을 high frequency 구조로 모델링하도록 제한할 동기를 부여한다. 로컬 이미지 패치 구조를 가지는 것으로 high frequency에 집중한 모델링할 수 있기에 패치 규모로만 수행되는 판별 모델 구조를 설계합니다. 이 판별 모델은 N x N 패치가 실제인지 가짜인지 분류하며 모든 응답의 평균을 $D$의 출력으로 내보냅니다.

4.4에서 patch의 크기가 이미지의 전체 크기보다 훨씬 작은 경우에도 고품질의 결과를 생성할 수 있음을 보여줍니다. 작은 PatchGAN은 파라미터 수가 작고 속도가 빠르며 임의의 큰 이미지에도 적용할 수 있다는 장점이 있습니다.

이런 판별 모델은 지정된 patch 크기보다 멀리 있는 픽셀 간은 독립되어 있다 가정하며 Markov random field 에 효과적입니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
L1 loss를 사용해 low frequency(구조 등)를 잘 포착하나 high frequency(edge, color와 같은 디테일한 부분)는 잘 포착하지 못하니 판별 모델은 high frequency 위주로 가짜 이미지를 판별하도록 해 생성 모델이 high frequency 를 잘 포착해 이미지를 생성할 수 있도록 학습합니다.<br><br>

PatchGAN은 말 그대로 이미지를 patch로 나눠 patch 부분만 확인해 진짜인지 가짜인지 확인하기 때문에 patch 크기에 따라 이미지의 디테일한 부분(high frequency)을 집중적으로 보게 되며 설정된 patch 외부의 이미지에 대해서는 독립되어 있다 가정하므로 Markov Random Field라 할 수 있습니다.
</font>

### 3.3. Optimization and inference
<a href="https://arxiv.org/abs/1406.2661" target="_blank">Generative adversarial nets</a>의 접근법을 따르며 $D$와 $G$를 번갈아 가며 gradient descent를 수행합니다. 기존 GAN 논문에서 제안된 것처럼 $log(1-D(x, (G(x, z))))$를 최소화하도록 $G$를 학습하는 대신 $log(x, G(x, z))$를 최대화하도록 학습합니다. 또한 $D$의 학습단계에서 목적 함수(=loss)를 2로 나누어 $D$의 학습 속도를 늦춥니다. 학습에서는 mini batch SGD를 사용하고 learning rate가 0.0002이고 momentum의 파라미터가 $\beta_1=0.5$, $\beta_2=0.999$인 <a href="https://arxiv.org/abs/1412.6980" target="_blank">Adam solver</a>를 사용합니다.

inference 시 학습 단계와 정확히 같은 방식으로 생성 모델 네트워크를 실행한다. 즉 test 때에도 dropout을 적용하며 training batch가 아닌 test batch를 사용해 batch normalization을 적용한다는 점에서 일반적인 프로토콜과는 다릅니다. batch 크기를 1로 설정할 때 batch normalization을 'instance normalization'이라 하며 이미지 생성에서 효과적인 것으로 <a href="https://arxiv.org/abs/1607.08022" target="_blank">Instance Normalization: The Missing Ingredient for Fast Stylization</a>에서 입증되었습니다. 본 논문에서는 실험에 따라 batch 크기가 1에서 10 사이를 사용합니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
GAN, DCGAN의 설정을 따라 $log(x, G(x,z))$를 최대화하도록 $G$를 학습하고 mini batch SGD, Adam을 사용합니다.<br><br>

batch normalization은 test 시에는 학습 때 사용했던 mini batch에서 얻은 평균과 분산 값을 이용해 normalization하는 것이 일반적입니다. 논문에서는 학습 시 사용한 통계가 아닌 test 에서 사용하는 데이터의 평균과 분산을 이용해 normalization 합니다. batch size는 1 ~ 10 로 사용하며 이 중 batch size가 1인 경우 이미지 생성에 좋다는 것이 증명되었으며 이를 instance normalization이라 합니다.
</font>

## 4. Experiments
조건부 GAN의 일반성을 탐구하기 위해 사진 생성과 같은 그래픽 작업과 semantic segmentation 과 같은 비전 작업을 포함한 다양한 작업과 데이터 셋에서 테스트를 진행했습니다.
- semantic labels $\leftrightarrow$ photo
- architectural $\leftrightarrow$ photo
- map $\rightarrow$ aerial photo
- BW $\rightarrow$ color photos
- Edges $\rightarrow$ photo
- sketch $\rightarrow$ photo
- day $\rightarrow$ night
- thermal $\rightarrow$ color
- photo with missing pixels $\rightarrow$ inpainted photo

<br>
<div>
  <img src="/assets/images/posts/pix2pix/paper/fig14.png" width="530" height="370">
</div>
> Figure 14: ground truth와 비교한 facades labels $\rightarrow$ photo 의 결과

<div>
  <img src="/assets/images/posts/pix2pix/paper/fig15.png" width="530" height="270">
</div>
> Figure 15: ground truth와 비교한 day $\rightarrow$ night의 결과

작은 데이터 셋에서도 종종 괜찮은 결과를 얻을 수 있었다 합니다. facade 학습 셋은 400 개의 이미지로만 구성되어 있으며 day $\rightarrow$ night 학습 셋은 91 개의 웹캠 이미지로만 구성되어 있음에도 위와 같은 결과를 얻었습니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
비전, 그래픽 등 다양한 과제들에서 테스트를 진행했으며 위의 Figure 14와 Figure 15의 경우 적은 데이터셋만으로도 좋은 결과를 얻었습니다.
</font>

### 4.1. Evaluation metrics
생성된 이미지의 품질을 평가하는 것은 미해결된 어려운 문제입니다. pixel 당 mean-squared error와 같은 전통적인 메트릭은 결과의 구조물들을 통계나 구조를 고려하지 않기 때문에 한계가 존재합니다.

결과를 전체적으로 평가하기 위해 2가지 전략을 사용합니다. 첫번째는 Amazon Mechanical Turk(AMT)로 'real vs fake'에 대해 사람이 평가하는 인식테스트입니다. 두번째는 'FCN-score'로 기존의 분류 모델을 합성된 이미지 안의 물체를 인식할 수 있을 정도로 합성된 이미지가 현실적인지 여부를 측정하는 방법입니다.

**AMT perceptual studies**<br>
AMT의 경우 <a href="https://arxiv.org/abs/1603.08511" target="_blank">Colorful image colorization</a>의 방법을 따릅니다. Turker는 작업자를 의미합니다. Turker는 알고리즘에 의해 생성된 'fake' 이미지와 'real' 이미지를 비교하는 실험을 진행합니다. 각 실험에서 이미지는 1초 동안 보여진 후 turker들은 어떤 것이 가짜인지에 대해 판단할 수 있는 시간이 주어집니다. 각 세션마다 처음 10개 이미지는 연습이며 연습마다 turker들은 피드백을 받아 정답을 학습하게 됩니다. 이후 40개의 이미지가 주어지며 이에 대한 피드백은 제공되지 않습니다. 50명 이하의 turker들이 알고리즘을 평가하며 <a href="https://arxiv.org/abs/1603.08511" target="_blank">Colorful image colorization</a>와는 다르게 vigilance trials를 포함하지 않았습니다.

**"FCN score"**<br>
생성 모델의 정량적 평가는 어려운 것으로 알려져 있지만 최근 연구들은 생성 이미지들의 판별 가능성을 측정하고자 유사 메트릭으로 사전 학습된 semantic classifier를 사용했습니다. 생성된 이미지가 현실적이라면 실제 이미지에 대해 학습된 판별 모델도 생성된 이미지를 정확하게 분류할 수 있다는 것이 아이디어입니다. 이를 위해 semantic segmentation으로 인기 있는 <a href="https://arxiv.org/abs/1411.4038" target="_blank">Fully Convolutional Networks for Semantic Segmentation</a>의 FCN-8s 구조를 채택해 Cityscapes 데이터 셋으로 학습시킨 후 생성된 이미지에 대한 분류 정확도로 생성된 사진의 점수를 측정합니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
생성 이미지 품질을 평가하는 절대적인 방법은 없지만 논문에서는 AMT와 FCN으로 정성적 / 정량적 평가를 합니다.<br><br>

AMT는 사람이 주어진 이미지를 보고 진짜인지 가짜인지 판단하는 실험입니다. 생성된 이미지가 얼마나 많은 사람들에게 진짜 이미지처럼 보여 사람들을 속일 수 있었는지에 대한 퍼센트를 통해 분석합니다.<br><br>

FCN은 기존의 semantic segmentation 모델을 사용해 생성 이미지 내의 object들을 얼마나 정확하게 클래스 별로 픽셀에 나타내는 지를 실험합니다. 생성된 이미지과 현실의 이미지와 유사할 수록 segmentation 모델 또한 더 정확하게 이미지 내의 object 들을 segmentation 할 수 있을 것이라는 아이디어로 segmentation 결과의 pixel, class 별 정확도 등을 분석합니다.
</font>


### 4.2. Analysis of the objective function
ablation study(모델의 구조나 feature를 제거해 가며 해당 요소가 성능에 얼마나 영향을 미치는지 확인해보는 실험)를 실행해 L1 term, GAN term의 영향을 분리하고 조건부 판별 모델(cGAN, Eqn. 1)과 조건 없는 판별 모델(GAN, Eqn. 2)을 사용하는 것을 비교합니다.

<div>
  <img src="/assets/images/posts/pix2pix/paper/fig4.png" width="530" height="400">
</div>
> Figure 4: loss에 따른 결과의 품질 차이. 각 column은 다른 loss로 학습된 결과입니다.

Figure 4는 labels $\rightarrow$ photo 문제에 대해 비교에 대한 질적 효과를 보여줍니다. L1 만으로도 합리적인 결과긴 하지만 흐릿함을 볼 수 있습니다. 조건부 GAN 단독(Eqn. 4에서 $\lambda=0$으로 설정) 사용은 훨씬 더 sharp한 결과를 제공하지만 시각적 아티팩트(노이즈)가 발견됩니다. 두 term을 모두 추가하면($\lambda=100$) 아티팩트가 줄어듭니다.
<br>

<div>
  <img src="/assets/images/posts/pix2pix/paper/table1.png" width="350" height="150">
</div>
> Table 1: Cityscapes labels $\leftrightarrow$ photos 에서 loss에 따른 FCN 점수 평가

Table 1은 Cityscape $\leftrightarrow$ photo의 FCN 점수를 사용해 정량화한 것입니다. 조건부 GAN인 cGAN은 L1에 비해 더 높은 점수를 달성했으며 생성 이미지에 모델이 인식 가능한 구조들이 있음을 의미하고 더 현실적인 이미지를 생성했다 할 수 있습니다. 조건부가 없는 GAN의 경우 결과를 조사하면 입력 사진과 상관없이 거의 동일한 출력을 생성하는 model collapsed 가 생성 모델에 발생했음을 알 수 있었다 합니다. L1 term을 추가하면 L1 loss가 출력이 정답과의 거리를 비교해 loss에 더하기 때문에 더 좋은 결과가 나옴을 볼 수 있습니다.


**colorfulness**

<div>
  <img src="/assets/images/posts/pix2pix/paper/fig7.png" width="600" height="160">
</div>
> Figure 7: Cityscape에서 테스트된 cGAN의 Color distribution matching property와 histogram intersection scores.<br>

L1이 edge를 정확히 어디에 위치시킬지 불확실할 때 흐릿함을 발생시키는 것과 같이 픽셀이 어떤 색을 취해야하는지 불확실할 때 평균적인 회색을 띄는 색을 발생시킵니다. term을 최소화하기 위해 확률 밀도 함수의 중위수를 선택하게 되고 이 수치는 회색에 가까운 색이기 때문입니다. 반면 adversarial loss는 회색 출력이 비현실적이라는 것은 판별 모델이 인식하게 되니 실제 색상 분포와 일치하는 방향으로 학습할 수 있습니다. Figure 7을 통해 색상에 대한 loss의 효과를 볼 수 있습니다. L1이 ground truth보다 더 좁은 분포로 이어지며 이는 L1이 평균적인 회색빛을 조장한다는 가설을 확인시켜줍니다. 반면 cGAN은 출력 분포를 실제 값에 더 가깝게 생성합니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
labels $\rightarrow$ photo에 대한 Figure 4를 통해 loss 설정에 대한 결과를 비교할 수 있습니다. L1 loss만 사용한 경우 흐릿하고 cGAN loss만 사용한 경우 선명하나 시각적 아티팩트가 보이지만 $\lambda=100$으로 설정해 L1 loss와 cGAN loss를 모두 사용한 경우 시각적 아티팩트가 줄어든 선명한 이미지가 결과로 나옴을 볼 수 있습니다.<br><br>

cityscape $\leftrightarrow$ photo에 대한 loss 별 FCN 점수를 나타낸 Table 1에서도 L1 loss와 cGAN loss를 모두 사용한 경우 점수가 가장 높음을 볼 수 있습니다.<br><br>

Color distribution 그래프의 L은 brightness(명도), a는 Red/Green Value, b는 Blue / Yellow Value를 의미합니다. a의 경우 양수 a 값이 커질 수록 붉은색 / 음수 a 값이 커질 수록 녹색에 가까워지고 b의 경우 양수 b 값이 커질 수록 파란색 / 음수 b 값 커질 수록 노란색에 가까워집니다.<br>
점선(ground truth)와 각 loss 간에 histogram 그래프가 겹치는 정도를 나타낸 것이 histogram intersection scores입니다. 그래프가 가장 많이 겹치는 것은 그만큼 원본 이미지의 명도, 색감을 잘 표현했다 할 수 있습니다. L1의 경우 픽셀이 어떤 값을 취할지 불확실할 때 color의 중위수인 회색을 출력하므로 이미지가 흐릿해지게 되는데 Figure 7의 (b), (c)에서도 L1이 가장 뾰족한 그래프를 나타내는 것 볼 수 있으며 다양한 색을 표현하지 못함을 알 수 있습니다.
</font>

### 4.3. Analysis of the generator architecture

<div>
  <img src="/assets/images/posts/pix2pix/paper/fig5.png" width="500" height="300">
</div>
> Figure 5: encoder-decoder에 skip connection을 추가해 만든 'U-Net'의 결과가 훨씬 좋은 품질을 가지고 있다.

<div>
  <img src="/assets/images/posts/pix2pix/paper/table2.png" width="400" height="100">
</div>
> Table 2: Cityscape labels $\leftrightarrow$ photos 에서 평가된 생성 모델 구조(및 loss) 별 FCN 점수.<br>
U-Net(L1+cGAN)는 이 실험에서는 batch 크기가 10이지만 다른 실험에서는 batch 크기가 1이였기 때문에 다른 Table에서와 점수가 다르다.

Figure 5와 Table2로 U-Net과 encoder-decoder를 비교해 skip connection의 효과를 볼 수 있습니다. L1 loss로만 학습할 때, L1 + cGAN loss로 학습할 때 모두 U-Net이 더 우수한 결과를 나타냄을 볼 수 있습니다.

### 4.4. From PixelGANs to PatchGANs to ImageGANs

<div>
  <img src="/assets/images/posts/pix2pix/paper/fig6.png" width="600" height="100">
</div>
> Figure 6: patch 크기의 변화에 따른 결과 이미지.

<div>
  <img src="/assets/images/posts/pix2pix/paper/table3.png" width="400" height="120">
</div>
> Table 3: Cityscape labels $\rightarrow$ photos 에서 평가된 다양한 receptive field(=patch 크기)에 대한 FCN 점수. 입력 이미지는 256 x 256 픽셀이며 만약 이보다 더 큰 receptive field의 경우 0으로 패딩됩니다.

판별 모델의 receptive fields인 patch 크기 N을 1 x 1 'PixelGAN'에서 286 x 286 'ImageGAN'으로 변경했을 때의 효과를 테스트했습니다. Figure 6은 이 테스트의 결과를 보여주며 Table 3은 FCN 점수를 사용해 결과를 정량화합니다. 본 논문은 다른 곳에서 명시되지 않는 한 모든 실험이 70 x 70 PatchGAN을 사용했으며 이 섹션에서는 모든 실험이 L1 + cGAN loss를 사용했다는 것에 유의해야 합니다.

PixelGAN은 spatial sharpness에는 영향이 없지만 색상의 다양성을 증가시킵니다(Figure 7). 예시로 Figure 6의 버스는 네트워크가 L1 loss 였을 때 회색으로 칠해지지만 PixelGAN loss 에서는 빨간색이 된 것을 볼 수 있습니다.

16 x 16 PatchGAN을 사용하면 sharp한 출력이 가능하고 좋은 FCN 점수를 달성할 수 있지만 tiling 아티팩트가 나오기도 합니다. 70 x 70 PatchGAN을 사용하면 tiling 아티팩트를 완화하고 조금 더 나은 점수를 달성합니다. 286 x 286 ImageGAN은 결과의 시각적 품질을 향상시키지 않는 것으로 보이며 실제로 FCN 점수가 상당히 낮은 것을 Table 3에서 확인할 수 있습니다.


<div>
  <img src="/assets/images/posts/pix2pix/paper/fig8.png" width="570" height="300">
</div>
> Figure 8: 512 x 512 해상도의 구글 지도에 대한 예시 결과(모델은 256 x 256 해상도의 이미지로 학습되었으며 테스트 시 더 큰 이미지에서 convolution 됩니다). 선명도를 위해 이미지의 대비가 조정되었습니다.

**Fully-convolutional translation**<br>
PatchGAN의 장점은 고정 크기 patch 판별 모델을 임의의 큰 이미지에 적용할 수 있다는 것입니다. 생성 모델 또한 학습에 사용한 이미지보다 더 큰 이미지에 convolution처럼 적용할 수 있습니다. 우리는 map $\leftrightarrow$ aerial 작업에서 테스트했습니다. 256 x 256 이미지에 생성 모델을 학습할 후 512 x 512 이미지에 대해 테스트했으며 Figure 8을 통해 확인할 수 있습니다.

<font color='41 69 E1'>
<b>[정리]</b><br><br>
<img src="/assets/images/posts/pix2pix/paper/tiling.png" width="380" height="200">
<br>
여러 patch 크기를 테스트했으며 FCN 점수로 가장 좋은 것은 70 x 70 PatchGAN 으로 16 x 16 PatchGAN에서 나타나던 tiling 아티팩트 또한 70 x 70 PatchGAN에서는 줄어들었다고 합니다. 그림과 같이 타일들이 나눠진 것처럼 보이는 tiling 아티팩트는 blocking 아티팩트라고도 불리며 영상의 압축률이 높은 경우 인접한 블록들의 경계에서 불연속성이 보여 화질의 열화를 일으키게 됩니다.<br><br>

논문에서 명시되지 않는 한 70 x 70 PatchGAN을 사용했으며 PatchGAN의 장점은 Patch 단위로 이미지를 처리하기 때문에 모델이 이미지 크기에 상관 없이 사용할 수 있다는 것입니다. 이에 관해 256 x 256 이미지로 학습한 모델을 512 x 512 이미지에 대해서 테스트했으며 결과를 Figure 8에서 확인할 수 있습니다.<br><br>

참고 : <a href="https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART001238011" blank="_blank">고압축 영상의 블로킹 아티팩트 잡음 제거</a>

</font>

### 4.5. Perceptual validation

<div>
  <img src="/assets/images/posts/pix2pix/paper/table4.png" width="450" height="100">
</div>
> Table 4: maps $\leftrightarrow$ aerial photos 데이터 기반으로 실행한 AMT 'real vs fake' 실험 결과

map $\leftrightarrow$ aerial 과 grayscale $\leftrightarrow$ color 작업에 대한 사실성을 검증합니다. map $\leftrightarrow$ photo에 대한 AMT 실험 결과는 Table 4를 통해 확인할 수 있습니다. L1 + cGAN loss로 생성된 사진은 L1 baseline보다 훨씬 더 높은 18.9%의 실험자들을 속였으며 L1은 흐릿한 결과를 생성해 실험자들을 거의 속이지 못했습니다. 대조적으로 photo $\rightarrow$ map 에서 L1 + cGAN loss로 생성한 이미지는 6.1% 실험자들을 속였으며 L1 basline 성능과 크게 다르지 않았습니다. 이는 사소한 구조의 오류가 실제 항공 사진보다 기하학 구조를 가진 지도에서 더 잘 보이기 때문일 수 있습니다.

<div>
  <img src="/assets/images/posts/pix2pix/paper/table5.png" width="400" height="110">
</div>
> Table 5: colorization 데이터 기반으로 실행한 AMT 'real vs fake' 실험 결과

ImageNet에서 <a href="https://arxiv.org/abs/1603.08511" target="_blank">Colorful image colorization</a>에서 도입한 테스트 분할에 대해 실험을 진행했습니다. L1 + cGAN loss로 만든 이미지는 22.5%의 실험자들을 속였음을 Table 5에서 확인할 수 있습니다. cGAN과 L2 regression은 <a href="https://arxiv.org/abs/1603.08511" target="_blank">Colorful image colorization</a>과 유사한 점수를 얻었지만 27.8% 실험자를 속인 <a href="https://arxiv.org/abs/1603.08511" target="_blank">Colorful image colorization</a>에는 미치지 못했습니다. <a href="https://arxiv.org/abs/1603.08511" target="_blank">Colorful image colorization</a>가 colorization을 수행하도록 특별히 설계된 구조이며 pix2pix는 범용적인 구조라는 것에 주목해야 합니다.

<font color='41 69 E1'>
<b>[정리]</b><br><br>
<div>
  <img src="/assets/images/posts/pix2pix/paper/fig9.png" width="320" height="300">
</div>
<blockquote>Figure 9: 조건부 GAN vs [62]의 L2 regression와 [64]의 방법(classification with rebalancing). cGAN은 강렬한 colorization(첫 번째, 두 번째 행)을 생성할 수 있지만 grayscale 또는 색의 포화도가 적은 결과(마지막 행)를 생성하는 실패 또한 가지고 있다.</blockquote>

AMT 실험에서 L1 loss를 쓴 것보다 L1 + cGAN loss를 쓴 경우 실험자들을 훨씬 더 많이 속일 수 있었지만 map $\rightarrow$ photo와 photo $\rightarrow$ map 간의 점수 차이가 꽤나 큽니다. map의 경우 기하학적 도형들로만 이루어져있다보니 사소한 오류가 더 잘 느껴졌을 것이라 논문에서는 말하고 있습니다.<br><br>

colorization 에서는 <a href="https://arxiv.org/abs/1603.08511" target="_blank">Colorful image colorization</a>의 방법보다는 조금 떨어지는 AMT 결과가 나왔으나 pix2pix는 <a href="https://arxiv.org/abs/1603.08511" target="_blank">Colorful image colorization</a>와 같이 colorization만을 위한 방법이 아닌 다양한 과제에 사용할 수 있는 방법이라는 것을 고려해야 합니다. Figure 9를 통해 생성된 결과를 비교할 수 있는데 L1 + cGAN을 사용한 논문의 방법(3열)의 1, 2행에서는 매우 현실적인 사진을 생성할 수 있음을 볼 수 있습니다.
</font>

### 4.6. Semantic segmentation

<div>
  <img src="/assets/images/posts/pix2pix/paper/fig10.png" width="500" height="230">
</div>
> Figure 10: 조건부 GAN을 semantic segmentation에 적용한 결과. cGAN은 ground truth처럼 보이는 sharp한 이미지를 생성하는 것처럼 보이나 실제로는 실제로 존재하지 않은 자잘한 물체를 많이 생성합니다.

<div>
  <img src="/assets/images/posts/pix2pix/paper/table6.png" width="450" height="120">
</div>
> Table 6: cityscapes 데이터 기반의 photo $\rightarrow$ labels 성능 비교

조건부 GAN은 이미지 처리 및 그래픽 작업에서 출력이 상세한 현실적인 과제들에 효과적으로 보입니다. 그렇다면 출력이 입력보다 덜 복잡한 semantic segmentation과 같은 문제에서도 효과적일까요?

이 실험을 시작하기 전 cityscape photo $\rightarrow$ label 데이터 셋으로 cGAN(L1 loss가 있는 경우 / 없는 경우)을 학습시킵니다. Figure 10은 이에 대한 질적 결과를 보여주며 정량적 분류 정확도는 Table 6을 통해 확인할 수 있습니다. 흥미롭게도 L1 loss 없이 학습된 cGAN으로도 합리적인 정확도로 문제를 해결할 수 있음을 보였습니다. 본 논문은 기존 segmentation 방법과 달리 cGAN으로 라벨을 생성하는 semantic segmentation을 성공적으로 생성한 첫 번째 시도입니다. cGAN 만으로 학습한 것보다 L1 + cGAN이 조금 더 나은 결과를 보이지만 단순히 L1 regression을 사용한 방법이 가장 높은 정확도를 달성했습니다.


<font color='41 69 E1'>
<b>[정리]</b><br>
cGAN만으로 semantic segmentation을 처음으로 시도했다 주장합니다. 결과를 Figure 10에서 볼 수 있는데 첫번째 행에서 cGAN만을 사용한 방법은 우측 부분에 오류들이 생겨났음을 볼 수 있습니다. cGAN만으로도 합리적인 결과를 나타낸다 하지만 가장 정확도가 높은 방법은 L1 regression을 사용한 것임을 Table 6을 통해 확인할 수 있습니다.
</font>

### 4.7. Community-driven Research

<div>
  <img src="/assets/images/posts/pix2pix/paper/fig11.png" width="550" height="150">
</div>
> Figure 11: 온라인 커뮤니티에서 pix2pix 코드 베이스라인을 기반으로 개발한 어플리케이션 예제.

<div>
  <img src="/assets/images/posts/pix2pix/paper/fig12.png" width="350" height="200">
</div>
> Figure 12: 'Learning to see : Gloomy Sunday' pix2pix 코드 베이스라인을 기반으로 개발한 대화형 예술 데모.

논문과 pix2pix 코드 베이스를 최초 공개한 이후, 컴퓨터 비전과 그래픽 실무자 뿐만 아니라 비주얼 아티스트를 포함한 트위터 커뮤니티는 논문의 범위를 훨씬 벗어나 다양하고 새로운 image-to-image 변환 작업에 프레임워크를 성공적으로 적용했습니다. Figure 11과 Figure 12는 배경 제거, 팔레트 생성, 스케치 $\rightarrow$ 초상화, 포즈 전송, #edges2cats 등을 포함한 예시를 보여줍니다.

## 5. Conclusion
본 논문의 결과는 조건부 adversarial network가 많은 imag-to-image 변환 작업, 특히 구조화된 그래픽 출력을 포함하는 작업에 유망한 접근 방식임을 시사합니다.

---


<div>
  <img src="/assets/images/posts/pix2pix/paper/fig21.png" width="600" height="250">
</div>
> Figure 21: 실패 예시. 각 영상에서 왼쪽은 입력 이미지, 오른쪽은 출력 이미지를 나타냅니다. 이 예시들은 작업에서 최악의 결과 중 일부입니다. 일반적으로 실패하는 경우에는 비정상적인 입력이거나 이미지 내 영역이 많이 비어있는 경우 생기는 아티팩트들이 출력되는 경우가 포함됩니다.

추가로 Figure 21에는 실패한 사례들이 나와있습니다. 입력이 비정상적이거나 입력 이미지 내의 일부 영역이 많이 비어있어 어떤 정보를 나타내는 지 모델이 알기 어려울 때 실패하는 경우가 많다 합니다.<br>
논문 리뷰는 여기까지입니다!

논문의 마지막 부분인 6. Appendix에는 네트워크의 구조와 학습에 사용한 파라미터들이 적혀있는데 해당 부분은 다음 글인 논문 구현에서 소개해드리겠습니다:)
