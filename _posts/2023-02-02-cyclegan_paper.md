---
layout: post
title: CycleGAN(1) - 논문 리뷰
# subtitle:
categories: gan
tags: [gan, cyclegan, 생성 모델, 논문 리뷰]
# sidebar: []
use_math: true
---

## 소개

이번 글은 CycleGAN 논문 리뷰로 기본적으로는 논문의 내용을 이해하기 위해 내용 번역을 논문 내용을 적고 내용 요약과 추가 설명이 필요하다 생각되는 세션에는 <font color='41 69 E1'>[정리]</font>로 설명을 적겠습니다.
<br><br>

---

## Abstract
Image-to-Image 변환은 비전과 그래픽스 분야의 해결 과제 중 하나로 페어 이미지를 학습해 입력 이미지와 출력 이미지 사이의 매핑을 학습하는 것입니다. 하지만 많은 과제들에서 페어 이미지로 이루어진 학습 데이터를 사용할 수 없습니다. 우리는 페어 이미지의 데이터가 없는 경우 도메인 $X$에서 도메인 $Y$로 변환하는 방법을 학습하기 위한 접근 방식을 제안합니다. 우리의 목표는 $G : X \rightarrow Y$인 매핑 $G$를 학습하는 것으로 adversarial loss를 사용한 $G(X)$의 이미지 분포가 분포 $Y$와 구별할 수 없도록 하는 것입니다. 이 매핑을 제약이 매우 낮기 때문에 우리는 역 매핑인 $F : Y \rightarrow X$를 함께 사용하며 $F(G(X)) \approx X$(또는 그 반대)가 가능하도록 하는 cycle consistency loss를 소개합니다. style transfer, object transfiguration, season transfer, photo enhancement 등 학습 시 페어 데이터를 사용하지 않는 여러 과제에 대한 좋은 결과를 제시합니다. 몇몇 방법과 비교를 통해 우리의 접근 방식에 대한 우수성을 보여줍니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
기존의 GAN 논문들과 마찬가지로 adversarial loss를 사용해 도메인 $X$의 데이터를 도메인 $Y$로 변화시키는 매핑 $G : X \rightarrow Y$를 사용합니다.<br>
여기서 논문의 제목의 근간이 된 cycle consistency loss를 추가로 사용합니다. 매핑 $G$의 역 매핑인 $F$는 도메인 $Y$를 도메인 $X$로 변화시키는 $F : Y \rightarrow X$의 역할을 합니다.<br>
$F(G(X))$는 도메인 변화가 $X \rightarrow Y \rightarrow X$가 될 것이며 다시 도메인 $X$가 된 데이터는 원본 데이터와 비슷하기를 바라는 $F(G(X)) \approx X$를 cycle consistency loss로 사용합니다.
</font>
<br><br>

---

## 1. Introduction
<div>
  <img src="/assets/images/posts/cyclegan/paper/fig1.png" width="500" height="250">
</div>
> Figure 1: 두 개의 이미지 데이터 $X$와 $Y$가 주어지면, 우리의 알고리즘은 이미지를 자동으로 하나에서 다른 것으로 "변환"하는 방법을 배웁니다. (왼쪽)Flickr의 모네 그림과 풍경 사진, (가운데)ImageNet의 얼룩말과 말, (오른쪽)Flickr의 Yosemite의 여름과 겨울 사진. (아래) : 유명한 예술가의 그림을 사용해 우리의 방법은 자연 사진을 각 스타일로 렌더링하는 방법을 배운다.

클로드 모네는 1873년 기분 좋은 봄에 Argenteuil 근처 Seine 강둑에 그의 이젤을 놓았을 때 무엇을 보았을까요?(Fgirue 1, top-left) 만약 컬러 사진이 발명되었다면, 상쾌한 푸른 하늘과 그것을 반사하는 유리 같은 강을 기록했을지도 모릅니다. 모네는 밝은 팔레트를 통해 그 장면에 대한 그의 느낌을 전달했습니다.

만약 모네가 시원한 여름 저녁에 Cassis의 작은 항구에서 일어나게 되었다면 어땠을까요?(Figure 1, bottom-left) 모네의 그림 갤러리를 잠시 확인해본다면 파스텔 색조, 갑작스러운 페인트 얼룩, 다소 잠잠한 역동성 등 그가 어떤 장면을 연출했을지 상상할 수 있습니다.

우리는 모네가 그린 장면에 대한 실제 사진 옆에 모네 그림이 나란히 있는 예시를 본 적이 없음에도 불구하고 이 모든 것을 상상할 수 있습니다. 우리에게는 모네 그림과 풍경 사진 데이터 셋에 대한 지식을 가지고 있습니다. 우리는 두 데이터 셋 사이의 스타일적인 차이에 대해 추론할 수 있기에 한 데이터 셋에서 다른 데이터셋으로 "변환"한다면 어떤 장면이 어떻게 보일지 상상할 수 있습니다.

본 논문에서는 위의 이론을 동일하게 학습할 수 있는 방법을 제시합니다. 즉, 한 이미지 데이터 셋의 특수성을 포착하고 이런 특성이 학습에 사용할 페어 데이터가 없는 경우에도 다른 이미지 데이터 셋으로 어떻게 변환될 수 있는지를 파악합니다.

<div>
  <img src="/assets/images/posts/cyclegan/paper/fig2.png" width="300" height="180">
</div>
> Figure 2: 페어를 이루는 학습 데이터(왼쪽)은 $x_i$와 $y_i$ 사이에 관계가 존재하는 pix2pix에서 사용하며 학습 예제로 $\{ x_i, y_i\} ^N _{i=1}$로 이루어져 있습니다. 우리는 대신 $x_i$와 $y_i$ 간의 매칭되는 정보가 주어지지 않은 source 데이터 셋 $\{ x_i \} ^N _{i=1} (x_i \in X)$와 target 데이터 셋 $\{ y_j \} ^M _{j=1} (y_j \in Y)$로 이루어진 페어를 이루지 않는 학습 데이터(오른쪽)을 사용합니다.

이 문제는 주어진 장면의 한 표현인 $x$에서 다른 표현 $y$로 이미지를 변환하는 grayscale to color, image to semantic labels, edge-map to photograph와 같은 <a href="https://arxiv.org/abs/1611.07004" target="_blank">image-to-image 변환</a>으로 더 광범위하게 설명될 수 있습니다. 컴퓨터 비전, 이미지 처리, 컴퓨터 그래픽스에 대한 수년간의 연구는 지도학습 아래에서 페어 이미지 $\{ x_i, y_i \} ^N _{i=1}$가 가능한 강력한 변환 시스템을 만들었습니다(Figure 2, left). 하지만 학습을 위한 페어 데이터를 얻는 것은 어렵고 비용이 많이 들 수 있습니다. 예를 들어, semantic segmentation과 같은 작업을 위한 데이터 셋을 오직 몇 개만 존재하며 상대적으로 수가 적습니다. 예술적 스타일 변환과 같은 그래픽 작업을 위한 입출력 페어 데이터를 얻는 것은 원하는 출력이 매우 복잡하며 일반적으로 저작권이 필요하기 때문에 훨씬 더 어려울 수 있습니다. 객체 변환(예시 : 얼룩말 $\leftrightarrow$ 말, Figure 1, top-middle)과 같은 많은 과제들의 경우 원하는 출력이 잘 정의되기 힘듭니다.

따라서 우리는 페어를 이루는 입력-출력 예제 없이 도메인 간 변환을 학습할 수 있는 알고리즘을 찾습니다(Figure 2, right). 우리는 도메인 사이에 어떤 기본 관계가 있다고 가정합니다 예를 들어 도메인은 동일한 기본 장면의 두 가지 다른 렌더링이라고 가정하고 그 관계를 배우고자 합니다. 지도 학습에서 페어 데이터를 사용한 예시가 부족할 수 있지만 우리는 도메인 $X$의 이미지 셋과 다른 이미지 셋인 도메인 $Y$에 대한 데이터 셋이 제공되는 상태에서 지도학습을 이용할 수 있습니다. 우리는 $y$와 $\hat{y}$를 구별하도록 학습된 모델이 출력 $\hat{y} = G(x), x \in X$와 이미지 $y \in Y$를 구별할 수 없도록 매핑 $G : X \rightarrow Y$를 학습합니다. 이론적으로, 매핑 $G$는 분포 $p _{data}(y)$와 일치하는 $\hat{y}$에 대한 출력 분포를 유도할 수 있습니다. 따라서 최적의 $G$는 도메인 $X$를 $Y$와 동일하게 분포된 도메인 $\hat{Y}$로 변환합니다. 그러나 이러한 변환은 입력 $x$와 출력 $y$가 유의미하게 페어를 이룬다는 것을 보장하지 않으며 $\hat{y}$에 대해 동일한 분포를 유도하는 무한히 많은 매핑 $G$가 존재합니다. 더욱이 실제로 우리는 adversarial 목적 함수를 분리하여 최적화하는 것이 어렵다는 것을 알게 되었다. 일반적인 방법은 종종 모든 입력 이미지가 동일한 출력 이미지에 매핑되고 최적화가 이뤄지지 않는 mode collapse로 잘 알려진 문제를 초래합니다.

이러한 문제들은 우리의 목적 함수에 더 많은 구조를 추가할 것을 요구합니다. 따라서 우리는 변환 작업에 '주기적으로 일관되어야 한다(cycle consistent)'는 특성을 사용합니다. 예를 들어 영어에서 프랑스어로 문장을 번역한 다음, 프랑스어에서 영어로 다시 번역한다면 원래의 문장으로 돌아가야 합니다. 수학적으로, 만약 우리가 번역기 $G : X \rightarrow Y$와 또 다른 번역기 $F : \rightarrow X$를 가지고 있을 때 $G$와 $F$는 서로 반대의 기능을 수행해야 하며 두 매핑은 전단사 함수여야 합니다. 우리는 매핑 $G$와 $F$를 동시에 학습하고 'cycle consistency loss'를 추가해 $F(G(X)) \approx x$와 $G(F(y)) \approx y$가 가능하도록 합니다. 이 cycle consistency loss가 도메인 $X$와 $Y$의 adversarial loss와 결합하면 페어를 이루지 않은 image-to-image 변환에 대한 우리의 전체 목적 함수를 만들어 냅니다.

우리는 style transfer, object transfiguration, season transfer, photo enhancement를 포함한 응용 프로그램에 우리의 방법을 적용합니다. 또한 스타일과 컨텐츠의 factorization 또는 shared embedding functions에 의존하는 이전 접근 방식과 비교하여 우리의 방법이 이런 기준선을 능가한다는 것을 보여줍니다. 우리는 <a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix" target="_blank">pytorch</a>와 <a href="https://github.com/junyanz/CycleGAN" target="_blank">torch</a> 구현을 모두 제공합니다. 우리의 <a href="https://junyanz.github.io/CycleGAN/" target="_blank">웹사이트</a>에서 더 많은 결과를 확인하세요.

<font color='41 69 E1'>
<b>[정리]</b><br>
페어 데이터는 데이터 셋의 수가 작기도 작고 만드는 데 비용이 많이 들며 출력에 대한 정의 또한 어렵습니다. 따라서 페어를 이루지 않는 데이터 셋을 사용하는 image-to-image 변환에 대한 알고리즘을 만들고자 합니다.<br><br>

adversarial loss로 $p _{data}(y)$에 일치하는 $\hat{y}$에 대한 출력 분포를 유도할 수 있지만 입력 $x$와 출력 $y$가 유의미하게 페어를 이룬다는 것을 보장하지 않으며 mode collapse 문제가 발생할 수 있습니다. 이런 문제를 해결해하기 위해 $F(G(X)) \approx x$와 $G(F(y)) \approx y$가 가능하도록 cycle consistency loss를 추가합니다.<br><br>

adversarial loss와 cycle consistency loss를 사용해 페어를 이루지 않은 데이터셋에서 image-to-image 변환이 가능하도록 합니다.
</font>
<br><br>

---

## 2. Related work
**Generative Adversarial Networks(GANs)**
[16, 63]은 image generation[6, 39], image editing[66], representation learning[39, 43, 37]에서 인상적인 결과를 달성했습니다. 최근의 방법은 text2image[41], image inpainting[38], future prediction[36]과 같은 조건부 이미지 생성 뿐만 아니라 비디오[54], 3D 데이터[57]와 같은 다른 도메인에 대해서도 동일한 아이디어를 채택합니다. GANs의 성공 핵심은 생성된 이미지가 원칙적으로 실제 사진과 구별할 수 없도록 하는 adversarial loss에 대한 아이디어입니다. 이 loss는 많은 컴퓨터 그래픽이 최적화하려는 목표이기 때문에 특히 이미지 생성 작업에 강력하게 작용합니다. 우리는 변환된 이미지가 대상 도메인과 구별할 수 없도록 매핑을 학습하기 위해 adversarial loss를 채택해 사용합니다.

**Image-to-Image Translation**
Image-to-Image 변환의 아이디어는 단일 입력-출력 페어 이미지에 non-parametric texture model을 사용한 Hertzmann의 Image Analogire[19] 연구까지 거슬러 올라갑니다. 보다 최근의 접근 방식은 CNN을 사용하여 parametric translation function을 학습하기 위해 입력-출력 데이터 셋을 사용합니다(예시 : [33]). 우리의 접근 방식은 Isola의 'pix2pix' 프레임 워크를 기반으로 하며[22], 이건은 조건부 adversarial generation nerwork[16]을 사용하여 입력에서 출력 이미지로의 매핑을 학습합니다. 비슷한 아이디어들이 sketch로부터 photographs를 생성하는 [44] 또는 attribute와 semantic layout으로부터 photographs를 생성하는 [25]와 같은 작업들에 적용되었습니다. 하지만 이전의 연구들과는 다르게 우리는 페어를 이루는 학습 데이터 없이 매핑하는 방법을 학습하고자 합니다.

**Unpaired Image-to-Image Translation**
페어를 이루지 않는 환경을 다루는 데 목표는 $X$와 $Y$라는 두 데이터 도메인을 연관시키는 것입니다. Rosales[42]는 source image에서 계산된 patch 기반의 Markov random field와 여러 style image에서 얻은 likelihood term을 기반으로 하는 Bayesian framework를 제안합니다. 더 최근에는 CoGAN[32]과 cross-modal scene network[1]가 weight-sharing 전략을 사용해 도메인 간의 공통된 representation을 학습했습니다. 우리의 방법과 동시에 Liu[31]은 variational autoencoders[27]와 generative adversarial networks[16]의 조합으로 프레임워크를 확장했습니다. 다른 계열에서 동시의 연구들[46, 49, 2]은 입력과 출력이 'style'은 다를 수 있지만 특정 'content' feature를 공유하도록 장려합니다. 또한 이러한 방법은 class label space[2], image pixel space[46], image feature space[49]와 같이 사전 정의된 메트릭 공간에서 입력에 가까운 출력을 강제하는 추가적인 term과 함께 adversarial network를 사용합니다.

위의 접근 방식과 달리, 우리의 공식은 입력과 출력 사이의 작업 별 사전 정의된 유사성 함수에 의존하지 않으며 입력과 출력이 동일한 저차원 임베딩 공간에 있어야 한다고 가정하지도 않습니다. 이는 우리의 방법은 많은 비전 및 그래픽 작업을 위한 범용 솔루션으로 만듭니다. 우리는 Section 5.1.에서 이전의 몇가지 연구들과 현재 접근법을 직접 비교합니다.

**Cycle Consistency**
`구조를 가진 데이터`를 정규화하는 방법으로 transitivity를 사용하는 아이디어는 오랜 역사를 가지고 있습니다. visual tracking에서 단순한 forward-backward consistency를 시행하는 것은 수십 년 동안 표준적으로 사용한 방법이였습니다[24, 48]. 연어 영역에서 "back translation, reconciliation"을 통해 번역을 검증하고 개선하는 것은 인간 번역가[3] (재미있게도 MarkTwain[51]을 포함합니다)와 기계[17]에 의해 사용되는 기술힙니다. 보다 최근에는 motion[61], 3D shape matching[21], co-segmentation[55], dense semantic alignment[65, 64], depth estimation[14]의 구조에서 cycle consistency가 사용되었습니다.

**Neural Style Transfer**
[13, 23, 52, 12]는 image-to-image 변환을 수행하는 또 다른 방법으로 사전 학습된 deep feature의 gram matrix와 일치하는 것을 기반으로 한 하나의 이미지의 콘텐츠를 다른 이미지의 스타일(일반적으로 그림)과 결합하여 새로운 이미지를 합성합니다.


<font color='41 69 E1'>
<b>[정리]</b><br>
논문의 목적인 image-to-image 변환 관련 연구로는 gram matrix를 사용해 이미지를 합성하는 Neural Style Transfer와 조건부 adversarial 모델을 사용하는 연구들이 있었습니다.<br>
조건부 adversarial 모델 관련 연구들은 페어 데이터 셋을 사용했거나 페어를 이루지 않은 데이터 셋을 사용하더라도 연구 주제에 따라 연구 주제에 적합하도록 정의된 유사성 함수에 의존했습니다.<br>
본 논문은 페어를 이루지 않는 학습 데이터 셋을 사용하며 연구 주제에 따라 유사성 함수를 따로 설정하지 않아도 되는 범용 솔루션을 제안합니다.<br><br>

논문은 실제 데이터와 구별할 수 없도록 만드는 GANs의 adversarial loss와 Cycle consistency loss를 사용합니다.
</font>
<br><br>

---

## 3. Formulation
<div>
  <img src="/assets/images/posts/cyclegan/paper/fig3.PNG" width="600" height="180">
</div>
> Figure 3: (a) 우리의 모델은 두 매핑 $G : X \rightarrow Y$와 $F : Y \rightarrow X$를 포함하고 adversarial discriminator인 $D_Y$와 $D_X$와 연관되어 있습니다. $D_Y$는 $G$가 $X$를 도메인 $Y$와 구별할 수 없는 결과로 변환하도록 장려하며 $D_X$와 $F$는 그 반대입니다. 매핑을 더욱 정교하게 하기 위해, 우리는 한 도메인에서 다른 도메인으로 변환하고 다시 되돌리면 우리가 시작한 곳에 도달해야 한다는 직관을 가진 두 가지 cycle consistency loss를 소개합니다. (b) forward cycle-consistency loss : $x \rightarrow G(X) \rightarrow F(G(x)) \approx x$, (c)backward cycle-consistency loss : $y \rightarrow F(y) \rightarrow G(F(y)) \approx y$

우리의 목표는 학습 데이터로 $x_i \in X$인 $\lbrace x_i \rbrace ^N _{i=1}$ 와 $y_j \in Y$인 $\lbrace y_j \rbrace ^M _{j=1}$를 가진 두 도메인 $X$와 $Y$ 사이를 매핑하는 함수를 학습하는 것입니다. 우리는 데이터 분포를 $x \sim p _{data}(x)$와 $y \sim p _{data}(y)$로 나타냅니다. Figure 3에서 표현된 것처럼 우리의 모델은 두 매핑 $G : X \rightarrow Y$와 $F : Y \rightarrow X$를 포함합니다. 추가로 우리는 2개의 adversarial discriminator인 $D_X$와 $D_Y$를 소개합니다. $D_X$는 이미지 $\lbrace x \rbrace$와 변환된 이미지 $\lbrace F(y) \rbrace$를 구별하는 것이 목적이고 같은 방법으로 $D_Y$는 이미지 $\lbrace y \rbrace$와 $\lbrace  G(x) \rbrace$를 구별하는 것이 목적입니다. 우리의 목적은 생성된 이미지의 분포와 target 도메인의 데이터 분포를 매칭하기 위한 adversarial loss[16]와 학습된 매핑 $G$와 $F$가 서로 부정하는 것을 막기 위한 cycle consistency loss, 총 두 종류의 term을 담고 있습니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
도메인 $X$에서 도메인 $Y$로 변환하기 위한 매핑 $G$, 도메인 $Y$에서 도메인 $X$로 변환하기 위한 매핑 $F$를 adversarial loss를 이용해 학습합니다.<br>
또한 두 매핑은 두 도메인 $X$와 $Y$ 간의 변환이므로 데이터 $x$에게 $G$와 $F$를 적용한 $F(G(x))$는 다시 도메인 $X$에 대한 데이터가 되기를 바라는 forward cycle consistency loss와 데이터 $y$에게 $F$와 $G$를 적용한 $G(F(x))$는 다시 도메인 $Y$에 대한 데이터가 되기를 바라는 backward cycle consistency loss를 cycle consistency loss로 적용해 학습합니다.
</font>

### 3.1. Adversarial Loss
우리는 adversarial loss[16]을 두 매핑 함수 모두에 적용합니다. 우리는 매핑 함수 $G : X \rightarrow Y$와 판별 모델 $D_Y$를 목적 함수로 아래와 같이 표현합니다.

$$
\mathcal{L} _{GAN}(G, D_Y, X, Y) = \mathbb{E} _{y \sim p _{data}(y)}[log D_Y(y)] + \mathbb{E} _{x \sim p _{data}(x)}[log(1-D_Y(G(x)))]
\tag{1}
$$

$G$는 도메인 $Y$의 이미지들과 비슷해보이도록 이미지 $G(x)$를 생성하고 $D_Y$는 생성된 이미지 $G(x)$와 실제 이미지 $y$를 구별하는 것에 목적을 둡니다. $G$는 목적함수를 최소화하며 이에 대항하는 $D$는 목적함수를 최대화합니다. 즉, $\min_G \max _{D_Y} \mathcal{L} _{GAN} (G, D_Y, X, Y)$입니다. 우리는 매핑 함수 $F : Y \rightarrow X$ 뿐 만 아니라 이에 대한 판별 모델 $D_X$에 대해서도 유사한 adversarial loss를 도입합니다. 즉, $\min_F \max _{D_Y} \mathcal{L} _{GAN} (F, D_X, Y, X)$입니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
GAN의 adversarial loss와 같은 식으로 $G$와 $D$의 기능 또한 같지만 매핑이 2가지라는 것에 차이가 있습니다.<br><br>

생성 모델 $G$는 데이터 $x$를 도메인 $Y$의 분포와 비슷하도록 만들고 판별 모델 $D_Y$는 도메인 $Y$의 데이터인지 아닌지 $G(x)$에 대해서 판별합니다. 역 매핑에 대해서도 마찬가지로 생성 모델 $F$는 데이터 $y$를 도메인 $X$의 분포와 비슷하도록 만들고 판별 모델 $D_X$는 $F(y)$가 도메인 $X$의 데이터인지 판별합니다.
</font>

### 3.2. Cycle Consistency Loss
이론적으로 adversarial 학습은 각각 대상 도메인 $Y$와 $X$로 동일하게 분포된 출력을 생성하는 매핑 $G$와 $F$를 학습할 수 있습니다[15]. 그러나 복잡한 관계를 표현할 수 있는 정보를 담을 용량이 충분한 네트워크는 target 분포와 일치하는 출력 분포를 유도할 수 있지만 동일한 입력 이미지 셋을 target 도메인의 임의의 이미지에 매핑할 수 있습니다. 따라서 adversarial loss 만으로는 학습된 개별 함수가 개별 입력 $x_i$를 원하는 출력 $y_i$에 매핑할 수 있다고 보장할 수 없습니다. 가능한 매핑 함수의 공간을 줄이기 위해, 우리는 학습된 매핑 함수는 cycle-consistency 여야 한다고 주장합니다. Figure 3의 (b)에서 보여준 것처럼 도메인 $X$의 각각의 이미지 $x$에 대해서 이미지 변환 사이클은 원본 이미지 $x$로 다시 돌릴 수 있어야 합니다. 즉, $x \rightarrow G(x) \rightarrow F(G(X)) \approx x$입니다. 우리는 이것을 forward cycle consistency라 부릅니다. 유가하게 표현되는 Figure 3 (c)는 도메인 $Y$의 각각의 이미지 $y$에 대해서 $G$와 $F$는 backward cycle consistency : $y \rightarrow F(y) \rightarrow G(F(y)) \approx y$를 만족해야 합니다. 우리는 cycle consistency loss를 사용해 이를 장려합니다.

$$
\mathcal{L} _{cyc} (G, F) = \mathbb{E} _{x \sim p _{data}(x)}[\| F(G(x))-x \|_1] + \mathbb{E} _{y \sim p _{data}(y)}[\| G(F(x))-y \|_1]
\tag{2}
$$

선행 연구들에서 시도했던 것처럼 우리 또한 loss에 L1 norm을 $F(G(x))$와 $x$, $G(F(y))$와 $y$ 사이의 adversarial loss로 대체하려 시도했지만 향상된 성능을 관찰하진 못했습니다.


<div>
  <img src="/assets/images/posts/cyclegan/paper/fig4.png" width="350" height="350">
</div>
> Figure 4: 다양한 실험에서의 입력 이미지 $x$, 출력 이미지 $G(x)$ 그리고 재구성된 이미지 $F(G(x))$. 위에서부터 아래까지 순서대로 photo $\leftrightarrow$ Cezanne, horses $\leftrightarrow$ zebras, winter Yosemite $\leftrightarrow$ summer Yosemite, aerial photos $\leftrightarrow$ google maps

cycle consistency loss로 인한 결과를 Figure 4에서 확인할 수 있습니다. 재구성된 $F(G(x))$는 입력 이미지 $x$와 유사하게 매칭됩니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
생성 모델 $G$가 $\lbrace x_i \rbrace$를 도메인 $Y$의 분포와 일치하도록 변환한 $G(x)$가 원하는 출력인 $\lbrace y_i \rbrace$에 일치하지 않을 수 있습니다. 대표적인 예시로는 생성 모델이 어떤 입력에서도 같은 출력을 내놓는 Mode Collapse 가 있습니다.
이 문제를 해결하기 위해 생성한 $G(x)$를 생성 모델 $F$에 입력해 다시 도메인 $X$로 변환한 $F(G(x))$와 원본 데이터인 $x$와 L1 loss를 사용해 최대한 일치하도록 만드는 cycle consistency loss를 도입합니다.<br><br>

매핑 함수가 2개이니 cycle consistency loss도 2개로 구별할 수 있습니다.<br>
forward cycle consistency는 source 데이터 $x$와 $F(G(x))$와의 L1 loss를 계산하며 backward cycle consistency는 target 데이터 $y$와 $F(G(y))$와의 L1 loss를 계산합니다.
</font>


### 3.3. Full Objective
우리의 목적 함수는 아래와 같습니다.

$$
\mathcal{L}(G, F, D_X, D_Y) = \mathcal{L} _{GAN} (G, D_Y, X, Y) +  \mathcal{L} _{GAN} (F, D_X, Y, X) + \lambda \mathcal{L} _{cyc}(G, F)
\tag{3}
$$

여기서 $\lambda$는 두 목적의 상대적 중요성을 제어합니다. 우리는 아래를 해결하는 것을 목표로 합니다.

$$
G ^* , F ^* = arg \min _{G, F} \max _{D_x, D_Y} \mathcal{L}(G, F, D_X, D_Y)
\tag{4}
$$

우리의 모델은 "autoencoders"[20]를 학습하는 것으로 볼 수 있습니다. 우리는 하나의 autoencoder $F \circ G : X \rightarrow X$와 다른 antoencoder $G \circ F : Y \rightarrow Y$를 함께 학습합니다. 그러나 이런 autoencoder들은 각각 이미지를 다른 도메인으로 변환하는 중간 표현을 통해 이미지를 자신에게 매핑하는 특별한 내부 구조를 가지고 있습니다. 이러한 설정은 임의의 target 분포와 일치하도록 autoencoder의 bottleneck 레이어를 학습하기 위해 adversarial loss를 사용하는 "adversarial autoencoder"[34]의 특별한 경우로 볼 수 있습니다. 이 경우 $X \rightarrow X$를 수행하는 autoencoder의 target 분포는 도메인 $Y$의 분포입니다.

Section 5.1.4 에서는 adversarial loss $\mathcal{L} _{GAN}$을 단독으로 사용한 것과 cycle consistency loss $\mathcal{L} _{cyc}$를 단독으로 사용한 것을 포함해 목적 함수의 일부와 전체 목적 함수를 비교하고, 두 목적이 고품질 결과에 도달하는 데 중요한 역할을 한다는 것을 경험적으로 보여줍니다. 또한 우리는 한 방향의 cycle loss만 가지고 우리의 방법을 평가하고 단일 주시로는 제한되지 않는 문제에 대한 학습을 정규화하기에 충분하지 않다는 것을 보여줍니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
최종 목적 함수는 Equation 3으로 위에서 설명된 GAN loss(adversarial loss)와 cyc loss(cycle consistency loss)를 사용합니다.
<br><br>

autoencoder는 input $x$ $\rightarrow$ encoder $\rightarrow$ $z$ $\rightarrow$ decoder $\rightarrow$ output $x$의 구조를 가집니다. 여기서 encoder를 $G$, decoder를 $F$로 이해한다면 입력 값이 $G$(encoder)를 통해 다른 값으로 변하고 다시 $F$(decoder)를 통해 원래의 값으로 돌아오는 모양이 되며 CycleGAN의 모델과 같은 모양이라 볼 수 있게 됩니다.<br>
중간 표현인 $z$가 bottleneck 레이어를 통해 나오게 되는데 이 $z$가 도메인 $Y$, input $x$와 output $x$를 도메인 $X$로 본다면 autoencoder로 도메인 변화가 $X \rightarrow Y \rightarrow X$가 되므로 논문과 일치한다 할 수 있습니다.
<br><br>

adversarial loss만 사용한 경우, cycle consistency loss만 사용한 경우, adversarial + forward cycle consistency loss를 사용한 경우, adversarial + backward cycle consistency loss를 사용한 경우, adversarial + cycle consistency loss를 사용한 경우를 실험한 결과를 Section 5.1.4에서 확인할 수 있습니다.

</font>
<br><br>

---

## 4. Implementation
**Network Architecture**
우리는 neural style transfer와 super resolution에 대해 인상적인 결과를 보여준 Johnson[23]의 생성 네트워크를 채택합니다. 이 네트워크는 3개의 convolution, 여러 residual blocks[18], $\frac{1}{2}$의 stride를 가진 2개의 fractionally-strided convolution 그리고 feature map을 RGB에 매핑하는 convolution 하나를 포함합니다. 128x128 이미지에는 6개의 block을 사용하고 256x256 이상의 고해상도 학습 이미지에는 9개의 block을 사용합니다. Johnson[23]과 비슷하게 우리는 instance normalization[53]을 사용합니다. 판별 모델 네트워크의 경우 70x70 PatchGAN[22, 30, 29]를 사용하며 70x70 부분 이미지 패치가 실제인지 가짜인지를 분류합니다. 이런 패치 수준 판별 모델 아키텍처를 전체 이미지 판별 모델보다 매개 변수가 적으며, fully convolutional fashion[22] 방식으로 임의의 크기의 이미지에서 작용할 수 있습니다.

**Training details**
우리는 모델 학습 절차를 안정화하기 위해 최근 연구의 2가지 기술을 적용합니다. 첫번째로 $\mathcal{L} _{GAN}$ (Equation 1)의 경우 negative log likelihood 목적을 least-squares loss[53]로 대체합니다. 이 손실은 훈련 중에 더 안정적이고 더 높은 품질의 결과를 생성합니다. 특히 GAN loss $\mathcal{L} _{GAN}(G, D, X, Y)$를 위해 우리는 $\mathbb{E} _{x \sim p _{data}(x)}[(D(G(x)) - 1) ^2]$를 최소화하도록 $G$를 학습하고 $\mathbb{E} _{y \sim p _{data}(y)}[(D(y) - 1) ^2] + \mathbb{E} _{x \sim p _{data}(x)}[D(G(x)) ^2]$를 최소화하기 위해 $D$를 학습합니다.

두번째로 model oscillation[15]를 줄이기 위해, 우리는 Shrivastava의 전략[46]을 따르고 마지막 생성 모델이 생성한 하나의 이미지가 아닌 생성된 이미지 히스토리를 사용해 판별 모델을 업데이트한다. 우리는 이전에 생성된 50개의 이미지를 저장하는 이미지 버퍼를 가지고 있는다.

모든 실험에서, 우리는 Equation 3의 $\lambda = 10$으로 설정했다. 우리는 배치 크기가 1인 Adam solver[26]을 사용한다. 모든 네트워크는 0.0002의 학습률을 가지고 학습되었다. 우리는 처음 100 epoch 동안 학습률을 유지하고 다음 100 epoch 동안 학습률을 0으로 감소시킨다. 데이터 셋, 아키텍처 및 학습 순서에 대한 자세한 내용은 부록(Section 7)을 참고하라.


<font color='41 69 E1'>
<b>[정리]</b><br>

</font>
<br><br>

---

## 5. Results
우리는 input-output 페어를 이루고 있는 데이터 셋에 페어를 이루지 않은 image-to-image 변환에 대해서 최근의 연구들과 우리의 접근 방식을 비교합니다. 이후 adversarial loss와 cycle consistency loss의 중요성을 연구하고 전체 loss에 대해 변형된 방법들과 비교합니다. 미자믹으로, 우리는 페어를 이룬 데이터가 존재하지 않는 광범위한 응용 프로그램에서 알고리즘의 일반성을 입증합니다. 우리는 우리의 방법을 간략하게 CycleGAN이라 부릅니다. <a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix" target=_blank>PyTorch</a> 및 <a href="https://github.com/junyanz/CycleGAN" target=_blank>Torch</a> 코드, 모델, 전체 결과는 우리의 <a href="https://junyanz.github.io/CycleGAN/" target=_blank>웹사이트</a>에서 확인할 수 있습니다.

### 5.1. Evaluation
pix2pix[22]와 동일한 데이터 셋 및 메트릭을 사용해 우리의 방법을 질적으로나 양적으로 여러 기준 모델들과 비교합니다. 작업에는 Cityscapes 데이터 셋에서의 semantic labels $\leftrightarrow$ photo와 Google Maps에서 스크랩한 데이터의 map $\leftrightarrow$ aerial photo가 포함됩니다. 또한 우리는 전체 loss 함수에 대한 `ablation study`를 수행합니다.


<details>
<summary>ablation study</summary>
<span style="color:gray">
  ddd
  <br><br>

  참고<br>
  - Fintecuriosity님의 <a href="https://fintecuriosity-11.tistory.com/73" target="_blank">[데이터 사이언스]"Ablation study"란 무엇인가?</a><br>
  <br>
</span>
</details>


#### 5.1.1 Evaluation Metrics
**AMT perceptual studies**


**FCN score**


**Semantic segmentation metrics**

<font color='41 69 E1'>
<b>[정리]</b><br>

</font>

#### 5.1.2 Baselines
**CoGAN[32]**


**SimGAN[46]**


**Feature loss + GAN**

**BiGAN / ALI{9, 7}**


**pix2pix[22]**

<font color='41 69 E1'>
<b>[정리]</b><br>

</font>

#### 5.1.3 Comparison against Baselines
<div>
  <img src="/assets/images/posts/cyclegan/paper/fig5.png" width="600" height="280">
</div>
> Figure 5:

<div>
  <img src="/assets/images/posts/cyclegan/paper/fig6.png" width="600" height="280">
</div>
> Figure 6:


<div>
  <img src="/assets/images/posts/cyclegan/paper/table1.png" width="600" height="280">
</div>
> Table 1:

<div>
  <img src="/assets/images/posts/cyclegan/paper/table2.png" width="600" height="280">
</div>
> Table 2:

<div>
  <img src="/assets/images/posts/cyclegan/paper/table3.png" width="600" height="280">
</div>
> Table 3:



<font color='41 69 E1'>
<b>[정리]</b><br>

</font>

#### 5.1.4 Analysis of the loss function
<div>
  <img src="/assets/images/posts/cyclegan/paper/table4.png" width="600" height="280">
</div>
> Table 4:

<div>
  <img src="/assets/images/posts/cyclegan/paper/table5.png" width="600" height="280">
</div>
> Table 5:


<font color='41 69 E1'>
<b>[정리]</b><br>

</font>

#### 5.1.5 Image reconstruction quality


<font color='41 69 E1'>
<b>[정리]</b><br>

</font>

#### 5.1.6 Additional results on paired datasets



<font color='41 69 E1'>
<b>[정리]</b><br>

</font>

### 5.2 Applications


<font color='41 69 E1'>
<b>[정리]</b><br>

</font>

<br><br>

---

## 6. Limitations and Discussion
<div>
  <img src="/assets/images/posts/cyclegan/paper/fig17.png" width="600" height="280">
</div>
> Figure 17: 우리의 방법에서 전형적인 실패 사례들. 왼쪽 : 개 $\rightarrow$ 고양이 변환 작업에서 CycleGAN은 입력에서 최소한의 변경만 만들어 낼 수 있었습니다. 오른쪽 : CycleGAN은 말 $\rightarrow$ 얼룩말 예제에서도 실패 사례가 나왔는데 모델이 학습 중 승마에 대한 이미지를 보지 못했지 때문입니다.

우리의 방법이 많은 경우레 설득력 있는 결과를 얻을 수 있었지만, 결과들이 모든 곳에서 긍정적인 결과인 것과는 거리가 있습니다. Figure 17은 몇 가지 실패 사례를 보여줍니다. 위에서 언급한 것들처럼 색상과 질감 변화를 수반하는 변환 작업에서 우리의 방법은 종종 성공적이였습니다. 우리는 기하학적 변화가 필요한 작업을 탐구했지만 거의 성공하지 못했습니다. 예를 들어 개 $\rightarrow$ 고양이(Figure 17, 좌측) 변환 작업에서 모델은 입력을 최소한의 변경만 하는 것으로 퇴화합니다. 이 실패는 외관 변경에 대한 우수한 성능을 위해 조정된 생성 모델 구조로 인해 발생할 수 있습니다. 더 다양하고 극단적인 변화, 특히 기하하적 변화를 다루는 것은 향후 작업에 중요한 문제입니다.

일부 실패 사례는 학습 데이터 셋의 분포 특성으로 인해 발생합니다. 예를 들어, 우리의 방법은 말 $\rightarrow$ 얼룩말 예시(Figure 17, 우측)에서 혼란스러워하는 모습을 보였는데, 이는 모델이 말이나 얼룩말을 타는 사람의 이미지를 포함하지 않은 ImageNet의 *wild horse*와 *zebra*에 대해 학습되었기 때문입니다.

우리는 또한 페어 데이터로 달성할 수 있는 결과와 페어를 이루지 않는 방법으로 달성된 결과 사이의 지속적인 격차를 관찰했습니다. 어떤 경우에는 이 간격을 좁히기가 매우 어렵거나 심지어 불가능할 수도 있었습니다. 예를 들어, 우리의 방법은 때때로 photo $\rightarrow$ label 의 작업의 출력에서 나무나 빌딩의 label을 바꾸었습니다. 이 모호성을 해결하려면 어떤 형태로든 약간 semantic supervision이 필요할 수도 있습니다. 약한 지도 또는 준-지도 데이터를 통합하면 훨씬 더 강력한 변환 모델이 나올 수 있으며 준-지도 데이터를 여전히 완전 지도 시스템(fully-supercised system) 비용의 일부에 불과합니다.

그럼에도 불구하고, 많은 경우에 페어를 이루지 않은 데이터는 충분히 이용 가능하며 사용되어야 합니다. 본 논문은 "감독되지 않은" 환경에서 가능한 작업의 경계를 확장합니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
색상, 질감 변화의 작업에는 적합하지만 외관 자체가 변화하는 것에는 적합하지 않다. 또한 학습한 이미지 데이터 셋에 없던 객체가 이미지에 나타난다면 모델이 혼란스러워 하며 결과가 좋지 않다.<br><br>

semantic segmentation과 같은 작업에서는 페어를 이룬 데이터를 사용한 결과를 따라잡을 수 없었다. 이를 해결하기 위해서는 준-지도 데이터 사용을 통해 가능할 것으로 생각되며 준-지도 데이터를 사용한다고 하더라도 풀-지도 데이터에 비하면 비용이 싸다.<br><br>

하지만 많은 경우에 페어를 이루지 않은 데이터를 사용해 모델이 좋은 결과를 내고 있다. 본 논문은 비 지도 환경에서 가능한 작업의 경계를 확장했다.
</font>

<br><br>

---
