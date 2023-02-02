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

---

## Abstract
Image-to-Image 변환은 비전과 그래픽스 분야의 해결 과제 중 하나로 페어 이미지를 학습해 입력 이미지와 출력 이미지 사이의 매핑을 학습하는 것입니다. 하지만 많은 과제들에서 페어 이미지로 이루어진 학습 데이터를 사용할 수 없습니다. 우리는 페어 이미지의 데이터가 없는 경우 도메인 $X$에서 도메인 $Y$로 변환하는 방법을 학습하기 위한 접근 방식을 제안합니다. 우리의 목표는 $G : X \rightarrow Y$인 매핑 $G$를 학습하는 것으로 adversarial loss를 사용한 $G(X)$의 이미지 분포가 분포 $Y$와 구별할 수 없도록 하는 것입니다. 이 매핑을 제약이 매우 낮기 때문에 우리는 역 매핑인 $F : Y \rightarrow X$를 함께 사용하며 $F(G(X)) \approx X$(또는 그 반대)가 가능하도록 하는 cycle consistency loss를 소개합니다. style transfer, object transfiguration, season transfer, photo enhancement 등 학습 시 페어 데이터를 사용하지 않는 여러 과제에 대한 좋은 결과를 제시합니다. 몇몇 방법과 비교를 통해 우리의 접근 방식에 대한 우수성을 보여줍니다.

<font color='41 69 E1'>
<b>[정리]</b><br>
기존의 GAN 논문들과 마찬가지로 adversarial loss를 사용해 도메인 $X$의 데이터를 도메인 $Y$로 변화시키는 매핑 $G : X \rightarrow Y$를 사용합니다.<br>
여기서 논문의 제목의 근간이 된 cycle consistency loss를 추가로 사용합니다. 매핑 $G$의 역 매핑인 $F$는 도메인 $Y$를 도메인 $X$로 변화시키는 $F : Y \rightarrow X$의 역할을 합니다.<br>
$F(G(X))$는 도메인 변화가 $X \rightarrow Y \rightarrow X$가 될 것이며 다시 도메인 $X$가 된 데이터는 원본 데이터와 비슷하기를 바라는 $F(G(X)) \approx X$를 cycle consistency loss로 사용합니다.
</font>

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

## 2. Related work
**Generative Adversarial Networks(GANs)**
[16, 63]은 image generation[6, 39], image editing[66], representation learning[39, 43, 37]에서 인상적인 결과를 달성했습니다. 최근의 방법은 text2image[41], image inpainting[38], future prediction[36]과 같은 조건부 이미지 생성 뿐만 아니라 비디오[54], 3D 데이터[57]와 같은 다른 도메인에 대해서도 동일한 아이디어를 채택합니다. GANs의 성공 핵심은 생성된 이미지가 원칙적으로 실제 사진과 구별할 수 없도록 하는 adversarial loss에 대한 아이디어입니다. 이 loss는 많은 컴퓨터 그래픽이 최적화하려는 목표이기 때문에 특히 이미지 생성 작업에 강력하게 작용합니다. 우리는 변환된 이미지가 대상 도메인과 구별할 수 없도록 매핑을 학습하기 위해 adversarial loss를 채택해 사용합니다.

**Image-to-Image Translation**
Image-to-Image 변환의 아이디어는


**Unpaired Image-to-Image Translation**


**Cycle Consistenc**


**Neural Style Transfer**


## 3.


## 4.


## 5.


## 6.
