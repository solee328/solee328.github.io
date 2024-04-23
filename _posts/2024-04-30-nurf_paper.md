---
layout: post
title: NeRF - 논문 리뷰
# subtitle:
categories: gan
tags: [nerf, 3D, volume rendering, scene representation, 논문 리뷰]
# sidebar: []
use_math: true
---

이번 논문은 NeRF: Representing Scenes as Nueral Radiance Fields for View Synthesis 입니다!<br>
논문 이름에서도 볼 수 있듯이 NeRF는 Neural Radiance Fields를 줄임말로 2D 이미지를 입력으로 주었을 때 마치 3D를 보는 것처럼 다른 시점에서 본 입력 이미지를 생성하는 View Synthesis task를 다룹니다.

지금부터 NeRF에 대해서 하나씩 살펴보겠습니다:eyes:
<br><br>

---

## 소개

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/b882d6ff-de14-4e37-ae94-e45369f4d3e0" width="500" height="250">
</div>
> 지점($x, y, z$)과 시각 방향($\theta, \phi$)의 시각화.<br>
$\theta$는 물체를 바라보는 시선을 $xy$ 평면에 projection했을 때 $x$ 축과 이루는 각도를 나타내고, $\phi$는 물체를 바라보는 시선과 $z$축과의 각도를 의미합니다.

NeRF는 공간의 각 지점($x, y, z$)에 대해서 시각 방향($\theta, \phi$)을 모델 입력으로 받아


 Deep Learning을 통해 계산해 각 지점의 색상($R, G, B$)과 밀도($\sigma$)로 표현하는 방식을 사용합니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/d13e3235-6535-4f6c-9442-718f232953b8" width="800" height="250">
</div>
> Figure 2. Neural Radiance Field scene representation 개요와 rendering 절차.<br>

ray 설명



NeRF는 Convolution이 아닌 Fully-connected layer를 쌓은 MLP 구조를 사용


<br><br>



---

## 모델

---


## Loss


<br><br>

---


## 실험 설정

### Dataset


### Baseline


### Result


### Ablation study


---

## 정리

<br><br>

---
