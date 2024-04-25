---
layout: post
title: NeRF - 논문 리뷰
# subtitle:
categories: gan
tags: [nerf, 3D, volume rendering, scene representation, 논문 리뷰]
# sidebar: []
use_math: true
---

```
예시 이미지
```

이번 논문은 <a href="https://arxiv.org/abs/2003.08934" target="_blank">NeRF: Representing Scenes as Nueral Radiance Fields for View Synthesis</a> 입니다!<br>
논문 이름에서도 볼 수 있듯이 NeRF는 Neural Radiance Fields를 줄임말로 2D 이미지를 입력으로 주었을 때 마치 3D를 보는 것처럼 다른 시점에서 본 입력 이미지를 생성하는 View Synthesis task를 다룹니다.

처음으로 3D 관련 논문을 보게 되었는데, 생소한 단어가 많았습니다:hushed: 단어 선택이나 해석에 문제를 보셨다면 알려주시면 감사하겠습니다!
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


```
Figure 1
```
NeRF의 목표는 입력으로 한 object를 여러 시각으로 바라본 사진을 학습시킨다면 학습되지 않은 다른 방향에서 바라본 경우를 계산할 수 있게 하는 것.


<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/d13e3235-6535-4f6c-9442-718f232953b8" width="800" height="250">
</div>
> Figure 2. Neural Radiance Field scene representation 개요와 rendering 절차.<br>

3D

ray 설명
공간 상에서 점 하나와 각도가 정해지면 직선을 정의할 수 있고, 직선 위에 점 집합을 network의 입력이 되고 출력으로 $RGB\sigma$를 계산하는 것이 목표
네트워크가 한번 돌아가면 ray 하나에 대한 값이 나온다.
이때 Volume Rendering을 사용해 계산된 Ray의 $RGB\sigma$ 점들을 하나의 픽셀로 변환한다. 하나의 픽셀로 변환한 값과 Ground Truth의 픽셀 값과의 차이를 계산해 Loss로 사용하며 Volume Rendering 계산 과정이 미분 가능하기 때문에 Back propagation을 통해 모델을 학습한다.

NeRF는 Convolution이 아닌 Fully-connected layer를 쌓은 MLP 구조를 사용


<br><br>

---

## Volume Rendering

Volume Rendering은 ~~~~을 말합니다. 여러 방식이 있지만 NeRF에서는 Ray Casting 방법을 사용합니다.
Ray Casting은 ~~~입니다.

[참고]
<a href="https://nuguziii.github.io/cg/CG-001/" target="_blank">ZZEN, [구현하기] Volume Ray Castring 개념 및 구현</a>
<a href="https://lobotomi.tistory.com/35" target="_blank">lobotomi, Volume Rendering</a>


### Sampling
stratified Sampling을 사용

$$
t_i \sim \mathcal{U} \left[ t_n + \frac{i-1}{N}(t_f - t_n), t_n + \frac{i}{N}(t_f - t_n) \right]
$$



### Volume Rendering 수식

$$
C(\mathrm{r}) = \int^{t_f} _{t_n}T(t)\sigma(\mathrm{r}(t))\mathrm{c}(\mathrm{r}(t), \mathrm{d})dt, \  \mathrm{where} \  T(t)=\exp(- \int^t _{t_n}\sigma(\mathrm{r}(s))ds).
$$

- $C(\mathrm{r})$ Volume Rendering의 결과인 픽셀 값.
- $t_n$ 근거리 경계? / $t_f$ 원거리 경계?
- $T(t)$ 누적 투과도(transmittance). 광선이 물체 일부 입자(particle)에 부딪히지 않고 이동할 확률
- $\sigma(\mathrm{x})$ 밀도(volume density).
- $r(t)$ 광선(ray). $\mathrm{r}(t) = \mathrm{o} + t\mathrm{d}$
- $\mathrm{c}$
- $\mathrm{d}$

stratified sampling을 사용한다면 연속적인 위치(position)가 가능해 연속적인 장면 표현을 나타낼 수 있습니다. Max[26]에서 논의된 quadrature rule로 $C(\mathrm{r})$을 추정한다면 식을 아래와 같이 표현할 수 있습니다.

$$
\hat{C}(\mathrm{r}) = \sum^n _{i=1}T_i(1-\exp(-\sigma_i \delta_i))c_i, \ \mathrm{where} \ T_i=\exp \left( -\sum^{i-1} _{j=1}\sigma_j \delta_i\right)
$$

$\delta_i$는 인접한 sample 사이 거리로 $\delta_i = t_{i+1} - t_i$를 의미합니다.
$\alpha_i$는 ~~~로 alpha compositing...? $\alpha_i = 1-\exp(-\sigma_i\delta_i)$...?



### Volume Rendering 효과

```
Figure 3
```

색상 $c$는 $c = RGB$이니, NeRF의 MLP Network $F _{\Theta}$의 목적을 $F _{\Theta} : (\mathrm{x}, \mathrm{d}) \rightarrow (\mathrm{c}, \sigma)$로 표현할 수 있습니다.
$c$는 지점 $\mathrm{x}$, 시각 방향 $\mathrm{d}$에 관계되며, $\sigma$는 지점 $\mathrm{x}$에 대해서만 관여를 하기 때문에 한 지점을 고정하고 바라보는 시각 방향만 바뀌었을 때에도 연속적으로 표현이 가능합니다.

예시로 Figure 3과 같은 경우를 볼 수 있습니다. 배와 바다가 표현된 하나의 scene에서 (a)와 (b)는 같은 배와 바다를 표현하고 있지만 시각 방향 $\mathrm{d}$가 다른 경우입니다.
시각 방향이 바뀌었지만 Figure 3의 (c)에서 볼 수 있듯이 연속적인 색상 변화를 볼 수 있습니다?


<br><br>

---

## More Trick


### Positional encoding

### Hierarchical volume sampling

---

## Loss & Model

### Loss

$$
\mathcal{L} = \sum _{\mathrm{r}\in \mathcal{R}}\left[ \|\hat{C} _c(\mathrm{r}) - C(\mathrm{r})\|^2_2 + \|\hat{C} _f(\mathrm{r}) - C(\mathrm{r})\|^2_2 \right]
$$

### Model

---





## 실험 설정

### Dataset

- Diffuse Synthetic 360$^{\circ}$

### Baseline


### Result


### Ablation study


---

## 정리

<br><br>

---
