---
layout: post
title: NeRF - 논문 리뷰
# subtitle:
categories: gan
tags: [nerf, 3D, volume rendering, scene representation, 논문 리뷰]
# sidebar: []
use_math: true
---


이번 논문은 <a href="https://arxiv.org/abs/2003.08934" target="_blank">NeRF: Representing Scenes as Nueral Radiance Fields for View Synthesis</a> 입니다!<br>
논문 이름에서도 볼 수 있듯이 NeRF는 Neural Radiance Fields를 줄임말로 2D 이미지를 입력으로 주었을 때 마치 3D를 보는 것처럼 다른 시점에서 본 입력 이미지를 생성하는 View Synthesis task를 다룹니다.


처음으로 3D 관련 논문을 보게 되었는데, 생소한 단어가 많았습니다:hushed: 정의하는 부분을 제 나름대로 정리했는데 잘못된 단어 선택이나 해석에 문제를 보신다면 알려주시면 감사하겠습니다!
지금부터 NeRF에 대해서 하나씩 살펴보겠습니다:eyes:
<br><br>

---

## 소개

<div>
  <iframe src="http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/website_renders/synth_grid_3.mp4" width="700" height="393" frameboarder="0" allowfullscreen allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>
</div>
> NeRF 공식 <a href="https://www.matthewtancik.com/nerf" target="_blank">프로젝트 페이지</a>의 영상

NeRF에서 제시한 결과 영상들입니다. 여러 Object들을 360$^\circ$ 회전시키며 마치 3D Object를 보는 것 같지 않나요?<br>

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/a1249dbe-cdfb-46bc-a3fe-1bc341500646" width="800" height="200">
</div>
> Figure 1.


NeRF는 하나의 2D object에 대해 여러 방향에서 찍힌 이미지들을 학습해 학습하지 않은 방향에 대해서 object 모습을 예측합니다.
NeRF의 목표는 입력으로 한 object를 여러 시각으로 바라본 사진을 학습시킨다면 학습되지 않은 다른 방향에서 바라본 경우를 계산할 수 있게 하는 것. 3D Object를 생성하는 것이 아니라 object를 어떤 방향에서 바라보는 장면을 생성하는 View Synthesis 기술입니다. 연속된 방향 변화에 대해서 생성할 수 있기 때문에 자연스러운 Rendering이 가능하다는 것이 장점.


<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/d13e3235-6535-4f6c-9442-718f232953b8" width="800" height="250">
</div>
> Figure 2. Neural Radiance Field scene representation 개요와 rendering 절차.<br>

NeRF의 동작 방식을 나타낸 Figure입니다.
학습 데이터(2D image)에서 임의의 지점에 ray를 쏩니다. 여기서 ray는 화면 상에 투영되는 이미지를 생성하기 위해 object와 상호 작용되는 빛을 의미합니다. Figure 2의 (a), (b)에서 빨간 선 하나로 표현되어 있는 것이 ray입니다.


<br>

### ray
공간 상에서 점 하나와 각도가 정해지면 직선을 정의할 수 있고, 직선 위에 점 집합을 network의 입력이 되고 출력으로 $RGB\sigma$를 계산하는 것이 목표
네트워크가 한번 돌아가면 ray 하나에 대한 값이 나온다.
이때 Volume Rendering을 사용해 계산된 Ray의 $RGB\sigma$ 점들을 하나의 픽셀로 변환한다. 하나의 픽셀로 변환한 값과 Ground Truth의 픽셀 값과의 차이를 계산해 Loss로 사용하며 Volume Rendering 계산 과정이 미분 가능하기 때문에 Back propagation을 통해 모델을 학습한다.

### 5D Input

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/b882d6ff-de14-4e37-ae94-e45369f4d3e0" width="500" height="250">
</div>
> 지점($x, y, z$)과 시각 방향($\theta, \phi$)의 시각화.<br>
$\theta$는 물체를 바라보는 시선을 $xy$ 평면에 projection했을 때 $x$ 축과 이루는 각도를 나타내고, $\phi$는 물체를 바라보는 시선과 $z$축과의 각도를 의미합니다.

NeRF는 공간의 각 지점($x, y, z$)에 대해서 시각 방향($\theta, \phi$)을 모델 입력으로 받아

 Deep Learning을 통해 계산해 각 지점의 색상($R, G, B$)과 밀도($\sigma$)로 표현하는 방식을 사용합니다.

<br><br>

---

## Volume Rendering

Volume Rendering은 그래픽스 분야에서 3D 데이터를 2D 투시로 보여주는 시각화 기술을 말합니다.
여러 방식이 있지만 NeRF에서는 Ray Casting 방법을 사용합니다.
Ray Casting은 ~~~입니다.

[참고]
- ZZEN님의 <a href="https://nuguziii.github.io/cg/CG-001/" target="_blank">[구현하기] Volume Ray Casting 개념 및 구현</a>
- lobotomi님의 <a href="https://lobotomi.tistory.com/35" target="_blank">Volume Rendering</a>


### Sampling
stratified Sampling을 사용
dicrete하지 않고 continuous sampling을 위함?


$$
t_i \sim \mathcal{U} \left[ t_n + \frac{i-1}{N}(t_f - t_n), t_n + \frac{i}{N}(t_f - t_n) \right]
$$

n개의 bin으로 나누고 bin 안에서 1개의 점을 random하게 샘플링함.
학습할 때마다 sampling되는 point들이 조금씩 달라지기 떄문에 continuous에 대해 학습이 가능함.


### Volume Rendering 수식

$$
C(\mathrm{r}) = \int^{t_f} _{t_n}T(t)\sigma(\mathrm{r}(t))\mathrm{c}(\mathrm{r}(t), \mathrm{d})dt, \  \mathrm{where} \  T(t)=\exp(- \int^t _{t_n}\sigma(\mathrm{r}(s))ds).
$$

- $C(\mathrm{r})$ : Volume Rendering의 결과인 픽셀 값.
- $t_n$ : ray 시작점 / $t_f$ : ray 끝점
- $t$ : 목표 점?
- $T(t)$ : 누적 투과도(transmittance). 광선이 물체 일부 입자(particle)에 부딪히지 않고 이동할 확률
- $\sigma(\mathrm{x})$ : 밀도(volume density).
- $r(t)$ : 광선(ray). $\mathrm{r}(t) = \mathrm{o} + t\mathrm{d}$
- $\mathrm{c}$ : RGB
- $\mathrm{d}$ :

stratified sampling을 사용한다면 연속적인 위치(position)가 가능해 연속적인 장면 표현을 나타낼 수 있습니다. Max[26]에서 논의된 quadrature rule로 $C(\mathrm{r})$을 추정한다면 식을 아래와 같이 표현할 수 있습니다.

$$
\hat{C}(\mathrm{r}) = \sum^n _{i=1}T_i(1-\exp(-\sigma_i \delta_i))c_i, \ \mathrm{where} \ T_i=\exp \left( -\sum^{i-1} _{j=1}\sigma_j \delta_i\right)
$$

$\delta_i$는 인접한 sample 사이 거리로 $\delta_i = t_{i+1} - t_i$를 의미합니다.
$\alpha_i$는 ~~~로 alpha compositing...? $\alpha_i = 1-\exp(-\sigma_i\delta_i)$...?



### Volume Rendering 효과

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/b3ba9edf-93bc-4bca-80e0-75bd2a62c738" width="800" height="250">
</div>
> Figure 3.

색상 $c$는 $c = RGB$이니, NeRF의 MLP Network $F _{\Theta}$의 목적을 $F _{\Theta} : (\mathrm{x}, \mathrm{d}) \rightarrow (\mathrm{c}, \sigma)$로 표현할 수 있습니다.
$c$는 지점 $\mathrm{x}$, 시각 방향 $\mathrm{d}$에 관계되며, $\sigma$는 지점 $\mathrm{x}$에 대해서만 관여를 하기 때문에 한 지점을 고정하고 바라보는 시각 방향만 바뀌었을 때에도 연속적으로 표현이 가능합니다.

예시로 Figure 3과 같은 경우를 볼 수 있습니다. 배와 바다가 표현된 하나의 scene에서 (a)와 (b)는 같은 배와 바다를 표현하고 있지만 시각 방향 $\mathrm{d}$가 다른 경우입니다.
시각 방향이 바뀌었지만 Figure 3의 (c)에서 볼 수 있듯이 연속적인 색상 변화를 볼 수 있습니다?


<br><br>

---

## More Trick


### Positional encoding

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/a1766a96-0591-493d-a0c1-bea3a4d8d0f5" width="800" height="250">
</div>
> Figure 4.

high frequency 영역까지 표현할 수 있도록 positional encoding을 사용한다.

$$
\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \cdots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))
$$

하나의 변수($p$)를 사용해 정보를 더 늘리기 쉬워짐.
L이 10이라면 sin, cos이 10개씩 생겨 20개가 됨. -> location 값 x, y, z를 $p$에 넣어 $\gamma$를 계산하면 20개씩 총 60개 차원이 완성됨.
density $\sigma$는 location에만 의존하는 값이므로 먼저 추출, 추가로 direction에 대한 positional encoding 값을 추가로 넣어 RGB 값을 추출함.


### Hierarchical volume sampling

$$
\hat{C} _c(\mathrm{r}) = \sum ^{N_c} _{i=1}w_ic_i, \quad w_i=T_i(1-\exp(-\sigma_i \delta_i))
$$

학습을 2번 한다. coarse하게 1번, fine하게 1번
최종 결과물은 coarse 때 sampling한 $N_c$와 fine 때 샘플링한 $N_f$를 합한 $N_c + N_f$ 샘플을 사용함.

<br>

---

## Loss & Model

### Loss

$$
\mathcal{L} = \sum _{\mathrm{r}\in \mathcal{R}}\left[ \|\hat{C} _c(\mathrm{r}) - C(\mathrm{r})\|^2_2 + \|\hat{C} _f(\mathrm{r}) - C(\mathrm{r})\|^2_2 \right]
$$

sampling camera ray -> hierarchical sampling -> volume rendering -> computing loss의 단계를 반복하며 모델을 학습시킨다.
Coarse, fine을 MSE loss한 모습.
loss는 굉장히 단순하나 100-300k iteration에 V100 GPU를 사용해도 1~2일 정도가 소요된다.

### Model




---





## 실험 설정

### Dataset

- Diffuse Synthetic 360$^{\circ}$
- Diffuse Real 360$^{\circ}$
- Diffuse LLFF

### Baseline

- Neural Volumes(NV)
- Scene Representation Networks(SRN)
- Local Light Field Fusion(LLFF)

### Result

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/296df67f-d0ff-4580-a2cb-eee3e92a2668" width="800" height="1000">
</div>
> Figure 5.

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/f3415fed-ec3f-4a5e-9c8f-d9bf690daf7d" width="800" height="880">
</div>
> Figure 6.

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/8ab76847-2696-449c-94c4-dc7936b7f783" width="800" height="150">
</div>
> Table 1.


### Ablation study

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/f9d24bce-f090-4d8e-af9e-f41dcab8fe9d" width="800" height="300">
</div>
> Table 2.

<br>

---

## 정리

<br><br>

---
