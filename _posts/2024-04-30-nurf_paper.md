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
논문 이름에서도 볼 수 있듯이 NeRF는 Neural Radiance Fields를 줄임말로 2D 이미지를 입력으로 주었을 때 마치 3D를 보는 것처럼 다른 시점에서 본 입력 이미지를 생성하는 View Synthesis task를 다룹니다. 모델은 MLP 형식으로 단순한 구조로 3D scene representation을 성공한 것으로도 유명합니다.

처음으로 3D 관련 논문을 보게 되었는데, 생소한 단어가 많았습니다:hushed: 정의하는 부분을 제 나름대로 정리했는데 잘못된 단어 선택이나 해석에 문제를 보신다면 알려주시면 감사하겠습니다!
지금부터 NeRF에 대해서 하나씩 살펴보겠습니다:eyes:
<br><br>

---

## 소개

<div>
  <iframe src="http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/website_renders/synth_grid_3.mp4" width="700" height="393" frameboarder="0" allowfullscreen allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"></iframe>
</div>
> NeRF 공식 <a href="https://www.matthewtancik.com/nerf" target="_blank">프로젝트 페이지</a>의 영상

NeRF에서 제시한 결과 영상들입니다.여러 Object들을 360$^\circ$ 회전시키며 보여주는 결과가 마치 3D Object Asset를 보는 것 같지 않나요?<br>

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/a1249dbe-cdfb-46bc-a3fe-1bc341500646" width="800" height="200">
</div>
> Figure 1. 입력 이미지들로부터 연속적인 5D neural radiance field representation scene을 최적화하는 방법.<br>
volume rendering을 사용해 ray를 따라 scene의 sample을 축적해 모든 시점에서의 scene을 렌더링합니다. 반구에서 무작위로 포착한 드럼 이미지 100개를 시각화하고 최적화된 NeRF에서 렌더링한 새로운 뷰를 보여줍니다.

NeRF는 하나의 2D object에 대해 여러 방향에서 찍힌 이미지들을 학습해 학습하지 않은 방향에 대해서 object 모습을 예측합니다. 3D Object를 생성하는 것이 아니라 object를 새로운 방향에서 바라보는 장면을 생성하는 View Synthesis 기술입니다.

Figure 1과 프로젝트 공식 영상에서 Drum에 대한 입력 이미지로 NeRF를 학습해 최적화한 이후에는 학습하지 않았던 새로운 방향에서 본 이미지를 생성한 것을 볼 수 있습니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/d13e3235-6535-4f6c-9442-718f232953b8" width="800" height="250">
</div>
> Figure 2. Neural Radiance Field scene representation 개요와 rendering 절차.<br>
(a) ray를 따라 5D 좌표(위치 + 시각 방향)를 샘플링해 영상을 합성<br>
(b) MLP에 좌표를 통과시켜 색상(RGB)와 밀도(density)를 생성<br>
(c) volume rendering 기술을 이용해 생성한 값을 이미지로 합성<br>
(d) 합성된 영상과 실제 학습 데이터 사이의 차이를 최소화해 scene representation를 최적화

NeRF의 동작 방식을 Figure 2에서 볼 수 있습니다. Figure 2에서 표현되어 있는 ray와 입력 데이터 $x, y, z, \theta, \phi$에 대해 간단하게 짚고 넘어가겠습니다!

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/4809b5ca-3500-4e46-ab75-84846a528a1e" width="800" height="250">
</div>
> <a href="https://arxiv.org/abs/2103.13415" target="_blank">Mip-NeRF</a>의 Figure 1.

ray는 화면 상에 투영되는 이미지를 생성하기 위해 object와 상호 작용되는 빛을 의미합니다. $\mathrm{r}(t) = \mathrm{o} + t\mathrm{d}$로 표현하며, 여기서 $\mathrm{o}$는 원점(카메라), $\mathrm{d}$는 시각 방향, $t$는 샘플링되는 지점(원점에서 시각 방향으로 특정 거리만큼 이동)을 의미합니다.

현실에서는 ray가 object가 닿는 순간 어떤 방향으로 반사되는 것까지 계산을 해야하지만, NeRF는 ray가 object가 부딪힐 때 해당 object에 닿는 빛의 양을 추측하는 형태의 radiance field rendering 방식을 사용합니다. radiance field에서 하나의 object는 수많은 작은 입자로 이루어져 있어, 공간의 특정 지점에서 ray가 object와 충돌하는 것은 입자(particle)와 충돌이 발생할 확률로 근사화됩니다. 모든 광선은 입자가 닿을 때까지 field를 통과하며 최종적으로 종료되었을 시 해당 particle에서 카메라를 향새 반사되는 색을 반환합니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/b882d6ff-de14-4e37-ae94-e45369f4d3e0" width="400" height="200">
</div>
> 지점($x, y, z$)과 시각 방향($\theta, \phi$)의 시각화.

NeRF는 ray를 정의하기 위해 지점($x, y, z$, position)과 시각 방향($\theta, \phi$, direction)을 사용합니다. $\theta$는 물체를 바라보는 시선을 $xy$ 평면에 projection했을 때 $x$ 축과 이루는 각도를 나타내고, $\phi$는 물체를 바라보는 시선과 $z$축과의 각도를 의미합니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/53cf4e6a-b869-47c4-ad53-3f93530695c8" width="400" height="390">
</div>
> Figure 2.(b) MLP에 좌표를 통과시켜 색상(RGB)와 밀도(density)를 생성

NeRF은 하나의 Ray를 정의한 5D Input을 입력으로 받아 해당 Ray를 RGB$\sigma$ 표현한 것을 출력합니다. Figure 2.(b)에서 모델 $F_{\Theta}$는 위치 $\mathrm{x}=(x, y, z)$와 시각 방향 $\mathrm{d}=(\theta, \phi)$를 입력받아 색상 $\mathrm{c}=$RGB와 밀도 $\sigma$를 출력하는 $F_{\Theta} : (\mathrm{x}, \mathrm{d}) \rightarrow (\mathrm{c}, \sigma)$로 표현할 수 있습니다.


$RGB$는 색상 값, $\sigma$는 밀도를 의미하며 radiance field에서 particle이 얼마나 object에 밀집되어 있는지를 표현합니다. 밀도 $\sigma$는 투명도의 역수를 의미하며 밀도가 높은 object는 철과 같은 고체로 선명하게 픽셀에 표현됩니다. 반대로 밀도가 낮은 object는 물과 같은 액체 또는 유리와 같이 투명도가 높은 고체가 해당하며 뒤에 있는 물체까지 픽셀에 표현되어야 하기 때문에 상대적으로 픽셀에 흐릿하게 표현됩니다.

radiance field에서 particle과의 충돌은 확률적이기 때문에 ray를 한 방향으로 반복해서 쏘게 된다면 매번 다른 색상이 나타납니다. 따라서 이미지 한 픽셀 당 하나의 ray만 쏴서 색을 찾는 것이 아니라 특정 방향에서 발사된 모든 가능한 예상 색을 계산합니다. Volume Rendering 기술을 사용해 색상을 계산하며 아래에서 더 자세하게 살펴보겠습니다.

<br>

---

## Volume Rendering

Volume Rendering은 그래픽스 분야에서 3D 데이터를 2D 투시로 보여주는 시각화 기술을 말합니다.<br>
NeRF는 volume rendering을 하기 위해 radiance field 환경의 ray를 사용하는데, ray에 어떤 지점을 가져와 rendering에 사용할 지 정해야하며 이때 Stratified Sampling을 사용해 지점들을 sampling합니다.

### Stratified Sampling

$$
t_i \sim \mathcal{U} \left[ t_n + \frac{i-1}{N}(t_f - t_n), t_n + \frac{i}{N}(t_f - t_n) \right]
$$
> $N$ : bin 개수<br>
$t_n$ : ray sampling 시작 위치<br>
$t_f$ : ray sampling 끝 위치

ray에서 sampling이 가능한 시작 위치를 $t_n$, 끝 위치를 $t_f$로 두었을 때, 두 값을 뺀 값($t_f - t_n$)을 bin 개수인 $N$으로 나누어 구간을 정합니다(논문에서는 $N=64$ 사용). 이 구간에서 weight 지정 없이 1개의 지점을 random하게 sampling하기 때문에 uniform sampling을 하게 됩니다. uniform sampling이기 때문에 sampling 할 때마다 sampling되는 지점이 계속해서 달라지게 되고 이로 인해 ray에 대해 continuous한 학습이 가능하다고 합니다.

### Volume Rendering 수식

$$
C(\mathrm{r}) = \int^{t_f} _{t_n}T(t)\sigma(\mathrm{r}(t))\mathrm{c}(\mathrm{r}(t), \mathrm{d})dt, \  \mathrm{where} \  T(t)=\exp(- \int^t _{t_n}\sigma(\mathrm{r}(s))ds).
$$
> $C(\mathrm{r})$ : ray $\mathrm{r}$의 volume rendering 예상 색상 값<br>
$t$ : 현재 sampling된 위치<br>
$T(t)$ : 누적 투과도(transmittance)로 ray $\mathrm{r}$가 물체 일부 입자(particle)에 부딪히지 않고 $t$까지 이동할 확률<br>
$\sigma(\mathrm{r}(t))$ : 밀도로 ray $\mathrm{r}$이 위치 $t$에서 입자와 충돌할 확률<br>
$\mathrm{d}$ : 시각 방향($\theta, \phi$)<br>
$\mathrm{c}(\mathrm{r}(t), \mathrm{d})$ : 방향 $\mathrm{d}$인 ray $\mathrm{r}$의 위치 $t$에서의 색상 값<br>

$T(t)$가 ray가 위치 $t$까지 진행하며 다른 입자에 부딪히지 않을 확률이고 $\sigma(\mathrm{r}(t))$가 위치 $t$에서 입자와 ray가 충돌할 확률이니 $T(t)\sigma(\mathrm{r}(t))$는 ray가 위치 $t$에서 입자와 충돌해 ray가 종료될 확률을 의미합니다.

$\mathrm{c}(\mathrm{r}(t), \mathrm{d})$는 방향 $\mathrm{d}$인 ray $\mathrm{r}$의 위치 $t$에서의 색상 값을 의미합니다.


stratified sampling을 사용한다면 연속적인 위치(position)가 가능해 연속적인 장면 표현을 나타낼 수 있어 위 수식과 같이 적분을 사용할 수 있습니다. <a href="https://ieeexplore.ieee.org/abstract/document/468400" target="_blank">Optical models for direct volume rendering</a>에서 논의된 quadrature rule을 사용해 유한한 수의 구간(bin) 합으로 $C(\mathrm{r})$을 추정하는 방법을 사용해 위의 식을 아래와 같이 discrete하게 변형했습니다.

<br>

$$
\hat{C}(\mathrm{r}) = \sum^n _{i=1}T_i(1-\exp(-\sigma_i \delta_i))c_i, \ \mathrm{where} \ T_i=\exp \left( -\sum^{i-1} _{j=1}\sigma_j \delta_i\right)
$$
> $i$ : 구간(bins) 순서<br>
$T_i$ : 이전 bins에 부딪히지 않고 현재 bins까지 이동할 확률<br>
$\sigma_i$ : i번째 bin의 밀도(=$\sigma(\mathrm{r}(t_i))$)<br>
$\delta_i$ : i번째 sample과 i+1번째 sample의 거리(=t_{i+1} - t_i)<br>
$c_i$ : i번째 bin을 나타내는 색상(=$\mathrm{c}(\mathrm{r}(t_i))$)<br>

$1-\exp(-\sigma_i \delta_i)$은 ray가 이전 충돌과 무관하게 i번째 bin에서 충돌할 확률을 의미합니다.

이후 alpha compositing(알파 합성)으로 $\alpha_i = 1-\exp(-\sigma_i\delta_i)$을 계산한다....?

<br>

---

## More Trick
복잡한 scene에 대해서 높은 해상도 표현이 가능하도록 NeRF를 최적화하기 위해 2가지 기법을 추가적으로 사용합니다.

MLP가 higher frequency function을 표현할 수 있도록 Positional encoding을 사용해 입력 5D 좌표를 변환합니다. 또한 high-frequency scene representation을 적절하게 샘플링하는 데 필요한 쿼리 수를 줄이기 위해 hierarchical sampling을 사용합니다.

### Positional encoding

NeRF 모델 입력인 $xyz\theta\phi$을 사용할 시, 모델이 lower frequency function을 학습하는 것에 편향되어 있다는 것을 발견함.
[35]의 연구 또한 마찬가지였는데, [35]의 경우 high frequency function을 사용해 입력 값을 더 높은 고차원 공간에 매핑하면 high frequency variation이 포함된 데이터를 더 잘 맞출 수 있음을 보여주었습니다.

따라서 NeRF 또한 같은 이유로 입력을 고차원 공간으로 매핑하기 위한 $\gamma$를 사용해 모델 $F_{\Theta}$을 $F_\Theta = F_\Theta' \circ \gamma$로 변화시킵니다.

<br>

$$
\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \cdots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))
$$

하나의 변수($p$)를 사용해 정보를 더 늘리기 쉬워짐.

NeRF에서 $\gamma(\mathrm{d})$에는 $L=4$, $\gamma(\mathrm{x})$에는 $L=10$을 적용했습니다.
L이 10이라면 sin, cos이 10개씩 생겨 20개가 됨. -> location 값 x, y, z를 $p$에 넣어 $\gamma$를 계산하면 20개씩 총 60개 차원이 완성됨.
density $\sigma$는 location에만 의존하는 값이므로 먼저 추출, 추가로 direction에 대한 positional encoding 값을 추가로 넣어 RGB 값을 추출함.


<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/a1766a96-0591-493d-a0c1-bea3a4d8d0f5" width="800" height="250">
</div>
> Figure 4. view dependence와 positional encoding에 따른 결과 시각화.

positional encoding을 사용한 경우에 대한 결과 차이를 볼 수 있습니다.


### Hierarchical volume sampling

ray sampling에서 결과 퀄리티를 높이고 싶으면 더 많은 sampling을 진행하는 것이지만, 렌더링된 이미지에 기여하지 않는 물체가 없는 부분(free space), 가려진 부분(occluded region)이 반복적으로 샘플링될 수 있어 비효율적입니다.

NeRF는 volume redering 초기 연구[20]에서 영감을 받아 효율적인 샘플링 방식인 hierarchical representation을 제안합니다. 샘플링을 2번에 나눠서하는 방식으로 첫번째를 coarse network, 두번째를 fine network라 합니다.

coarse network는 stratified sampling을 사용해 $N_c$를 샘플링합니다. coarse network인 $N_c$로 $C(\mathrm{r})$를 생성합니다.

$$
\hat{C} _c(\mathrm{r}) = \sum ^{N_c} _{i=1}w_ic_i, \quad w_i=T_i(1-\exp(-\sigma_i \delta_i))
$$

이 가중치를 $\hat{w}_i = w_i / \sum ^{N_c} _{j=1}w_j$로 normalize하면 piecewise-constant PDF가 생성됩니다. 역변환 샘플링(inverse transform sampling)을 사용해 이 분포에서 두번째 샘플링 $N_f$을 진행합니다.

```
설명해 추가할 그림
```

최종적으로 모든 샘플링인 $N_c + N_f$이 fine network가 되며 이를 사용해 $\hat{C} ^f(r)$을 계산합니다. 이를 통해 free space, occluded region이 아닌 object가 포함될 것으로 예상되는 영역에서 더 많은 샘플링이 가능해 효율적인 샘플링이 가능합니다.

<br>

---

## Loss & Model

### Loss

$$
\mathcal{L} = \sum _{\mathrm{r}\in \mathcal{R}}\left[ \|\hat{C} _c(\mathrm{r}) - C(\mathrm{r})\|^2_2 + \|\hat{C} _f(\mathrm{r}) - C(\mathrm{r})\|^2_2 \right]
$$

- $\mathcal{R}$ : 각 batch에 ray set
- $C(\mathrm{r})$ : ray $\mathrm{r}$에 대한 GT
- $\hat{C} _c(\mathrm{r})$ : coarse volume 예측 값
- $\hat{C} _f(\mathrm{r})$ : fine volume 예측 값

sampling camera ray -> hierarchical sampling -> volume rendering -> computing loss의 단계를 반복하며 모델을 학습시킨다.
Coarse, fine을 MSE loss한 모습.

NeRF에서는 $\mathcal{R}$=4096, $N_c$=64, $N_f$=128를 사용했다고 합니다.
loss는 굉장히 단순하나 100-300k iteration에 V100 GPU를 사용해도 1~2일 정도가 소요된다.

<br>

### Model
<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/9af7f9ae-58e2-4073-8e1d-d586d2f585a2" width="800" height="360">
</div>
> Figure 7. fully-connected network 구조 시각화.

Figure 7에서 NeRF의 모델 구조를 볼 수 있습니다. 모델에 입력되는 값은 녹색, 출력되는 값은 빨간색으로 표시되어 있으며 검은 실선 화살표는 ReLU activation, 주황색 실선 화살표는 activation이 없고 검은색 점선 화살표는 Sigmoid activation이 있는 layer입니다.

위치 $\mathrm{x}$로만 밀도 $\sigma$를 예측하도록 모델을 제한하기 위해서 밀도 $\sigma$는 중간 layer에서 출력되며, 색상 $\mathrm{c}$는 위치 $\mathrm{x}$와 시각 방향 $\mathrm{d}$를 모두 활용해 예측하도록 합니다. 색상은 물체를 바라보는 방향에 따라 영향을 받으며 밀도는 시각 방향에 영향을 받지 않기 때문에 밀도와 관련된 물체의 존재 여부와 물성 또한 영향을 받지 않습니다.

NeRF는 DeepSDF[32] 구조를 따르며 5번째 layer의 activation과 $\gamma(\mathrm{x})$이 concat되는 skip connection이 있습니다. 중간 출력인 밀도 $\sigma$는 음수가 될 수 없으므로 ReLU가 추가적으로 사용된 후 출력되며, 시각 방향 $\gamma(\mathrm{d})$이 9번째 layer에서 추가로 입력됩니다. 마지막 layer의 ~~~ 이후 색상 RGB 값이 출력됩니다.


<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/b3ba9edf-93bc-4bca-80e0-75bd2a62c738" width="800" height="210">
</div>
> Figure 3. view-dependent radiance 시각화.

색상 $c$는 $c = RGB$이니, NeRF의 MLP Network $F _{\Theta}$의 목적을 $F _{\Theta} : (\mathrm{x}, \mathrm{d}) \rightarrow (\mathrm{c}, \sigma)$로 표현할 수 있습니다.
$c$는 지점 $\mathrm{x}$, 시각 방향 $\mathrm{d}$에 관계되며, $\sigma$는 지점 $\mathrm{x}$에 대해서만 관여를 하기 때문에 한 지점을 고정하고 바라보는 시각 방향만 바뀌었을 때에도 연속적으로 표현이 가능합니다.

예시로 Figure 3과 같은 경우를 볼 수 있습니다. 배와 바다가 표현된 하나의 scene에서 (a)와 (b)는 같은 배와 바다를 표현하고 있지만 시각 방향 $\mathrm{d}$가 다른 경우입니다.
시각 방향이 바뀌었지만 Figure 3의 (c)에서 볼 수 있듯이 연속적인 색상 변화를 볼 수 있습니다. <a href="https://ko.wikipedia.org/wiki/%EB%9E%8C%EB%B2%A0%EB%A5%B4%ED%8A%B8_%EB%B0%98%EC%82%AC" target="_blank">Lambertian effect</a>가 적용되었다면, 바라보는 방향이 바뀌었더라도 같은 색상으로 보여야하지만, NeRF의 색상 $\mathrm{c}$는 지점 $\mathrm{x}$과 시각 방향 $\mathrm{d}$에 의존하기 때문에 색상이 바뀌었으며 non-Lambertian임을 나타냅니다.
<br><br>

---

## 실험 설정

### Dataset
NeRF를 학습하기 위해서는 object가 다양한 각도에서 촬영된 이미지와 함께 카메라의 pose, intrinsic parameter, scene bounds에 대한 정보가 필요합니다.

NeRF는 실제 데이터에 대한 이런 parameter를 추정하기 위해 COLMAP structure-from-motion package[39]를 사용했다고 합니다.

- Diffuse Synthetic 360$^{\circ}$<br>
  Diffuse Synthetic 360$^{\circ}$는 DeepVoxels[41]데이터로 간단한 구조를 가진 Lambertian 객체 4개가 포함되어 있습니다. 각 객체에는 반구형(hemisphere)에서 시점이 샘플링되며 512x512 픽셀을 가집니다. 객체 하나(scene)의 전체 이미지에서 학습으로 479, 테스트로 100개를 사용합니다.

- Realistic Synthetic 360$^{\circ}$<br>
  복잡한 구조를 가진 non-Lambertian 객체 8개를 NeRF 팀에서 만들어 사용한 데이터셋입니다. 객체 8개 중 6개 객체는 반구형(hemisphere), 2개는 전체 (full sphere) 시점에서 샘플링되며 800x800 픽셀을 가집니다. 객체 하나(scene)의 전체 이미지에서 학습으로 100, 테스트로 200개를 사용합니다.

- Real Forward-Facing<br>
  현실의 복잡한 scene 데이터로 핸드폰으로 촬영된 데이터셋입니다. LLFF[28]에서 만든 5개와 NeRF 팀에서 만든 3개 scene이 있으며 1008x756 픽셀을 가집니다. scene 당 20~62개의 이미지가 있으며 이중 1/8을 테스트로 사용합니다.

### Baseline

NeRF와 비교할 Baseline 모델들입니다. Local Light Field Fusion을 제외한 방법은 scene에 대해 별도의 network를 학습한 다음 test time에 새로운 scene의 입력 이미지를 처리하는 방식입니다.

Baseline들의 가장 큰 tradeoff는 시간(time)과 공간(space)로 scene 하나에 하나의 모델을 사용하는 NV와 SRN은 scene 하나를 학습하는데 최소 12시간이 걸립니다. 대조적으로 LLFF는 작은 입력 데이터 셋을 10분 이내에 처리할 수 있습니다. 하지만 LLFF는 모든 입력 이미지에 대해 거대한 3D voxel grid를 생성하기 때문에 엄청난 저장 공간을 요구합니다. 하나의 Realistic Synthetic scene에 대해서 15GB이 필요합니다.

- <a href="https://github.com/facebookresearch/neuralvolumes" target="_blank">Neural Volumes(NV)</a><br>
  Neural Volumes(NV)[24]는 deep 3D convolutional network로 1283개의 샘플을 이산화된 RGB$\alpha$ voxel grid와 323개의 3D warp gird 샘플에 대해 예측합니다. 알고리즘은 warped voxel grid를 통해 카메라 광선을 행진하며 새로운 view를 만듭니다.

- <a href="https://github.com/vsitzmann/scene-representation-ne" target="_blank">Scene Representation Networks(SRN)</a><br>
  Scene Representation Networks(SRN)[42]은 연속적인 scene을 불투명한 표면(opaque surface)로 표현하며, recurrent neural network로 임의의 3D 좌 $(x, y, z)$에서 feature vector를 사용해 다음 step size를 예측합니다. SRN은 DeepVoxels[41]과 동일 저자로 더 나은 성능을 가지는 후속연구이므로 NeRF에서 DeepVoxels를 baseline에 포함하지 않았습니다.

- <a href="https://github.com/Fyusion/LLFF" target="_blank">Local Light Field Fusion(LLFF)</a>.<br>
  Local Light Field Fusion(LLFF)[28]는 3D convolutional network로 입력에 대해 이산화돤 frustum sampling RGB$\alpha$ grid(multiplane image 또는 MPI[52])를 예측한 다음, 근처 MPIs를 새로운 view로 alpha 합성하고 혼합해 새로운 view를 만듭니다.

### Result

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/8ab76847-2696-449c-94c4-dc7936b7f783" width="800" height="150">
</div>
> Table 1. NeRF와 baseline들의 PSNR/SSIM/LPIPS 수치.

NV는 제한된 volume 내의 객체만 구성하기 때문에 Real Forward-Facing 데이터에서 평가할 수 없어 제외되었습니다.

scene 당 별도의 network를 최적화하는 NV와 SRN에 대해서는 모든 수치가 NeRF가 다른 두 모델을 능가함을 볼 수 있습니다.
LLFF는 Real Forward-Facing 데이터셋 결과에서 LPIPS 수치가 NeRF보다 더 좋지만 NeRF의 저자들은 NeRF의 결과가 더 나은 일관성을 가지고 있으며 더 적은 artifact를 생성한다고 언급합니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/296df67f-d0ff-4580-a2cb-eee3e92a2668" width="800" height="1000">
</div>
> Figure 5. Realistic synthetic 데이터셋에 대한 test set 결과 비교.

NeRF는 $Ship$'s rigging, $Lego$'s gear와 treads, $Microphone$'s shiny stand과 mesh grille, $Material$'s non Lambertian reflectance 모두에서 디테일한 점까지 만들어냄을 볼 수 있습니다.

LLFF의 경우 $Microphone$'s stand와 $Material$'s object edges에 banding artifact, $Ship$'s mast와 $Lego$ 내부에 ghosting artifact를 생성했습니다. LLFF는 입력 view 간 간격(disparity)가 64 픽셀을 넘지 않도록 "sampling guideline"을 제공하기 때문에 view 간격이 최대 400~500 픽셀인 synthetic 데이터셋에서 정확한 기하학적 구조를 추정하지 못하는 경우가 많았다고 합니다.

SRN은 모든 경우에 흐릿하고 왜곡된 렌더링을 생성했습니다. SRN은 심하게 smooth한 질감을 생성하며 하나의 ray에서 단일 depth와 color만 선택하기 때문에 view synthesis를 위한 표현력이 제한됩니다.

Neural Volumes는 $Microphone$'s grille 또는 $Lego$'s gear의 디테일을 생성할 수 없으며, $Ship$'s rigging에 대한 기하학적 구조를 완전히 생성하지 못했습니다. 1283개의 voxel grid를 사용하는 것이 명시되어 있어 고해상도에서 자세한 디테일을 나타내기 위해 확장할 수 없는 것이 문제가 됩니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/f3415fed-ec3f-4a5e-9c8f-d9bf690daf7d" width="800" height="880">
</div>
> Figure 6. real world scene 데이터셋에 대한 test set 결과 비교

LLFF는 실제 환경을 정면으로 촬영한 이 데이터셋을 위해 특별히 설계된 모델입니다.

NeRF는 $Fern$의 잎 부분과 $T-rex$의 skeleton rib와 railing에서 LLFF보다 일관되고 디테일한 정보를 표현할 수 있습니다. 또한 $Fern$의 잎 뒤에 있는 노란색 선반과 $Orchid$ 배경의 녹색 잎과 같이 LLFF가 어려움을 겪는 object에 가려진 영역에 대해서도 올바르게 재구성합니다.

LLFF는 서로 다른 view를 렌더링하기 위해 서로 다른 scene representation 사이에서 블렌딩되는데, 여러 렌더링을 블렌딩하면 LLFF의 $Orchid$의 위쪽 결과와 같이 물체 가장자리 경계가 반복되는 현상이 발생할 수도 있습니다.

SRN은 각 scene에서 low-frequency geometry와 색상 변화(color variation)을 잡아낼 수 있지만 미세한 영역은 재현할 수 없습니다.



### Ablation study

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/f9d24bce-f090-4d8e-af9e-f41dcab8fe9d" width="800" height="300">
</div>
> Table 2. NeRF의 ablation study.

Realistic Synthetic 360$^\circ$에 대한 ablation study 결과를 Table 2에서 볼 수 있습니다.

1행은 Positional Encoding(PE), View Dependence(VD), Hierarchical sampling(H)가 없는 모델을 보여주며, 2~4행에서는 3가지 구성 요소를 하나씩 제거한 모델 결과를 볼 수 있습니다.

5, 6행에서 입력 이미지 수가 감소함에 따른 성능 차이를 볼 수 있고 7, 8행에서 Positional Encoding에 사용되는 maximum frequency $\mathrm{L}$에 따른 성능 차이를 볼 수 있습니다. frequency를 10보다 작게 사용한 7행과 10보다 크게 15를 사용한 8행 모두 성능이 저하되었습니다. 저자들은 maximum frequency가 $2^ {\mathrm{L}}$을 초과하도록 샘플링된다면 $\mathrm{L}$를 늘리는 것의 이점이 제한된다고 합니다.

<br>

---

NeRF 논문 리뷰가 끝났습니다! 끝까지 봐주셔서 감사합니다:) <br>
모델 구조만 봤을 때는 단순해서 쉽네!라고 생각했지만, 3D 분야를 처음 알게 되니 개념을 익히는 것부터 수학적 공식을 이해하는 것까지 생각보다 어려웠어요...:speak_no_evil:

3D 관련 task가 최근에 많이 보이는 것 같아요. 이렇게 조금씩 정리해서 3D 분야에도 손을 담궈볼 수 있도록...! 노력해보겠습니다:wink:
