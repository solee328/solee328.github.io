---
layout: post
title: SAGAN(1) - 논문 리뷰
# subtitle:
categories: gan
tags: [gan, SAGAN, attention, spectral, TTUR, 생성 모델, 논문 리뷰]
# sidebar: []
use_math: true
---

이번 논문은 self-attention을 Generative model에 적용한 SAGAN(<a href="https://arxiv.org/abs/1805.08318" target="_blank">Self-Attention Generative Adversarial Network</a>)입니다. 지금부터 SAGAN을 살펴보겠습니다:lemon:
<br><br>

---
## 소개
GANs는 ImageNet과 같은 multi-class를 모델링 시 문제는 가지고 있었습니다. 당시 class conditional image generation task에서 SOTA인 <a href="https://arxiv.org/abs/1802.05637" target="_blank">CGANS with Projection Discriminator</a>는 간헐천, 계곡과 같이 객체의 구조적 제약이 거의 없는 이미지(텍스처(질감)으로 구별 가능한 바다, 하늘과 같은 풍경) 생성은 탁월하지만, 하프, 크로스워드 등 일부 클래스(개와 같은 클래스의 이미지는 개의 털 텍스처(질감)은 성공적으로 생성되지만 일부 발이 생성되지 않는 경우가 발생)에서 기하학적/구조적 패턴을 파악하지 못합니다 실제로 위 논문의 Figure 7에서 퓽경의 FID는 낮지만 객체에 대한 FID는 높은 것을 확인할 수 있습니다.

 ```
 Figure 7 사진 넣기
 ```

 이에 대한 가능한 설명은 모델이 서로 다른 이미지 영역에 걸쳐 종속성(dependency)를 모델링하기 위해 convolution에 크게 의존한다는 것입니다.


<details>
<summary>FID(Fréchet Inception Distance)</summary>
<span style="color:gray">
  <a href="https://arxiv.org/abs/1706.08500" target="_blank">Fréchet Inception Distance(FID)</a>는 생성 모델에서 생성된 이미지의 품질을 평가하는 데 사용되는 metric으로 Inception Score(IS)를 개선하기 위해 제안되었습니다. 두 분포의 거리를 계산하는 metric으로 값이 낮을 수록 분포가 가까워 생성 이미지가 실제 이미지와 유사함을 의미해 좋습니다.<br><br>

  FID를 계산하기 위해서 우선 pretrain된 Inception V3를 사용해 실제 이미지와 생성된 이미지의 (2048, ) 크기의 feature map을 계산합니다. 계산된 이 feature map들의 분포 차이를 계산하기 위해 정규분포(Gaussian distribution)를 사용합니다. 정규분포는 평균과 분산이 주어져 있을 때 엔트로피를 최대화하는 분포이므로 다차원 정규분포를 따른다고 가정해 두 feature map의 평균(mean)과 공분산(covariance) 차이를 이용해 두 분포의 차이를 계산합니다. 이때 차이는  Wasserstein-2 distance라고도 불리는 Fréchet distance로 계산합니다. 아래 수식이 Fréchet distance를 활용한 FID의 수식입니다.<br><br>

  $
  d^2((m,C),(m_w, C_w)) = \| m-m_w \|^2_2 + Tr(C + C_w - 2(CC_w)^{1/2})
  $

  <br><br>
  $m$, $C$는 실제 이미지의 feature map의 평균과 공분산이고 $m_w$, $C_w$는 생성된 이미지의 feature map의 평균과 공분산이며, Tr은 행렬의 대각합(trace)를 의미합니다.<br><br>

  참고<br>
  - 페이오스님의 <a href="https://m.blog.naver.com/chrhdhkd/222013835684" target="_blank">GAN 평가지표</a><br>
  - viriditass.log님의 <a href="https://velog.io/@viriditass/GAN%EC%9D%80-%EC%95%8C%EA%B2%A0%EB%8A%94%EB%8D%B0-%EA%B7%B8%EB%9E%98%EC%84%9C-%EC%96%B4%EB%96%A4-GAN%EC%9D%B4-%EB%8D%94-%EC%A2%8B%EC%9D%80%EA%B1%B4%EB%8D%B0-How-to-evaluate-GAN" target="_blank">GAN은 알겠는데, 그래서 어떤 GAN이 더 좋은건데?</a>
  <br>
</span>
</details>
<br>

Self-Attention Generative Adversarial Network(SAGAN)은 이미지 생성 작업에 대한 attention-driven, long-range dependency 모델링을 위해 제안된 모델입니다. 기존의 convolution GAN은 convolution을 사용하다보니 lower-resolutional feature maps에서 공간적으로 local 부분을 사용해 high-resolution 세부 정보를 생성했습니다. SAGAN은 ---, attention layer를 시각화해 생성 모델이 고정된 모양의 local 영역이 아닌 객체 모양에 해당하는 영역을 활용한다는 것을 확인할 수 있습니다.

또한 GAN의 생성 모델에 spectral normliazation을 사용해 training dynamic를 향상시켜 Inception score, Frechet Inception Distance 모두에서 기존 State-of-the-art를 능가하는 성능을 보여주었습니다.
<br><br>

---
## 데이터셋


<br><br>
---
## 모델

<br><br>

---
## Loss

<br><br>

---
## 결과


<br><br>

---
ㅇㅇㅇ
