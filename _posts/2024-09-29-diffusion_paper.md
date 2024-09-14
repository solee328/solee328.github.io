---
layout: post
title: DDPM - 논문 리뷰
# subtitle:
categories: diffusion
tags: [diffusion, ddpm, 논문 리뷰]
# sidebar: []
use_math: true
---

안녕하세요:lemon: 오랜만에 들고 온 논문은 DDPM으로 불리는 <a href="https://arxiv.org/abs/2006.11239" target="_blank">Denoising Diffusion Probabilistic Models</a>입니다!

DDPM은 Diffusion Model이라 불리는 Diffusion Probabilistic Model[53]을 개선한 모델로 Variational Autoencoders(VAEs), Generative Adversarial Networks(GANs)와 같은 Generative model로서 널리 사용되고 있습니다. 지금부터 하나하나 살펴보겠습니다:eyes:
<br>

---

## Diffusion Model?

nonequilibrium thermodynamics(비평형 열역학) 연구에서 영감을 얻은 latent variable model에 diffusion probabilistic model을 사용해 고품질 이미지 합성 결과를 내었다고 합니다....하나도 모르겠죠?

diffusion probabilistic model과 Langevin dynamics의 denoising score matching를 연결해 설계한 weighted variational bound에 대한 학습을 통해 얻어졌으며 autoregressive decoding을 일반화(generalization)로 해석될 수 있는 decompression scheme로 받아들여 진다고 합니다.

abstract에서 모든 영어 단어를 모르는 상태입니다...
하나씩 단어부터 개념을 정리해보겠습니다.


### nonequilibrium thermodynamics


### latent variable model


### denoising score matching
Langvebin dynamics


### diffusion probabilistic Model


### autoregressive decoding


### 정리


<br>

---

## Dataset
LSUN, CelebA-HQ, CIFAR10 등


<br>


---

<br>
