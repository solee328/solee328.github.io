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

DDPM은 Diffusion Model이라 불리는 Diffusion Probabilistic Model[53]을 개선한 모델로 Variational Autoencoders(VAEs), Generative Adversarial Networks(GANs)와 같은 Generative model로서 널리 사용되고 있습니다. 지금부터 하나하나 살펴보겠습니다 :eyes:
<br>

---

## Diffusion Model?

nonequilibrium thermodynamics 연구에서 영감을 얻은 latent variable model에 diffusion probabilistic model을 사용해 고품질 이미지 합성 결과를 내었다고 합니다.

diffusion probabilistic model과 Langevin dynamics의 denoising score matching를 연결해 설계한 weighted variational bound에 대한 학습을 통해 가장 좋은 결과를 얻을 수 있었으며, 모델은 autoregressive decoding을 일반화(generalization)로 해석될 수 있는 decompression scheme로 받아들여 진다고 합니다.

abstract 내용만 가져왔습니다.... 나오는 모든 단어를 정말 하나도 모르겠네요... :joy:
우선 단어의 개념부터 하나씩 정리해보겠습니다

### diffusion probabilistic model
논문에서 말하는 nonequilibrium thermodynamics(비평형 열역학) 연구는 <a href="https://arxiv.org/abs/1503.03585" target="_blank">Deep Unsupervised Learning using Nonequilibrium Thermodynamics</a>를 말합니다.

논문에서는 확률론적 모델(probabilistic model)이 tractability와 flexibility의 trade-off를 해결하기 위해 "diffusion probabilistic model"을 사용했습니다.

non-equilibrium statistical physics에서 영감을 받은 diffusion probabilistic model의 기본 아이디어는 2가지 단계를 가지고 있습니다.

첫번째는 반복적인 forward diffusion process를 통해 데이터 분포를 천천히 파괴하는 것입니다. 두번째 과정은 reverse diffusion process로 파괴한 데이터 구조를 복원하는 것으로 매우 flexible하고 tractable한 generative model이 학습된다고 합니다.

이 접근 방식은 수천 개의 layer 또는 time step을 가진 deep generative model에서 probabilistic을 신속하게 train, evaluate가 가능하며 학습된 모델에서 posterior probabilistic을 계산할 수 있습니다.


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
