---
layout: post
title: MUNIT(1) - 논문 리뷰
# subtitle:
categories: gan
tags: [gan, munit, multimodal, unsupervised, 생성 모델, 논문 리뷰]
# sidebar: []
use_math: true
---

이번 논문은 <a href="https://arxiv.org/abs/1804.04732" target="_blank">MUNIT</a>(Multimodal Unsupervised Image-to-Image Translation) 입니다.


---

## 소개

---

## 모델

### Assumption
MUNIT은 도메인 $\mathcal{X}_1$에서 온 이미지 $x_1$ ($x_1 \in \mathcal{X}_1$)과 도메인 $\mathcal{X}_2$에서 온 이미지 $x_2$ ($x_2 \in \mathcal{X}_2$)를 사용합니다. 이미지 $x_1$을 도메인 $\mathcal{X}_1$에서 도메인 $\mathcal{X}_2$로 변환하는 모델 $p(x _{1 \rightarrow 2} | x_1)$과 이미지 $x_2$을 도메인 $\mathcal{X}_2$에서 도메인 $\mathcal{X}_1$로 변환하는 모델 $p(x _{2 \rightarrow 1} | x_2)$을 사용해 $p(x_1 | x_2)$와 $p(x_2 | x_1)$을 추정하는 것이 MUNIT의 목표입니다.

$p(x_1 | x_2)$, $p(x_2 | x_1)$는 복잡한 multimodal distribution으로


### Auto-Encoder

### GAN

---

## Loss

### Bidirectional Reconstruction Loss

#### Image Reconstruction

#### Latent Reconstruction

### Adversarial Loss

### Total Loss

### Domain-invariant perceptual loss
고해상도 이미지일 경우 사용

---

## 이론 가정

---

## 사용 데이터셋

---

## 평가 지표

### Human preference

### LPIPS distance

### (Conditional) Inception Score

---

## 결과


### Baseline 모델


### edges $\leftrightarrow$ shoes/handbags

### animal translation

### Cityscape $\leftrightarrow$ SYNTHIA

### Yosemite summer $\leftrightarrow$ winter
