---
layout: post
title: Pix2Pix(1) - 논문 구2
# subtitle:
categories: gan
tags: [gan, pix2pix, 생성 모델, 논문 구현]
# sidebar: []
use_math: true
---

pix2pix의 논문 구현 글로 논문의 내용을 따라 구현했습니다.<br>
공식 코드로는 논문에 나온 <a href="https://github.com/phillipi/pix2pix" target="_blank">phillipi/pix2pix</a>와 PyTorch로 pix2pix와 CycleGAN을 구현한 <a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix" target="_blank">junyanz의 pytorch-CycleGAN-and-pix2pix</a>가 있습니다.<br>
데이터셋 설명 모델 설명을 코드와 함께 설명해드리겠습니다!

## 데이터 셋

### 논문 데이터셋

논문에서는 총 8개의 데이터셋을 사용합니다. 이중에서 10,000장 이하의 작은 데이터셋은 3개가 있습니다.
1. Cityscapes labels $\rightarrow$ photo / 2975장
2. Architectural labels $\rightarrow$ photo / 400장
3. Maps $\leftrightarrow$ aerial photograph / 1096장

3가지 데이터셋 모두 random jitter와 mirroring을 사용해 200 epoch만큼 학습했다고 합니다. random jitter와 mirroring 모두 data augmentation을 위해 사용합니다.

random jitter로는 이미지를 286 x 286으로 resize한 다음 256 x 256으로 random cropping해 사용하며 <a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/125" target="_blank">issues</a>를 보니 286 x 286으로 지정한 특별한 이유는 없다고 답변이 되어 있었습니다. random crop을 위해 원하는 크기보다 적당히 큰 정도면 될 것 같아요!

mirroring은 flip을 의미합니다. augmentation의 대표적인 방법으로 데이터셋에 따라 horizontal flip과 vertical flip을 추가했다 이해했습니다.

### 사용 데이터셋
pix2pix 관련 프로젝트들은 <a href="https://phillipi.github.io/pix2pix/" target="_blank">Image-to-Image Translation with Conditional Adversarial Nets</a>에서 확인하실 수 있습니다.

'#edges2cats'나 'Interactive Anime', 'Suggestive Drawing'처럼 스케치를 색칠해주거나 특정 물체로 완성해주는 프로젝트들은 많아 보이는데 특정 물체를 마치 캐리커쳐처럼 스케치해주는 프로젝트는 보이지 않아 스케치 프로젝트를 하기로 정했습니다.

pix2pix는 pair 데이터를 사용하는데 1000장 이상의 스케치와 관련된 pair 데이터셋을 찾기 힘들었습니다... 대신 장수가 적더라도 데이터 퀄리티가 마음에 들었던 <a href="http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html" blank="_blank">CUHK Face Sketch Database(CUFS)</a>를 발견했습니다.

데이터 셋은 Honk Kong Chinese University의 학생들로 이루어진 CUHK 데이터 셋(188장), AR database(123장), XM2VTS database(295장)로 이루어져 있으며 3종류의 얼굴 데이터 셋에서 각각의 얼굴을 스케치한 이미지로 (사진, 스케치) pair 되어 있습니다. 3종류의 데이터셋 모두 스케치 파일을 다운받을 수 있는데 원본 사진의 경우 CUHK만 다운 가능하며 AR과 XM2VTS는


## 모델


### Generator
**transposedconv**

**cgan**
cgan이지만 z를 쓰지 않음

### Discriminator
**receptive field**

**cgan**
cgan이므로 포토와 스케치 이미지를 같이 입력받으므로 첫번째 cnn의 in_channel 값이 3+3으로 6이 됨

## 추가 설정 및 학습

### 추가 설정


### 학습



## 결과
