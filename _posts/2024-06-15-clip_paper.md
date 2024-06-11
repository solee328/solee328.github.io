---
layout: post
title: CLIP - 논문 리뷰
# subtitle:
categories: gan
tags: [clip, zero-shot, representation learning, contrastive learning, 논문 리뷰]
# sidebar: []
use_math: true
---

안녕하세요:lemon: 이번에 다룰 논문은 CLIP입니다!
논문 전체 이름은 <a href="https://arxiv.org/abs/2103.00020"_blank">Learning Transferable Visual Models From Natural Language Supervision" target</a>으 GPT와 같이 자연어 분야에서 이뤄지던 대규모 데이터를 학습해 pre-training 후 다양한 task로 zero-shot transfer를 실험한 것으로 유명합니다.

여러 Diffusion 모델에서 이미지를 임베딩할 때는 다들 CLIP 모델을 사용하길래 알게 되어서 논문을 살펴보게 되었습니다. 어떤 구조를 가지고 있기래 모두 CLIP 모델을 사용하는지 지금부터 살펴보게 되었습니다:eyes:

<br><br>

---

## 소개
자연어 분야에서 모델이 raw text를 학습하는 방법을 사용하며 GPT-3와 같은 모델들이 개발되었습니다. 이런 모델들은 text-to-text task의 경우 해당 task에 맞춰 출력 head를 설정하거나 다른 데이터셋을 학습할 필요 없이 zero-shot transfer가 가능한 것이 특징입니다.

<details>
<summary>zero-shot transfer</summary>
<span style="color:gray">
  Transfer learning은 미리 학습된 모델(pre-train)을 가지고 fine-tuning하는 과정을 거칩니다. zero-shot transfer는 fine-tuning 과정 없이 본 적 없는 데이터에 대해서도 task를 수행할 수 있습니다.
</span>
</details>

<br>
Vision 분야에서는 ImageNet처럼 라벨링된 데이터셋을 사용해 모델을 학습하는 방법을 주로 사용하기 때문에 task 별로 task에 맞는 데이터셋을 모델에 학습시키는 방법을 사용했습니다. 또한 데이터셋 라벨링에는 큰 노력이 필요합니다. 대표적인 데이터셋인  ImageNet의 경우 1,400만개 이미지에 주석을 달기 위해 25,000명 이상의 작업자가 필요했습니다. CLIP은 Vision 분야에도 자연어 분야처럼 raw text를 학습하는 방법을 사용해 추가적인 라벨링 비용이 들지 않고 높은 성능을 얻고자 합니다.

유사하게 자연어에서 직접 이미지 representation을 학습하는 연구들이 있습니다. VirTex (Desai & Johnson, 2020), ICMLM (Bulent Sariyildiz et al., 2020), ConVIRT (Zhang et al., 2020)의 경우 transformer 기반 언어 모델링, masked language modeling, contrastive objectives에서 텍스트로 이미지 representation을 학습하는 가능성을 입증했습니다. 하지만 세 모델 모두 1~20만개의 이미지를 사용했습니다. 하지만 weak supervised 모델인 Mahajan et al. (2018), Kolesnikov et al. (2019)은 수백만개에서 수십 억개의 이미지에 대해 학습을 해 학습 양에 차이가 큰 것을 볼 수 있습니다.

CLIP에서는 Vision 모델이지만 large scale의 데이터셋을 사용하며 약 4억 개의 (image, text) 페어 데이터셋을 새롭게 구성하고 Contrastive Language-Image 방법을 사용해 모델을 학습합니다. GPT 모델과 유사하게 CLIP이 학습에서 OCR, geo-localization, action recognition 등 다양한 task를 수행하는 방법을 배웠으며 강력한 zero-shot 성능을 결과에서 보여줍니다.  

학습 방법과 결과에 대해서 지금부터 자세하게 살펴보겠습니다:stuck_out_tongue_winking_eye:

---

## Dataset


고품질의 라벨링 데이터셋으로는 MS-COCO (Lin et al., 2014), Visual Genome (Krishna et al., 2017)이 있지만 각각 약 100,000장으로 데이터셋 크기가 작습니다. 다른 데이터셋으로 35억개의 인스타 그램 사진(Mahajan et al., 2018), 1억개의 사진 YFCC100M (Thomee et al., 2016)이 있지만 각 이미지마다 품질 차이가 크며 많은 이미지가 "20160716_113957.JPG"와 같이 자동으로 생성된 이름("title") 또는 카메라 설정에 대한 메타 데이터("description")를 가지고 있습니다. CLIP은 (image, text) 페어 데이터셋을 사용하기 때문에 이미지에 대한 정보가 담긴 텍스트가 포함되어 있어야 합니다. 영어로 표시된 title 또는 description을 가진 이미지들만 남기도록 필터링했을 때 데이터셋은 600~1500만 장으로 크기가 크게 줄어들었다고 합니다.

따라서 CLIP은 공개적으로 사용 가능한 다양한 소스들에서 데이터 셋을 모아 4억개의 (image, text) 페어 데이터셋을 구축합니다. 영어 위키 데이터셋에서 최소 100번 이상 발생하는 모든 단어와 WordNet synset을 합쳐 만든 500,000개 쿼리 목록을 만들고 해당 쿼리 목록에서 적어도 하나를 포함하는 (이미지, 텍스트) 객체를 만들었다고 합니다. 쿼리 하나 당 최대 20,000개의 (이미지, 텍스트) 객체가 나오며 이 데이터 셋은 GPT-2를 학습하는 데 사용되는 WebbText 데이터 셋과 유사한 총 단어 수를 갖기 때문에 저자들은 이 데이터셋을 WIT(WebImageText)라 부릅니다.

---

## Contrastive learning


Contrative learning은 ConVIRT의 단순화된 버전을 사용합니다.


---

## Model(Encoder)


---

## 결과


---
