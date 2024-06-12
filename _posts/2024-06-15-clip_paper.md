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
논문 전체 이름은 <a href="https://arxiv.org/abs/2103.00020"_blank">Learning Transferable Visual Models From Natural Language Supervision target</a>으 GPT와 같이 자연어 분야에서 이뤄지던 대규모 데이터를 학습해 pre-training 후 다양한 task로 zero-shot transfer를 실험한 것으로 유명합니다.

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

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/cb4a35d0-1103-46e1-b471-e0a16a0d79c5" width="300" height="500">
</div>

따라서 CLIP은 공개적으로 사용 가능한 다양한 소스들에서 데이터 셋을 모아 4억개의 (image, text) 페어 데이터셋을 구축합니다. 영어 위키 데이터셋에서 최소 100번 이상 발생하는 모든 단어와 WordNet synset을 합쳐 만든 500,000개 쿼리 목록을 만들고 해당 쿼리 목록에서 적어도 하나를 포함하는 (이미지, 텍스트) 객체를 만들었다고 합니다. 쿼리 하나 당 최대 20,000개의 (이미지, 텍스트) 객체가 나오며 이 데이터 셋은 GPT-2를 학습하는 데 사용되는 WebbText 데이터 셋과 유사한 총 단어 수를 갖기 때문에 저자들은 이 데이터셋을 WIT(WebImageText)라 부릅니다.

공개된 데이터로만 만들었다고 해서 혹시나 WIT를 공개했나 싶어서 찾아봤더니 다른 <a href="https://github.com/google-research-datasets/wit" target="_blank">WIT(Wikipedia-based Image Text Dataset)</a>이 나오네요... CLIP의 4억개까지는 도달하지 못하지만 약 3700만개의 image-text 데이터셋입니다. 다른 유사 데이터셋으로느 CLIP implementation인 <a href="https://github.com/mlfoundations/open_clip?tab=readme-ov-file" target="_blank">OpenCLIP</a>에서 사용한 데이터셋인 LAION(<a href="https://arxiv.org/abs/2111.02114" target="_blank">LAION-400M</a>, <a href="https://arxiv.org/abs/2210.08402" target="_blank">LAION-5B</a>)과 <a href="https://arxiv.org/abs/2304.14108" target="_blank">DataComp</a>가 있습니다.

<br>

---

## Contrastive learning

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/0eacecde-f651-4ebc-b240-973472ef6a10" width="700" height="500">
</div>
> Figure 1.(1). 일반적인 이미지 모델은 image feature extractor와 linear classifier를 학습시키지만, CLIP은 image encoder와 text encoder를 동시에 학습시키며 올바른 (image, text) 페어 예측하는 방식을 사용합니다.

Contrastive learning은 self-supervised representation learning에서 사용되는 방법으로 유사한 것끼리는 가깝도록, 다른 것끼리는 멀리 있도록 representation space를 학습하는 방법입니다.

CLIP은 (image, text) 페어로 구성된 데이터셋을 사용하니 서로 가깝게 있어야 할 대상은 페어로 된 데이터입니다. 따라서 이미지 $I_i$는 텍스트 $T_i$와 가까워야 합니다. 반대로 이미지 $I_i$는 텍스트 $T_i$가 아닌 모든 텍스트와 멀리 떨어져 있어야 합니다. 이를 시각화하면 Figure 1.(1).과 같아집니다.

(image, text) 페어 데이터셋은 각각 pre-train된 Image, Text encoder에 입력되어 image, text embedding을 계산하게 됩니다. 이를 linear projection 시켜 $[I_1, ..., I_N]$과 $[T_1, ..., T_N]$를 구합니다. 각각의 페어 데이터에 해당하는 $(I_i, T_i)$의 값은 최대가 되도록, 그 외의 페어에 대해서는 최소가 되도록 학습하는 방식입니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/fbe7d60f-e688-4415-802f-5a4f7fbdea47" width="500" height="500">
</div>
> Figure 3. CLIP 학습을 위한 pseudo code

ConVIRT에서는 INFONCE loss를 사용했지만 CLIP에서는 cross entropy loss를 사용하는 것이 차이점입니다. 논문에서는 symmetric cross entropy를 사용한다고 언급되는데 image 관점에서의 loss인 loss_i와 text 관점에서의 loss인 loss_t 모두에 대해 CE loss를 계산하기 때문이라고 이해했습니다.

Figure 3에서 CLIP 구현에 대한 pseudo 코드를 확인할 수 있습니다.

image, text encoder로 각각 image embedding(I_f), text embedding(T_f)을 생성합니다. 이후 `np.dot`으로 행렬곱을 계산해 linear projection을 수행합니다. batch 개수(n)만큼 linear embedding이 생성되며 생성된 두 embedding을 행렬곱을 통해 $n \times n$ 행렬로 생성합니다.

joint multimodal embedding 과정에서 `L2_normalize`를 수행했기 때문에 `np.dot`의 결과가 [-1, 1] 범위로 나오게 되고 이 값을 cosine similarity 값이 됩니다. L2_normalize로 값의 범위를 제한했기 때문에 `np.exp(t)`로 다시 [-$\infty$, $\infty$]로 scaling해주며 t 또한 학습 가능한 learnable parameter입니다.

하나의 이미지를 설명하는 하나의 텍스트가 (image, text) 형태로 batch에 들어가기 때문에 같은 index에 존재하는 이미지와 텍스트가 실제 정답 값이 되기 때문에 `np.arange(n)`으로 라벨을 생성합니다. `np.arange(n)`으로 생성한 라벨 값과 `cross_entropy_loss`를 계산해 loss를 계산합니다.


<br>

---

## Zero-shot

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/ae00c1d4-2f11-4643-8fc8-2042aa287f87" width="700" height="500">
</div>

모델 학습 이후 classification에 사용할 시에는 클래스 이름을 임베딩해 zero-shot linear classifier를 만들어 사




---

## Model(Encoder)


---

## 결과


---
