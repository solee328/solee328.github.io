---
layout: post
title: CLIP - 논문 리뷰
# subtitle:
categories: gan
tags: [clip, contrastive learning, zero-shot transfer, representation learning, 논문 리뷰]
# sidebar: []
use_math: true
---

안녕하세요:lemon: 이번에 다룰 논문은 CLIP입니다!
논문 전체 이름은 <a href="https://arxiv.org/abs/2103.00020" target="_blank">Learning Transferable Visual Models From Natural Language Supervision target</a>으로 GPT와 같이 자연어 분야에서 이뤄지던 대규모 데이터를 사용해 pre-training 후 다양한 task로 zero-shot transfer를 실험한 것으로 유명합니다.

여러 Diffusion 모델에서 이미지를 임베딩할 때는 다들 CLIP 모델을 사용하길래 알게 되어서 논문을 살펴보게 되었습니다. 어떤 구조와 특징을 가지고 있길래 모두 CLIP 모델을 사용하는지 지금부터 살펴보게 되었습니다:eyes:

<br>

---

## 소개
자연어 분야에서 모델이 raw text를 학습하는 방법을 사용하며 GPT-3와 같은 너무나도 유명한 모델들이 개발되었고 이런 모델들은 text-to-text task 분야에 한해서라면 task에 맞춰 출력 head를 설정하거나 다른 데이터셋을 학습할 필요 없이 zero-shot transfer가 가능한 것이 특징입니다.

<details>
<summary>zero-shot transfer</summary>
<span style="color:gray">
  Transfer learning은 미리 학습된 모델(pre-train)을 가지고 fine-tuning하는 과정을 거칩니다. zero-shot transfer는 fine-tuning 과정 없이 본 적 없는 데이터에 대해서도 task를 수행할 수 있는 능력을 말합니다.
</span>
</details>

<br>
이런 자연어 분야와는 다르게 Vision 분야에서는 ImageNet처럼 라벨링된 데이터셋으로 모델을 학습하는 방법을 주로 사용하기 때문에 task 별로 task에 맞는 데이터셋을 모델에 학습시켰습니다. 문제는 데이터셋 라벨링에는 큰 노력이 필요하다는 것으로 대표적인 데이터셋인  ImageNet의 경우 1,400만개 이미지에 주석을 달기 위해 25,000명 이상의 작업자가 필요했습니다. 이런 문제를 회피하기 위해 CLIP은 Vision 분야에도 자연어 분야처럼 raw text를 학습하는 방법을 사용해 추가적인 라벨링 비용이 들지 않고 높은 성능을 얻고자 했습니다.

유사하게 자연어에서 직접 이미지 representation을 학습하는 연구들이 있었으며 Bert를 사용해 추출한 text representation을 visual representation로 변환하는 <a href="https://arxiv.org/abs/2006.06666" target="_blank">VirTex</a>와 masked language modeling으로 이미지에서 마스킹된 단어를 예측하는 <a href="https://arxiv.org/abs/2008.01392" target="_blank">ICMLM</a>, 의료이미지에 contrastive objective를 활용한 <a href="https://arxiv.org/abs/2010.00747" target="_blank">ConVIRT</a>가 대표적입니다. 세 모델들 모두 텍스트로 이미지 representation을 학습하는 가능성을 입증했지만 모두 1~20만개의 이미지를 사용해 자연어 분야에서 사용하는 데이터양에 비하면 굉장히 작은 숫자입니다.

CLIP에서는 Vision 모델이지만 large scale의 데이터셋을 사용하며 약 4억 개의 (image, text) 페어 데이터셋을 새롭게 구성하고 Contrastive Language-Image 방법을 사용해 모델을 학습합니다. GPT 모델과 유사하게 CLIP이 pre-train으로 OCR, geo-localization, action recognition 등 다양한 task를 수행하는 zero-shot이 가능한 것이 특징입니다.  

CLIP의 모델 구조, 학습 방법과 결과에 대해서 지금부터 자세하게 살펴보겠습니다 :stuck_out_tongue_winking_eye:

<br>

---

## Dataset


고품질의 라벨링 데이터셋으로는 <a href="https://arxiv.org/abs/1405.0312" target="_blank">MS-COCO</a>, <a href="https://arxiv.org/abs/1602.07332" target="_blank">Visual Genome</a>이 있지만 각각 약 100,000장으로 데이터셋 크기가 상대적으로 작았습니다. 다른 데이터셋으로 35억개의 인스타그램 사진 데이터셋인 <a href="https://arxiv.org/abs/1805.00932" target="_blank">IG-3.5B-17.K</a>, 1억개의 사진 <a href="https://arxiv.org/abs/1503.01817" target="_blank">YFCC100M</a>이 있지만 각 이미지마다 품질 차이가 크며 많은 이미지가 "20160716_113957.JPG"와 같이 자동으로 생성된 이름("title") 또는 카메라 설정에 대한 메타 데이터("description")를 가지고 있습니다.

CLIP은 (image, text) 페어 데이터셋을 사용하기 때문에 이미지에 대한 정보가 담긴 텍스트가 포함되어 있어야 하기 때문에 정보가 되지 않는 텍스트에 대해서는 필터링이 필요합니다. 영어로 표시된 title 또는 description을 가진 이미지들만 남기도록 필터링했을 때 데이터셋은 600~1500만 장으로 크기가 크게 줄어들었다고 합니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/1b7ddd15-c0a0-4a63-a7dc-c19d66014519" width="300" height="350">
</div>
> Figure 1.(1). 텍스트와 이미지가 batch마다 pair로 입력되는 것을 볼 수 있습니다.

따라서 CLIP은 공개적으로 사용 가능한 다양한 소스들에서 데이터 셋을 모아 4억개의 (image, text) 페어 데이터셋을 새롭게 구축했습니다. 영어 위키 데이터셋에서 최소 100번 이상 발생하는 모든 단어와 WordNet synset을 합쳐 만든 500,000개 쿼리 목록을 만들고 해당 쿼리 목록에서 적어도 하나를 포함하는 (이미지, 텍스트) 객체를 만들었다고 합니다. 쿼리 하나 당 최대 20,000개의 (이미지, 텍스트) 객체가 나오며 이 데이터 셋은 GPT-2를 학습하는 데 사용되는 WebbText 데이터 셋과 유사한 총 단어 수를 갖기 때문에 저자들은 이 데이터셋을 WIT(WebImageText)라 부릅니다.

공개된 데이터로만 만들었다고 해서 혹시나 WIT를 공개했나 싶어서 찾아봤더니 다른 <a href="https://github.com/google-research-datasets/wit" target="_blank">WIT(Wikipedia-based Image Text Dataset)</a>이 나오네요... CLIP의 4억개까지는 도달하지 못하지만 약 3700만개의 image-text 데이터셋입니다. 다른 유사 데이터셋으로는 CLIP implementation인 <a href="https://github.com/mlfoundations/open_clip?tab=readme-ov-file" target="_blank">OpenCLIP</a>에서 사용한 데이터셋인 LAION(<a href="https://arxiv.org/abs/2111.02114" target="_blank">LAION-400M</a>, <a href="https://arxiv.org/abs/2210.08402" target="_blank">LAION-5B</a>)과 <a href="https://arxiv.org/abs/2304.14108" target="_blank">DataComp</a>가 있습니다.

<br>

---

## Contrastive learning

이렇게 만든 데이터셋을 CLIP은 Contrasitive learning 방식을 사용해 모델을 학습시킵니다. Contrastive learning은 self-supervised representation learning에서 사용되는 방법으로 유사한 것끼리는 가깝도록, 다른 것끼리는 멀리 있도록 representation space를 학습하는 방법입니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/0eacecde-f651-4ebc-b240-973472ef6a10" width="700" height="500">
</div>
> Figure 1.(1). 일반적인 이미지 모델은 image feature extractor와 linear classifier를 학습시키지만, CLIP은 image encoder와 text encoder를 동시에 학습시키며 올바른 (image, text) 페어 예측하는 방식을 사용합니다.

CLIP은 (image, text) 페어로 구성된 데이터셋을 사용하니 서로 가깝게 있어야 할 대상은 페어로 된 데이터입니다. 따라서 이미지 $I_i$는 텍스트 $T_i$와 가까워야 합니다. 반대로 이미지 $I_i$는 텍스트 $T_i$가 아닌 모든 텍스트와 멀리 떨어져 있어야 합니다. 이를 시각화하면 Figure 1.(1).과 같아집니다.

(image, text) 페어 데이터셋은 각각 Image, Text encoder에 입력되어 image, text embedding으로 출력됩니다. 이를 linear projection 시켜 $[I_1, ..., I_N]$과 $[T_1, ..., T_N]$를 구합니다. 각각의 페어 데이터에 해당하는 $(I_i, T_i)$의 값($N$개)은 cosine similarity가 최대가 되도록, 그 외의 페어($N^2 - N$)에 대해서는 최소가 되도록 학습하는 방식입니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/fbe7d60f-e688-4415-802f-5a4f7fbdea47" width="500" height="500">
</div>
> Figure 3. CLIP 학습을 위한 pseudo code

논문에서 pseudo code를 Figure 3으로 넣어놔서 코드로 더 자세하게 과정을 볼 수 있었습니다. 코드를 자세하게 보자면 우선, image, text encoder로 각각 image embedding(I_f), text embedding(T_f)을 생성합니다. 이후 weight와 `np.dot`으로 행렬곱을 계산해 linear  projection을 수행합니다. batch 개수(n)만큼 linear embedding이 생성되며 생성된 두 embedding을 행렬곱을 통해 $n \times n$ 행렬을 생성합니다.

joint multimodal embedding 과정에서 `L2_normalize`를 수행했기 때문에 `np.dot`의 결과가 [-1, 1] 범위로 나오게 되고 이 값이 cosine similarity 값이 됩니다. L2_normalize로 값의 범위를 제한했기 때문에 `np.exp(t)`로 다시 [-$\infty$, $\infty$]로 scaling해주며 t 또한 학습 가능한 learnable parameter입니다.

하나의 이미지를 설명하는 하나의 텍스트가 (image, text) 형태로 batch에 들어가기 때문에 같은 index에 존재하는 이미지와 텍스트가 실제 정답 값이 되기 때 `np.arange(n)`으로 라벨을 생성합니다. `np.arange(n)`으로 생성한 라벨 값과 `cross_entropy_loss`를 계산해 loss를 계산합니다.

 CLIP과 같은 Contrastive learning 방식을 사용한 ConVIRT에서는 INFONCE loss를 사용했지만 CLIP에서는 cosine similarity에 대해 cross entropy loss를 사용하는 것이 차이점입니다. 또한 논문에서는 symmetric cross entropy를 사용한다고 언급되는데 image 관점에서의 loss인 loss_i와 text 관점에서의 loss인 loss_t 모두에 대해 CE loss를 계산하기 때문이라고 이해했습니다.

<br>

---

## Model(Encoder)
contrastive learning에 대해 살펴봤으니 이제 CLIP이 사용하는 Image, Text encoder에 대해서 알아보겠습니다.

### Image encoder
우선 Image encoder입니다! Image encoder로는 ResNet과 ViT를 사용합니다.

#### ResNet-50

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/e7bcd486-7a9d-47f6-bf96-7eac5668101c" width="220" height="400">
</div>
> ResNet-D의 구조

ResNet 모델로는 <a href="https://arxiv.org/abs/1512.03385" target="_blank">ResNet-50</a>을 기반으로 한 <a href="https://arxiv.org/abs/1812.01187" target="_blank">ResNet-D</a>를 사용합니다. ResNet이 1x1 patch 2 stride의 convolution이 feature map의 3/4에 대해 계산하지 않고 넘어가기 때문에, ResNet-D는 1x1 convolution 전에 2x2 average pooling layer를 추가하고 1x1 convolution의 stride를 1로 변경한 구조를 사용합니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/97b25ebc-c541-4361-a934-95f825ebc42d" width="800" height="380">
</div>
> (Top) Anti-aliased max pooling.<br>
(Bottom) BlurPool.

추가로 <a href="https://arxiv.org/abs/1904.11486" target="_blank">antialiased rect-2 blur pooling</a>를 사용했다고 합니다. Maxpooling 만을 사용하면서 aliasing이 발생하는 데 low-pass filter(blur)를 사용해 연산을 유지하며 anti-aliasing을 달성하도록 합니다.

또한 ResNet의 마지막 layer인 global average pooling layer를 attention pooling mechanism으로 대체해 multi-head QKV attention의 single layer로 구현했습니다. <a href="https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/model.py#L58" target="_blank">Github link</a>에서 attention pooling 부분 코드를 볼 수 있습니다.

#### ViT

<a href="https://arxiv.org/abs/2010.11929" target="_blank">ViT</a>는 ResNet과 달리 구조를 변경한 것은 없고 layer normalization을 추가한 것이 끝입니다. ViT 부분 코드는 <a href="https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/model.py#L195" target="_blank">Github link</a>에서 볼 수 있습니다.


#### 학습 설정
Image encoder는 <a href="https://arxiv.org/abs/1905.11946" target="_blank">EfficientNet</a> 접근 방식처럼 width, depth, resolution을 높여 연산 수를 늘린 모델을 함께 비교합니다. ResNet-50 연산의 4배, 16배, 64배인 모델 3개를 학습시켜 각각을 RN50x4, RN50x16, RN50x64로 표기합니다. ViT도 마찬가지로 3개 모델을 학습시켜 각각을 ViT-B/32, ViT-B/16, ViT-L/14로 표기합니다. ViT-L/14의 경우 더 높은 해상도로 학습한 ViT-L/14@336px로 표기한 모델을 추가 학습했으며 가장 좋은 결과를 가진다고 합니다.

모든 모델은 32 epoch 학습되며 <a href="https://arxiv.org/abs/1711.05101" target="_blank">decoupled weight decay regularization</a>이 적용된 <a href="https://arxiv.org/abs/1412.6980" target="_blank">Adam optimizer</a>와 <a href="https://arxiv.org/abs/1608.03983" target="_blank">cosine scheduler</a>를 사용했다고 합니다. 또한 학습 가능한 <a href="https://arxiv.org/abs/1805.01978" target="_blank">temperature parameter $\tau$</a>는 0.07로 초기화되어 학습되며 불안정성을 방지하기 위해 100 이상이 되는 경우 clip 했습니다. 32,768 mini batch를 사용하는데 메모리를 절악하고 학습을 가속화하기 위해 <a href="https://arxiv.org/abs/1710.03740" target="_blank">mixed-precision</a>을 사용하고 추가적인 메모리 절약을 위해 <a href="https://arxiv.org/abs/1604.06174" target="_balnk">gradient checkpointing</a>과 <a href="https://arxiv.org/abs/2005.00341" target="_blank">half-precision Adam ststistics</a>를 사용했습니다.

이후 논문 결과에서 추가로 명시하지 않은 "CLIP"으로 표시된 결과는 가장 좋은 결과를 가졌던 ViT-L/14@336px를 의미합니다.


### Text encoder
다음은 Text encoder입니다.  Text encoder는 <a href="https://arxiv.org/abs/1706.03762" target="_blank">Transformer</a>를 기반으로 <a href="https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank">GPT-2</a>의 구조를 사용합니다. CLIP이 사용한 기본 모델은 8개의 attention heads가 있는 63M parameter 12-layer 512-wide 모델입니다. transformer는 49,152 vocab 크기의 text를 <a href="https://arxiv.org/abs/1508.07909" target="_blank">lower-cased byte pair encoding(BPE) representation</a>으로 표현하며 계산 효율성을 위해 sequence lenth는 76으로 제한되었습니다.

추가로 Text encoder 또한 scaling 작업으로 여러 크기의 모델을 학습하지만 Text encoder의 capacity가 CLIP의 성능에는 큰 영향을 미치지 않았기 때문에 모델의 width를 image encoder의 width 증가에 비례하도록 계산하지만 depth는 확장하지 않았다고 합니다.

<br>

---

## 결과

### Zero-shot Transfer
Image classification task에서 zero-shot은 학습하지 않은 클래스의 이미지에 대해 unseen category로 분류하는 거나 학습하지 않은 클래스의 이름을 추가로 입력받아 수행합니다. CLIP에서는 조금 다르게 특정 데이터셋에 대한 fine-tuning 없이 task를 수행하는 zero-shot transfer로 더 넓은 의미의 zero-shot을 수행합니다.

CLIP은 (image, text) 데이터에 대해 cross entropy로 학습되기 때문에 이미지가 어떤 텍스트랑, 반대로 텍스트가 어떤 이미지랑 서로 가장 관련이 깊은지 예측이 가능합니다. 이를 사용해 Zero-shot transfer task로 Image classification을 시도합니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/ae00c1d4-2f11-4643-8fc8-2042aa287f87" width="700" height="500">
</div>
> Figure 1.(2). 입력 이미지와 클래스 이름들을 embedding해 zero-shot classification이 가능합니다.

Zero-shot classification에 대해서 위의 Figure 1.(2)가 잘 표현하고 있습니다. 모델은 pre-train된 CLIP을 사용하며 주어진 이미지를 image encoder로 embedding할 때 사용합니다. 정답이 될 클래스 이름들 또한 하나의 문장으로 취급해 이미지와 마찬가지로 pre-train된 CLIP의 text encoder로 embedding해 클래스 별 representation 값을 추출합니다. 이후 두 embedding의 cosine similarity를 계산하고 temperature parameter $\tau$로 스케일링을 한 이후, softmax를 통해 normalize합니다. 이미지와 가장 관련이 깊은 클래스의 embedding 값이 가장 높은 값, 즉 높은 유사도를 가지고 있게 되어 해당 클래스를 답으로 출력하는 방식입니다.

classification을 수행할 데이터 셋이 고정되어 있다면 클래스 값 또한 고정되어 있기 때문에 text encoder에 의해 embedding된 클래스 값 또한 변하지 않습니다. 예를 들어 ImageNet이라면 ImageNet 안의 클래스의 이름에 대해서만 text encoder를 사용해 text embedding을 생성하면 됩니다. 따라서 text embedding을 한번 생성하고 나면 이후에는 image encoder만 계산하는 방식으로 계산을 최적화했다고 합니다.

<br>

#### Prompt Engineering

Figure 1.(2)의 그림에서 설명하지 않은 것이 하나 있습니다! 클래스 값들을 CLIP의 text encoder에 넣기 전에 하나의 단계를 추가합니다. 모든 텍스트를 "A photo of a {object}" 형태로 변경해주는 prompt engineering 과정인데, 단순해 보이는 이 과정의 효과가 굉장히 컸습니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/ff0fe258-d258-4730-971d-0c2d147f8d63" width="600" height="570">
</div>
> Figure 4. Prompt engineering과 ensemble은 zero-shot 성능을 향상시킵니다.

Figure 4는 입력 텍스트의 형태를 변경하는 prompt engineering과 ensemble을 적용했을 때의 결과를 보여줍니다. ensemble은 다양한 프롬프트(“A photo of a big {label}”, “A photo of a small {label}”)를 적용해 학습한 embedding space를 ensemble하는 방식으로 평균 text embedding을 사용하며 평균 embedding을 사용하니 test time에는 계산 비용이 추가로 들지 않는 것 또한 장점이 됩니다.

prompt engineering을 사용한 경우 ImageNet 기준 1.3%의 정확도가 올라갔으며, ensemble까지 추가 사용한 경우 3.5% 정확도가 올라갔다고 합니다. Figure 4에서 RN50x16 기준 5%가 올라간 것을 볼 수 있으며, 9.9GFLOPs인 RN101가 prompt engineering+ensemble로 21.5GFLOPs인 기본 RN50x4의 성능보다도 좋아져 계산 효율까지 좋아진것을 볼 수 있습니다.

prompt engineering은 단어 다의성(polysemy) 문제를 해결하기 위해서도 필요합니다. Oxford-IIIT Pet 데이터셋에는 "Boxer" 클래스가 있습니다. Pet 데이터셋이기 때문에 여기서 "Boxer"는 복서라 불리는 강아지 종에 대한 이미지를 나타내지만 이런 정보를 가지지 못한채 "Boxer"라는 단어를 받은 text encoder는 권투 선수 의미로 단어를 받아들일 수 있게 됩니다. 저자들은 context 제공을 위해 "A Photo of a {label}, a type of pet"을 사용해 잘 작동되는 것을 확인했으며 classification 뿐만 아니라 OCR의 경우 텍스트나 숫자에 따옴표(quote)를 넣게 되면 성능이 향상되었다고 합니다.



#### Zero-shot 결과
<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/5850edb1-46e9-4f1e-a230-b9e4cbf3d942" width="600" height="170">
</div>
> Table 1. CLIP 이전 zero-shot transfer image classification 모델인 Visual N-Grams와의 결과 비교

Table 1에서 image classification 데이터셋에 대한 zero-shot transfer를 연구한 논문인 <a href="https://arxiv.org/abs/1612.09161" target="_blank">Visual N-Grams</a>과 CLIP의 성능을 비교합니다. CLIP은 Yahoo, ImageNet, SUN 데이터셋 모두에 대해서 Visual N-Grams보다 성능이 뛰어남을 볼 수 있지만, CLIP은 Visual N-Grams 때에 존재하지 않았던 transformer 기반 모델이며 10배 더 큰 데이터셋을 사용해 학습되었고 예측 당 거의 100배 더 많은 계산이 필요한 모델이기 때문에 Visual N-Grams와 직접적인 성능 차이보다 CLIP의 zero-shot 성능이 좋다 정도를 봐야 한다고 언급합니다.

CLIP은 ImageNet에서 76.2%의 정확도를 기록했으며 이 수치는 ImageNet은 supervised로 학습한 original ResNet-50의 성능과 일치하며 Top-1이 아닌 Top-5일 경우 CLIP의 정확도는 95%로 Inception-V4와 일치하는 수준입니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/820d163d-9b20-4a22-85e0-250cd98b80b7" width="650" height="630">
</div>
> Figure 5. Zero-shot CLIP과 fully supervised baseline인 ResNet-50과의 성능 비교.

더 자세한 성능을 살펴보기 위해 supervised ResNet-50과의 데이터셋 별 성능을 비교합니다.

CLIP은 전체 27개 데이터셋 중 16개 데이터셋에 대해서 ResNet보다 성능이 앞섰습니다. Zero-Shot CLIP의 성능이 현저하게 떨어지는 부분은 위성 이미지 분류(EuroSAT, RESISC45), 림프절 종양 탐지(PatchCamelyon), counting objects(CLEVRCounts), 독일 표지판 인식(GTSRB), 거리 인식(KITTI Distance)으로 전문적이고 복잡한 작업이 필요한 경우들이였습니다.

반대로 CLIP은 사람 행동이 표현된 비디오 데이터셋인 Kinetics700, UCF101에서 ResNet보다 각각 14.5%, 7.7%을 능가하는데 CLIP이 학습한 자연어가 명사 중심의 Image Classification에 비해 동사와 관련된 visual concept에 대해 더 넓게 학습했기 때문이라 저자들은 추측했습니다.

Image Classification에 대해서는 CLIP이 더 좋은 성능을 보인 경우(Stanford Cars, Food101), 비슷한 경우(OxfordPets, Birdsnap), 성능이 떨어지는 경우(Flowers102, FGVC Aircraft)로 나뉘며 데이터셋에 따라 성능 차이가 컸습니다. 이런 차이는 classification 하고자 하는 특정 object에 대해 CLIP의 데이터셋인 WIT가 가지고 있는 수와 해당 데이터셋이 가진 수, 즉 supervision 양이 다르기 때문이라고 언급되며 일반적인 object classification인 ImageNet, CIFAR10/100, STL10, PascalVOC2007에 대해서는 성능이 비교적 유사하며 모든 경우에 대해서 Zero-shot CLIP이 조금 더 좋았다고 합니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/4d45f0bb-d35f-4790-be80-13394c241622" width="700" height="400">
</div>
> Figure 21의 일부. Zero-shot CLIP의 예측 시각화.

Figure 21에서 Zero-shot CLIP의 결과와 prompt 값을 함께 확인할 수 있습니다. Figure21이 너무 큰 이미지여서 일부만 가지고 왔으며 전체 Figure에는 geolocalization, optical character recognition, facial emotion recognition, action recognition task가 포함되어 있습니다. 이미지 위에 데이터셋 이름, 정답 라벨, 모델의 정답 라벨 예측 확률이 표시되며 ground truth 라벨은 녹색, 예측이 잘못된 경우 주황색으로 표시됩니다. CIFAR-100의 뱀 사진은 화질이 좋지 않은데도 모델이 잘 예측하는 걸 볼 수 있습니다.

<br>

#### Zero vs Few
<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/f15e6a27-4b26-47f0-82d8-dddd89754a07" width="650" height="630">
</div>
> Figure 6.Zero-shot CLIP은 동일한 feature space에서 학습한 4-shot linear classifier의 평균 성능과 일치하며 BiT-M(ImageNet-21K)의 16-shot linear classifier와 거의 일치합니다.

기존 classification 모델들이 zero-shot 모델들이 아니다보니 few-shot을 이용한 기존 모델들과 CLIP의 zero-shot, few-shot에 대해 비교합니다.

few-shot은 clssification을 위한 initialize된 linear layer를 모든 모델 끝에 추가하고 클래스 당 {1, 2, 4, 8, 16}개의 이미지로 해당 layer를 학습하는 방식을 사용합니다. 새로운 layer를 모델에 추가해 학습하는 방식을 사용하다보니 layer를 학습시키기에 클래스 당 1, 2장은 굉장히 작은 숫자이기 때문에 CLIP의 few-shot(1-shot, 2-shot) 성능이 zero-shot 보다 일때보다 더 나빠지는 것을 볼 수 있습니다.

기존 classification 모델들에도 같은 방식으로 few-shot 능력을 측정했습니다. 가장 성능이 좋은 ImageNet-21K에서 학습한 BiT-M(ResNet-152x2)의 16-shot과 zero-shot CLIP의 성능이 비슷했으며 더 큰 데이터 셋인 JFT-300M로 학습한 BiT-M의 성능이 더 좋을 것이라 생각되어 사용하고 싶었지만 공개되지 않아 실험까지 이어지지 못했다고 합니다.

<br>

#### Scaling in Zero-shot

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/443ba504-ba5b-42ea-81fd-5bbfece3995b" width="650" height="430">
</div>
> Figure 9. Zero-shot CLIP의 성능은 model computing 성능에 따라 scaling됩니다.

지난 몇 년간 딥러닝에 대한 경험적 연구들에 따르면 학습에 사용되는 컴퓨터 자원 및 데이터 셋 크기가 커진다면 모델의 성능 또한 좋아졌습니다. CLIP은 Image Encoder를 변경해 컴퓨터 자원(GFLOPs)를 늘렸을 때 성능 변화에 대해 실험했으며 이 결과를 Figure 9에서 볼 수 있습니다. 36개 데이터셋에 대해 5개의 ResNet CLIP 모델의 평균 error rate를 확인했을 때 GFLOPs가 44배 증가하는 동안 log-log linear scaling trend가 있음을 발견했다고 합니다.


### Representation
모델이 학습을 통해 입력의 Representation을 얼마나 잘 만들어내는가에 대한 평가를 진행합니다. classification task에서는 모델에 linear layer를 추가해 classifier로 사용하는 방법과 end-to-end fine-tuning을 사용하는 방법이 있습니다. fine-tuning을 사용하는 것이 linear classifier를 사용하는 것보다 성능이 뛰어나지만 fine-tuning 과정에서 pre-train된 모델의 representation을 데이터셋에 맞게 변형하게 되어 평가가 어려워진다는 단점이 있습니다. 또한 CLIP의 경우 27개의 데이터셋과 baseline까지 66개의 서로 다른 모델에 대한 다양한 기술들에 대해 공정하게 평가하는 것이 어렵고 계산 비용이 크다는 이유로 linear classifier 방식을 선택했습니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/36e19b46-94be-45dd-a568-1e1896155619" width="800" height="420">
</div>
> Figure 10. state-of-the-art computer vision 모델과 CLIP의 Linear probe 성능 비교.

EfficientNet, MoCo, ResNeXt, BiT, ViT, SimCLRv2, BYOL 등 vision 모델과 CLIP 모델에 linear classifier 방식으로 Representation Learning의 성능을 실험해 Figure 10에서 결과를 볼 수 있습니다. Figure 10의 왼쪽은 ImageNet에서 좋은 성능을 보인 모델에 대해 Transfer learning으로 모델의 성능을 다시 측정한 <a href="https://arxiv.org/abs/1805.08974" target="_blank">Do Better ImageNet Models Transfer Better?</a>에서 사용한 12개 데이터 셋에 대한 평균, 오른쪽은 CLIP이 새로운 데이터셋을 추가한 총 27개 데이터셋에 대한 평균입니다.

CLIP의 작은 크기의 image encoder인 ResNet-50과 ResNet101의 경우 ImageNet-1K에서 학습된 다른 ResNet(BiT-S)들보다 성능이 뛰어나지만 ImageNet-21K에서 학습된 ResNet(BiT-M)보다는 성능이 낮습니다. CLIP의 image encoder 중 가장 큰 모델인 ResNet-50x64는 점수와 계산 효율성 모두에서 당시 SOTA 모델인 a Noisy Student EfficientNet-L2보다 성능이 좋았으며 Transformer 구조를 가진 ViT encoder가 ResNet encoder보다 더 좋은 계산 효율성과 성능을 보였습니다.


### Robustness
Image Classification의 대표적인 데이터셋인 ImageNet에 대해서 딥러닝 모델은 사람의 성능을 뛰어넘었다고 하지만 연구들의 대부분이 ImageNet에서 학습된 모델을 사용해 평가했으며 이런 모델들은 다른 데이터셋에서 성능이 크게 저하됩니다. CLIP은 매우 큰 데이터셋인 WIT에서 학습되었으며 zero-shot이 가능하니 이를 이용해 다른 각도에서 모델을 평가합니다.

Robustness 연구를 한 <a href="https://arxiv.org/abs/2007.00644" target="_blank">Measuring Robustness to Natural Distribution Shifts in Image Classification</a>는 effective robustness와 relative robustness 모두에 대해서 개선할 수 있어야 한다고 제안합니다. effective robustness는 학습한 데이터 셋에 대한 in-distribution 성능과 그 외 데이터셋인 out-of-distribution에 대한 예측되는 정확도를 초과하는 지 측정합니다. 데이터셋에 변경되어 distribution shift가 되는 상황에서 예측되는 정확도보다 높게 나올 때 effective robustness하다고 할 수 있습니다. relative robustness는 in-distribution과 무관하게 out-of-distribution 성능 개선에 대해서만 측정합니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/8e0f83d8-da1d-41de-aa7b-ecc50da30fa1" width="600" height="560">
</div>
> Figure 15. Few-shot CLIP은 기존 ImageNet 모델보다 effective robustness가 증가하지만 Zero-shot CLIP보다는 감소합니다.

robustness를 파악하기 위해 ImageNet 데이터에 대해 0-shot, 1-shot, 2-shot, 4-shot, ..., 128-shot부터 fully supervised(all)까지 성능을 시각화합니다. ImageNet 데이터셋의 학습 데이터 양을 최소화하면 relative robustness가 감소하는 대신 effective robustness가 증가한다고 합니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/e8c536a1-b1fd-458f-b225-b457b03cf478" width="800" height="330">
</div>
> Figure 13. Zero-shot CLIP은 ResNet101보다 distribution shift에 대해 robust합니다.

CLIP의 effective robustness에 대해서는 Figure 13에서 더 직관적으로 확인할 수 있습니다. ImageNet 뿐만 아니라 관련된 다른 데이터셋인 ImageNetV2, ImageNet-R, ObjectNet, ImageNet Sketch, ImageNet-A에 대해서 성능 변화을 확인해 distribution shift 상황을 실험했습니다. Zero-shot CLIP은 기존 모델에 비해 데이터셋이 변화되어도 성능이 크게 낮아지지 않아 effective robustness를 크게 향상시켜 distribution shift 발생에도 ResNet101과 비교해 정확도 사이 간격을 최대 74.4%까지 줄였습니다.


### Limitation
논문에는 CLIP의 겪고 있는 한계에 대해서 정리해놓았습니다.

우선 CLIP은 scaling으로 더 많은 데이터셋, 더 많은 컴퓨팅(큰 모델)로 꾸준히 성능을 개선했지만 zero-shot CLIP이 전체적인 state-of-the-art에 도달하기 위해서는 약 100배의 컴퓨팅 자원이 필요할 것으로 예상한다고 합니다. CLIP의 계산 및 데이터 효율성 향상에 대한 추가 연구가 필요합니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/a1e3c28d-a505-4b4d-b616-7e68f0223f70" width="600" height="300">
</div>
> Table 14. 5개 데이터셋에 대한 OCR 성능.

CLIP의 pre-train 데이터셋에 포함될 가능성이 높은 ImageNet과 같은 단순 Classification에 대해서는 Zero-shot CLIP의 성능이 좋지만, 반대로 pre-train 데이터셋에 포함될 가능성이 낮은 추상적이고 체계적인 task에 대해서는 CLIP의 성능은 보장되지 않습니다. 예시로 OCR task에서 CLIP은 단순한 logistic regression basline보다 성능이 낮아 MNIST 숫자에 대해 오직 88% 정확도를 달성했으며 관련 결과 테이블을 Table 14에서 볼 수 있습니다.

저자들은 nearest-duplicate nearset neighborhood retrieval을 통해 pre-trian 데이터셋에서 MNIST와 유사한 이미지가 거의 없음을 확인했으며 이는 CLIP이 딥러닝 모델의 일반화에 약한 brittle generalization이라는 근본적인 문제를 해결하지 못함을 시사한다고 합니다.

다른 문제로는 Zero-shot evaluate가 있습니다. CLIP은 Zero-shot 평가를 위해 기존 supervised 데이터셋을 사용했으며 이는 평가 데이터셋으로 구축된 것이 아니기 때문에 적합하지 않으며 fine-tuning 방식에서는 validation set의 많은 샘플들에서도 반복적으로 학습되어 Zero-shot 평가가 제대로 이루어지지 않았다고 합니다. Zero-shot transfer 기능을 평가하기 위한 새로운 벤치마크의 필요성을 언급합니다. 또한 CLIP의 학습이 인터넷의 다양한 source들을 가지고 오기 때문에 사회적 편견(인종, 나이)을 학습하게 된다는 문제가 있다고 합니다.

<br>

---

<br>

CLIP 리뷰가 끝이 났습니다!	끝까지 봐주셔서 감사합니다:satisfied:

긴 논문이지만 GPT에서 사용하던 web scale 데이터셋을 이미지 관련 모델에 적용한 첫번째 모델이다보니 모델 자체에 관한 내용보다는 task에 대한 결과 분석 내용이 훨씬 더 많았던 거 같아요. 리뷰에서 넣을까 말까 고민하다 뺀 내용이 CLIP의 한계에서 언급된 부분으로 "7. Broader Impacts" 내용인데 인종, 나이, surveillance task처럼 사회적 민감성이 있는 부분들에 대해서 모델을 분석한 부분이였습니다. 해당 부분에 대해서는 <a href="https://arxiv.org/abs/2103.00020" target="_blank">CLIP 논문</a>을 확인해주시면 감사하겠습니다:wink:
