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

(image, text) 페어 데이터셋은 각각 Image, Text encoder에 입력되어 image, text embedding을 계산하게 됩니다. 이를 linear projection 시켜 $[I_1, ..., I_N]$과 $[T_1, ..., T_N]$를 구합니다. 각각의 페어 데이터에 해당하는 $(I_i, T_i)$의 값($N$개)은 cosine similarity가 최대가 되도록, 그 외의 페어($N^2 - N$)에 대해서는 최소가 되도록 학습하는 방식입니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/fbe7d60f-e688-4415-802f-5a4f7fbdea47" width="500" height="500">
</div>
> Figure 3. CLIP 학습을 위한 pseudo code

ConVIRT에서는 INFONCE loss를 사용했지만 CLIP에서는 cross entropy loss를 사용하는 것이 차이점입니다. 논문에서는 symmetric cross entropy를 사용한다고 언급되는데 image 관점에서의 loss인 loss_i와 text 관점에서의 loss인 loss_t 모두에 대해 CE loss를 계산하기 때문이라고 이해했습니다.

Figure 3에서 CLIP 구현에 대한 pseudo 코드를 확인할 수 있습니다.

image, text encoder로 각각 image embedding(I_f), text embedding(T_f)을 생성합니다. 이후 `np.dot`으로 행렬곱을 계산해 linear  projection을 수행합니다. batch 개수(n)만큼 linear embedding이 생성되며 생성된 두 embedding을 행렬곱을 통해 $n \times n$ 행렬로 생성합니다.

joint multimodal embedding 과정에서 `L2_normalize`를 수행했기 때문에 `np.dot`의 결과가 [-1, 1] 범위로 나오게 되고 이 값을 cosine similarity 값이 됩니다. L2_normalize로 값의 범위를 제한했기 때문에 `np.exp(t)`로 다시 [-$\infty$, $\infty$]로 scaling해주며 t 또한 학습 가능한 learnable parameter입니다.

하나의 이미지를 설명하는 하나의 텍스트가 (image, text) 형태로 batch에 들어가기 때문에 같은 index에 존재하는 이미지와 텍스트가 실제 정답 값이 되기 때문에 `np.arange(n)`으로 라벨을 생성합니다. `np.arange(n)`으로 생성한 라벨 값과 `cross_entropy_loss`를 계산해 loss를 계산합니다.

<br>

---

## Model(Encoder)
contrastive learning에 대해 살펴봤으니 이제 CLIP이 사용하는 Image, Text encoder에 대해서 알아보겠습니다.

### Image encoder
우선 Image encoder입니다! Image encoder는 ResNet과 ViT를 사용해 비교했습니다.

#### ResNet-50

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/e7bcd486-7a9d-47f6-bf96-7eac5668101c" width="220" height="400">
</div>
> ResNet-D의 구조

<a href="https://arxiv.org/abs/1512.03385" target="_blank">ResNet-50</a>를 기반으로 <a href="https://arxiv.org/abs/1812.01187" target="_blank">ResNet-D</a>를 사용합니다. ResNet-D는 1x1 patch 2 stride의 convolution이 feature map의 3/4에 대해 계산하지 않고 넘어가기 때문에, 1x1 convolution 전에 2x2 average pooling layer를 추가하고 1x1 convolution의 stride를 1로 변경했습니다.

<br>

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/97b25ebc-c541-4361-a934-95f825ebc42d" width="800" height="380">
</div>
> (Top) Anti-aliased max pooling.<br>
(Bottom) BlurPool.

<a href="https://arxiv.org/abs/1904.11486" target="_blank">antialiased rect-2 blur pooling</a>를 사용했다고 합니다. Maxpooling 만을 사용하면서 aliasing이 발생하는 데 low-pass filter(blur)를 사용해 연산을 유지하며 anti-aliasing을 달성하도록 합니다.

```python
# https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/model.py#L58

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
```

또한 ResNet의 마지막 layer인 global average pooling layer를 attention pooling mechanism으로 대체해 multi-head QKV attention의 single layer로 구현했습니다.

#### ViT

```python
# https://github.com/openai/CLIP/blob/dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1/clip/model.py#L195

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

```

<a href="https://arxiv.org/abs/2010.11929" target="_blank">ViT</a>를 기반으로 layer normalization을 추가했다고 합니다.

#### 학습 설정
Image encoder는 <a href="https://arxiv.org/abs/1905.11946" target="_blank">EfficientNet</a> 접근 방식처럼 with, depth, resolution을 높여 연산 수를 늘린 모델을 함께 비교합니다. ResNet-50 연산의 4배, 16배, 64배인 모델 3개를 추가 학습했으며 각각을 RN50x4, RN50x16, RN50x64로 표기합니다. ViT도 마찬가지로 3개 모델을 추가학습해 각각을 ViT-B/32, ViT-B/16, ViT-L/14로 표기합니다. ViT-L/14의 경우 더 높은 336 pixel 해상도로 학습한 ViT-L/14@336px로 표기한 모델을 추가 학습했으며 가장 좋은 결과를 가진다고 합니다.

모든 모델은 32 epoch 학습되며 <a href="https://arxiv.org/abs/1711.05101" target="_blank">decoupled weight decay regularization</a>이 적용된 <a href="https://arxiv.org/abs/1412.6980" target="_blank">Adam optimizer</a>와 <a href="https://arxiv.org/abs/1608.03983" target="_blank">cosine scheduler</a>를 사용했다고 합니다. 또한 학습 가능한 <a href="https://arxiv.org/abs/1805.01978" target="_blank">temperature parameter $\tau$</a>는 0.07로 초기화되어 학습되며 불안정성을 방지하기 위해 100 이상이 되는 경우 clip 했습니다. 32,768 mini batch를 사용하는데 메모리를 절악하고 학습을 가속화하기 위해 <a href="https://arxiv.org/abs/1710.03740" target="_blank">mixed-precision</a>을 사용하고 추가적인 메모리 절약을 위해 <a href="https://arxiv.org/abs/1604.06174" target="_balnk">gradient checkpointing</a>과 <a href="https://arxiv.org/abs/2005.00341" target="_blank">half-precision Adam ststistics</a>를 사용했습니다.

이후 논문 결과에서 추가로 명시하지 않은 "CLIP"으로 표시된 결과는 가장 잘 수행되었던 ViT-L/14@336px를 의미합니다.


### Text encoder
다음은 Text encoder입니다.  Text encoder는 <a href="https://arxiv.org/abs/1706.03762" target="_blank">Transformer</a>를 기반으로 <a href="https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" target="_blank">GPT-2</a>의 구조를 사용합니다. CLIP이 사용한 기본 크기는 8개의 attention heads가 있는 63M parameter 12-layer 512-wide 모델입니다. transformer는 49,152 vocab 크기의 text를 <a href="https://arxiv.org/abs/1508.07909" target="_blank">lower-cased byte pair encoding(BPE) representation</a>으로 표현합니다(Sennrich et al., 2015). 계산 효율성을 위해 sequence lenth는 76으로 제한되었습니다.

text encoder의 capacity가 CLIP의 성능에는 큰 영향을 미치지 않았기 때문에 모델의 width를 image encoder의 width 증가에 비례하도록 계산하지만 depth는 확장하지 않았다고 합니다.

<br>

---

## 결과

### Task Learning(zero-shot)

<div>
  <img src="https://github.com/solee328/solee328.github.io/assets/22787039/ae00c1d4-2f11-4643-8fc8-2042aa287f87" width="700" height="500">
</div>

모델 학습 이후 classification에 사용할 시에는 클래스 이름을 임베딩해 zero-shot linear classifier를 만들어 사

<br>

### Representation Learning


---
