---
layout: post
title: MUNIT(2) - 논문 구현
# subtitle:
categories: gan
tags: [gan, munit, multimodal, unsupervised, 생성 모델, 논문 구현]
# sidebar: []
use_math: true
---

저번 글인 <a href="https://solee328.github.io/gan/2023/04/19/munit_paper.html" target="_blank">MUNIT(1) - 논문 리뷰</a>에 이은 MUNIT 코드 구현입니다! 논문의 공식 코드는 <a href="https://github.com/NVlabs/MUNIT" target="_blank">github</a>에서 제공되고 있습니다.
<br><br>

---

## 1. 데이터셋
논문에서 사용된 데이터 셋 중에서 Animal Translation dataset을 사용해 동물 간의 변환을 구현하는 것으로 목표를 잡았습니다. 이유는 하나입니다. 귀여우니까요 :see_no_evil: :dog: :cat: :tiger:

하지만 MUNIT의 <a href="https://github.com/NVlabs/MUNIT/issues/22" target="_blank">issue</a>에서 ImageNet 데이터셋의 저작권 때문에 사용한 데이터는 공개되지 않는다고 합니다. 유사한 데이터셋을 찾아보다 <a href="https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq" target="_blank">Animal-Faces-HQ(AFHQ)</a>를 찾게 되어 AFHQ 데이터셋을 사용했습니다.

### 논문
UNIT의 <a href="https://github.com/mingyuliutw/UNIT/issues/27" target="_blank">issue</a>에서 데이터 처리 방법을 알 수 있었습니다. VGG를 이용해 Template matching을 통해 개와 고양이 품종의 머리 부분의 이미지를 찾았으며 각 카테고리 별로 이미지는 1000장 ~ 10000장이였다 합니다. 또한 종이 다양하게 섞여있도록 데이터셋을 만들었는데 예시로 고양이 카테고리 안에는 이집트 고양이, 페르시안 고양에, 범무늬 고양이 등 다양한 종을 섞어 사용했다 합니다.

### AFHQ
AFHQ는 StarGAN-v2에서 공개한 데이터로 고양이, 강아지, 야생의 동물들 3개의 도메인에 각각 약 5000장의 이미지가 있는 데이터셋입니다. AFHQ-v2도 공개되어 있는데 기존의 nearest neighbor downsampling이 아닌 Lanczos resampling을 이용해 더 좋은 퀄리티의 AFHQ 데이터셋으로 두 데이터셋 모두 쉽게 다운 받을 수 있어 이용하기 용이했습니다.

AFHQ 데이터셋을 사용해 강아지와 고양이 두 도메인 변환을 시도해보겠습니다!

### AFHQ 처리
AFHQ는 'train', 'val'로 폴더가 나뉘어져 있으며 각각의 폴더 내부에 'dog', 'cat', 'wild' 폴더가 존재합니다.

```python
class AFHQ(Dataset):
    def __init__(self, path, target, transforms=None):
        self.path_dataset = path + '\\' + target
        self.transforms = transforms
        self.images = glob(self.path_dataset + '\\*.jpg')

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        return image

    def __len__(self):
        return len(self.images)


transform = transforms.Compose([
  transforms.Resize((256, 256)),
  transforms.RandomHorizontalFlip(0.5),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

DATA_PATH = 'E:\\DATASET\\afhq\\train'

dataset_dog = AFHQ(path=DATA_PATH, target='dog', transforms=transform)
dataset_cat = AFHQ(path=DATA_PATH, target='cat', transforms=transform)

dataloader_dog = DataLoader(dataset=dataset_dog, batch_size=batch_size, shuffle=True)
dataloader_cat = DataLoader(dataset=dataset_cat, batch_size=batch_size, shuffle=True)
```

dataloader_dog와 dataloader_cat에서 이미지 한 장씩을 확인해보았습니다.
```
데이터 이미지
```


---

## 2. 모델
<div>
  <img src="/assets/images/posts/munit/code/fig3.png" width="700" height="250">
</div>
> **Fig.3** auto-encoder 구조

<a href="https://arxiv.org/abs/1804.04732" target="_blank">MUNIT 논문</a>의 Fig.3과 **B. Training Details**에서 간략한 모델 구조를 확인할 수 있습니다.

- Generator architecture
  - Content encoder: $\mathsf{c7s1-64, d128, d256, R256, R256, R256, R256}$
  - Style encoder: $\mathsf{c7s1-64, d128, d256, d256, d256, GAP, fc8}$
  - Decoder: $\mathsf{R256, R256, R256, R256, u128, u64, c7s1-3}$
- Discriminator architecture : $\mathsf{d64, d128, d256, d512}$

표기법이 어디선가 봤다 했더니 <a href="https://solee328.github.io/gan/2023/02/28/cyclegan_code.html#h-22-generator" target="_blank">CycleGAN 코드 구현</a> 글이네요. CycleGAN과 유사한 표기와 convolution을 사용합니다. 추가된 것으로는 Global average pooling을 의미하는 $\mathsf{GAP}$과 fully connected layer를 의미하는 $\mathsf{fck}$가 있으며 Decoder에 사용되는 $\mathsf{uk}$의 경우 CycleGAN과 표기는 같으나 모듈 차이가 존재합니다.

<br>
MUNIT에서 정의한 모듈 내용은 다음과 같습니다.
- $\mathsf{c7s1-k}$ : k filter와 1 stride를 가진 7x7 Convolution - InstanceNorm - ReLU
- $\mathsf{dk}$ : k filter와 2 stride를 가진 4x4 Convolution - InstanceNorm - ReLU
- $\mathsf{Rk}$ : [k filter를 가진 3x3 Convolution - InstanceNorm - ReLU] x 2 (Residual block)
- $\mathsf{uk}$ : nearest-neighbor upsampling - k filter와 1 stride를 가진 5x5 convolution - LayerNorm - ReLU
- $\mathsf{GAP}$ : Global Average Pooling
- $\mathsf{fck}$ : k filter를 가진 fully connected layer

<br>

Generator와 Discriminator에서 사용하는 $\mathsf{c7s1-k}$, $\mathsf{dk}$, $\mathsf{uk}$는 `class xk`를 만들어 쉽게 호출할 수 있도록 구현했습니다.

```python
class xk(nn.Module):
    def __init__(self, name, in_feature, out_feature, norm_mode='in'):
        super(xk, self).__init__()

        model = []
        norm = [nn.InstanceNorm2d(out_feature)]
        relu = [nn.ReLU()]

        if name == 'c7s1':
            conv = [nn.Conv2d(in_feature, out_feature, kernel_size=7, stride=1, padding=3, padding_mode='reflect')]

        elif name == 'dk':
            conv = [nn.Conv2d(in_feature, out_feature, kernel_size=4, stride=2, padding=1, padding_mode='reflect')]

        elif name == 'uk':
            conv = []
            conv += [nn.Upsample(scale_factor=2, mode='nearest')]
            conv += [nn.Conv2d(in_feature, out_feature, kernel_size=5, stride=1, padding=2)]

            norm = [LayerNorm(out_feature)]

        model += conv
        model += norm
        model += relu

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
```

<br>

$\mathsf{Rk}$는 이전 논문 구현글에서 사용했던 Residual block 코드에서 norm_mode가 추가되어 어떤 normalization을 사용할 지 인자로 선택할 수 있도록 구현했습니다. in은 Instance Normalization으로 이전과 동일하지만 adain으로 불리는 Adaptive Instace Normalization이 추가되었습니다. Decoder에서만 사용되는 normalization으로 Decoder에서 더 자세하게 설명하겠습니다.

```python
class Residual(nn.Module):
    def __init__(self, in_feature, out_feature, norm_mode='in'):
        super(Residual, self).__init__()

        if norm_mode == 'in':
            norm = nn.InstanceNorm2d(out_feature)
        elif norm_mode == 'adain':
            norm = AdaptiveInstanceNorm2d()

        conv = nn.Conv2d(in_feature, out_feature, kernel_size=3, padding=1, padding_mode='reflect')
        relu = nn.ReLU()

        model = []
        for _ in range(2):
            model += [conv, norm, relu]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)
```

생성 모델에 해당하는 Content Encoder, Style Encoder, Decoder와 판별 모델인 Discriminator까지 하나하나 살펴보겠습니다!



### Content Encoder
<div>
  <img src="/assets/images/posts/munit/code/content_encoder.png" width="500" height="280">
</div>
> Fig.3에서 표현된 Content Encoder 부분

Content Encoder는 이미지를 입력받아 이미지의 내용을 담고 있는 Content code를 만드는 것이 목적입니다. 크게 Down-sampling, Residual Blocks 2단계로 이루어져 있음을 볼 수 있습습니다.

Down-sampling 부분은 $\mathsf{c7s1-64}$, $\mathsf{d128}$, $\mathsf{d256}$으로 위의 `class xk`로 객체를 만들었습니다. Residual Blocks는 $\mathsf{R256}$, $\mathsf{R256}$, $\mathsf{R256}$, $\mathsf{R256}$으로 `class Residual`로 객체를 만들었습니다. Content Encoder에서는 normalization으로 Instance Normalization을 사용하므로 norm_mode 인자 값으로 'in'을 넣어주었습니다.


```python
class ContentEncoder(nn.Module):
    def __init__(self):
        super(ContentEncoder, self).__init__()

        self.model = nn.Sequential(
            # Down-sampling
            xk('c7s1', 3, 64),
            xk('dk', 64, 128),
            xk('dk', 128, 256),

            # Residual Blocks
            Residual(256, 256, norm_mode='in'),
            Residual(256, 256, norm_mode='in'),
            Residual(256, 256, norm_mode='in'),
            Residual(256, 256, norm_mode='in')
        )

    def forward(self, x):
        return self.model(x)
```


### Style Encoder

<div>
  <img src="/assets/images/posts/munit/code/style_encoder.png" width="500" height="280">
</div>
> Fig.3에서 표현된 Style Encoder 부분

Style Encoder는 이미지를 입력으로 받아 이미지의 스타일을 나타내는 Style code를 출력하는 것이 목표입니다. Down-sampling, Global pooling, Fully connected layer 단계로 이루어져 있습니다.

Style Encoder에는 Global Average Pooling가 구조로 포함되어 있습니다.
JINSOL KIM님의 <a href="https://gaussian37.github.io/dl-concept-global_average_pooling/" target="_blank">Global Average Pooling</a>을 참고했습니다.

스타일 코드의 차원 : 8


```python
class StyleEncoder(nn.Module):
    def __init__(self):
        super(StyleEncoder, self).__init__()

        self.model = nn.Sequential(
            # Down-sampling
            xk('c7s1', 3, 64),
            xk('dk', 64, 128),
            xk('dk', 128, 256),
            xk('dk', 256, 256),
            xk('dk', 256, 256)
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # FC
        self.fc = nn.Linear(256, 8)

    def forward(self, x):
        x = self.model(x)
        x = self.gap(x)
        x = self.fc(x.flatten(start_dim=1))

        return x
```

### Decoder

<div>
  <img src="/assets/images/posts/munit/code/decoder.png" width="550" height="350">
</div>


upsampling과 convolution이 번갈아 나옴.
AdaIN 논문에서도 등장하는 내용으로 checker-board effect를 감소시키기 위해 decoder의 pooling layer를 nearest-up sampling 방식으로 교체함
cycleGAN에서 convtranspose2d를 사용하던 것과 차이가 남.
residual block에 MLP로 인해 학습된는 parameter인 AdaIN 사용
기존 AdaIN은 고정 값이지만 MUNIT에서는 MLP에 의해 생성된다.
decoder에는 AdaIN

```python
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            # Residual
            Residual(256, 256, norm_mode='adain'),
            Residual(256, 256, norm_mode='adain'),
            Residual(256, 256, norm_mode='adain'),
            Residual(256, 256, norm_mode='adain'),

            # Upsampling
            xk('uk', 256, 128),
            xk('uk', 128, 64),

            # c7s1-3
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
```

```python
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self):
        super(AdaptiveInstanceNorm2d, self).__init__()

        self.eps = 1e-5
        self.y_mean = None
        self.y_std = None

    def forward(self, x):
        assert self.y_mean is not None and self.y_std is not None, "Set AdaIN first"

        n, c, h, w = x.size()

        x_mean = x.view(n, c, -1).mean(dim=2).view(n, c, 1, 1).expand(x.size())
        x_var = x.view(n, c, -1).var(dim=2) + self.eps
        x_std = x_var.sqrt().view(n, c, 1, 1).expand(x.size())

        y_mean = self.y_mean.view(n, c, 1, 1).expand(x.size())
        y_std = self.y_std.view(n, c, 1, 1).expand(x.size())

        norm = y_std * ((x - x_mean) / (x_std + self.eps)) + y_mean
        return norm
```


```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2048)
        )

    def forward(self, x):
        x = self.model(x)
        return x
```

```python
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
```


### Discriminator
Discriminator는 Pix2PixHD의 Multi-scale discriminator를 사용합니다. 고해상도의 이미지를 처리하기 위해 더 깊은 네트워크 또는 더 큰 convolution kernel을 사용해하나 두 가지 방법 모두 네트워크 용량을 증가시키고 잠재적으로 overfitting을 유발할 수 있으며 더 큰 메모리 공간을 필요로 한다는 단점을 언급하며 Pix2PixHD에서는 이 문제를 해결하기 위해 multi-scale discriminator를 제안합니다.

Multi scale discriminator는 말그대로 여러 개의 판별 모델을 사용하는 방법입니다. 네트워크 구조(PatchGAN)은 동일하지만 서로 다른 크기의 이미지에서 동작하는 판별 모델을 사용합니다. 원본 이미지 크기에서 동작하는 $D_1$, 원본 이미지의 높이, 너비가 절반이 된 이미지에서 동작하는 $D_2$, 원본 이미지의 높이, 너비가 1/4가 된 이미지에서 동작하는 $D_3$를 사용하며 이때 모든 판별 모델의 구조는 동일합니다.

정말 이미지 크기만 다르게 입력으로 넣게 되면서 $D_1$, $D_2$, $D_3$의 receptive field 크기가 달라지 되는데 가장 큰 receptive field를 가진 $D_1$은 이미지를 전체적으로 보면서 생성 모델이 일관된 이미지를 생성하도록 유도할 수 있으며 가장 작은 $D_3$는 생성 모델이 더 디테일을 생성할 수 있도록 유도할 수 있습니다.

Discriminator의 구조는 <a href="https://solee328.github.io/gan/2023/02/28/cyclegan_code.html#h-23-discriminator" target="_blank">CycleGAN - 코드구현의 Discriminator</a>와 광장히 유사합니다. 0.2 slope인 LeakyReLU와 InstanceNorm을 사용하며 convolution의 channel, kernel size, stride, padding 크기 모두 일치합니다. 하지만 모델 마지막의 convolution에 차이가 있습니다.

CycleGAN, Pix2Pix에서는 ConvTranspose2d를 사용해 receptive field 크기를 70x70으로 맞춘 70x70 PatchGAN을 사용했습니다. 하지만 MUNIT에서는 receptive field 크기가 논문에 언급된 것이 없었고 Multi scale discriminator를 사용해 다양한 receptive field를 가진 여러 $D$를 사용하므로 굳이 Convtranspose2d를 사용해 70x70으로 맞출 필요가 사라졌습니다.

따라서 MUNIT에서는 Convtranspose2d를 사용하지 않고 Conv2d로 channel 수를 512에서 1로 만듭니다. Discriminator의 구조를 아래 코드에서 확인하실 수 있습니다.

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.D1 = self.set_model()
        self.D2 = self.set_model()
        self.D3 = self.set_model()

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def set_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=1)
        )

        return model

    def forward(self, x):
        out_d1 = self.D1(x)
        x = self.downsample(x)
        out_d2 = self.D2(x)
        x = self.downsample(x)
        out_d3 = self.D3(x)

        return [out_d1, out_d2, out_d3]
```

이미지 크기는 nn.Avgpool2d로 바꿔주며 위 코드의 경우 256 x 256 크기의 이미지가 forward의 x로 들어오면 $D_1$, $D_2$, $D_3$마다 receptive field 크기는 각각 ~~~~가 됩니다.
<br><br>


### 전체 모델

```python
class MUNIT(nn.Module):
    def __init__(self):
        super(MUNIT, self).__init__()

        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder()
        self.mlp = MLP()
        self.decoder = Decoder()  # Generator
        self.discriminator = Discriminator()

        self.gen_params = list(self.content_encoder.parameters()) + list(self.style_encoder.parameters()) + list(self.mlp.parameters()) + list(self.decoder.parameters())
        self.dis_params = list(self.discriminator.parameters())

    def encode(self, x):
        content = self.content_encoder(x)
        style = self.style_encoder(x)
        return content, style

    def decode(self, content, style):
        param = self.mlp(style)
        self.set_adain(param)

        return self.decoder(content)

    def discriminate(self, x):
        return self.discriminator.forward(x)

    def loss_gan(self, results, target):
        loss = 0

        for result in results:
            loss += torch.mean((result - target) ** 2)

        return loss

    def set_adain(self, param):
        cnt = 0
        for m in self.decoder.modules():
            if m.__class__.__name__ == 'AdaptiveInstanceNorm2d':
                m.y_mean = param[:, cnt*256:(cnt+1)*256]
                m.y_std = param[:, (cnt+1)*256:(cnt+2)*256]
                cnt += 2
```


---

## 3. Loss

loss 별 lambda 값

### Adversarial Loss
LSGAN?


### Bidirectional Reconstruction Loss
- Image reconstruction
- Latent reconstruction


### Generator


### Discriminator


### 추가
512 x 512 이상의 이미지에서 효과가 있다고 합니다. config를 통해 데이터셋마다 적용 여부를 확인할 수 있었는데 cityscape 데이터셋에만 적용했음을 확인할 수 있었습니다.


---

## 4. 학습

### scheduler
```python
scheduler_gen = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=8, gamma=0.5)
scheduler_dis = torch.optim.lr_scheduler.StepLR(optimizer_dis, step_size=8, gamma=0.5)
```


### 학습

---

## 5. 결과


### 결과 1


### 결과 2


### 결과 3