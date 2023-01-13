---
layout: post
title: Pix2Pix(2) - 논문 구현
# subtitle:
categories: gan
tags: [gan, pix2pix, 생성 모델, 논문 구현]
# sidebar: []
use_math: true
---

pix2pix의 논문 구현 글로 논문의 내용을 따라 구현했습니다.<br>
공식 코드로는 논문에 나온 <a href="https://github.com/phillipi/pix2pix" target="_blank">phillipi/pix2pix</a>와 PyTorch로 pix2pix와 CycleGAN을 구현한 <a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix" target="_blank">junyanz의 pytorch-CycleGAN-and-pix2pix</a>가 있습니다.<br>
데이터셋 설명 모델 설명을 코드와 함께 설명해드리겠습니다!

## 1. 데이터 셋

### 1.1. 논문 데이터셋

논문에서는 총 8개의 데이터셋을 사용합니다. 이중에서 10,000장 이하의 작은 데이터셋은 3개가 있습니다.
1. Cityscapes labels $\rightarrow$ photo / 2975장
2. Architectural labels $\rightarrow$ photo / 400장
3. Maps $\leftrightarrow$ aerial photograph / 1096장

3가지 데이터셋 모두 random jitter와 mirroring을 사용해 200 epoch만큼 학습했다고 합니다. random jitter와 mirroring 모두 data augmentation을 위해 사용합니다.

random jitter로는 이미지를 286 x 286으로 resize한 다음 256 x 256으로 random cropping해 사용하며 <a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/125" target="_blank">issues</a>를 보니 286 x 286으로 지정한 특별한 이유는 없다고 답변이 되어 있었습니다. random crop을 위해 원하는 크기보다 적당히 큰 정도면 될 것 같아요!

mirroring은 flip을 의미합니다. augmentation의 대표적인 방법으로 데이터셋에 따라 horizontal flip과 vertical flip을 추가했다 이해했습니다.

### 1.2. 사용 데이터셋
pix2pix 관련 프로젝트들은 <a href="https://phillipi.github.io/pix2pix/" target="_blank">Image-to-Image Translation with Conditional Adversarial Nets</a>에서 확인하실 수 있습니다.

'#edges2cats'나 'Interactive Anime', 'Suggestive Drawing'처럼 스케치를 색칠해주거나 특정 물체로 완성해주는 프로젝트들은 많아 보이는데 특정 물체를 마치 캐리커쳐처럼 스케치해주는 프로젝트는 보이지 않아 스케치 프로젝트를 하기로 정했습니다.

pix2pix는 pair 데이터를 사용하는데 1000장 이상의 스케치와 관련된 pair 데이터셋을 찾기 힘들었습니다... 대신 장수가 적더라도 데이터 퀄리티가 마음에 들었던 <a href="http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html" blank="_blank">CUHK Face Sketch Database(CUFS)</a>를 발견했습니다.

```
예시 사진
```
데이터 셋은 Honk Kong Chinese University의 학생들로 이루어진 CUHK 데이터 셋(188장), AR database(123장), XM2VTS database(295장)로 이루어져 있으며 3종류의 얼굴 데이터 셋에서 각각의 얼굴을 스케치한 이미지로 (사진, 스케치) pair 되어 있습니다. 3종류의 데이터셋 모두 스케치 파일을 다운받을 수 있는데 원본 사진의 경우 CUHK만 다운 가능하며 AR과 XM2VTS는 메일을 통해 신청하거나 돈을 내야 데이터셋을 사용할 수 있습니다. 추가로 < a href="http://mmlab.ie.cuhk.edu.hk/archive/cufsf/" target="_blank">FERET</a> 데이터셋을 이용한 스케치 파일을 구할 수 있지만 AR과 마찬가지로 원본 이미지는 메일을 통해 신청해 받을 수 있습니다.


```
사진 전처리
```
저는 바로 원본 사진 이미지와 스케치 이미지를 다운받을 수 있는 CUHK 데이터셋만을 이용했습니다. 총 188장의 이미지로 여성 54장, 남성 134장으로 이루어져 있습니다. 원본 사진 이미지의 경우 파란색 배경을 없애고 싶고 배경 제거 작없을 진행했고 스케치 이미지의 경우 이미지 상단에 원본 이미지의 파일 명의 씌여있는 걸 없애고 싶어 이미지 크롭 작업을 진행했습니다.

```
사진 transform
```
```python
transform = transforms.Compose([
    transforms.Resize((572, 572)),
    transforms.RandomCrop((512, 512)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset_CUHK = CUHK(photo_path=data_path + '/photo',
                    sketch_path=data_path + '/sketch',
                    transform=transform)

sampler = RandomSampler(dataset_CUHK)

dataloader = DataLoader(dataset=dataset_CUHK,
                        batch_size=batch_size,
                        sampler=sampler)
```
데이터셋은 논문에서 사용한 random jitter와 mirroring을 사용했습니다. 논문에서는 286 x 286 resize $\rightarrow$ 256 x 256 randomcrop이였지만 저는 572 x 572 $\rightarrow$ 512 x 512 randomcrop을 사용해 random jitter를 RandomHorizontalFlip을 통해 mirroring을 구현했습니다.


## 2. 모델
Generator와 Discriminator의 자세한 구조는 논문의 Appendix에서 확인할 수 있습니다.

$ck$는 k개의 필터를 가진 Convolution-BatchNorm-ReLU 레이어를 의미하고 $cdk$는 k개의 필터를 가진 Convolution-BatchNorm-Dropout-ReLU 레이어를 의미하며 이때 dropout rate는 50%입니다. 모든 convolution들은 stride 값은 2이고 4x4의 filter를 사용합니다. encoder(generator)와 discriminator에서 convolution으로 downsample되는 지수는 2이고 decoder(generator)에서 convolution으로 upsample되는 지수 또한 2입니다.

 convolution을 통해 downsample된다는 것은 이미지의 크기를 줄이는 것은 말하며 이때 downsample 지수가 2이므로 encoder와 discriminator에서 이미지가 convolution을 통과할 때마다 이미지의 크기는 절반이 됩니다. 반대로 decoder에서 이미지가 convolution을 통과하면 이미지의 크기는 2배가 됩니다. 더 자세한 내용은 아래의 코드와 함께 확인해보겠습니다.

### 2.1. Generator
**Unet**
generator는 Unet의 구조를 사용하며

**transposedconv**

**cgan**
cgan이지만 z를 쓰지 않음

```python
class BlockCK(nn.Module):
  def __init__(self, in_ch, out_ch, is_encoder=True, is_batchnorm=True, is_dropout=False):
    super(BlockCK, self).__init__()

    self.is_encoder = is_encoder

    if is_encoder:
      conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
      relu = nn.LeakyReLU(0.2)
    else:
      conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
      relu = nn.ReLU()

    batchnorm = nn.InstanceNorm2d(out_ch)
    dropout = nn.Dropout(0.5)

    model = [conv]

    if is_batchnorm:
      model += [batchnorm]
    if is_dropout:
      model += [dropout]

    model += [relu]

    self.model = nn.Sequential(*model)

  def forward(self, x, skip=None):
    if self.is_encoder:
      return self.model(x)
    else:
      return torch.cat((self.model(x), skip), 1)
```


```python

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.down_1_C64 = BlockCK(3, 64, is_batchnorm=False)
    self.down_2_C128 = BlockCK(64, 128)
    self.down_3_C256 = BlockCK(128, 256)
    self.down_4_C512 = BlockCK(256, 512)
    self.down_5_C512 = BlockCK(512, 512)
    self.down_6_C512 = BlockCK(512, 512)
    self.down_7_C512 = BlockCK(512, 512)
    self.down_8_C512 = BlockCK(512, 512)

    self.up_7_CD512 = BlockCK(512, 512, is_encoder=False, is_dropout=True)
    self.up_6_CD512 = BlockCK(1024, 512, is_encoder=False, is_dropout=True)
    self.up_5_CD512 = BlockCK(1024, 512, is_encoder=False, is_dropout=True)
    self.up_4_CD512 = BlockCK(1024, 512, is_encoder=False)
    self.up_3_C256 = BlockCK(1024, 256, is_encoder=False)
    self.up_2_C128 = BlockCK(512, 128, is_encoder=False)
    self.up_1_C64 = BlockCK(256, 64, is_encoder=False)

    self.conv = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
    self.tan = nn.Tanh()

  def forward(self, input):
    down_1 = self.down_1_C64(input)
    down_2 = self.down_2_C128(down_1)
    down_3 = self.down_3_C256(down_2)
    down_4 = self.down_4_C512(down_3)
    down_5 = self.down_5_C512(down_4)
    down_6 = self.down_6_C512(down_5)
    down_7 = self.down_7_C512(down_6)
    down_8 = self.down_8_C512(down_7)

    up_7 = self.up_7_CD512(down_8, skip=down_7)
    up_6 = self.up_6_CD512(up_7, skip=down_6)
    up_5 = self.up_5_CD512(up_6, skip=down_5)
    up_4 = self.up_4_CD512(up_5, skip=down_4)
    up_3 = self.up_3_C256(up_4, skip=down_3)
    up_2 = self.up_2_C128(up_3, skip=down_2)
    up_1 = self.up_1_C64(up_2, skip=down_1)

    conv = self.conv(up_1)
    tan = self.tan(conv)

    return tan
```

### 2.2. Discriminator
**receptive field**

**cgan**
cgan이므로 포토와 스케치 이미지를 같이 입력받으므로 첫번째 cnn의 in_channel 값이 3+3으로 6이 됨


### 2.3. Weight Initialize

```python
```


## 3. 추가 설정 및 학습

### 3.1. 추가 설정


### 3.2. 학습



## 4. 결과
