---
layout: post
title: CycleGAN(2) - 논문 구현
# subtitle:
categories: gan
tags: [gan, cyclegan, 생성 모델, 논문 구현]
# sidebar: []
use_math: true
---

pix2pix의 논문 구현 글로 이전 글인 <a href="" target="_blank">CycleGAN(1) - 논문 리뷰</a>의 논문의 내용을 따라 구현해 보았습니다.

공식 코드로는 <a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix" target="_blank">Pytorch</a>와 <a href="https://github.com/junyanz/CycleGAN" target="_blank">Torch</a>가 있습니다.

<br>

---

## 1. 데이터 셋
이전 글이였던 <a href="https://solee328.github.io/gan/2023/01/27/pix2pix_code.html" target="_blank">pix2pix</a>와 성능과 비교해보고자 이전의 <a href="https://solee328.github.io/gan/2023/01/27/pix2pix_code.html" target="_blank">pix2pix</a>에서 사용했던 Facade 데이터 셋을 사용했습니다.


pix2pix 논문에서는 random jitter와 mirroring을 사용했다는 말이 있었지만 CycleGAN에는 씌여있지 않아 augmentation은 적용하지 않았으며 resize와 normalize만 적용했습니다.

```python
class Facade(Dataset):
  def __init__(self, path, transform = None):
    self.filenames = glob(path + '/*.jpg')
    self.transform = transform

  def __getitem__(self, idx):
    photoname = self.filenames[idx]
    sketchname = self.filenames[idx][:-3] + 'png'
    photo = Image.open(photoname).convert('RGB')
    sketch = Image.open(sketchname).convert('RGB')

    if self.transform:
      photo = self.transform(photo)
      sketch = self.transform(sketch)

    return photo, sketch, (photoname, sketchname)

  def __len__(self):
    return len(self.filenames)

    data_path = './Facade'

transform = transforms.Compose([
  transforms.Resize((256, 256)),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


dataset_Facade = CUHK(path = data_path,
                        transform=transform)

dataloader = DataLoader(dataset=dataset_Facade,
                        batch_size=batch_size,
                        shuffle=True)
```


<div>
  <img src="/assets/images/posts/cyclegan/code/data.png" width="500" height="250">
</div>

<br>

---

## 2. 모델
### 2.1. Weight

```python
def init_weight(module):
    classname = module.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

### 2.2. Generator

```python
class XK(nn.Module):
  def __init__(self, in_feature, out_feature, name, drop=False):
    super(XK, self).__init__()

    if name == 'ck':
      conv = nn.Conv2d(in_feature, out_feature, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

    elif name == 'dk':
      conv = nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=2, padding=1, padding_mode='reflect')

    elif name == 'Rk':
      conv = nn.Conv2d(in_feature, out_feature, kernel_size=3, padding=1, padding_mode='reflect')

    elif name == 'uk':
      conv = nn.ConvTranspose2d(in_feature, out_feature, kernel_size=3, stride=2, padding=1, output_padding=1)
    else:
      raise Exception('Check name')

    norm = nn.InstanceNorm2d(out_feature)
    relu = nn.ReLU()
    model = [conv, norm, relu]

    if name == 'Rk':
      model += [conv, norm, relu]
    if drop:
      model = [conv]

    self.model = nn.Sequential(*model)

  def forward(self, x):
    return self.model(x)
```
```python
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.model = nn.Sequential(
      XK(3, 64, name='ck'),
      XK(64, 128, name='dk'),
      XK(128, 256, name='dk'),
      XK(256, 256, name='Rk'),
      XK(256, 256, name='Rk'),
      XK(256, 256, name='Rk'),
      XK(256, 256, name='Rk'),
      XK(256, 256, name='Rk'),
      XK(256, 256, name='Rk'),
      XK(256, 256, name='Rk'),
      XK(256, 256, name='Rk'),
      XK(256, 256, name='Rk'),
      XK(256, 128, name='uk'),
      XK(128, 64, name='uk'),
      XK(64, 3, name='ck', drop=True),
      nn.Tanh()
    )

  def forward(self, x):
    return self.model(x)
```

### 2.3. Discriminator
Discriminator는 Pix2Pix와 마찬가지로 70x70 PatchGAN을 사용합니다. 이전 글인 Pix2Pix

```python
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2),
        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(0.2),
        nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(512),
        nn.LeakyReLU(0.2),
        nn.ConvTranspose2d(512, 1, kernel_size=4, stride=4, dilation=3),
        nn.Sigmoid()
    )

  def forward(self, x):
    return self.model(x)
```

<br>

---

## 3. 학습

### 3.1. Buffer
Pix2Pix를 포함한 이전 모델에서 판별 모델의 업데이트는 minibatch의 이미지에서만 이루어졌습니다. CycleGAN은 <a href="https://arxiv.org/pdf/1612.07828.pdf" target="_blank">SimGAN</a>의 Buffer를 사용합니다.

D를 학습시키면서 최근 생성된 이미지들에 집중하는 경향이 생기거나 G가 D를 잘 속일 수 있는 artifact를 발견해 artifact를 계속 생성하는 문제가 발생할 수 있습니다. 이를 해결하기 위해 생성된 이미지 히스토리를 사용해 판별 모델을 업데이트하는 방법으로 adversarial 학습의 안정성을 향상시키고자 도입한 방법이 Buffer입니다.

SimGAN에서는 G가 생성한 fake 이미지를 D가 학습할 때 buffer를 사용합니다. minibatch 크기를 b라 했을 때 G로 부터 생성된 이미지에서 b/2, 버퍼에서 b/2만큼 이미지를 샘플링해 판별 모델 업데이트에 사용합니다. 업데이트 이후 버퍼 내부 이미지들 중 b/2 만큼 샘플링 해 새로 생성된 이미지로 무작위 교체합니다.

CycleGAN의 경우 mini batch가 1이므로 생성 모델에서 생성된 이미지 1장과 버퍼에서 무작위로 1장을 추출해 D를 학습하도록 했습니다. 학습 후 버퍼에 이미지 장수가 50장 미만이라면 새롭게 생성했던 이미지 1장을 버퍼에 넣고 50장 이상이라면 무작위로 교체하도록 했습니다. 50장은 버퍼의 크기로 <a href="https://solee328.github.io/gan/2023/02/09/cyclegan_paper.html#h-4-implementation" target="_blank">논문 4.Implementation</a>에 언급된 크기를 사용했습니다.


```python
class Buffer():
  def __init__(self):
    self.history = []

  def insert(self, x):
    if len(self.history) < 50:
      self.history.append(x)
    else:
      idx = int(random() * 50)
      self.history[idx] = x

  def sampling(self, image):
    if len(self.history) == 0:
      return image

    idx = int(random() * len(self.history))
    return torch.cat((image, self.history[idx]), dim=0)

  def __len__(self):
    return len(self.history)
```

### 3.2. Scheduler
learning rate는 0.0002로 첫 100 epoch 동안 유지되며 이후 100 epoch 동안 0 까지 linear하게 learning rate를 줄어들게 한다고 합니다. scheduler를 사용해 learning rate를 관리할 수 있습니다. 다양한 scheduler가 있지만 지정한 함수에 따라 자유롭게 lr를 조절할 수 있는게 LambdaLR을 사용했습니다.

```python
scheduler_lambda = lambda x: 0.95 ** (x-99) if x > 99 else 1
scheduler_G_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G_G, lr_lambda=scheduler_lambda)
scheduler_G_F = torch.optim.lr_scheduler.LambdaLR(optimizer_G_F, lr_lambda=scheduler_lambda)
scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda=scheduler_lambda)
scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda=scheduler_lambda)
```

lambdaLR을 사용하기 위해서는 lr에 적용할 함수 또는 바뀔 lr 값에 대한 리스트를 입력으로 주어야 합니다. `scheduler_lambda`를 만들어 함수로 넣어주었습니다. 100 epoch 이후 lr 값에 0.95씩 곱해지게 되어 점점 lr 값이 작아지게 하는 것이 목표입니다.



<div>
  <img src="/assets/images/posts/cyclegan/code/history_lr.png" width="450" height="300">
</div>
scheduler에 의해 변화된 lr 값을 epoch에 따라 기록한 결과 이미지입니다. 초기 lr 값인 0.0002로 100 epoch 간 유지되다 이후 점점 줄어드는 것을 확인할 수 있습니다.



### 3.3. GAN Loss
GAN loss는 sigmoid cross entropy loss function을 사용했었습니다. CycleGAN에서 GAN loss로는 <a href="https://arxiv.org/pdf/1611.04076.pdf" target="_blank">LSGAN</a>에서 소개된 Least Square Loss를 사용합니다.

G가 생성한 가짜 이미지를 D가 진짜라고 판별했다 하더라도 다양한 이유로 진짜 이미지와의 차이가 커 생성 이미지의 퀄리티가 떨어질 수 있습니다. D의 학습이 부족했을 수도 있고 G가 D를 잘 속이는 artifact를 생성했을 가능성도 있습니다. 이미지 퀄리티 자체를 least square를 이용해 높이고 실제 데이터 분포와 가까운 이미지를 생성하는 것이 목표입니다. 또한 sigmoid cross entropy loss와는 달리 least square loss를 오직 한 점에서만 최소값을 갖기 때문에 학습이 안정적이라는 장점이 있습니다.

$$
\min_D V _{LSGAN}(D) = \frac{1}{2}\mathbb{E} _{x \sim p _{data}(x)}[(D(x)-b)^2] + \frac{1}{2}\mathbb{E} _{z \sim p_z(z)}[(D(G(z))-a)^2]
$$

$$
\min_G V _{LSGAN}(G) \frac{1}{2}\mathbb{E} _{z \sim p_z(z)}[(D(G(z))-c)^2]
$$

첫번째 수식에서 a는 G가 생성한 가짜 데이터 라벨, b는 실제 데이터 라벨을 의미합니다. 첫번째 수식은 D의 학습에 대한 식으로 실제 데이터인 b(1)이 입력된다면 1로 결과를 출력하고 가짜 데이터인 a(0)이 입력된다면 0으로 결과를 출력해 식을 최소화하도록 합니다. 두번째 수식에서 c는 D가 실제라고 믿기를 바라는 G가 생성한 가짜 데이터를 의미합니다. G는 D가 속을 수 있도록 실제 데이터이길 바라는 c(1)와 D의 결과가 최대한 1에 가깝게 출력되도록 식을 최소화합니다. 수식이 Mean squared Loss와 모양이 같은 것을 볼 수 있으며 따라서 코드로는 nn.MSELoss()를 사용해 구현했습니다.

<a href="https://jaejunyoo.blogspot.com/2017/03/lsgan-1.html" target="_blank">Jaejun Yoo님의 초짜 대학원생의 입장에서 이해하는 LSGAN</a>에서 LSGAN의 장점에 대해 자세한 설명이 있으니 함께 보시는 걸 추천드립니다. :thumbsup:


<br>

---

## 4. 결과
### 4.1. history

<div style="float:left;margin:0 10px 20px 0">
  <img src="/assets/images/posts/cyclegan/code/history_loss_D.png" width="300" height="200">
</div>
<div style="margin:0 10px 20px 0">
  <img src="/assets/images/posts/cyclegan/code/history_loss_G.png" width="300" height="200">
</div>

<div align="center">
  <video muted controls width="600" height="300">
    <source src="/assets/images/posts/cyclegan/code/history_image.mp4" type="video/mp4">
  </video>
</div>


### 4.2. cycleGAN vs pix2pix



### 5. 추가
### 5.1. identity loss / cycle consistency loss

---
