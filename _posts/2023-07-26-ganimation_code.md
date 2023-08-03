---
layout: post
title: GANimation(2) - 논문 구현
# subtitle:
categories: gan
tags: [gan, ganimation, 생성 모델, 논문 구현]
# sidebar: []
use_math: true
---

:wave::wave: <a href="https://solee328.github.io/gan/2023/07/04/ganimation_paper.html" target="_blank">GANimation(1) - 논문 리뷰</a>에 이은 GANimation 논문 구현 글입니다!

공식 코드는 <a href="https://github.com/albertpumarola/GANimation/tree/master" target="_blank">github</a>에서 제공되고 있습니다.
<br><br>

---

## 1. 데이터셋
논문 구현 글의 시작을 알리는 데이터셋입니다ㅎㅎㅎ<br>
논문의 경우 EmotioNet을 사용했다하는데 찾아보니 <a href="http://cbcsl.ece.ohio-state.edu/EmotionNetChallenge/" target="_blank">EmotioNet Challenge</a>이 있었습니다. <a href="http://cbcsl.ece.ohio-state.edu/dbform_emotionet.html" target="_blank">EmotioNet Database Access Form</a>에 데이터를 신청할 수 있지만 연구자도 뭣도 아닌 저는 만인의 데이터셋인 <a href="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" target="_blank">CelebA</a>를 사용하는 것으로 결정했습니다:joy:

GANimation은 action unit이라는 condition을 사용하는 condition GAN입니다. 조건을 사용하기 위해 CelebA를 다운받은 후에는 AU 라벨링 과정을 거쳐야 하며 <a href="https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units" target="_blank">OpenFace</a>를 사용해 AU 라벨링이 가능합니다.

OpenFace를 사용하기 위해서는 모델을 다운받아야 하는데 <a href="https://github.com/TadasBaltrusaitis/OpenFace/wiki/Model-download" target="_blank">github</a>에서 관련 설명을 볼 수 있습니다. 링크에는 PowerShell(Windows), bash script(Mac)를 사용해 자동으로 다운받거나 OneDrive, GoogleDrive를 통해 다운받을 수 있는 링크가 있었습니다. 저는 PowerShell을 이용해서 다운받았습니다.

이후 `FaceLandmarkImg.exe -fdir imgs_in -out_dir imgs_out -aus -verbose -simsize 128 -simalign -nomask -format_aligned png` 명령어를 실행해 Action Unit을 추출할 수 있습니다. 명령어 설정을 <a href="https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments#facelandmarkimg" target="_blank">Documentation</a>에서 확인하실 수 있습니다.
- `-fdir` : 입력 이미지들이 저장된 폴더명
- `-out_dir` : 결과 이미지들을 저장할 폴더명
- `-aus` : Facial Action Units 값을 출력합니다.
- `-verbose` : 프로세싱하는 동안 얼굴 부분, HOG 특징 등을 시각화해 보여줍니다.
- `-simsize` : 이미지의 높이/너비 (default 112)
- `-simalign` : 얼굴 부분에 초점을 맞춰 정렬되어 결과 이미지가 출력됩니다.
- `-nomask` : 결과에 배경이 포함되도록 합니다. 하지 않으면 얼굴을 제외한 배경은 검은 색으로 처리됩니다.
- `-format_aligned` : 결과 이미지의 출력 포맷 (default bmp)

```
처리 과정 이미지 넣기
```

처리 중 CelebA의 253, 725, 1135, 1222, 1294 등 823 장의 이미지에서는 얼굴이 2개 이상 추출되었는데 처음 발견된 얼굴이 주요한 얼굴이고 나머지는 배경으로 같이 찍힌 얼굴들임을 확인했습닌다. 823 장의 이미지에서는 첫번째 얼굴만 사용하고 나머지 얼굴을 사용하지 않았습니다.

결과 엑셀을 확인했을 때 AU annotation는 해당 Action Unit의 존재 여부만 알려주는 AU00_c와 Action Unit의 표현 강도 값이 표기된 AU00_r,총 2가지 종류가 있습니다. 이 중 GANimation에서는 AU00_r을 사용하며 1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45, 총 17가지 AUs를 사용합니다.

```
엑셀 이미지 넣기
```

이미지에서 얼굴 부분이 작아 Action Unit이 추출되지 않는 경우도 있었는데, 이 경우 csv 파일과 png 파일이 나오지 않으므로 파일의 존재를 확인 후 존재하지 않으면 해당 번호의 이미지를 건너뛰어 전체 Action Unit 라벨링이 된 csv 파일을 만들었습니다. 데이터 수는 202,599에서 202,055로 줄어 총 544장의 이미지가 걸러지게 되었습니다.


```python
class CelebA(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms

        self.path_imgs = glob.glob(self.path + '/img/*.png')
        self.aus = dict()
        self.read_csv()

    def __getitem__(self, idx):
        image = Image.open(self.path_imgs[idx]).convert('RGB')
        if self.transforms:
            image = self.transforms(image)

        au = self.aus[os.path.basename(self.path_imgs[idx][:-4])]
        return image, au

    def __len__(self):
        return len(self.path_imgs)

    def read_csv(self):
        with open('./img_openface/aus.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.aus[row[0]] = np.asarray(row[1:], dtype=np.float32)
```

데이터를 불러올 때는 위의 csv 파일을 열어 aus 값을 확인해 파일명을 key 값으로 AUs를 value로 aus dictation에 저장합니다. 나중에 특정 이미지에 맞는 AUs 값을 확인할 때는 이미지의 이름을 이용해 aus dictation에서 불러올 수 있도록 만들었습니다.


```
데이터 확인 이미지
```

<br><br>

---

## 2. 모델
모델은 Generator와 Discriminator(Critic) 모두 이전의 논문에서 사용한 모델과 유사한 모델을 사용합니다. Generator의 경우 CycleGAN이 사용한 Perceptual Losses for Real-Time Style Transfer and Super-Resolution 논문의 모델을 사용합니다. Discriminator는 Pix2Pix에서 사용한 PatchGAN을 사용합니다. 하지만 CycleGAN과 Pix2Pix 모두 condition GAN이 아니였기 때문에


### Generator
CycleGAN이 사용한 Perceptual Losses for Real-Time Style Transfer and Super-Resolution 모델은 이미지 크기에 따라 사용하는 Residual block 수가 달라지는데 128x128 크기의 이미지 데이터에서는 6개의 residual block을 사용하며 256x256 크기 이상의 해상도 이미지 데이터에서는 9개의 Residual block을 사용합니다.

128x128로 crop한 CelebA 이미지 데이터를 사용하므로 6개의 Residual block으로 이루어진 네트워크는 c7s1-64, d128, d256, R256*6, u128, u64, c7s1-3 입니다. CycleGAN(2) - 논문 구현에서 사용한 것과 유사합니다.

CycleGAN - 논문 구현(2)에서 c7s1-64, dk, uk에 해당하는 모듈을 쉽게 가져다 쓸 수 있는 `class XK`와 Residual 모듈인 `class Residual`에 해당하는 코드를 만들었었습니다. 달라진 점이 없어 두 모듈은 CycleGAN의 글과 같은 코드입니다.

```python
class XK(nn.Module):
    def __init__(self, in_feature, out_feature, name):
        super(XK, self).__init__()

        if name == 'ck':
            conv = nn.Conv2d(in_feature, out_feature, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
        elif name == 'dk':
            conv = nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        elif name == 'uk':
            conv = nn.ConvTranspose2d(in_feature, out_feature, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            raise Exception('Check module name')

        norm = nn.InstanceNorm2d(out_feature)
        relu = nn.ReLU()
        model = [conv, norm, relu]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
```

```python
class Residual(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Residual, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_feature),
            nn.ReLU(),
            nn.Conv2d(in_feature, out_feature, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_feature),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.model(x)
```

CycleGAN(2) - 논문 구현에서 사용한 것과 3가지 차이점이 존재합니다.

첫번째 차이점은 이미지 해상도에 따른 Residual block의 개수로 CycleGAN에서는 256x256 해상도 이미지를 사용해 9개의 Residual block을 사용했으며 GANimation에서는 128x128 크기의 이미지 데이터를 사용해 6개의 Residual block을 사용합니다.

위의 ck, dk, uk, residual을 사용해 6개의 Residual block을 사용한 모델을 만들어주었습니다.

```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            XK(20, 64, name='ck'),
            XK(64, 128, name='dk'),
            XK(128, 256, name='dk'),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            XK(256, 128, name='uk'),
            XK(128, 64, name='uk'),
        )

        self.color = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        # x + c
        c = c.view(*c.size(), 1, 1) # [25, 17, 1, 1]
        c = c.repeat(1, 1, 128, 128) # [25, 17, 1, 1]
        z = torch.cat((x, c), 1) # [25, 20, 128, 128]

        feature = self.model(z)
        color = self.color(feature)
        attention = self.attention(feature)
        return color, attention
```

두번째 차이점은 condition이 입력에 추가되어 입력 이미지와 합쳐지는 것입니다. CycleGAN은 condition GAN이 아니라 condition이 입력되지 않았지만 GANimation은 Action Unit 값이 condition으로 입력되기에 입력 x와 condition c를 합쳐주어야 합니다.

```
x + c 그림
```

마지막 차이점은 출력 layer가 color mask, attention mask를 출력하기 위해 병렬 layer로 구조된다는 것입니다. color mask는 RGB color 이미지니까 channel을 3으로 줄이기 위해 nn.Conv2d(64, 3)을 적용한 후 nn.Tanh()로 이미지를 생성합니다. attention mask는 gray scale이므로 channel을 1로 줄이기 위해 nn.Conv2d(64, 1)을 적용한 후 Sigmoid()를 적용해 이미지를 생성합니다.

### Discriminator

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.cls = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.aus = nn.Conv2d(256, len_aus, kernel_size=16)

    def forward(self, x):
        feature = self.model(x)
        cls = self.cls(feature)
        aus = self.aus(feature)
        return cls, aus.squeeze()
```

<br><br>

---

## 3. Loss
Loss는 생성된 이미지의 분포를 학습 이미지 분포로 변화시키는 WGAN-GP[9]의 Adversarial loss, attention mask가 포화되는 것을 예방하고 결과 이미지의 질감을 매끄럽게 표현하기 위한 attention loss, 생성된 이미지들이 원하는 조건(AUs)의 이미지를 표현하도록 하는 Conditional expression loss, 입력 이미지의 사람의 정체성을 보존하기 위한 identity loss로 이루어져 있습니다.


### Adversarial Loss

Adversarial loss는 StarGAN에서도 사용했던 <a href="https://arxiv.org/abs/1704.00028" target="_blank">WGAN-GP</a>를 사용합니다.

$$
\mathbb{E} _{I _{y_o} \sim \mathbb{P}_o} [D _I(G(I _{y_o} | y_f))] - \mathbb{E} _{I _{y_o} \sim \mathbb{P}_o}[D _I(I _{y_o})] + \lambda _{gp} \mathbb{E} _{\tilde{I} \sim \mathbb{P} _{\tilde{I}}}[(\| \nabla _{\tilde{I}} D _I(\tilde{I})\| -1 )^2]
$$


수식 설명

```python
def gradient_penalty(x, y):
    gradients, *_ = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=y.new_ones(y.shape),
                                        create_graph=True)

    gradients = gradients.view(gradients.size(0), -1)  # norm 계산을 위한 reshape
    norm = gradients.norm(2, dim=-1)  # L2 norm
    return torch.mean((norm -1) ** 2)  # mse (norm - 1)
```

코드 설명


### Attention Loss

Attention loss는 2가지 목적이 있습니다.

$$
\lambda_{TV} \mathbb{E} _{I _{y_o} \sim \mathbb{P} _o} \left[ \sum ^{H, W} _{i, j}[(A _{i+1, j} - A _{i,j})^2 + (A _{i, j+1} - A _{i, j})^2] \right] + \mathbb{E} _{I _{y_o} \sim \mathbb{P}_o}[\| A \|_2]
$$



```python
def total_variation_loss(img):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
    return tv_h + tv_w
```


### Conditional Expression Loss


$$
\mathbb{E} _{I _{y_o} \sim \mathbb{P}_o} [\| D _y(G(I _{y_o}|y_f)) - y_f \|^2_2] + \mathbb{E} _{I _{y_o} \sim \mathbb{P}_o} [\| D _y(I _{y_o}) - y_o \|^2_2]
$$

### Identity Loss

$$
\mathcal{L} _{idt}(G, I _{y_o}, y_o, y_f) = \mathbb{E} _{I_{y_o} \sim \mathbb{P}_o}[\| G(G(I _{y_o}|y_f)|y_o) \|_1]
$$


###  Full Loss

D의 경우 코드

G의 경우 코드

---

## 4. 학습

### scheduler

### 학습
<br><br>

---

## 5. 결과


### 시도_1

### 시도_2


<br><br>

---

<br>
