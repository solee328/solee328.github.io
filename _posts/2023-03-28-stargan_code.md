---
layout: post
title: StarGAN(2) - 논문 구현
# subtitle:
categories: gan
tags: [gan, stargan, 생성 모델, 논문 구현]
# sidebar: []
use_math: true
---

<a href="https://solee328.github.io/gan/2023/03/13/stargan_paper.html" target="_blank">StarGAN(1) - 논문 리뷰</a>에 이은 StarGAN 논문 구현 글입니다! 공식 코드는 <a href="https://github.com/yunjey/stargan" target="_blank">Github</a>에서 확인하실 수 있습니다.
<br><br>

---

## 1. 데이터 셋
논문에서는 <a href="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html" target="_blank">The CelebFaces Attributes dataset(CelebA)</a>와 <a href="https://rafd.socsci.ru.nl/RaFD2/RaFD?p=main" target="_blank">The Radboud Faces Database(RaFD)</a>를 사용해 다중 데이터셋을 사용하는 모델을 구현했습니다. 이 중 RaFD는 대학에서 일하는 연구자임을 연구실 웹페이지 또는 최근 논문들을 이메일을 통해 보여주고 데이터를 얻을 수 있습니다. 저는 소속이 없으니 쿨하게 RaFD 데이터셋 사용을 포기했습니다 :joy:

하지만 다중 데이터셋을 사용한 모델을 만들어본 적이 없어 이번 기회에 시도해봐야겠다 생각되어 쉽게 얻을 수 있고 이미지의 양도 상당한 CelebA 데이터 셋을 나누기로 했습니다.


### 데이터셋 분리
기존 CelebA 데이터셋에는 총 40개의 속성이 있는데 이 중 몇몇 속성을 선택해 CelebA와 CelebB로 데이터셋을 나누었습니다. CelebA에는 머리 스타일과 관련된 속성들로 `'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'` 5개의 도메인을 선택하고 CelebB에는 얼굴에 추가로 얹을 수 있는 속성, 얼굴의 필수 요소가 아닌 속성들로 `'Wearing_Earrings', 'Wearing_Hat', 'Mustache', 'Eyeglasses'` 4개의 도메인을 선택했습니다. ~~사실 CelebB에 `Wearing_Necktie` 속성도 넣었었는데 이미지를 확인해보니 목 아래는 crop으로 대부분 잘리길래 빼버렸습니다.~~


기존 CelebA 데이터셋에서 라벨 속성이 CelebB의 속성이나 CelebA의 속성에 겹치는 것이 있다면 label_celebB와 label_celebA에 모아두고 celebB와 celebA 어디에도 속하지 않는 경우 label_except에 우선 모아두도록 했습니다.

```python
attr = open(path_base + '\\list_attr_celeba.txt', "r")
lines = attr.readlines()
labelname = lines[1].split()

label_celebA = []
label_celebB = []
label_except = []

for line in tqdm(lines[2:]):
  unit = line.split()
  filename = unit[0]
  labels = unit[1:]
  indexes = set([i for i in range(40) if labels[i]=='1'])  # 현재 이미지 라벨에 해당하는 속성만 모아둠

  # celebB 데이터
  if len(celebB & indexes) > 0:  # celebB와 겹치는 속성이 있는 경우
    label_celebB.append(line)
    shutil.copyfile(path_dataset + '\\' + filename, path_celebB + '\\' + filename)

  # celebA 데이터
  elif len(celebA & indexes) > 0:  # celebB와는 겹치지 않으나 celebA와 겹치는 속성이 있는 경우
    label_celebA.append(line)
    shutil.copyfile(path_dataset + '\\' + filename, path_celebA + '\\' + filename)

  # except -> 이후 celebA와 celebB 데이터 수에 따라 어디에 넣을지 정해지기 위한 용도
  else:
    label_except.append(line)

print(len(label_celebA), len(label_celebB), len(label_except))
```
<br>

 StarGAN 논문에서 CelebA 데이터 수와 RaFD 데이터 수가 크게 차이가 나 각자 데이터 셋을 학습하는 수에 차이를 두었습니다. CelebA는 총 20 epoch, RaFD는 총 200 epoch을 학습했다 되어있습니다.

 코드의 마지막 줄인 print() 함수로 CelebA, CelebB, 어디에도 속하지 않는 except의 수를 확인한 결과 각각 96197, 64068, 42334가 출력되었습니다. except에 해당하는 데이터를 CelebB로 옮긴다면 CelebA와 CelebB의 데이터 수가 비슷해 학습 epoch 수 차이가 없어도 될 것 같다 생각해 except의 데이터를 아래의 코드로 CelebB에 합쳤습니다.


```python
# except 데이터 celebB로 이동
for line in tqdm(label_except):
  unit = line.split()
  filename = unit[0]
  label_celebB.append(line)
  shutil.copyfile(path_dataset + '\\' + filename, path_celebB + '\\' + filename)

print(len(label_celebA), len(label_celebB))
```
<br>

이미지는 shutil.copyfile로 옮겼으니 이미지와 이미지에 해당하는 라벨을 인식할 수 있도록 원본 파일과 같은 형식의 .txt 파일로 라벨 정보를 저장할 수 있도록 아래 코드를 사용했습니다.

```python
def write_label(path_file, label_name, labels):
  file = open(path_file, "w")

  file.write(str(len(labels)) + '\n')
  file.write(' '.join(label_name) + '\n')
  for line in labels:
    file.write(line)

  file.close()

attr_celebA = 'E:\\DATASET\\celeba\\attr_celebA.txt'
attr_celebB = 'E:\\DATASET\\celeba\\attr_celebB.txt'

write_label(attr_celebA, labelname, label_celebA)
write_label(attr_celebB, labelname, label_celebB)
```


### 데이터셋 처리
데이터셋을 원하는 속성들로 CelebA와 CelebB로 나누었으니 이미지와 라벨을 데이터로 사용할 수 있도록 Dataset과 DataLoader를 만들어 봅시다!

CelebA와 CelebB 모두 기존 CelebA와 같은 형식의 데이터셋이니 같은 방법으로 처리할 수 있도록 `Class Celeb` Dataset을 만들었습니다. `attr_label` 함수를 통해 라벨 값이 있는 txt 파일을 읽은 뒤 한 줄씩 라벨 인식을 진행합니다.

목표 라벨의 idx 값을 읽어 해당 값이 1인 경우 목표 라벨 속성에 해당하는 이미지이니 True로, 값이 0인 경우는 False로 라벨 값을 작성합니다. 나중에 dataloader의 호출로 `getitem` 함수로 이미지와 라벨을 출력할 때는 FloatTensor로 라벨 값을 변경하기 때문에 True, False가 아닌 1.0, 0.0의 값으로 출력됩니다.

```python
class Celeb(Dataset):
  def __init__(self, path, name, target, transform = None):
    self.path_dataset = path + '\\' + name  # 이미지 폴더 위치
    self.path_attr = path + '\\attr_' + name + '.txt'  # 라벨 텍스트 위치
    self.target = target
    self.transform = transform

    self.images = []
    self.labels = []
    self.label_name = []
    self.target_idx = []

    self.attr_label()

  def attr_label(self):
    lines = [line.rstrip() for line in open(self.path_attr, 'r')]  # 라벨 값을 전부 읽는다
    self.label_name = lines[1].split()
    self.target_idx = [self.label_name.index(item) for item in self.target]

    for line in lines[2:]:
      unit = line.split()

      self.images.append(unit[0])
      self.labels.append([unit[1:][idx] == '1' for idx in self.target_idx])  # 목표 라벨의 해당하는 값을 True, 그 외는 False으로 기록

  def __getitem__(self, idx):
    image = Image.open(self.path_dataset + '\\' + self.images[idx])
    label = self.labels[idx]

    if self.transform:
      image = self.transform(image)

    return image, torch.FloatTensor(label)

  def __len__(self):
    return len(self.labels)
```
<br>

Dataset을 만들었으니 데이터에 적용할 transform과 데이터를 불러올 Dataloader를 만들어야 합니다.

StarGAN에서는 178x218 크기의 원본이미지를 178x178 크기로 crop한 후 128x128 크기로 이미지를 resize하며, $p=0.5$로 Horizontal Flip을 수행합니다. transforms를 이용해 같은 작업을 수행하도록 작성했습니다.

Celeb Dataset에 데이터셋의 위치 값과 원하는 목표 도메인, transform까지 인자로 넣어 CelebA와 CelebB 데이터셋에 대한 객체를 만들어 준 후 DataLoader에 batchsize(16), shuffle, droplast와 함께 인자로 넣어 dataloader를 만들었습니다.

```python
transform = transforms.Compose([
  transforms.CenterCrop((178, 178)),
  transforms.Resize((128, 128)),
  transforms.RandomHorizontalFlip(0.5),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

targetA = ['Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
targetB = ['Wearing_Earrings', 'Wearing_Hat', 'Mustache', 'Eyeglasses']

datasetA = Celeb('E:\DATASET\celeba', 'celebA', targetA, transform=transform)
datasetB = Celeb('E:\DATASET\celeba', 'celebB', targetB, transform=transform)

dataloaderA = DataLoader(dataset=datasetA, batch_size=batch_size, shuffle=True, drop_last=True)
dataloaderB = DataLoader(dataset=datasetB, batch_size=batch_size, shuffle=True, drop_last=True)
```
<br>

마지막으로 데이터 이미지와 라벨이 잘 들어갔는지 확인하기 위해 아래의 코드로 dataloader의 출력 값을 확인해봤습니다. CelebA와 CelebB 모두 9장씩  총 18장의 이미지와 라벨 값을 출력해보겠습니다.

```python
imageA, labelA = next(iter(dataloaderA))
imageB, labelB = next(iter(dataloaderB))

def show_data(image, label, label_name):
  plt.figure(figsize=(8,8))
  plt.suptitle(label_name)

  for i in range(1, 10):
    plt.subplot(3, 3, i)
    plt.imshow(transforms.functional.to_pil_image(0.5 * image[i] + 0.5))
    plt.title(re.sub(r'[^\d]', '', str(label[i])))
    plt.axis('off')

  plt.tight_layout()
  plt.savefig('./history/data.png', dpi=300)
  plt.show()

show_data(imageA, labelA, targetA)
show_data(imageB, labelB, targetB)
```

<div>
  <img src="/assets/images/posts/stargan/code/data.png" width="700" height="450">
</div>

CelebA의 (2, 2)에는 머리 상단 부분을 억지로 늘린 이미지, CelebB의 (2, 3)에는 워터마크가 포함된 이미지가 보이네요. 이미지의 퀄리티가 다 좋은 것만은 아닌 것 같습니다. CelebB의 (3, 2)에는 물안경을 쓴 남자 이미지가 있는데 Eyeglasses의 라벨 값으로는 0이 들어가 있는 걸보니 물안경은 안경에 포함되지 않는 것 같습니다. :eyeglasses:

이미지를 억지로 늘린 데이터도 존재한다는 것
물안경은 eyeglass로 포함되지 않는다는 것
워터마크가 있는 이미지도 있다는 것


<br><br>

---

## 2. 모델
StarGAN의 생성 모델은 CycleGAN[33]의 구조를, 판별 모델은 PatchGANs[7]의 구조를 사용합니다. 두 모델의 구조는 Table 4와 Table 5를 통해 확인할 수 있습니다.

Table에서 사용하는 표기에서 $nd$는 도메인의 수, $nc$는 도메인 라벨의 차원을 의미합니다. 제 경우 celebA의 도메인 수는 -- 개이고 CelebB의 도메인 수는 -- 개이므로 $nd = --$가 됩니다. $nc$는 사용하는 데이터 셋의 수에 따라 달라지는데 저는 2개의 데이터 셋을 사용하므로 $nd$ 값에 2를 더해 $nc = --$가 됩니다.
nd : 도메인 수
nc : 도메인 라벨의 차원
(CelebA와 RaFD 데이터셋을 모두 사용해 학습할 경우 n+2, 그렇지 않으면 nd와 동일)


### Generator
<div>
  <img src="/assets/images/posts/stargan/code/table4.png" width="600" height="350">
</div>
<br>

Bottleneck 부분의 Residual Block은 <a href="https://solee328.github.io/gan/2023/02/28/cyclegan_code.html#h-22-generator" target="_blank">CycleGAN(2) - 논문구현</a>에서 사용했던 Residual Block 코드를 사용했습니다. 변경된 점은 padding으로 CycleGAN에서는 Reflection padding을 사용했었습니다. 하지만 StarGAN에서는 라벨이 이미지와 concat되어 주어지므로 Reflection Padding을 사용하게 된다면 모델의 라벨 인식에 영향을 주어 라벨이 제대로 동작을 하지 않을 수 있습니다. 따라서 padding을 주되 CycleGAN처럼 Reflection padding을 사용하지는 않습니다.

```python
class Residual(nn.Module):
  def __init__(self, in_feature, out_feature):
    super(Residual, self).__init__()

    self.model = nn.Sequential(
      nn.Conv2d(in_feature, out_feature, kernel_size=3, padding=1),
      nn.InstanceNorm2d(out_feature),
      nn.ReLU(),
      nn.Conv2d(in_feature, out_feature, kernel_size=3, padding=1),
      nn.InstanceNorm2d(out_feature),
      nn.ReLU()
    )

  def forward(self, x):
    return x + self.model(x)
```
<br>

그 외의 Down-sampling과 Up-sampling 부분에는 Convolution과 DeConvolution(Transposed Convolution)은 각각 클래스로 만들어 'Convolution - InstanceNorm - ReLU' 순으로 적용할 수 있도록 구현했습니다.

```python
class Conv(nn.Module):
  def __init__(self, in_feature, out_feature, k, s, p):
    super(Conv, self).__init__()

    self.model = nn.Sequential(
      nn.Conv2d(in_feature, out_feature, kernel_size=k, stride=s, padding=p),
      nn.InstanceNorm2d(out_feature),
      nn.ReLU()
    )

  def forward(self, x):
    return self.model(x)


class Deconv(nn.Module):
  def __init__(self, in_feature, out_feature, k, s, p):
    super(Deconv, self).__init__()

    self.model = nn.Sequential(
      nn.ConvTranspose2d(in_feature, out_feature, k, s, p),
      nn.InstanceNorm2d(out_feature),
      nn.ReLU()
    )

  def forward(self, x):
    return self.model(x)
```
<br>

위의 클래스들을 가져와 Table 4와 같은 구조로 구현한 Generator입니다.

```python
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.model = nn.Sequential(
      # down-sampling
      Conv(14, 64, 7, 1, 3),
      Conv(64, 128, 4, 2, 1),
      Conv(128, 256, 4, 2, 1),

      # 6 residual block(bottleneck)
      Residual(256, 256),
      Residual(256, 256),
      Residual(256, 256),
      Residual(256, 256),
      Residual(256, 256),
      Residual(256, 256),

      # up-sampling
      Deconv(256, 128, 4, 2, 1),
      Deconv(128, 64, 4, 2, 1),
      nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
      nn.Tanh()
    )

  def forward(self, x, c):
    c = c.view(*c.size(), 1, 1)  # [16, 11, 1, 1]
    c = c.repeat(1, 1, 128, 128)  # [16, 11, 128, 128]
    x = torch.cat((x, c), 1)  # [16, 14(3+11), 128, 128]

    return self.model(x)
```

생성 모델 $G$의 입력으로 들어오는 것은 원본 이미지인 $x$와 목표 라벨인 $c$입니다. $x$와 $c$의 채널을 합치기 위해서 라벨 $c$의 모양을 [batchsize, nc]에서 [batchsize, nc, h, w]로 변형해 줍니다. 이후 $x$와 $c$를 concat해 모델의 최종 입력의 모양은 [batchsize, nc+3, h, w]가 됩니다. 제 경우 $nc = --$이고 $h, w = 128$이므로 최종 입력 값의 형태는 [16, 14, 128, 128]이 됩니다.




### Discriminator
<div>
  <img src="/assets/images/posts/stargan/code/table5.png" width="600" height="250">
</div>

<br>
판별 모델 또한 CycleGAN에서 사용했던 PatchGAN이지만 CycleGAN에서는 256x256 크기 이미지 기준 70x70 Patch을 사용하고 StarGAN에서는 128x128 크기 이미지 기준으로 2x2 Patch를 사용해 최종 형태가 다르다보니 CycleGAN에서 사용했던 코드를 그대로 사용하지는 못했지만 Table 5에 구조가 잘 나와있어 그대로 따라 구현해 보았습니다.

```python
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.hidden = nn.Sequential(
      # input layer
      nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(0.01),

      # hidden layer
      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(0.01),
      nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(0.01),
      nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(0.01),
      nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(0.01),
      nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(0.01)
    )

    # output layer
    self.src = nn.Conv2d(2048, 1, kernel_size=3, stride=1, padding=1)
    self.cls = nn.Conv2d(2048, 9, kernel_size=2, stride=1, padding=0)  # 5 : CelebA / 4 : CelebB

  def forward(self, x):
    hidden = self.hidden(x)
    x_src = self.src(hidden)
    x_cls = self.cls(hidden)

    return x_src, x_cls
```

이미지가 진짜인지 가짜인지 판별하는 PathGAN 역할은 x_src로 출력하고 이미지의 도메인이 무엇인지 예측하는 부분은 x_cls로 출력합니다. 도메인을 예측하는 cls는 각자 선택한 도메인의 수에 따라 달라지는데 저는 celebA의 도메인 수는 -- 개이고 CelebB의 도메인 수는 -- 개이므로 ---가 self.cls의 out_feature가 됩니다.
<br><br>

---

## 3. 학습

### Adversarial Loss
이전 글인 <a href="https://solee328.github.io/gan/2023/03/13/stargan_paper.html#h-loss-function" target="_blank">StarGAN(1) - 논문 리뷰</a>에서 나왔던 것처럼 StarGAN은 Adversarial loss로 <a href="https://arxiv.org/abs/1701.07875" target="_blank">WGAN</a>의 Wasserstein Distance(Earth Mover Distance)와 <a href="https://arxiv.org/abs/1704.00028" target="_blank">WGAN-GP</a>의 Gradient Penalty를 사용합니다.

$$
\mathcal{L} _{adv} = \mathbb{E} _x[D _{src}(x)] - \mathbb{E} _{x, c}[D _{src}(G(x, c))] - \lambda _{gp}\mathbb{E} _{\hat{x}}[(\| \nabla _{\hat{x}}D _{src}(\hat{x}) \|_2 - 1)^2]
$$

지금까지 계속 사용하던 adversarial loss가 변경되면서 $\mathbb{E} _x[log D(x)]$에서 $\mathbb{E} _x[D(x)]$로, $\mathbb{E} _{x, c}[log(1-D(G(x, c)))]$는 $\mathbb{E} _{x, c}[D(G(x, c))]$로 변경되어 log loss를 사용하지 않는 것을 확인할 수 있습니다.


```python
def gradient_penalty(x, y):
  gradients, *_ = torch.autograd.grad(outputs=y,
                                      inputs=x,
                                      grad_outputs=y.new_ones(y.shape),
                                      create_graph=True)

  gradients = gradients.view(gradients.size(0), -1)  # norm 계산을 위한 reshape
  norm = gradients.norm(2, dim=-1)  # L2 norm
  return torch.mean((norm -1) ** 2)  # mse (norm - 1)

# classify real
real_crs, real_cls = discriminator(images)
real_cls = real_cls[:, :len_targetA] if i%2 ==0 else real_cls[:, len_targetA:]

# classify fake
fake_images = generator(images, fake_condition)
fake_crs, _ = discriminator(fake_images)

# gradient penalty
eps = torch.rand(1).cuda()
x_hat = (eps * images + (1 - eps) * fake_images)
crs_hat, _ = discriminator(x_hat)

loss_gp = gradient_penalty(x_hat, crs_hat)

# adv loss
loss_adv = torch.mean(real_crs) - torch.mean(fake_crs) - lambda_gp * loss_gp
```


### Classification Loss
Classification Loss는 Discriminator


### Reconstruction Loss


### Scheduler

```python
scheduler_lambda = lambda epoch: 1.0 - max(0, epoch - n_epoch//2 - 1) / float(n_epoch)
scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=scheduler_lambda)
scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=scheduler_lambda)
```

Scheduler는 CycleGAN 때와 같습니다! StarGAN 논문에서 CelebA와 RaFD 데이터를 학습할 때 CelebA는 전체 20 epoch을 학습시키고 10 epoch 부터 learning rate를 linear하게 줄였고 RaFD는 전체 200 epoch을 학습시키고 중간인 100 epoch 부터 learning rate를 linear하게 줄였다고 합니다. 저는 CelebA 데이터를 제 임의로 나눈 CelebA와 CelebB 데이터를 사용했고 논문의 CelebA 데이터 같이 전체 20 epoch을 학습시키되 10 epoch 부터 learning rate를 줄이도록 설정했습니다. 아래의 그림을 통해 epoch에 따라 줄어드는 learning rate를 확인할 수 있습니다.

<div>
  <img src="/assets/images/posts/stargan/code/history_lr.png" width="450" height="300">
</div>


### 전체 학습 코드

- generate_label
- g, d 학습 비율 1:5 (https://github.com/clovaai/stargan-v2/issues/47)

```python
def generate_label(label):
  fake_label = torch.rand(label.size())
  fake_label = ((0.5 > fake_label).float() * 1).cuda()
  return fake_label


for epoch in range(n_epoch):
  time_start = datetime.now()

  for i in range(2):
    if i == 0:
      dataloader = dataloaderA
    else:
      dataloader = dataloaderB


    for idx, (images, labels) in enumerate(dataloader):
      images = images.cuda()
      labels = labels.cuda()

      '''
      label
      '''
      fake_labels = generate_label(labels)

      if i == 0:
        mask = torch.cat([ones, zeros], dim=1)
        label_ignore = torch.zeros(batch_size, 4).cuda()
        condition = torch.cat([labels, label_ignore, mask], dim=1)
        fake_condition = torch.cat([fake_labels, label_ignore, mask], dim=1)
      else:
        mask = torch.cat([zeros, ones], dim=1)
        label_ignore = torch.zeros(batch_size, 5).cuda()
        condition = torch.cat([label_ignore, labels, mask], dim=1)
        fake_condition = torch.cat([label_ignore, fake_labels, mask], dim=1)

      '''
      Discriminator
      '''
      optimizer_D.zero_grad()

      # classify real
      real_crs, real_cls = discriminator(images)
      real_cls = real_cls[:, :len_targetA] if i%2 ==0 else real_cls[:, len_targetA:]

      # classify fake
      fake_images = generator(images, fake_condition)
      fake_crs, _ = discriminator(fake_images)

      # gradient penalty
      eps = torch.rand(1).cuda()
      x_hat = (eps * images + (1 - eps) * fake_images)
      crs_hat, _ = discriminator(x_hat)

      loss_gp = gradient_penalty(x_hat, crs_hat)

      # adv loss
      loss_adv = torch.mean(real_crs) - torch.mean(fake_crs) - lambda_gp * loss_gp

      # cls loss
      loss_cls = loss_bce(real_cls.squeeze(), labels)

      # total
      loss_D = -loss_adv + lambda_cls * loss_cls
      history['D'].append(loss_D.item())

      # update
      loss_D.backward()
      optimizer_D.step()


      '''
      Generator
      '''
      if idx % 5 == 0:
        optimizer_G.zero_grad()

        # generate fake
        fake_images = generator(images, fake_condition)
        fake_crs, fake_cls = discriminator(fake_images)
        fake_cls = fake_cls[:, :len_targetA] if i % 2 == 0 else fake_cls[:, len_targetA:]

        # adv loss
        loss_adv = -torch.mean(fake_crs)

        # cls loss
        loss_cls = loss_bce(fake_cls.squeeze(), fake_labels)

        # rec loss
        recon_images = generator(fake_images, condition)
        loss_rec = loss_l1(recon_images, images)

        # total
        loss_G = loss_adv + lambda_cls * loss_cls + lambda_rec * loss_rec
        history['G'].append(loss_G.item())

        # update
        loss_G.backward()
        optimizer_G.step()

      else:
        history['G'].append(history['G'][-1])

      '''
      Save history
      '''
      if idx % 1000 == 0:
        print('[idx : %6d] loss_G: %.5f, loss_D: %.5f \n' %(idx, history['G'][-1], history['D'][-1]))

  '''
  Scheduler step
  '''
  scheduler_G.step()
  scheduler_D.step()

  '''
  Log
  '''
  time_end = datetime.now() - time_start
  history['lr'].append(optimizer_G.param_groups[0]['lr'])

  print('%2dM %2dS / Epoch %2d' %(*divmod(time_end.seconds, 60), epoch+1))
  print('-'*20)
```




<br><br>

---

## 4. 결과
### history



<br><br>

---
