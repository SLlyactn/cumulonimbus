# 深度学习模型与数据处理整理笔记

## 1. 数据集准备与处理

### 1.1 数据划分策略

#### 1.1.1 训练集与验证集划分  
采用 `train_test_split` 函数按比例切分数据集，确保训练与验证数据分布一致，有利于模型泛化性能评估。  
```python
from sklearn.model_selection import train_test_split
train_images, val_images = train_test_split(images, train_size=0.7, random_state=42)
```

#### 1.1.2 路径组织  
建议将训练集与验证集分别存放于明确的文件夹路径下，便于后续加载和维护。  
```python
train_dir = r'/image2/train'
val_dir = r'/image2/val'
```

### 1.2 数据加载与预处理

#### 1.2.1 自定义数据集类  
通过继承 `torch.utils.data.Dataset` 实现 `ImageTxtDataset` 类，利用txt文件组织图片路径与标签信息，便于灵活扩展。  
```python
class ImageTxtDataset(data.Dataset):
    def __init__(self, txt_path, folder_name, transform):
        self.transform = transform
        self.imgs_path = []
        self.labels = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                img_path, label = line.split()
                self.imgs_path.append(img_path)
                self.labels.append(int(label.strip()))
```

#### 1.2.2 数据增强与归一化  
采用 `transforms` 模块对图像进行统一缩放、归一化处理，确保输入格式一致。  
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])
```

## 2. 常见神经网络结构

### 2.1 GoogLeNet

#### 2.1.1 Inception模块设计  
```python
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_features):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
```

#### 2.1.2 模型堆叠结构  
```python
class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
```

### 2.2 MobileNetV2

#### 2.2.1 倒残差结构  
```python
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
```

#### 2.2.2 模型特点  
```python
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = [nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )]
```

### 2.3 MogaNet

#### 2.3.1 模块化设计  
```python
class MogaNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
```

#### 2.3.2 实践价值  
```python
self.layer1 = self._make_layer(64, 64, 2)
self.layer2 = self._make_layer(64, 128, 2, stride=2)
```

### 2.4 ResNet18

#### 2.4.1 残差连接  
```python
from torchvision.models import resnet18
model = resnet18(pretrained=True)
```

#### 2.4.2 迁移学习  
```python
model.fc = nn.Linear(model.fc.in_features, 10)
model.to(device)
```

## 3. 模型训练与评估

### 3.1 训练过程设置

#### 3.1.1 损失函数  
```python
criterion = nn.CrossEntropyLoss()
```

#### 3.1.2 优化器  
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 3.2 测试与评估

#### 3.2.1 准确率评估  
```python
correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
```

#### 3.2.2 日志与可视化  
```python
writer = SummaryWriter("logs/resnet18")
writer.add_scalar("Train Loss", train_loss, epoch)
writer.add_scalar("Test Acc", test_acc, epoch)
```

## 4. 激活函数与可视化

### 4.1 ReLU激活函数应用

#### 4.1.1 特性  
```python
self.relu = nn.ReLU()
```

#### 4.1.2 使用示例  
```python
output = self.relu(input)
```

### 4.2 数据可视化工具

#### 4.2.1 TensorBoard图像可视化  
```python
writer.add_images("input", imgs, global_step=step)
writer.add_images("output", output_sigmod, global_step=step)
```

## 5. 数据集辅助脚本

### 5.1 自动生成txt标签文件

#### 5.1.1 功能说明  
```python
def create_txt_file(root_dir, txt_filename):
    with open(txt_filename, 'w') as f:
        for label, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    img_path = os.path.join(category_path, img_name)
                    f.write(f"{img_path} {label}\n")
```

#### 5.1.2 调用方式  
```python
create_txt_file(r'/image2/train', 'train.txt')
create_txt_file(r'/image2/val', 'val.txt')
```