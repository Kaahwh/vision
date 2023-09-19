# 输入自己的图像测试模型性能，自己的图像在test文件夹中。

import torch
import torchvision
from PIL import Image

img_path =  './test/3.jpg.'
imgs = Image.open(img_path)
# 转换成灰度图像
imgs = imgs.convert('L')
trans = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)),
                                        torchvision.transforms.ToTensor()])
imgs = trans(imgs)
imgs = imgs.unsqueeze(0)

model = torch.load("./net.pth", map_location=torch.device('cpu'))
imgs = torch.reshape(imgs, (-1, 1, 28, 28))

with torch.no_grad():
    output = model(imgs)

print(output.argmax(1))