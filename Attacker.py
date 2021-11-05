"""无目标攻击"""
import torch

from adverDataSet import adverDataSet
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda")


class Attacker:
    def __init__(self, img_dir, label):
        """读入数据和训练模型"""
        self.mean = [0.485, 0.456, 0.406]  # 加快模型收敛速度
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            self.normalize
        ])
        self.dataset = adverDataSet(img_dir, label, transform)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=4, shuffle=False)
        # 预训练好的VGG16模型
        self.model = models.vgg16(pretrained=True)
        self.model.cuda()
        self.model.eval()

    def fgsm_atteck(self, image, epslion, data_grad):
        """fgsm攻击"""
        sign_data_grad = data_grad.sign()
        perturbed_image = image + sign_data_grad * epslion
        return perturbed_image

    def attack(self, epslion):
        """攻击图片开始"""
        adv_examples = []  # 存储成功攻击的图片
        wrong, false, success = 0, 0, 0
        for data, target in self.dataloader:
            data, target = data.to(device), target.to(device)
            data_raw = data
            data.requires_grad = True
            out = self.model(data)
            init_pre = out.max(1, keepdim=True)[1]

            # 图片的类别是否正确
            if init_pre.item() != target.item():
                wrong += 1
                continue

            # 图片类别正确，开始攻击
            loss = F.nll_loss(out, target)  # 与CrossEntropyLoss区别：不做softmax
            self.model.zero_grad()  # 梯度是累积计算的，每次batch需要归零
            loss.backward()
            data_grad = data.grad.data  # 找到每个像素的微分
            perturbed_data = self.fgsm_atteck(data, epslion, data_grad)     # data没有改变，还没有更新参数

            # 是否攻击成功
            final_pred = self.model(perturbed_data)

            if final_pred.item() == target.item():
                false += 1
            else:
                success += 1
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data * torch.tensor(self.std, device=device).view(3, 1, 1) + torch.tensor(
                        self.mean, device=device).view(3, 1, 1)
                    adv_ex = adv_ex.squeeze().detach().cpu().numpy()
                    data_raw = data_raw * torch.tensor(self.std, device=device).view(3, 1, 1) + torch.tensor(self.mean,
                                                                                                             device=device).view(
                        3, 1, 1)
                    data_raw = data_raw.squeeze().detach().cpu().numpy()
                    adv_examples.append((init_pre.item(), final_pred.item(), data_raw, adv_ex))
                final_acc = (false / (wrong + success + false))
                print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epslion, false, len(self.loader), final_acc))
                return adv_examples, final_acc
