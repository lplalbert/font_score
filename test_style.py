import os
from models import Msp_model
from models import MSP
import torch.nn as nn
from models import networks
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

gpu_ids=3,4
style_vgg = MSP.vgg
#style_vgg.load_state_dict(torch.load('models/style_vgg.pth'))
style_vgg = nn.Sequential(*list(style_vgg.children()))
netP = MSP.StyleExtractor(style_vgg,gpu_ids )
netP_style = MSP.Projector(gpu_ids)  
netP = networks.init_net(netP, 'normal', 0.02, gpu_ids) 
netP_style = networks.init_net(netP_style, 'normal', 0.02, gpu_ids)
# ===== 加载训练好的模型权重 =====
# 指定模型路径
netP_path = '/home/yly/workspace/DS-Font/output/checkpoints/Msp/latest_net_P.pth'
netP_style_path = '/home/yly/workspace/DS-Font/output/checkpoints/Msp/latest_net_P_style.pth'
# ===== 加载训练好的模型权重 =====
def smart_load(net, path, device='cuda:3'):
    state_dict = torch.load(path, map_location=device)
    
    is_dp_model = isinstance(net, torch.nn.DataParallel)
    has_module_prefix = list(state_dict.keys())[0].startswith('module.')

    # 情况1：模型是DataParallel，但state_dict没有module前缀 → 加前缀
    if is_dp_model and not has_module_prefix:
        print(f"[smart_load] Adding 'module.' prefix to match DataParallel for {path}")
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}

    # 情况2：模型不是DataParallel，但state_dict有module前缀 → 去掉前缀
    if not is_dp_model and has_module_prefix:
        print(f"[smart_load] Removing 'module.' prefix from {path}")
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    net.load_state_dict(state_dict)

# 加载参数
netP_path = '/home/yly/workspace/DS-Font/output/checkpoints/Msp/latest_net_P.pth'
netP_style_path = '/home/yly/workspace/DS-Font/output/checkpoints/Msp/latest_net_P_style.pth'

smart_load(netP, netP_path)
smart_load(netP_style, netP_style_path)


netP.eval()
netP_style.eval()
def transform_image(img):
    # transform = transforms.Compose([
        
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)), 
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # 读取图像并预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5), std = (0.5))])  # 归一化到[-1, 1]
   
    return transform(img).unsqueeze(0)  # 增加 batch 维度

def style_score(image1,image2):
    image1 = transform_image(image1)
    image2 = transform_image(image2)
    with torch.no_grad():
        features1 = netP(image1, [0, 1, 2])  # 假设是三层特征
        projections1 = netP_style(features1, [0, 1, 2])

        features2 = netP(image2, [0, 1, 2])  # 假设是三层特征
        projections2 = netP_style(features2, [0, 1, 2])

        # for i, proj in enumerate(features1):
        #     print(f"Projection1 {i} size:", proj.size())
        # for i, proj in enumerate(features2):
        #     print(f"Projection2 {i} size:", proj.size())
    return features1,features2,projections1,projections2
        # sims = []
        # for i in range(3):
        #     sim = F.cosine_similarity(projections1[i], projections2[i], dim=1)  # 返回 shape: [1]
        #     print(f"Layer {i} projections similarity:", sim.item())
        #     sims.append(sim.item())

        # # 计算平均相似度
        # avg_sim = sum(sims) / len(sims)
        # print("Average projections similarity:", avg_sim)
        # sims = []
        # for i in range(3):
        #     features1[i]=features1[i].view(1, -1)
        #     features2[i]=features2[i].view(1, -1)
        #     sim = F.cosine_similarity(features1[i], features2[i], dim=1)  # 返回 shape: [1]
        #     print(f"Layer {i} features similarity:", sim.item())
        #     sims.append(sim.item())

        # # 计算平均相似度
        # avg_sim = sum(sims) / len(sims)
        # print("Average features similarity:", avg_sim)
        
if __name__=='__main__':
    transform = transforms.Compose(
        [transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5), std = (0.5))])
    path1="/data/yly/font/test_unknown_style/chinese/Meng na Hei song(MSungHK-Medium)Font - Traditional Chinese/按.png"
    path2="/data/yly/font/test_unknown_style/chinese/Meng na Cu yuan(MYuenHK-Xbold) Font - Traditional Chinese/按.png"
    image1 = Image.open(path1)
    image1 = transform(image1).unsqueeze(0)

    image2 = Image.open(path2)
    image2 = transform(image2).unsqueeze(0)

