import torch
import torch.nn.functional as F
from torchvision import  transforms
import lpips
# from torchvision.models import vgg16, VGG16_Weights
# # 加载预训练的 VGG16 模型（去掉全连接层）
# #model = models.vgg16(pretrained=True)
# model = vgg16(weights=VGG16_Weights.DEFAULT)
# model = nn.Sequential(*list(model.features.children()))
# model.eval()  # 设置为评估模式

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
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
    ])
    return transform(img).unsqueeze(0)  # 增加 batch 维度

# 计算余弦相似度
def cosine_similarity_torch(vector1, vector2):
    vector1 = vector1.flatten(start_dim=1)  # 展平为一维
    vector2 = vector2.flatten(start_dim=1)
    return F.cosine_similarity(vector1, vector2).item()

def features_similarity(image1,image2):
    # 预处理
    image1_tensor = transform_image(image1)
    image2_tensor = transform_image(image2)
    # print(image1_tensor)
    
    # 提取特征
    loss_fn_vgg = lpips.LPIPS(net='alex')
    d = loss_fn_vgg(image1_tensor, image2_tensor)
    d_numpy = d[0][0][0][0].detach().cpu().numpy()
    # torch.no_grad()
    # vector1 = model(image1_tensor)
    # vector2 = model(image2_tensor)
    #print("lpips:",d_numpy)

    # 计算余弦相似度
    # similarity = cosine_similarity_torch(vector1, vector2)
    # print("similarity:",similarity)
    #print("cos",similarity)
    return d_numpy

if __name__=='__main__':
    # 创建两个 64x64 RGB 图像（全黑）
    img0 = torch.zeros(1, 3, 64, 64)  # 归一化到 [-1,1]
    #img1 = torch.zeros(1, 3, 64, 64)
    img1=torch.ones(1,3,64,64)
    img0_normalized = img0 * 2 - 1  # 0 -> -1
    img1_normalized = img1 * 2 - 1  # 1 -> 1
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    d = loss_fn_vgg(img0_normalized,img0_normalized)
    d_numpy = d[0][0][0][0].detach().cpu().numpy()
    d1 = loss_fn_vgg(img0_normalized,img1_normalized)
    d_numpy1 = d1[0][0][0][0].detach().cpu().numpy()
    print(d_numpy,d_numpy1)
    # torch.no_grad()
    # vector1 = model(img0 )
    # vector2 = model(img1)
   

    # #计算余弦相似度
    # similarity = cosine_similarity_torch(vector1, vector2)
    # print("similarity:",similarity)