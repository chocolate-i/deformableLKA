import os
import h5py
import numpy as np
import torch
import shutil
from tqdm import tqdm
from networks.d_lka_former.d_lka_net_synapse import D_LKA_Net
from networks.d_lka_former.transformerblock import TransformerBlock_3D_single_deform_LKA
from dataloaders.la_heart import LAHeart
from torchvision import transforms
from utils import losses

# 设置路径
data_path = "/root/chennuo/deformableLKA/3D/pancreas_code/dataset_pancreas/Pancreas"
model_path = "/root/chennuo/deformableLKA/3D/pancreas_code/model/pancreas1/d_lka_former_iter_6000.pth"  # 使用训练好的模型路径
output_dir = "/root/chennuo/deformableLKA/3D/pancreas_code/dataset_pancreas/selected_best"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 在脚本开始时
print(f"正在验证模型路径: {model_path}")
if not os.path.exists(model_path):
    print(f"错误：模型文件不存在: {model_path}")
    print(f"当前工作目录: {os.getcwd()}")
    # 查找可能的模型文件
    print("查找可能的模型文件:")
    if os.path.exists("./model/pancreas1/"):
        print(os.listdir("./model/pancreas1/"))
    exit(1)

# 如果指定的模型不存在，尝试查找可用的模型
if not os.path.exists(model_path):
    model_dir = os.path.dirname(model_path)
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
        if model_files:
            # 按文件名中的迭代次数排序
            model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if 'iter_' in x else 0, reverse=True)
            model_path = os.path.join(model_dir, model_files[0])
            print(f"找到可用的模型: {model_path}")
        else:
            print(f"错误: 在 {model_dir} 中没有找到 .pth 模型文件")
            exit(1)

# 检查数据目录
print(f"正在验证数据路径: {data_path}")
if not os.path.exists(data_path):
    print(f"错误：数据路径不存在: {data_path}")
    exit(1)

# 初始化模型
def create_model():
    net = D_LKA_Net(in_channels=1, 
                   out_channels=2, 
                   img_size=[96, 96, 96],
                   patch_size=(2,2,2),
                   input_size=[48*48*48, 24*24*24,12*12*12,6*6*6],
                   trans_block=TransformerBlock_3D_single_deform_LKA,
                   do_ds=False)
    model = net.cuda()
    return model

# 加载模型权重
model = create_model()
model.load_state_dict(torch.load(model_path))
model.eval()

# 加载数据集
dataset = LAHeart(
    base_dir=data_path,
    split='train',
    train_flod='train0.list',
    common_transform=transforms.Compose([]),  # 不需要随机裁剪
    sp_transform=None  # 不需要数据增强
)

print(f"数据集大小: {len(dataset)}")
select_count = max(1, int(len(dataset) * 0.02))  # 选择2%的样本，至少1个
print(f"将选择 {select_count} 个最佳样本")

# 验证一个样本
print("正在验证数据格式...")
try:
    sample = dataset[0]
    print(f"样本类型: {type(sample)}")
    if isinstance(sample, list):
        sample = sample[0]
    print(f"图像形状: {sample['image'].shape}, 标签形状: {sample['label'].shape}")
except Exception as e:
    print(f"数据验证失败: {e}")
    raise

# 评估每个样本的性能
scores = []
with torch.no_grad():
    for idx in tqdm(range(len(dataset)), desc="评估样本"):
        try:
            # 获取样本
            sample = dataset[idx]
            if isinstance(sample, list):
                sample = sample[0]  # 获取第一个样本（如果样本是列表）
                
            image, label = sample['image'], sample['label']
            
            # 转换为适合网络输入的格式
            image = torch.tensor(image, dtype=torch.float).unsqueeze(0).cuda()  # [1, 1, 96, 96, 96]
            label = torch.tensor(label, dtype=torch.long).cuda()  # [96, 96, 96]
            
            # 前向传播
            output = model(image)
            
            # 计算性能指标（Dice系数）
            output_softmax = torch.softmax(output, dim=1)
            dice_score = 1 - losses.dice_loss(output_softmax[0, 1], label == 1).item()
            
            # 记录样本索引和得分
            sample_name = dataset.image_list[idx]
            scores.append((idx, sample_name, dice_score))
            
        except Exception as e:
            print(f"处理样本 {idx} 时出错: {e}")
    
# 按得分排序（从高到低）
scores.sort(key=lambda x: x[2], reverse=True)
best_samples = scores[:select_count]

print("\n选择的最佳样本:")
for i, (idx, name, score) in enumerate(best_samples):
    print(f"{i+1}. 索引: {idx}, 文件名: {name}, Dice系数: {score:.4f}")

# 复制选中的样本到输出目录
for idx, name, score in tqdm(best_samples, desc="复制文件"):
    src_path = os.path.join(data_path, "pancreas_data", name)
    dst_path = os.path.join(output_dir, f"best_dice_{score:.4f}_{os.path.basename(name)}")
    
    try:
        # 如果目标文件已存在，则先删除
        if os.path.exists(dst_path):
            os.remove(dst_path)
        
        # 复制数据文件
        shutil.copy(src_path, dst_path)
        print(f"复制文件: {src_path} -> {dst_path}")
        
        # 创建元数据文件，记录性能信息
        meta_path = os.path.join(output_dir, f"best_dice_{score:.4f}_{os.path.basename(name)}.meta")
        with open(meta_path, 'w') as f:
            f.write(f"Original file: {name}\n")
            f.write(f"Dice score: {score:.6f}\n")
            f.write(f"Original index: {idx}\n")
        
    except Exception as e:
        print(f"复制样本 {name} 时出错: {e}")

print(f"\n完成! {len(best_samples)} 个最佳样本已保存到 {output_dir}")