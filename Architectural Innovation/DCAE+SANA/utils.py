import os
import sys
import torch
import argparse
from pathlib import Path
from torch.amp import autocast
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image
from datasets import load_dataset
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 添加项目根目录到系统路径
BASE_DIR = Path(__file__).parent.parent
sys.path.append(r"C:\Users\Lenovo\Desktop\Paper-recover\DCAE+SANA\efficientvit-main")

from diffusers import AutoencoderDC
from efficientvit.apps.utils.image import CustomImageFolder, DMCrop

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # 直接使用PIL图像对象
        image = item['image']
        if self.transform:
            image = self.transform(image)
        return image

class AEEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_environment()
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        
    def setup_environment(self):
        """设置环境变量和基本配置"""
        os.environ['USE_LIBUV'] = '0'  # 禁用 libuv
        torch.backends.cudnn.benchmark = True
        
    def load_model(self):
        """加载模型"""
        print(f"Loading model: {self.args.model}")
        self.model = AutoencoderDC.from_pretrained(r"G:\code\model\dc-ae-f32c32-sana-1.1-diffusers", torch_dtype=torch.float16).to('cuda')
        if self.args.half:
            self.model = self.model.half()
            
    def evaluate_reconstruction(self, image_path):
        """评估单张图像的重建质量"""
        # 图像预处理
        transform = transforms.Compose([
            DMCrop(self.args.resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])
        
        # 加载图像
        image = transform(Image.open(image_path)).unsqueeze(0).to(self.device)
        if self.args.half:
            image = image.half()
            
        # 编码和解码
        with torch.no_grad(), autocast('cuda', enabled=self.args.half):
            latent = self.model.encode(image)
            # 从EncoderOutput对象中获取latent
            if hasattr(latent, 'latent_dist'):
                latent = latent.latent_dist.sample()
            reconstructed = self.model.decode(latent)
            
        # 保存结果
        save_path = os.path.join(self.args.output_dir, f"reconstructed_{os.path.basename(image_path)}")
        save_image(reconstructed * 0.5 + 0.5, save_path)
        print(f"Saved reconstruction to {save_path}")
        
    def calculate_metrics(self, original, reconstructed):
        """计算评估指标"""
        # 转换为numpy数组用于计算PSNR和SSIM
        original_np = (original.cpu().numpy() * 0.5 + 0.5).transpose(0, 2, 3, 1)
        reconstructed_np = (reconstructed.cpu().numpy() * 0.5 + 0.5).transpose(0, 2, 3, 1)
        
        # 计算PSNR和SSIM
        psnr_value = np.mean([psnr(orig, recon) for orig, recon in zip(original_np, reconstructed_np)])
        ssim_value = np.mean([ssim(orig, recon, channel_axis=2) for orig, recon in zip(original_np, reconstructed_np)])
        
        # 计算LPIPS
        lpips_value = self.lpips_fn(original, reconstructed).mean().item()
        
        return psnr_value, ssim_value, lpips_value
        
    def evaluate_celeba(self):
        """评估CelebA数据集"""
        print("Loading CelebA dataset...")
        # 加载CelebA数据集
        data = load_dataset('G:/code/datasets/CelebA_faces')
        dataset = data['train'].select(range(10000))  # 选择前10000张图片
        
        # 图像预处理
        transform = transforms.Compose([
            DMCrop(self.args.resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])
        
        # 创建自定义数据集
        celeba_dataset = CelebADataset(dataset, transform)
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            celeba_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=0 if os.name == 'nt' else self.args.num_workers,  # Windows下禁用多进程
            pin_memory=True
        )
        
        # 评估指标
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        num_batches = 0
        
        print("Starting evaluation...")
        # 评估循环
        with torch.no_grad(), autocast('cuda', enabled=self.args.half):
            for batch in dataloader:
                batch = batch.to(self.device)
                if self.args.half:
                    batch = batch.half()
                
                # 编码和解码
                latent = self.model.encode(batch)
                # 从EncoderOutput对象中获取latent
                if hasattr(latent, 'latent_dist'):
                    latent = latent.latent_dist.sample()
                reconstructed = self.model.decode(latent)
                
                # 计算评估指标
                psnr_value, ssim_value, lpips_value = self.calculate_metrics(batch, reconstructed)
                total_psnr += psnr_value
                total_ssim += ssim_value
                total_lpips += lpips_value
                
                num_batches += 1
                
                # 保存一些示例图像
                if num_batches == 1:
                    save_image(reconstructed[:4] * 0.5 + 0.5, 
                             os.path.join(self.args.output_dir, 'reconstruction_samples.png'))
                
                # 打印进度
                if num_batches % 10 == 0:
                    print(f"Processed {num_batches} batches...")
        
        # 打印评估结果
        print("\nCelebA Evaluation Results:")
        print("-" * 50)
        print(f"PSNR: {total_psnr/num_batches:.4f}")
        print(f"SSIM: {total_ssim/num_batches:.4f}")
        print(f"LPIPS: {total_lpips/num_batches:.4f}")
        print("-" * 50)
        
    def run(self):
        """运行评估"""
        self.load_model()
        
        # 创建输出目录
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        if self.args.image_path:
            # 评估单张图像
            self.evaluate_reconstruction(self.args.image_path)
        else:
            # 评估CelebA数据集
            self.evaluate_celeba()

def main():
    parser = argparse.ArgumentParser(description="DCAE Autoencoder Evaluation Tool")
    
    # 基本参数
    parser.add_argument("--model", type=str, required=True,
                      help="Model name from HuggingFace (e.g., dc-ae-f32c32-in-1.0)")
    parser.add_argument("--resolution", type=int, default=512,
                      help="Image resolution")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for dataset evaluation")
    parser.add_argument("--half", action="store_true",
                      help="Use half precision (FP16)")
    parser.add_argument("--output_dir", type=str, default="./results",
                      help="Directory to save results")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of data loading workers")
    
    # 评估模式
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_path", type=str, required=False,
                      help="Path to single image for reconstruction evaluation")
    group.add_argument("--dataset_path", type=str, default='G:/code/datasets/CelebA_faces',
                      help="Path to CelebA dataset")
    
    args = parser.parse_args()
    
    # 创建评估器并运行
    evaluator = AEEvaluator(args)
    evaluator.run()

if __name__ == "__main__":
    main()