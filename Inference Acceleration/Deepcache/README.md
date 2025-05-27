# DeepCache: Accelerating Diffusion Models for Free

## 1.核心思想 
**重复利用缓存的深层特征图，使用Uniform或者Non-Uniform加速策略来training-free加速推理**

## 2.技术方法
1. **特征缓存机制** (Feature Caching Mechanism)：缓存UNet中深层块的特征图，避免重复计算
2. **均匀加速策略** (Uniform Acceleration)：在所有去噪步骤中使用相同的缓存间隔
3. **非均匀加速策略** (Non-Uniform Acceleration)：根据不同去噪阶段动态调整缓存间隔

## 3.实现细节
DeepCache通过以下方式实现加速：
- **特征重用**：识别并缓存计算密集型深层特征
- **动态缓存策略**：根据去噪阶段自适应调整缓存频率
- **内存优化**：平衡计算加速与内存使用

## 4.性能优势
- 在保持生成质量的同时实现**2-5倍**的推理加速
- 适用于多种扩散模型架构（SD 1.5/2.1/XL等）
- 与其他加速方法（如ControlNet、LoRA等）兼容

## 5.局限性
- 在某些高细节场景下可能出现轻微质量下降
- 加速效果受硬件内存限制影响
- 非均匀策略需要针对不同模型进行参数调优