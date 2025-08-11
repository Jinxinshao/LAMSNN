import math
import torch
import torch.nn as nn
import os
import logging
import time
import numpy as np
import traceback
from .submodules.layers import Conv3x3, Conv1x1, LIF, PLIF, BN, Linear, SpikingMatmul
from spikingjelly.activation_based import layer, functional, neuron
from typing import Any, List, Mapping, Optional, Dict, Tuple, Union
from timm.models.registry import register_model
import torch.nn.functional as F


# 新增：动态Tanh类，实现可学习的alpha参数
class DynamicTanh(nn.Module):
    """
    动态Tanh（DyT）模块，基于'Transformers without Normalization'论文
    
    包含一个可学习的缩放因子alpha，其初始值设置为1.0
    alpha预期会在训练过程中学习到一个在0.5-1.2范围内的最优值
    """
    def __init__(self, init_alpha=1.0, channels=None):
        super().__init__()
        # 可学习的缩放因子，初始化为1.0
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        
        # 如果指定了通道数，则添加通道特定的gamma和beta参数
        if channels is not None:
            self.use_affine = True
            self.gamma = nn.Parameter(torch.ones(channels))
            self.beta = nn.Parameter(torch.zeros(channels))
        else:
            self.use_affine = False
            
    def forward(self, x):
        # 应用动态缩放的tanh变换
        if self.use_affine:
            # 如果是多维输入，需要正确地广播gamma和beta
            if x.dim() > 2:
                # 调整gamma和beta的形状以便于广播
                # 假设输入形状为[batch, channels, height, width]
                gamma = self.gamma.view(1, -1, 1, 1)
                beta = self.beta.view(1, -1, 1, 1)
            else:
                gamma = self.gamma
                beta = self.beta
                
            return gamma * torch.tanh(self.alpha * x) + beta
        else:
            return torch.tanh(self.alpha * x)
            
    def get_alpha(self):
        """返回当前alpha值，用于日志记录"""
        return self.alpha.item()


class SNN_FiLM(nn.Module):
    """
    Spiking Neural Network兼容的Feature-wise Linear Modulation模块
    
    实现了对SNN特征图的逐特征线性调制，考虑了脉冲神经元的特性
    """
    def __init__(self, channels, condition_channels=None, activation=neuron.LIFNode):
        super().__init__()
        if condition_channels is None:
            condition_channels = channels
            
        # 全连接层生成调制参数gamma和beta
        self.param_generator = nn.Sequential(
            activation(step_mode='m'),
            layer.Linear(condition_channels, channels * 2, bias=True, step_mode='m')
        )
        
    def forward(self, x, condition):
        """
        对输入特征进行调制
        
        Args:
            x: 输入特征 [T, B, C, H, W]
            condition: 条件特征 [T, B, Cc] 或 [B, Cc]
            
        Returns:
            调制后的特征 [T, B, C, H, W]
        """
        T, B, C, H, W = x.shape
        
        # 处理条件信息以生成调制参数
        if condition.dim() == 2:  # [B, Cc]
            # 扩展到时间维度
            condition = condition.unsqueeze(0).repeat(T, 1, 1)  # [T, B, Cc]
            
        # 生成gamma和beta参数
        params = self.param_generator(condition)  # [T, B, C*2]
        gamma, beta = torch.chunk(params, 2, dim=2)  # 每个 [T, B, C]
        
        # 重塑gamma和beta以适应特征维度
        gamma = gamma.view(T, B, C, 1, 1)
        beta = beta.view(T, B, C, 1, 1)
        
        # 应用调制: y = gamma * x + beta
        x = gamma * x + beta + x
        
        return x


class TS_SNN_FiLM(nn.Module):
    """
    时空联合的Feature-wise Linear Modulation模块
    
    结合了时间和空间信息进行特征调制
    """
    def __init__(self, channels, activation=neuron.LIFNode, T=4):
        super().__init__()
        
        # 时间编码器 - 根据时间步生成调制参数
        self.temporal_modulator = nn.Sequential(
            layer.Linear(channels, channels, bias=True, step_mode='m'),
            activation(step_mode='m'),
            layer.Linear(channels, channels * 2, bias=True, step_mode='m')
        )
        
        # 空间编码器 - 从空间特征生成调制参数
        self.spatial_modulator = nn.Sequential(
            layer.Conv2d(channels, channels, kernel_size=3, padding=1, step_mode='m'),
            layer.BatchNorm2d(channels, step_mode='m'),
            activation(step_mode='m'),
            layer.AdaptiveAvgPool2d((1, 1), step_mode='m'),
            layer.Conv2d(channels, channels * 2, kernel_size=1, step_mode='m')
        )
        
        # 归一化和激活
        self.norm = layer.BatchNorm2d(channels, step_mode='m')
        self.activation = activation(step_mode='m')
        
        # 映射时间步到嵌入
        self.timestep_embedder = nn.Embedding(T + 10, channels)  # 支持最多T+10个时间步，确保安全性
        
    def forward(self, x):
        """
        应用时空联合调制
        
        Args:
            x: 输入特征 [T, B, C, H, W]
            
        Returns:
            调制后的特征 [T, B, C, H, W]
        """
        T, B, C, H, W = x.shape
        outputs = []
        
        for t in range(T):
            # 获取当前时间步的特征
            curr_x = x[t]  # [B, C, H, W]
            
            # 生成时间调制参数
            time_embedding = self.timestep_embedder(torch.tensor(t, device=x.device)).expand(B, -1)  # [B, C]
            temp_params = self.temporal_modulator(time_embedding)  # [B, C*2]
            temp_gamma, temp_beta = torch.chunk(temp_params, 2, dim=1)  # 各 [B, C]
            
            # 生成空间调制参数
            spat_params = self.spatial_modulator(curr_x)  # [B, C*2, 1, 1]
            spat_gamma, spat_beta = torch.chunk(spat_params, 2, dim=1)  # 各 [B, C, 1, 1]
            
            # 重塑时间参数以匹配空间维度
            temp_gamma = temp_gamma.view(B, C, 1, 1)
            temp_beta = temp_beta.view(B, C, 1, 1)
            
            # 归一化输入
            normalized_x = self.norm(curr_x)
            
            # 联合调制: y = (temp_gamma * spat_gamma) * x + (temp_beta + spat_beta)
            modulated = (temp_gamma * spat_gamma) * normalized_x + (temp_beta + spat_beta)
            
            # 应用激活
            modulated = self.activation(modulated)
            
            outputs.append(modulated)
        
        # 重新组合时间步
        return torch.stack(outputs, dim=0)  # [T, B, C, H, W]


class FiLMSNNBlock(nn.Module):
    """
    带有FiLM调制的SNN兼容块，可以直接替换原始的块
    """
    def __init__(self, dim, activation=neuron.LIFNode):
        super().__init__()
        
        # 原始的处理组件
        self.norm1 = layer.BatchNorm2d(dim, step_mode='m')
        self.norm2 = layer.BatchNorm2d(dim, step_mode='m')

        self.conv1 = layer.Conv2d(dim, dim, kernel_size=1, step_mode='m')
        self.activation1 = activation(step_mode='m')
        
        self.conv2 = layer.Conv2d(dim, dim, kernel_size=3, padding=1, step_mode='m') 
        self.activation2 = activation(step_mode='m')
        
        # FiLM调制模块
        self.film_module = SNN_FiLM(dim, dim, activation)
        
        # 生成调制条件的网络
        self.condition_generator = nn.Sequential(
            layer.Conv2d(dim, dim, kernel_size=1, step_mode='m'),
            layer.BatchNorm2d(dim, step_mode='m'),
            activation(step_mode='m'),
            layer.AdaptiveAvgPool2d((1, 1), step_mode='m')  # 全局池化生成条件
        )
        
        # 注意力机制
        self.ca = nn.Sequential(
            layer.AdaptiveAvgPool2d((1, 1), step_mode='m'),
            layer.Conv2d(dim, dim, kernel_size=1, step_mode='m'),
            layer.BatchNorm2d(dim, step_mode='m'),
            activation(step_mode='m'),
            layer.Conv2d(dim, dim, kernel_size=1, step_mode='m')
        )
        
    def forward(self, x):
        """
        前向传播，应用FiLM调制
        
        Args:
            x: 输入特征 [T, B, C, H, W]
            
        Returns:
            调制后的特征 [T, B, C, H, W]
        """
        identity = x
        
        # 第一阶段处理
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.activation2(x)
        x = self.conv2(x)
        
        # 生成条件信息
        condition = self.condition_generator(x)
        # 展平条件信息以匹配FiLM要求
        T, B, C, _, _ = condition.shape
        condition = condition.view(T, B, C)
        
        # 应用FiLM调制
        x = self.film_module(x, condition)
        
        # 应用注意力并添加残差连接
        ca = self.ca(x)
        x = ca * x
        
        # 残差连接
        x = x + identity
        
        return x


class TS_FiLMSNNBlock(nn.Module):
    """
    带有时空FiLM调制的SNN兼容块
    """
    def __init__(self, dim, activation=neuron.LIFNode, T=4):
        super().__init__()
        
        # 原始的处理组件
        self.norm1 = layer.BatchNorm2d(dim, step_mode='m')
        self.norm2 = layer.BatchNorm2d(dim, step_mode='m')

        self.conv1 = layer.Conv2d(dim, dim, kernel_size=1, step_mode='m')
        self.activation1 = activation(step_mode='m')
        
        self.conv2 = layer.Conv2d(dim, dim, kernel_size=3, padding=1, step_mode='m') 
        self.activation2 = activation(step_mode='m')
        
        # 时空FiLM调制模块
        self.ts_film_module = TS_SNN_FiLM(dim, activation, T=T)
        
        # 注意力机制
        self.ca = nn.Sequential(
            layer.AdaptiveAvgPool2d((1, 1), step_mode='m'),
            layer.Conv2d(dim, dim, kernel_size=1, step_mode='m'),
            layer.BatchNorm2d(dim, step_mode='m'),
            activation(step_mode='m'),
            layer.Conv2d(dim, dim, kernel_size=1, step_mode='m')
        )
        
    def forward(self, x):
        """
        前向传播，应用时空FiLM调制
        
        Args:
            x: 输入特征 [T, B, C, H, W]
            
        Returns:
            调制后的特征 [T, B, C, H, W]
        """
        identity = x
        
        # 第一阶段处理
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.activation2(x)
        x = self.conv2(x)
        
        # 应用时空FiLM调制
        x = self.ts_film_module(x)
        
        # 应用注意力
        # 对每个时间步分别应用注意力
        modulated_outputs = []
        for t in range(x.shape[0]):
            curr_x = x[t]  # [B, C, H, W]
            ca = self.ca(curr_x)
            curr_x = ca * curr_x
            modulated_outputs.append(curr_x)
        
        x = torch.stack(modulated_outputs, dim=0)
        
        # 残差连接
        x = x + identity
        
        return x


class GWFFN(nn.Module):
    def __init__(self, in_channels, num_conv=1, ratio=4, group_size=64, activation=neuron.LIFNode):
        super().__init__()
        # 验证输入参数
        assert callable(activation), f"activation必须是可调用的类，而不是{type(activation)}"
        
        inner_channels = in_channels * ratio
        self.up = nn.Sequential(
            activation(step_mode='m'),
            Conv1x1(in_channels, inner_channels),
            BN(inner_channels),
        )
        
        self.conv = nn.ModuleList()
        for _ in range(num_conv):
            self.conv.append(
                nn.Sequential(
                    activation(step_mode='m'),
                    Conv3x3(inner_channels, inner_channels, groups=inner_channels // group_size),
                    BN(inner_channels),
                ))
        
        self.down = nn.Sequential(
            activation(step_mode='m'),
            Conv1x1(inner_channels, in_channels),
            BN(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat_out = x.clone()
        x = self.up(x)
        x_feat_in = x.clone()
        for m in self.conv:
            x = m(x)
        x = x + x_feat_in
        x = self.down(x)
        x = x + x_feat_out
        return x


class DSSA(nn.Module):
    def __init__(self, dim, num_heads, spatial_size, patch_size, activation=neuron.LIFNode):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        assert callable(activation), f"activation must be a callable class, not {type(activation)}"
        
        self.dim = dim
        self.num_heads = num_heads
        
        # Calculate number of patches based on spatial size
        if isinstance(spatial_size, tuple):
            h, w = spatial_size
        else:
            h = w = spatial_size
            
        # Calculate length (num patches) - similar to original implementation
        self.lenth = (math.ceil(h / patch_size) * math.ceil(w / patch_size))
        
        # Initialize firing rates with small positive values
        self.register_buffer('firing_rate_x', torch.zeros(1, 1, num_heads, 1, 1))
        self.register_buffer('firing_rate_attn', torch.zeros(1, 1, num_heads, 1, 1))
        self.init_firing_rate_x = False
        self.init_firing_rate_attn = False
        self.momentum = 0.999

        self.activation_in = activation(step_mode='m')

        # Use strided convolution for patch extraction - exactly like the original
        self.W = layer.Conv2d(dim, dim * 2, patch_size, patch_size, bias=False, step_mode='m')
        self.norm = BN(dim * 2)
        
        # Attention operations
        self.matmul1 = SpikingMatmul('r')
        self.matmul2 = SpikingMatmul('r')
        self.activation_attn = activation(step_mode='m')
        self.activation_out = activation(step_mode='m')

        # Output projection - same as original
        self.Wproj = Conv1x1(dim, dim)
        self.norm_proj = BN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        x_feat = x.clone()
        x = self.activation_in(x)
        
        # Extract patches with strided convolution
        # This maintains the 5D structure [T, B, 2*C, H', W']
        y = self.W(x)
        y = self.norm(y)
        
        # Reshape for multi-head attention - exactly like original
        y = y.reshape(T, B, self.num_heads, 2 * C // self.num_heads, -1)
        y1, y2 = y[:, :, :, :C // self.num_heads, :], y[:, :, :, C // self.num_heads:, :]
        x = x.reshape(T, B, self.num_heads, C // self.num_heads, -1)

        # Update firing rates
        if self.training:
            firing_rate_x = x.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_x and torch.all(self.firing_rate_x == 0):
                self.firing_rate_x = firing_rate_x
            self.init_firing_rate_x = True
            self.firing_rate_x = self.firing_rate_x * self.momentum + firing_rate_x * (1 - self.momentum)
        
        # Calculate attention with scaling
        scale1 = 1. / torch.sqrt(self.firing_rate_x * (self.dim // self.num_heads))
        attn = self.matmul1(y1.transpose(-1, -2), x)
        attn = attn * scale1
        attn = self.activation_attn(attn)

        # Update attention firing rates
        if self.training:
            firing_rate_attn = attn.detach().mean((0, 1, 3, 4), keepdim=True)
            if not self.init_firing_rate_attn and torch.all(self.firing_rate_attn == 0):
                self.firing_rate_attn = firing_rate_attn
            self.init_firing_rate_attn = True
            self.firing_rate_attn = self.firing_rate_attn * self.momentum + firing_rate_attn * (1 - self.momentum)
        
        # Apply attention to values
        scale2 = 1. / torch.sqrt(self.firing_rate_attn * self.lenth)
        out = self.matmul2(y2, attn)
        out = out * scale2
        
        # Modified: Infer spatial dimensions from tensor size rather than using fixed calculations
        total_elements = out.numel()
        expected_elements = T * B * C
        spatial_elements = total_elements // expected_elements
        
        # Find proper spatial dimensions that are as square as possible
        out_h = int(math.sqrt(spatial_elements))
        # Ensure we get clean divisible dimensions
        while spatial_elements % out_h != 0:
            out_h -= 1
        out_w = spatial_elements // out_h
        
        # Reshape back to spatial dimensions
        out = out.reshape(T, B, C, out_h, out_w)
        out = self.activation_out(out)
        
        # Apply projection and normalization
        out = self.Wproj(out)
        out = self.norm_proj(out)
        
        # Ensure output size matches input for residual connection
        if out.shape[3:] != x_feat.shape[3:]:
            out = torch.nn.functional.interpolate(
                out.reshape(T * B, C, out.shape[3], out.shape[4]),
                size=(x_feat.shape[3], x_feat.shape[4]),
                mode='bilinear',
                align_corners=False
            ).reshape(T, B, C, x_feat.shape[3], x_feat.shape[4])
        
        # Add residual connection
        out = out + x_feat
        return out


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation=neuron.LIFNode) -> None:
        super().__init__()
        # 验证输入参数
        assert callable(activation), f"activation必须是可调用的类，而不是{type(activation)}"
        
        self.activation = activation(step_mode='m')
        self.stride = stride
        
        # 使用普通Conv1x1替代layer.Conv2d以避免形状问题
        self.conv = Conv1x1(in_channels, out_channels)
        self.norm = BN(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入x形状: [T, B, C, H, W]
        x = self.activation(x)
        
        T, B, C, H, W = x.shape
        # 计算目标输出大小
        out_h, out_w = H // self.stride, W // self.stride
        
        # 对每个时间步和批次分别处理
        outputs = []
        
        for t in range(T):
            batch_outputs = []
            
            for b in range(B):
                # 获取单个样本
                curr_x = x[t, b]  # [C, H, W]
                
                # 应用池化
                curr_pooled = nn.functional.adaptive_avg_pool2d(
                    curr_x.unsqueeze(0), (out_h, out_w)
                )  # [1, C, out_h, out_w]
                
                batch_outputs.append(curr_pooled.squeeze(0))  # [C, out_h, out_w]
            
            # 堆叠当前时间步的所有批次
            batch_tensor = torch.stack(batch_outputs, dim=0)  # [B, C, out_h, out_w]
            outputs.append(batch_tensor)
        
        # 堆叠所有时间步
        pooled = torch.stack(outputs, dim=0)  # [T, B, C, out_h, out_w]
        
        # 应用卷积和归一化
        conv_out = self.conv(pooled)
        out = self.norm(conv_out)
        
        return out


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, activation=neuron.LIFNode) -> None:
        super().__init__()
        # 验证输入参数
        assert callable(activation), f"activation必须是可调用的类，而不是{type(activation)}"
        
        self.activation = activation(step_mode='m')
        self.scale_factor = scale_factor
        
        # 使用双线性上采样
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
        # 使用Conv3x3代替layer.Conv2d以避免形状问题
        self.conv = Conv3x3(in_channels, out_channels)
        self.norm = BN(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入x形状: [T, B, C, H, W]
        x = self.activation(x)
        
        T, B, C, H, W = x.shape
        
        # 对每个时间步和批次分别处理
        outputs = []
        
        for t in range(T):
            batch_outputs = []
            
            for b in range(B):
                # 获取单个样本
                curr_x = x[t, b]  # [C, H, W]
                
                # 应用上采样
                curr_upsampled = self.upsample(curr_x.unsqueeze(0))  # [1, C, H*scale, W*scale]
                
                batch_outputs.append(curr_upsampled.squeeze(0))  # [C, H*scale, W*scale]
            
            # 堆叠当前时间步的所有批次
            batch_tensor = torch.stack(batch_outputs, dim=0)  # [B, C, H*scale, W*scale]
            outputs.append(batch_tensor)
        
        # 堆叠所有时间步
        x_upsampled = torch.stack(outputs, dim=0)  # [T, B, C, H*scale, W*scale]
        
        # 应用卷积和归一化
        x_conv = self.conv(x_upsampled)
        x_norm = self.norm(x_conv)
        
        return x_norm


class FeatureFusionBlock(nn.Module):
    """用于融合跳跃连接特征和上采样特征的块"""
    def __init__(self, skip_channels, up_channels, out_channels, activation=neuron.LIFNode):
        super().__init__()
        # 验证输入参数
        assert callable(activation), f"activation必须是可调用的类，而不是{type(activation)}"
        
        self.activation1 = activation(step_mode='m')
        self.activation2 = activation(step_mode='m')
        
        # 处理跳跃连接
        self.conv_skip = Conv1x1(skip_channels, out_channels)
        self.norm_skip = BN(out_channels)
        
        # 始终处理上采样特征以保持一致性
        self.conv_up = Conv1x1(up_channels, out_channels) 
        self.norm_up = BN(out_channels)
        
        # 融合后的进一步细化
        self.fusion_conv = Conv3x3(out_channels, out_channels)
        self.norm_fusion = BN(out_channels)
        
    def forward(self, x_up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x_up形状: [T, B, C_up, H_up, W_up]
        # skip形状: [T, B, C_skip, H_skip, W_skip]
        
        T, B, C_up, H_up, W_up = x_up.shape
        T_skip, B_skip, C_skip, H_skip, W_skip = skip.shape
        
        # 确保批次和时间步匹配
        assert T == T_skip and B == B_skip, "时间步和批次大小必须匹配"
        
        # 始终调整跳跃连接大小以匹配上采样特征维度
        if H_skip != H_up or W_skip != W_up:
            skip_reshaped = skip.reshape(T * B_skip, C_skip, H_skip, W_skip)
            skip_resized = nn.functional.interpolate(
                skip_reshaped,
                size=(H_up, W_up),
                mode='bilinear',
                align_corners=False
            )
            skip = skip_resized.reshape(T, B_skip, C_skip, H_up, W_up)
        
        # 处理跳跃连接
        skip = self.activation1(skip)
        skip = self.conv_skip(skip)
        skip = self.norm_skip(skip)
        
        # 处理上采样特征
        x_up = self.conv_up(x_up)
        x_up = self.norm_up(x_up)
        
        # 融合（添加）特征
        x = x_up + skip
        
        # 进一步处理
        x = self.activation2(x)
        x = self.fusion_conv(x)
        x = self.norm_fusion(x)
        
        return x


class DyTFiLMKBNSNNUNet(nn.Module):
    """
    基于KBN变换和DyT的FiLM-SNN-UNet，添加了DynamicTanh模块进行特征调制
    
    此模型使用输入图像的变换而不是直接生成输出图像
    输出层生成6个通道，分为K和B，应用变换: x = K * x - B + x
    """
    def __init__(
        self,
        encoder_layers: List[List[str]],
        encoder_planes: List[int],
        decoder_planes: List[int],
        num_heads: List[int],
        patch_sizes: List[int],
        img_size=256,
        T=4,
        in_channels=3,
        out_channels=3,
        prologue=None,
        group_size=64,
        activation=neuron.LIFNode,
        init_method='improved',
        use_film=True,  # 是否使用FiLM调制
        init_alpha=1.0,  # 初始alpha值
        **kwargs,
    ):
        super().__init__()
        # 验证关键输入参数
        if activation is None:
            raise ValueError("activation cannot be None")
        if not callable(activation):
            raise ValueError(f"activation must be a callable class, not {type(activation)}")
            
        self.T = T
        self.skip = ['prologue.0', 'output_conv', 'post_process']
        self.img_size = img_size
        self.init_method = init_method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_film = use_film
        self.init_alpha = init_alpha
        
        # 确保编码器和解码器有适当的参数
        assert len(encoder_planes) == len(encoder_layers) == len(num_heads) == len(patch_sizes)
        assert len(decoder_planes) == len(encoder_planes) - 1  # 解码器阶段比编码器少一个
        
        # 创建一个虚拟张量用于计算空间尺寸
        self.register_buffer('dummy', torch.zeros(1), persistent=False)
        
        # 如果未提供prologue，使用默认的
        if prologue is None:
            self.prologue = nn.Sequential(
                layer.Conv2d(in_channels, encoder_planes[0], 7, 2, 3, bias=False, step_mode='m'),
                layer.BatchNorm2d(encoder_planes[0], step_mode='m'),
                layer.MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode='m'),
            )
        else:
            self.prologue = prologue
            
        # 构建编码器路径
        self.encoder_stages = nn.ModuleList()
        
        # 计算空间维度
        curr_sizes = self._calculate_spatial_sizes(img_size, encoder_planes)
            
        # 构建编码器阶段
        curr_size = curr_sizes[0]  # prologue后的大小
        
        for idx in range(len(encoder_planes)):
            stage = nn.Sequential()
            # 除第一阶段外，添加下采样层
            if idx != 0:
                stage.append(
                    DownsampleLayer(encoder_planes[idx-1], encoder_planes[idx], stride=2, activation=activation)
                )
                curr_size = curr_sizes[idx]
                
            # 将块添加到阶段
            if use_film:
                # 如果使用FiLM，添加FiLM模块
                stage.append(FiLMSNNBlock(encoder_planes[idx], activation=activation))
            else:
                # 否则使用原始块
                for name in encoder_layers[idx]:
                    if name == 'DSSA':
                        stage.append(
                            DSSA(encoder_planes[idx], num_heads[idx], 
                                 spatial_size=curr_size,
                                 patch_size=patch_sizes[idx], activation=activation)
                        )
                    elif name == 'GWFFN':
                        stage.append(
                            GWFFN(encoder_planes[idx], group_size=group_size, activation=activation)
                        )
                    else:
                        raise ValueError(f"Unknown layer type: {name}")
                    
            self.encoder_stages.append(stage)
            
        # 计算解码器维度
        decoder_spatial_sizes = []
        for i in range(len(decoder_planes)):
            # 反转空间大小列表以获取解码器维度
            idx = len(curr_sizes) - 2 - i
            if idx >= 0:
                decoder_spatial_sizes.append(curr_sizes[idx])
            else:
                # 对于任何剩余的解码器阶段，将大小加倍
                decoder_spatial_sizes.append((
                    decoder_spatial_sizes[-1][0] * 2,
                    decoder_spatial_sizes[-1][1] * 2
                ))
            
        # 构建解码器路径
        self.decoder_stages = nn.ModuleList()
        bottleneck_channels = encoder_planes[-1]
        
        for idx, dec_channels in enumerate(decoder_planes):
            # 计算跳跃连接索引（反转）
            skip_idx = len(encoder_planes) - 2 - idx
            
            # 确定输入和跳跃通道
            if idx == 0:
                # 第一个解码器阶段从瓶颈层获取输入
                in_channels_decoder = bottleneck_channels
            else:
                # 其他阶段从前一个解码器阶段获取输入
                in_channels_decoder = decoder_planes[idx-1]
            
            # 获取跳跃连接通道
            if skip_idx >= 0:
                skip_channels = encoder_planes[skip_idx]
            else:
                skip_channels = encoder_planes[0]
                
            # 创建上采样层
            upsample = UpsampleLayer(in_channels_decoder, dec_channels, activation=activation)
            
            # 创建融合块
            fusion = FeatureFusionBlock(
                skip_channels=skip_channels,
                up_channels=dec_channels,
                out_channels=dec_channels,
                activation=activation
            )
            
            # 创建处理器块
            processor = nn.Sequential()
            
            if use_film:
                # 使用FiLM模块
                processor.append(FiLMSNNBlock(dec_channels, activation=activation))
            else:
                # 使用原始块
                # 添加DSSA块
                if skip_idx >= 0:
                    patch_size = patch_sizes[skip_idx]
                    spatial_size = decoder_spatial_sizes[idx]
                    num_head = num_heads[skip_idx]
                else:
                    patch_size = 1
                    spatial_size = decoder_spatial_sizes[idx]
                    num_head = 1
                    
                processor.append(
                    DSSA(
                        dec_channels, 
                        num_head,
                        spatial_size=spatial_size,
                        patch_size=patch_size,
                        activation=activation
                    )
                )
                
                # 添加GWFFN块
                processor.append(
                    GWFFN(dec_channels, group_size=group_size, activation=activation)
                )
            
            # 添加到模块列表
            self.decoder_stages.append(nn.ModuleDict({
                'upsample': upsample,
                'fusion': fusion,
                'processor': processor
            }))
            
        # 调整上采样路径以确保输出为完整分辨率
        # 第一个上采样阶段 - 更积极地减少通道
        self.final_upsample = UpsampleLayer(
            decoder_planes[-1], decoder_planes[-1] // 4, scale_factor=2, activation=activation
        )
        
        # 第二个上采样阶段以达到原始分辨率
        self.extra_upsample = UpsampleLayer(
            decoder_planes[-1] // 4, decoder_planes[-1] // 8, scale_factor=2, activation=activation
        )
        
        # 最后的卷积层，输出6个通道 (K和B各3个通道)
        self.output_conv = layer.Conv2d(
            decoder_planes[-1] // 8, out_channels * 2, kernel_size=3, padding=1, bias=True, step_mode='m'
        )
        
        # 替换后处理激活以使用DynamicTanh
        # 对K和B使用单独的DynamicTanh实例
        self.dyt_K = DynamicTanh(init_alpha=init_alpha)
        self.dyt_B = DynamicTanh(init_alpha=init_alpha)
        
        # 初始化权重
        self.init_weight()

    def init_weight(self):
        """基于指定的初始化方法初始化网络权重"""
        if self.init_method == 'default':
            # 原始初始化方法
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, layer.ConvTranspose2d)):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        elif self.init_method == 'improved':
            # 改进的初始化方法，更适合从头开始训练
            for m in self.modules():
                if isinstance(m, (nn.Linear, layer.Linear)):
                    # 线性层使用Kaiming初始化
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                
                elif isinstance(m, (nn.Conv2d, layer.Conv2d)):
                    # 注意力投影层使用较小的标准偏差
                    if m.kernel_size == (1, 1):  # 投影层
                        # DSSA投影层
                        if m.weight.size(0) == m.weight.size(1) * 2:  # W_proj层
                            nn.init.normal_(m.weight, std=0.01)
                        else:  # 其他1x1卷积
                            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    else:  # 其他卷积层
                        # 空间卷积使用Kaiming初始化
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                
                elif isinstance(m, (nn.BatchNorm2d, layer.BatchNorm2d)):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
                    
                # 初始化脉冲神经元参数
                elif hasattr(m, 'v_threshold') and isinstance(m.v_threshold, (torch.Tensor)):
                    nn.init.constant_(m.v_threshold, 1.0)
                elif hasattr(m, 'tau') and isinstance(m.tau, (torch.Tensor)):
                    nn.init.constant_(m.tau, 2.0)
        
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

    def _calculate_spatial_sizes(self, img_size, encoder_planes):
        """计算每个阶段的空间大小，避免模型前向传递"""
        if isinstance(img_size, int):
            h, w = img_size, img_size
        else:
            h, w = img_size
            
        # 仅执行数学计算而不是模型前向传递
        # 计算prologue大小 - 7x7卷积步长=2，然后最大池化步长=2
        h_after_prologue = h // 4
        w_after_prologue = w // 4
        
        sizes = [(h_after_prologue, w_after_prologue)]
        
        # 计算每个编码器阶段后的大小
        for i in range(len(encoder_planes)):
            if i > 0:  # 跳过第一阶段，因为它没有下采样
                # 每个下采样层将空间维度减半
                curr_h, curr_w = sizes[-1]
                sizes.append((curr_h // 2, curr_w // 2))
                    
        return sizes
    
    def load_pretrained_encoder(self, encoder_model_name, checkpoint_dir=None):
        """从预训练的编码器加载权重"""
        result = {
            "success": False,
            "message": "",
            "missing_keys": 0,
            "unexpected_keys": 0
        }
        
        if checkpoint_dir is None:
            result["message"] = "No checkpoint directory provided"
            return result
        
        try:
            # 查找与encoder_model_name匹配的检查点文件
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                               if f.endswith('.pth') and encoder_model_name in f]
            
            if not checkpoint_files:
                result["message"] = f"No checkpoint files found for {encoder_model_name}"
                return result
            
            # 使用找到的第一个匹配文件
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 检查checkpoint格式
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                result["message"] = "Checkpoint format not recognized"
                return result
            
            # 清理state_dict，删除'module.'前缀
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    cleaned_state_dict[k[7:]] = v
                else:
                    cleaned_state_dict[k] = v
            
            # 只加载匹配的编码器层
            encoder_dict = {}
            for k, v in cleaned_state_dict.items():
                if k.startswith('prologue') or k.startswith('encoder_stages'):
                    encoder_dict[k] = v
            
            # 加载编码器权重
            missing, unexpected = self.load_state_dict(encoder_dict, strict=False)
            
            # 设置结果
            result["success"] = True
            result["message"] = f"Loaded encoder weights from {checkpoint_path}"
            result["missing_keys"] = len(missing)
            result["unexpected_keys"] = len(unexpected)
            
            return result
            
        except Exception as e:
            import traceback
            result["message"] = f"Error loading weights: {str(e)}\n{traceback.format_exc()}"
            return result

    def transfer(self, state_dict: Mapping[str, Any]):
        """从预训练模型传输权重"""
        _state_dict = {k: v for k, v in state_dict.items() if 'output_conv' not in k and 'post_process' not in k}
        return self.load_state_dict(_state_dict, strict=False)

    def get_dyt_alpha_values(self):
        """获取当前DyT alpha参数值，用于日志记录"""
        return {
            'K_alpha': self.dyt_K.get_alpha(),
            'B_alpha': self.dyt_B.get_alpha()
        }

    def kbn_transform(self, x, feat):
        """应用KBN变换：x = K * x - B + x"""
        # 将特征分为K和B部分
        K, B = torch.split(feat, [self.out_channels, self.out_channels], dim=1)
        
        # 应用DyT将K和B限制在[-1,1]范围内
        K = self.dyt_K(K)
        B = self.dyt_B(B)
        
        # 应用变换: x = K * x - B + x
        out = K * x - B + x
        
        # 确保输出在有效范围内
        out = torch.clamp(out, 0, 1)
        
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传递函数
        
        Args:
            x: 输入张量，[B, C, H, W]或[T, B, C, H, W]
                
        Returns:
            输出张量[B, C, H, W]（在时间步上平均）
        """
        # 确保输入在正确的设备上
        device = next(self.parameters()).device
        x = x.to(device)
        
        # 存储原始张量形状
        original_shape = x.shape
        
        # 处理输入维度 - 确保5D张量格式[T, B, C, H, W]
        if x.dim() != 5:
            # [B, C, H, W] -> [T, B, C, H, W]
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        else:
            # [B, T, C, H, W] -> [T, B, C, H, W]
            if x.shape[0] != self.T and x.shape[1] == self.T:
                x = x.transpose(0, 1)
        
        # 存储输入形状以供参考
        input_shape = x.shape
        # 保存输入图像以供KBN变换使用
        original_input = x.clone()
        
        # 存储用于跳跃连接的编码器特征
        skip_features = {}
        
        # Prologue
        functional.reset_net(self.prologue)
        x = self.prologue(x)
        skip_features['prologue'] = x
        
        # 编码器 - 存储所有中间状态
        for i, stage in enumerate(self.encoder_stages):
            # 在每个阶段重置神经元状态
            functional.reset_net(stage)
            x = stage(x)
            skip_features[f'encoder_{i}'] = x
        
        # 瓶颈是最后一个编码器阶段的输出
        # 带跳跃连接的解码器
        for i, decoder_dict in enumerate(self.decoder_stages):
            # 计算跳跃连接的编码器特征索引
            skip_idx = len(self.encoder_stages) - 2 - i
            
            # 在每个阶段重置神经元状态
            functional.reset_net(decoder_dict['upsample'])
            functional.reset_net(decoder_dict['fusion'])
            functional.reset_net(decoder_dict['processor'])
            
            # 上采样
            x = decoder_dict['upsample'](x)
            
            # 获取跳跃连接
            if skip_idx >= 0:
                skip_key = f'encoder_{skip_idx}'
            else:
                skip_key = 'prologue'
            
            skip = skip_features[skip_key]
            
            # 特征融合
            x = decoder_dict['fusion'](x, skip)
            
            # 处理
            x = decoder_dict['processor'](x)
        
        # 第一个最终上采样
        functional.reset_net(self.final_upsample)
        x = self.final_upsample(x)
        
        # 第二个最终上采样以达到原始大小
        functional.reset_net(self.extra_upsample)
        x = self.extra_upsample(x)
        
        # 最终卷积 - 输出6个通道
        functional.reset_net(self.output_conv)
        x = self.output_conv(x)
        
        # 在时间维度上平均
        x_averaged = torch.mean(x, dim=0)  # [B, 2*C, H, W]
        
        # 应用KBN变换
        # 从encoder中获取相应时间步的输入
        input_for_transform = torch.mean(original_input, dim=0)  # [B, C, H, W]
        
        # 应用KBN变换
        output = self.kbn_transform(input_for_transform, x_averaged)
        
        # 验证输出空间维度匹配输入
        assert output.shape[2:] == input_shape[3:], \
            f"输出空间维度({output.shape[2:]})与输入({input_shape[3:]})不匹配"
        
        return output


class DyTTSFiLMKBNSNNUNet(nn.Module):
    """
    使用动态Tanh和矢量化时空FiLM的KBN-SNN-UNet模型
    
    这是DyTFiLMKBNSNNUNet的扩展，用高效的矢量化TS-FiLM模块
    替换标准FiLM模块和TS-FiLM模块
    """
    def __init__(
        self,
        encoder_layers,
        encoder_planes,
        decoder_planes,
        num_heads,
        patch_sizes,
        img_size=256,
        T=4,
        in_channels=3,
        out_channels=3,
        prologue=None,
        group_size=64,
        activation=neuron.LIFNode,
        init_method='improved',
        init_alpha=1.0,  # 初始alpha值
        **kwargs,
    ):
        super().__init__()
        # 验证关键输入参数
        if activation is None:
            raise ValueError("activation cannot be None")
        if not callable(activation):
            raise ValueError(f"activation must be a callable class, not {type(activation)}")
            
        self.T = T
        self.skip = ['prologue.0', 'output_conv', 'post_process']
        self.img_size = img_size
        self.init_method = init_method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_alpha = init_alpha
        
        # 确保编码器和解码器有适当的参数
        assert len(encoder_planes) == len(encoder_layers) == len(num_heads) == len(patch_sizes)
        assert len(decoder_planes) == len(encoder_planes) - 1  # 解码器阶段比编码器少一个
        
        # 创建一个虚拟张量用于计算空间尺寸
        self.register_buffer('dummy', torch.zeros(1), persistent=False)
        
        # 如果未提供prologue，使用默认的
        if prologue is None:
            self.prologue = nn.Sequential(
                layer.Conv2d(in_channels, encoder_planes[0], 7, 2, 3, bias=False, step_mode='m'),
                layer.BatchNorm2d(encoder_planes[0], step_mode='m'),
                layer.MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode='m'),
            )
        else:
            self.prologue = prologue
            
        # 构建编码器路径
        self.encoder_stages = nn.ModuleList()
        
        # 计算空间维度
        curr_sizes = self._calculate_spatial_sizes(img_size, encoder_planes)
            
        # 构建编码器阶段
        curr_size = curr_sizes[0]  # prologue后的大小
        
        for idx in range(len(encoder_planes)):
            stage = nn.Sequential()
            # 除第一阶段外，添加下采样层
            if idx != 0:
                stage.append(
                    DownsampleLayer(encoder_planes[idx-1], encoder_planes[idx], stride=2, activation=activation)
                )
                curr_size = curr_sizes[idx]
                
            # 添加矢量化TS_FiLM块
            stage.append(TS_FiLMSNNBlock(encoder_planes[idx], activation=activation, T=T))
                    
            self.encoder_stages.append(stage)
            
        # 计算解码器维度
        decoder_spatial_sizes = []
        for i in range(len(decoder_planes)):
            # 反转空间大小列表以获取解码器维度
            idx = len(curr_sizes) - 2 - i
            if idx >= 0:
                decoder_spatial_sizes.append(curr_sizes[idx])
            else:
                # 对于任何剩余的解码器阶段，将大小加倍
                decoder_spatial_sizes.append((
                    decoder_spatial_sizes[-1][0] * 2,
                    decoder_spatial_sizes[-1][1] * 2
                ))
            
        # 构建解码器路径
        self.decoder_stages = nn.ModuleList()
        bottleneck_channels = encoder_planes[-1]
        
        for idx, dec_channels in enumerate(decoder_planes):
            # 计算跳跃连接索引（反转）
            skip_idx = len(encoder_planes) - 2 - idx
            
            # 确定输入和跳跃通道
            if idx == 0:
                # 第一个解码器阶段从瓶颈层获取输入
                in_channels_decoder = bottleneck_channels
            else:
                # 其他阶段从前一个解码器阶段获取输入
                in_channels_decoder = decoder_planes[idx-1]
            
            # 获取跳跃连接通道
            if skip_idx >= 0:
                skip_channels = encoder_planes[skip_idx]
            else:
                skip_channels = encoder_planes[0]
                
            # 创建上采样层
            upsample = UpsampleLayer(in_channels_decoder, dec_channels, activation=activation)
            
            # 创建融合块
            fusion = FeatureFusionBlock(
                skip_channels=skip_channels,
                up_channels=dec_channels,
                out_channels=dec_channels,
                activation=activation
            )
            
            # 创建处理器块 - 使用矢量化TS-FiLM块
            processor = nn.Sequential(
                TS_FiLMSNNBlock(dec_channels, activation=activation, T=T)
            )
            
            # 添加到模块列表
            self.decoder_stages.append(nn.ModuleDict({
                'upsample': upsample,
                'fusion': fusion,
                'processor': processor
            }))
            
        # 调整上采样路径以确保输出为完整分辨率
        # 第一个上采样阶段 - 更积极地减少通道
        self.final_upsample = UpsampleLayer(
            decoder_planes[-1], decoder_planes[-1] // 4, scale_factor=2, activation=activation
        )
        
        # 第二个上采样阶段以达到原始分辨率
        self.extra_upsample = UpsampleLayer(
            decoder_planes[-1] // 4, decoder_planes[-1] // 8, scale_factor=2, activation=activation
        )
        
        # 最后的卷积层，输出6个通道 (K和B各3个通道)
        self.output_conv = layer.Conv2d(
            decoder_planes[-1] // 8, out_channels * 2, kernel_size=3, padding=1, bias=True, step_mode='m'
        )
        
        # 替换后处理激活以使用DynamicTanh
        # 对K和B使用单独的DynamicTanh实例
        self.dyt_K = DynamicTanh(init_alpha=init_alpha)
        self.dyt_B = DynamicTanh(init_alpha=init_alpha)
        
        # 初始化权重
        self.init_weight()

    def _calculate_spatial_sizes(self, img_size, encoder_planes):
        """计算每个阶段的空间大小，避免模型前向传递"""
        if isinstance(img_size, int):
            h, w = img_size, img_size
        else:
            h, w = img_size
            
        # 仅执行数学计算而不是模型前向传递
        # 计算prologue大小 - 7x7卷积步长=2，然后最大池化步长=2
        h_after_prologue = h // 4
        w_after_prologue = w // 4
        
        sizes = [(h_after_prologue, w_after_prologue)]
        
        # 计算每个编码器阶段后的大小
        for i in range(len(encoder_planes)):
            if i > 0:  # 跳过第一阶段，因为它没有下采样
                # 每个下采样层将空间维度减半
                curr_h, curr_w = sizes[-1]
                sizes.append((curr_h // 2, curr_w // 2))
                    
        return sizes

    def init_weight(self):
        """基于指定的初始化方法初始化网络权重"""
        if self.init_method == 'default':
            # 原始初始化方法
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d, layer.ConvTranspose2d)):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        elif self.init_method == 'improved':
            # 改进的初始化方法，更适合从头开始训练
            for m in self.modules():
                if isinstance(m, (nn.Linear, layer.Linear)):
                    # 线性层使用Kaiming初始化
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                
                elif isinstance(m, (nn.Conv2d, layer.Conv2d)):
                    # 注意力投影层使用较小的标准偏差
                    if m.kernel_size == (1, 1):  # 投影层
                        # DSSA投影层
                        if hasattr(m, 'weight') and m.weight.size(0) == m.weight.size(1) * 2:  # W_proj层
                            nn.init.normal_(m.weight, std=0.01)
                        else:  # 其他1x1卷积
                            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    else:  # 其他卷积层
                        # 空间卷积使用Kaiming初始化
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                
                elif isinstance(m, (nn.BatchNorm2d, layer.BatchNorm2d)):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
                    
                # 初始化脉冲神经元参数
                elif hasattr(m, 'v_threshold') and isinstance(m.v_threshold, (torch.Tensor)):
                    nn.init.constant_(m.v_threshold, 1.0)
                elif hasattr(m, 'tau') and isinstance(m.tau, (torch.Tensor)):
                    nn.init.constant_(m.tau, 2.0)
        
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")
    
    def load_pretrained_encoder(self, encoder_model_name, checkpoint_dir=None):
        """从预训练的编码器加载权重"""
        result = {
            "success": False,
            "message": "",
            "missing_keys": 0,
            "unexpected_keys": 0
        }
        
        if checkpoint_dir is None:
            result["message"] = "No checkpoint directory provided"
            return result
        
        try:
            # 查找与encoder_model_name匹配的检查点文件
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                               if f.endswith('.pth') and encoder_model_name in f]
            
            if not checkpoint_files:
                result["message"] = f"No checkpoint files found for {encoder_model_name}"
                return result
            
            # 使用找到的第一个匹配文件
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 检查checkpoint格式
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                result["message"] = "Checkpoint format not recognized"
                return result
            
            # 清理state_dict，删除'module.'前缀
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    cleaned_state_dict[k[7:]] = v
                else:
                    cleaned_state_dict[k] = v
            
            # 只加载匹配的编码器层
            encoder_dict = {}
            for k, v in cleaned_state_dict.items():
                if k.startswith('prologue') or k.startswith('encoder_stages'):
                    encoder_dict[k] = v
            
            # 加载编码器权重
            missing, unexpected = self.load_state_dict(encoder_dict, strict=False)
            
            # 设置结果
            result["success"] = True
            result["message"] = f"Loaded encoder weights from {checkpoint_path}"
            result["missing_keys"] = len(missing)
            result["unexpected_keys"] = len(unexpected)
            
            return result
            
        except Exception as e:
            import traceback
            result["message"] = f"Error loading weights: {str(e)}\n{traceback.format_exc()}"
            return result

    def transfer(self, state_dict):
        """从预训练模型传输权重"""
        _state_dict = {k: v for k, v in state_dict.items() if 'output_conv' not in k and 'post_process' not in k}
        return self.load_state_dict(_state_dict, strict=False)

    def get_dyt_alpha_values(self):
        """获取当前DyT alpha参数值，用于日志记录"""
        return {
            'K_alpha': self.dyt_K.get_alpha(),
            'B_alpha': self.dyt_B.get_alpha()
        }

    def kbn_transform(self, x, feat):
        """应用KBN变换：x = K * x - B + x"""
        # 将特征分为K和B部分
        K, B = torch.split(feat, [self.out_channels, self.out_channels], dim=1)
        
        # 应用DynamicTanh将K和B限制在[-1,1]范围内
        K = self.dyt_K(K)
        B = self.dyt_B(B)
        
        # 应用变换: x = K * x - B + x
        out = K * x - B + x
        
        # 确保输出在有效范围内
        out = torch.clamp(out, 0, 1)
        
        return out

    def forward(self, x):
        """
        前向传递函数，确保正确处理时间维度
        
        Args:
            x: 输入张量，[B, C, H, W]或[T, B, C, H, W]
                
        Returns:
            输出张量[B, C, H, W]（在时间步上平均）
        """
        # 确保输入在正确的设备上
        device = next(self.parameters()).device
        x = x.to(device)
        
        # 处理输入维度 - 确保5D张量格式[T, B, C, H, W]
        if x.dim() != 5:
            # [B, C, H, W] -> [T, B, C, H, W]
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        else:
            # [B, T, C, H, W] -> [T, B, C, H, W]
            if x.shape[0] != self.T and x.shape[1] == self.T:
                x = x.transpose(0, 1)
        
        # 存储输入形状以供参考
        input_shape = x.shape
        # 保存输入图像以供KBN变换使用
        original_input = x.clone()
        
        # 存储用于跳跃连接的编码器特征
        skip_features = {}
        
        # Prologue
        functional.reset_net(self.prologue)
        x = self.prologue(x)
        skip_features['prologue'] = x
        
        # 编码器 - 存储所有中间状态
        for i, stage in enumerate(self.encoder_stages):
            # 在每个阶段重置神经元状态
            functional.reset_net(stage)
            x = stage(x)
            skip_features[f'encoder_{i}'] = x
        
        # 瓶颈是最后一个编码器阶段的输出
        # 带跳跃连接的解码器
        for i, decoder_dict in enumerate(self.decoder_stages):
            # 计算跳跃连接的编码器特征索引
            skip_idx = len(self.encoder_stages) - 2 - i
            
            # 在每个阶段重置神经元状态
            functional.reset_net(decoder_dict['upsample'])
            functional.reset_net(decoder_dict['fusion'])
            functional.reset_net(decoder_dict['processor'])
            
            # 上采样
            x = decoder_dict['upsample'](x)
            
            # 获取跳跃连接
            if skip_idx >= 0:
                skip_key = f'encoder_{skip_idx}'
            else:
                skip_key = 'prologue'
            
            skip = skip_features[skip_key]
            
            # 特征融合
            x = decoder_dict['fusion'](x, skip)
            
            # 使用矢量化TS-FiLM处理器处理
            x = decoder_dict['processor'](x)
        
        # 第一个最终上采样
        functional.reset_net(self.final_upsample)
        x = self.final_upsample(x)
        
        # 第二个最终上采样以达到原始大小
        functional.reset_net(self.extra_upsample)
        x = self.extra_upsample(x)
        
        # 最终卷积 - 输出6个通道
        functional.reset_net(self.output_conv)
        x = self.output_conv(x)
        
        # 在时间维度上平均
        x_averaged = torch.mean(x, dim=0)  # [B, 2*C, H, W]
        
        # 应用KBN变换
        # 从encoder中获取相应时间步的输入
        input_for_transform = torch.mean(original_input, dim=0)  # [B, C, H, W]
        
        # 应用KBN变换
        output = self.kbn_transform(input_for_transform, x_averaged)
        
        # 验证输出空间维度匹配输入
        assert output.shape[2:] == input_shape[3:], \
            f"输出空间维度({output.shape[2:]})与输入({input_shape[3:]})不匹配"
        
        return output


# 注册新的模型名称
@register_model
def dyt_film_kbn_snn_unet_small(**kwargs):
    """基于SpikingResformer-S的DyT-FiLM-KBN-UNet"""
    return DyTFiLMKBNSNNUNet(
        encoder_layers=[
            ['DSSA', 'GWFFN'] * 1,  # 阶段1
            ['DSSA', 'GWFFN'] * 2,  # 阶段2
            ['DSSA', 'GWFFN'] * 3,  # 阶段3
        ],
        encoder_planes=[64, 256, 512],  # 与spikingresformer_s相同
        decoder_planes=[256, 128],  # 解码器平面 
        num_heads=[1, 4, 8],  # 与spikingresformer_s相同
        patch_sizes=[4, 2, 1],  # 与spikingresformer_s相同
        # in_channels=3,
        out_channels=3,
        use_film=True,  # 启用FiLM模块
        init_alpha=1.0,  # 初始alpha值设为1.0
        **kwargs,
    )


@register_model
def dyt_film_kbn_snn_unet_medium(**kwargs):
    """基于SpikingResformer-M的DyT-FiLM-KBN-UNet"""
    return DyTFiLMKBNSNNUNet(
        encoder_layers=[
            ['DSSA', 'GWFFN'] * 1,  # 阶段1
            ['DSSA', 'GWFFN'] * 2,  # 阶段2
            ['DSSA', 'GWFFN'] * 3,  # 阶段3
        ],
        encoder_planes=[64, 384, 768],  # 与spikingresformer_m相同
        decoder_planes=[384, 192],  # 解码器平面
        num_heads=[1, 6, 12],  # 与spikingresformer_m相同
        patch_sizes=[4, 2, 1],  # 与spikingresformer_m相同
        # in_channels=3,
        out_channels=3,
        use_film=True,  # 启用FiLM模块
        init_alpha=1.0,  # 初始alpha值设为1.0
        **kwargs,
    )


@register_model
def dyt_film_kbn_snn_unet_large(**kwargs):
    """基于SpikingResformer-L的DyT-FiLM-KBN-UNet"""
    return DyTFiLMKBNSNNUNet(
        encoder_layers=[
            ['DSSA', 'GWFFN'] * 1,  # 阶段1
            ['DSSA', 'GWFFN'] * 2,  # 阶段2
            ['DSSA', 'GWFFN'] * 3,  # 阶段3
        ],
        encoder_planes=[128, 512, 1024],  # 与spikingresformer_l相同
        decoder_planes=[512, 256],  # 解码器平面
        num_heads=[2, 8, 16],  # 与spikingresformer_l相同
        patch_sizes=[4, 2, 1],  # 与spikingresformer_l相同
        # in_channels=3,
        out_channels=3,
        use_film=True,  # 启用FiLM模块
        init_alpha=1.0,  # 初始alpha值设为1.0
        **kwargs,
    )


@register_model
def dyt_ts_film_kbn_snn_unet_small(**kwargs):
    """基于SpikingResformer-S的DyT-TS-FiLM-KBN-UNet"""
    return DyTTSFiLMKBNSNNUNet(
        encoder_layers=[
            ['DSSA', 'GWFFN'] * 1,  # 阶段1
            ['DSSA', 'GWFFN'] * 2,  # 阶段2
            ['DSSA', 'GWFFN'] * 3,  # 阶段3
        ],
        encoder_planes=[64, 256, 512],  # 与spikingresformer_s相同
        decoder_planes=[256, 128],  # 解码器平面 
        num_heads=[1, 4, 8],  # 与spikingresformer_s相同
        patch_sizes=[4, 2, 1],  # 与spikingresformer_s相同
        in_channels=3,
        out_channels=3,
        init_alpha=1.0,  # 初始alpha值设为1.0
        **kwargs,
    )


@register_model
def dyt_ts_film_kbn_snn_unet_medium(**kwargs):
    """基于SpikingResformer-M的DyT-TS-FiLM-KBN-UNet"""
    return DyTTSFiLMKBNSNNUNet(
        encoder_layers=[
            ['DSSA', 'GWFFN'] * 1,  # 阶段1
            ['DSSA', 'GWFFN'] * 2,  # 阶段2
            ['DSSA', 'GWFFN'] * 3,  # 阶段3
        ],
        encoder_planes=[64, 384, 768],  # 与spikingresformer_m相同
        decoder_planes=[384, 192],  # 解码器平面
        num_heads=[1, 6, 12],  # 与spikingresformer_m相同
        patch_sizes=[4, 2, 1],  # 与spikingresformer_m相同
        in_channels=3,
        out_channels=3,
        init_alpha=1.0,  # 初始alpha值设为1.0
        **kwargs,
    )


@register_model
def dyt_ts_film_kbn_snn_unet_large(**kwargs):
    """基于SpikingResformer-L的DyT-TS-FiLM-KBN-UNet"""
    return DyTTSFiLMKBNSNNUNet(
        encoder_layers=[
            ['DSSA', 'GWFFN'] * 1,  # 阶段1
            ['DSSA', 'GWFFN'] * 2,  # 阶段2
            ['DSSA', 'GWFFN'] * 3,  # 阶段3
        ],
        encoder_planes=[128, 512, 1024],  # 与spikingresformer_l相同
        decoder_planes=[512, 256],  # 解码器平面
        num_heads=[2, 8, 16],  # 与spikingresformer_l相同
        patch_sizes=[4, 2, 1],  # 与spikingresformer_l相同
        in_channels=3,
        out_channels=3,
        init_alpha=1.0,  # 初始alpha值设为1.0
        **kwargs,
    )