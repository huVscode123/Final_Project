"""
共用訓練工具模組 (Common Training Utilities)

提供機器學習與深度學習訓練過程中常用的工具類別與函式，
包含：EarlyStopping (早停機制)、向量化閾值搜尋 (Threshold Optimization)、
以及可選的混合精度上下文 (AMP Context)。
"""

import numpy as np
import torch
import torch.nn as nn
from contextlib import nullcontext

class EarlyStopping:
    """
    改良版早停機制。
    不再於每次改善時寫入磁碟，而是將最佳權重保存在記憶體中 (`detach().clone()`)，
    訓練結束時才透過 `restore_best` 將其寫入磁碟，以減少 I/O 負擔。
    """
    def __init__(self, patience=15, min_delta=1e-6, path="best_model.pt", save_every_improvement=False):
        """
        初始化 EarlyStopping。
        
        Args:
            patience (int): 容忍未改善的 epoch 數量。
            min_delta (float): 判斷為改善的最小變化量。
            path (str): 最佳模型的儲存路徑。
            save_every_improvement (bool): 是否在每次改善時寫入磁碟 (保留舊行為)。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.save_every_improvement = save_every_improvement
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        """
        每次驗證後呼叫此方法以判斷是否應該早停。
        
        Args:
            val_loss (float): 當前的驗證損失。
            model (torch.nn.Module): 當前訓練的模型。
            
        Returns:
            bool: 是否觸發早停。
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if self.save_every_improvement:
                torch.save(self.best_state, self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def restore_best(self, model, persist=True):
        """
        將最佳權重載入模型，並選擇性地儲存至磁碟。
        
        Args:
            model (torch.nn.Module): 欲載入權重的模型。
            persist (bool): 是否將最佳權重寫入磁碟。預設為 True。
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            if persist:
                torch.save(self.best_state, self.path)


def compute_optimal_threshold_vectorized(errors_normal, errors_attack, pct_range=(50, 100)):
    """
    向量化閾值搜尋：O(N log N) 取代 O(K×N)
    尋找使 F1-score 最大化的重建誤差閾值。
    
    Args:
        errors_normal (np.ndarray 或 list): 正常資料的重建誤差。
        errors_attack (np.ndarray 或 list): 異常(攻擊)資料的重建誤差。
        pct_range (tuple): 百分位數的搜尋範圍，例如 (50, 100)。
        
    Returns:
        tuple: (最佳閾值 (float), 最佳 F1-score (float), 最佳百分位數 (int))
    """
    thresholds = np.percentile(errors_normal, np.arange(*pct_range))
    sorted_normal = np.sort(errors_normal)
    sorted_attack = np.sort(errors_attack)
    fp = len(sorted_normal) - np.searchsorted(sorted_normal, thresholds, side="right")
    tp = len(sorted_attack) - np.searchsorted(sorted_attack, thresholds, side="right")
    fn = len(sorted_attack) - tp
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1)
    best_pct = pct_range[0] + best_idx
    return float(thresholds[best_idx]), float(f1[best_idx]), int(best_pct)


class AmpContext:
    """
    可選混合精度 (Automatic Mixed Precision, AMP) 上下文。
    提供與裝置相容的 AMP 設定，若不支援則優雅退回一般的精度訓練。
    """
    def __init__(self, device, enabled=True):
        """
        初始化 AmpContext。
        
        Args:
            device (torch.device 或 str): 訓練裝置。
            enabled (bool): 是否啟用混合精度。預設為 True。
        """
        self.device_type = device.type if hasattr(device, 'type') else str(device)
        self.enabled = enabled and self.device_type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.enabled) if self.enabled else None

    def autocast(self):
        """
        取得 AMP 的 autocast 上下文。
        
        Returns:
            context: torch.autocast 或 nullcontext。
        """
        if self.enabled:
            return torch.autocast(device_type=self.device_type, dtype=torch.float16)
        return nullcontext()

    def backward_step(self, loss, optimizer, model, clip_norm=1.0):
        """
        執行反向傳播、梯度裁剪與優化器更新。
        
        Args:
            loss (torch.Tensor): 損失值。
            optimizer (torch.optim.Optimizer): 優化器。
            model (torch.nn.Module): 訓練模型。
            clip_norm (float): 梯度裁剪的上限值。預設為 1.0。
        """
        optimizer.zero_grad(set_to_none=True)
        if self.enabled:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

__all__ = ["EarlyStopping", "compute_optimal_threshold_vectorized", "AmpContext"]
