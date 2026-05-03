# local_view_preprocessor.py
import numpy as np

class LocalViewPreprocessor:
    """
    实现《流场收敛曲线分类.docx》第6节的局部视图预处理逻辑
    输入: v2, v3 历史序列 (list or np.array)
    输出: 标准化后的 [2000, 2] 特征 (np.array, float32)
    """
    
    def __init__(self, window_size=2000, min_range=0.005, scale_factor=0.01):
        self.window_size = window_size
        self.min_range = min_range
        self.scale_factor = scale_factor

    def preprocess(self, v2_history, v3_history):
        """
        Args:
            v2_history: list of v2 values (length >= 1000)
            v3_history: list of v3 values (length >= 1000)
        Returns:
            processed_seq: np.array of shape [2000, 2], dtype=np.float32
        """
        if len(v2_history) < 1000:
            return None  # 不足1000步不处理
            
        # 转换为 numpy 数组
        v2 = np.array(v2_history, dtype=np.float32)
        v3 = np.array(v3_history, dtype=np.float32)
        N = len(v2)
        
        # 步骤1: 取末端2000步（或全部）
        end_idx = min(N, self.window_size)
        v2_local = v2[-end_idx:]
        v3_local = v3[-end_idx:]
        
        # 步骤2: 计算 Vavr（末端2000步均值）
        Vavr_v2 = np.mean(v2_local)
        Vavr_v3 = np.mean(v3_local)
        
        # 步骤3: 计算 Vmax/Vmin（从1000步后取极值）
        start_idx = min(1000, N)
        v2_tail = v2[start_idx:]
        v3_tail = v3[start_idx:]
        Vmax_v2 = np.max(v2_tail) if len(v2_tail) > 0 else Vavr_v2
        Vmin_v2 = np.min(v2_tail) if len(v2_tail) > 0 else Vavr_v2
        Vmax_v3 = np.max(v3_tail) if len(v3_tail) > 0 else Vavr_v3
        Vmin_v3 = np.min(v3_tail) if len(v3_tail) > 0 else Vavr_v3
        
        # 步骤4: 计算 range
        range_v2 = max(Vmax_v2 - Vmin_v2, self.min_range, abs(Vavr_v2) * self.scale_factor)
        range_v3 = max(Vmax_v3 - Vmin_v3, self.min_range, abs(Vavr_v3) * self.scale_factor)
        
        # 步骤5: 动态缩放
        v2_scaled = (v2_local - Vavr_v2) / (range_v2 + 1e-8)
        v3_scaled = (v3_local - Vavr_v3) / (range_v3 + 1e-8)
        
        # 合并为 [L, 2]
        local_seq = np.stack([v2_scaled, v3_scaled], axis=1)
        
        # 填充到2000步（前端填充）
        if len(local_seq) < self.window_size:
            pad = np.zeros((self.window_size - len(local_seq), 2), dtype=np.float32)
            final_seq = np.concatenate([pad, local_seq], axis=0)
        else:
            final_seq = local_seq.astype(np.float32)
            
        return final_seq