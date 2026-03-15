"""Модуль RIFE (Real-Time Intermediate Flow Estimation) интерполяции."""

import cv2
import numpy as np
from typing import List, Tuple

class FrameInterpolator:
    """
    Выполняет интерполяцию кадров (AI Frame Generation) 
    для заполнения пропущенных кадров, когда генератор вызывался 
    не для каждого кадра (skip_frames > 0) для экономии времени.
    
    В RunPod (где есть мощные GPU) можно использовать RIFE модель.
    Локально використовує optical flow (DIS или Farneback).
    """

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.dis_optical_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def interpolate_sequence(
        self, 
        base_frames: List[np.ndarray], 
        skip_frames: int
    ) -> List[np.ndarray]:
        """
        Интерполирует последовательность кадров, заполняя пропуски.
        
        Args:
            base_frames: сгенерированные кадры (длины N)
            skip_frames: количество кадров-заглушек между каждым сгенерированным
            
        Returns:
            полная последовательность длины N + (N-1)*skip_frames
        """
        if skip_frames <= 0 or len(base_frames) < 2:
            return base_frames
            
        result_sequence = []
        
        for i in range(len(base_frames) - 1):
            frame1 = base_frames[i]
            frame2 = base_frames[i + 1]
            
            # Добавляем начальный кадр
            result_sequence.append(frame1)
            
            # Генерируем промежуточные кадры
            interpolated = self._interpolate_between(frame1, frame2, num_intermediate=skip_frames)
            result_sequence.extend(interpolated)
            
        # Добавляем финальный кадр
        result_sequence.append(base_frames[-1])
        
        return result_sequence

    def _interpolate_between(
        self, 
        frame1: np.ndarray, 
        frame2: np.ndarray, 
        num_intermediate: int
    ) -> List[np.ndarray]:
        """
        Интерполяция между двумя кадрами на основе оптического потока
        (наивная, но быстрая реализация через OpenCV).
        """
        # Переводим в grayscale для вычисления потока
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        h, w = gray1.shape
        
        # Вычисляем прямое и обратное движение (flow)
        flow_forward = self.dis_optical_flow.calc(gray1, gray2, None)
        flow_backward = self.dis_optical_flow.calc(gray2, gray1, None)
        
        # Строим карту координат
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        interpolated_frames = []
        
        for t_step in range(1, num_intermediate + 1):
            t = t_step / (num_intermediate + 1)
            
            # Смешиваем кадры (Forward warping)
            # В идеале нужен backward warping для избежания дыр, но для MVP:
            
            map_x1 = (grid_x - t * flow_forward[..., 0]).astype(np.float32)
            map_y1 = (grid_y - t * flow_forward[..., 1]).astype(np.float32)
            warped1 = cv2.remap(frame1, map_x1, map_y1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
            map_x2 = (grid_x - (1 - t) * flow_backward[..., 0]).astype(np.float32)
            map_y2 = (grid_y - (1 - t) * flow_backward[..., 1]).astype(np.float32)
            warped2 = cv2.remap(frame2, map_x2, map_y2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
            # Простой блендинг (t = 0 означает больше frame1, t = 1 означает больше frame2)
            blended = cv2.addWeighted(warped1, 1 - t, warped2, t, 0)
            
            interpolated_frames.append(blended)
            
        return interpolated_frames
