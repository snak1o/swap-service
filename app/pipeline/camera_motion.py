"""Модуль оценки и компенсации движения камеры."""

import cv2
import numpy as np
from typing import List


class CameraProfiler:
    """
    Вычисляет движение камеры (панорамирование, наклон, зум)
    изображений/видеокадров, чтобы "стабилизировать" pose скелет, 
    чтобы Animate Anyone получал стационарную позу, а потом движение 
    возвращалось на этапе композитинга.
    """

    def __init__(self):
        # Используем ORB (Oriented FAST and Rotated BRIEF) для feature tracking,
        # так как это быстрый и эффективный встроенный метод OpenCV.
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def estimate_motion(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Вычисляет матрицы преобразования (гомографию или аффинные)
        между последовательными кадрами для оценки движения камеры.
        
        Возвращает список матриц 3x3, где каждая матрица M_i 
        переводит координаты из кадра i-1 в кадр i. M_0 будет единичной матрицей.
        """
        if not frames:
            return []

        transforms = [np.eye(3, dtype=np.float32)]

        for i in range(1, len(frames)):
            prev_frame = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

            # Находим ключевые точки и дескрипторы
            kp1, des1 = self.feature_detector.detectAndCompute(prev_frame, None)
            kp2, des2 = self.feature_detector.detectAndCompute(curr_frame, None)

            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                transforms.append(np.eye(3, dtype=np.float32))
                continue

            # Сопоставляем точки
            matches = self.matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Берем топ-50 лучших совпадений
            good_matches = matches[:50]

            if len(good_matches) < 4:
                transforms.append(np.eye(3, dtype=np.float32))
                continue

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Вычисляем аффинное преобразование (только сдвиг, вращение и масштаб),
            # чтобы избежать слишком сильных искажений от гомографии
            M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC)
            
            if M is not None:
                # Превращаем матрицу 2x3 в матрицу 3x3
                M_3x3 = np.vstack([M, [0, 0, 1]])
                transforms.append(M_3x3)
            else:
                transforms.append(np.eye(3, dtype=np.float32))

        return transforms

    def get_absolute_transforms(self, relative_transforms: List[np.ndarray]) -> List[np.ndarray]:
        """
        Преобразует относительные трансформации кадра к кадру
        в абсолютные трансформации относительно первого кадра.
        """
        if not relative_transforms:
            return []

        abs_transforms = [relative_transforms[0]]
        
        for i in range(1, len(relative_transforms)):
            # Перемножаем матрицы: M_abs_i = M_rel_i * M_abs_i-1
            abs_transform = np.matmul(relative_transforms[i], abs_transforms[i - 1])
            abs_transforms.append(abs_transform)
            
        return abs_transforms

    def stabilize_poses(
        self, 
        keypoints_list: List[List[List[float]]], 
        absolute_transforms: List[np.ndarray]
    ) -> List[List[List[float]]]:
        """
        Применяет инвертированные абсолютные трансформации камеры к скелетам позы, 
        чтобы поместить их в общую стабилизированную систему координат первого кадра.
        """
        if not keypoints_list or not absolute_transforms:
            return keypoints_list

        stabilized = []

        for i, (keypoints, transform) in enumerate(zip(keypoints_list, absolute_transforms)):
            if not keypoints:
                stabilized.append([])
                continue
                
            try:
                # Находим обратную матрицу
                inv_transform = np.linalg.inv(transform)
                
                new_kps = []
                for kp in keypoints:
                    x, y, conf = kp
                    if conf > 0:
                        # Применяем трансформацию: [x', y', 1]^T = M_inv * [x, y, 1]^T
                        pt = np.array([x, y, 1.0])
                        new_pt = np.matmul(inv_transform, pt)
                        new_x, new_y = new_pt[0], new_pt[1]
                        new_kps.append([new_x, new_y, conf])
                    else:
                        new_kps.append([x, y, conf])
                
                stabilized.append(new_kps)
            except np.linalg.LinAlgError:
                # Ошибка обращения матрицы, оставляем позу как есть
                stabilized.append(keypoints)

        return stabilized

    def apply_motion_to_generated(
        self, 
        generated_frames: List[np.ndarray], 
        absolute_transforms: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Применяет движение камеры обратно к сгенерированным кадрам,
        которые были сгенерированы в стабилизированном пространстве.
        """
        if not generated_frames or not absolute_transforms:
            return generated_frames

        result = []

        for i, (frame, transform) in enumerate(zip(generated_frames, absolute_transforms)):
            h, w = frame.shape[:2]
            
            # Применяем прямую трансформацию
            transformed_frame = cv2.warpPerspective(
                frame, 
                transform, 
                (w, h), 
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            result.append(transformed_frame)

        return result
