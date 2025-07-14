# player_reid_system.py

import numpy as np
from scipy.optimize import linear_sum_assignment

class PlayerTracker:
    def __init__(self, max_disappeared=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_disappeared
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_counter = 0
        self.frame_count = 0

        # By default, placeholder feature extractor
        self.extract_features = lambda frame, bbox: np.zeros(128)

    def update(self, frame, detections):
        self.frame_count += 1

        if len(detections) == 0:
            for track in self.tracks:
                track['age'] += 1
                track['hits_since_update'] = 0
            self.tracks = [t for t in self.tracks if t['age'] < self.max_age]
            return {}

        detections = np.array(detections)
        if detections.ndim == 1:
            detections = detections.reshape(1, -1)

        current_features = []
        for detection in detections:
            bbox = detection[:4].astype(int)
            feature = self.extract_features(frame, bbox)  # ✅ CORRECT USAGE
            current_features.append(feature)

        current_features = np.array(current_features)

        if len(self.tracks) > 0:
            similarity_matrix = np.zeros((len(self.tracks), len(detections)))

            for i, track in enumerate(self.tracks):
                for j, detection in enumerate(detections):
                    spatial_sim = self.calculate_iou(track['bbox'], detection[:4])
                    feature_sim = self.calculate_feature_similarity(
                        track['feature'], current_features[j]
                    )
                    similarity_matrix[i, j] = 0.3 * spatial_sim + 0.7 * feature_sim

            cost_matrix = 1 - similarity_matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_detections = set()

            for row, col in zip(row_indices, col_indices):
                if similarity_matrix[row, col] > 0.5:
                    track = self.tracks[row]
                    detection = detections[col]
                    track['bbox'] = detection[:4]
                    track['feature'] = current_features[col]
                    track['age'] = 0
                    track['hits_since_update'] += 1
                    track['total_hits'] += 1
                    matched_tracks.add(row)
                    matched_detections.add(col)

            for i, detection in enumerate(detections):
                if i not in matched_detections:
                    new_track = {
                        'id': self.track_id_counter,
                        'bbox': detection[:4],
                        'feature': current_features[i],
                        'age': 0,
                        'hits_since_update': 1,
                        'total_hits': 1,
                        'trajectory': [self.get_center(detection[:4])]
                    }
                    self.tracks.append(new_track)
                    self.track_id_counter += 1

            for i, track in enumerate(self.tracks):
                if i not in matched_tracks:
                    track['age'] += 1
                    track['hits_since_update'] = 0

        else:
            for i, detection in enumerate(detections):
                new_track = {
                    'id': self.track_id_counter,
                    'bbox': detection[:4],
                    'feature': current_features[i],
                    'age': 0,
                    'hits_since_update': 1,
                    'total_hits': 1,
                    'trajectory': [self.get_center(detection[:4])]
                }
                self.tracks.append(new_track)
                self.track_id_counter += 1

        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]

        valid_tracks = {}
        for track in self.tracks:
            if track['hits_since_update'] >= self.min_hits or track['total_hits'] >= self.min_hits:
                center = self.get_center(track['bbox'])
                track['trajectory'].append(center)
                valid_tracks[track['id']] = type('Player', (object,), {
                    'bbox': track['bbox'],
                    'center': center,
                    'confidence': min(track['hits_since_update'] / self.min_hits, 1.0),
                    'trajectory': track['trajectory']
                })
        return valid_tracks

    def get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def calculate_iou(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0
        intersection = (x2_int - x1_int) * (y2_int - y1_int)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return float(intersection / union) if union > 0 else 0.0

    def calculate_feature_similarity(self, f1, f2):
        norm1, norm2 = np.linalg.norm(f1), np.linalg.norm(f2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return max(0.0, float(np.dot(f1, f2) / (norm1 * norm2)))