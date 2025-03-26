import numpy as np
from collections import defaultdict
import time
from scipy.optimize import linear_sum_assignment

class FaceTracker:
    def __init__(self, max_age=30, iou_threshold=0.4):
        self.tracks = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.track_history = defaultdict(list)
        
    def update(self, detections):
        # Age all current tracks
        for track in self.tracks:
            track['age'] += 1
        
        # Hungarian algorithm matching
        matched_pairs = self._hungarian_matching(detections)
        matched_det_indices = set(p[0] for p in matched_pairs)
        matched_track_indices = set(p[1] for p in matched_pairs)
        
        # Update matched tracks
        for det_idx, track_idx in matched_pairs:
            self.tracks[track_idx].update({
                'box': detections[det_idx]['box'],
                'confidence': detections[det_idx]['confidence'],
                'age': 0,
                'last_seen': time.time()
            })
            self._update_history(self.tracks[track_idx]['id'], detections[det_idx])
        
        # Create new tracks for unmatched detections
        unmatched_detections = [i for i in range(len(detections)) 
                               if i not in matched_det_indices]
        for idx in unmatched_detections:
            self._init_new_track(detections[idx])
        
        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t['age'] < self.max_age]
        
        return self.tracks
    
    def _hungarian_matching(self, detections):
        if not self.tracks or not detections:
            return []
            
        # Cost matrix (1 - IOU)
        cost_matrix = np.ones((len(detections), len(self.tracks)))
        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks):
                cost_matrix[i,j] = 1 - self._calculate_iou(det['box'], track['box'])
        
        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Filter by threshold
        return [(row, col) for row, col in zip(row_ind, col_ind)
               if cost_matrix[row, col] < (1 - self.iou_threshold)]
    
    def _calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = ((box1[2]-box1[0])*(box1[3]-box1[1]) + 
                (box2[2]-box2[0])*(box2[3]-box2[1]) - inter)
        return inter / union if union > 0 else 0
    
    def _init_new_track(self, detection):
        new_track = {
            'id': self.next_id,
            'box': detection['box'],
            'confidence': detection['confidence'],
            'age': 0,
            'last_seen': time.time(),
            'best_quality': -1
        }
        self.tracks.append(new_track)
        self.next_id += 1
        self._update_history(new_track['id'], detection)
    
    def _update_history(self, track_id, detection):
        self.track_history[track_id].append({
            'timestamp': time.time(),
            'box': detection['box'],
            'confidence': detection['confidence']
        })