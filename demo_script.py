#!/usr/bin/env python3

import cv2
import numpy as np
import json
import argparse
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlayerReIDDemo:
    """Demo class for player re-identification system"""

    def __init__(self, video_path: str, output_dir: str = "outputs"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.setup_output_directory()

        # Initialize system components
        self.detector = None
        self.tracker = None
        self.feature_extractor = None
        self.initialize_components()

        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'unique_players': 0,
            'processing_time': 0,
            'avg_fps': 0
        }

    def setup_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "data"), exist_ok=True)

    def initialize_components(self):
        """Initialize detector, tracker, and feature extractor"""
        try:
            from yolo_integration import YOLOPlayerDetector, EnhancedFeatureExtractor
            self.detector = YOLOPlayerDetector()
            self.feature_extractor = EnhancedFeatureExtractor()
            logger.info("YOLO detector initialized successfully")
        except Exception as e:
            logger.warning(f"YOLO initialization failed: {e}")
            logger.info("Using fallback detector")
            from player_reid_system import MockObjectDetector  # Ensure you have this!
            self.detector = MockObjectDetector()
            self.feature_extractor = None

        from player_reid_system import PlayerTracker
        self.tracker = PlayerTracker(
            max_disappeared=30,
            min_hits=3,
            iou_threshold=0.3
        )

        if self.feature_extractor:
            self.tracker.extract_features = lambda frame, bbox: self.feature_extractor.extract_features(frame, bbox)

    def process_video(self, show_video=True, save_video=True):
        """Main video processing loop"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

        out = None
        output_video_path = None
        if save_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video_path = os.path.join(self.output_dir, "videos", f"tracked_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0
        tracking_data = []

        import time
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            detections = self.detector.detect_players(frame)
            total_detections += len(detections)

            players = self.tracker.update(frame, detections)

            frame_data = {
                'frame': frame_count,
                'timestamp': frame_count / fps,
                'players': []
            }

            for pid, player in players.items():
                frame_data['players'].append({
                    'id': pid,
                    'bbox': list(map(int, player.bbox)),
                    'center': tuple(map(int, player.center)),
                    'confidence': player.confidence,
                    'trajectory_length': len(player.trajectory)
                })

            tracking_data.append(frame_data)

            self.draw_visualization(frame, players, frame_count, fps)

            if show_video:
                cv2.imshow('Player Re-ID Demo', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_video and out:
                out.write(frame)

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        end_time = time.time()
        elapsed = end_time - start_time

        self.stats.update({
            'total_frames': frame_count,
            'total_detections': total_detections,
            'unique_players': self.tracker.track_id_counter,
            'processing_time': elapsed,
            'avg_fps': frame_count / elapsed if elapsed > 0 else 0
        })

        self.save_results(tracking_data, output_video_path)
        logger.info("Video processing completed!")
        self.print_statistics()

    def draw_visualization(self, frame, players, frame_count, fps):
        """Draw enhanced visualization with type-safe coordinates"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]

        header_height = 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], header_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        cv2.putText(frame, f"Frame: {frame_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Players: {len(players)}", (150, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Time: {frame_count / fps:.1f}s", (300, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for pid, player in players.items():
            x1, y1, x2, y2 = map(int, player.bbox)
            color = colors[pid % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{pid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cx, cy = map(int, player.center)
            cv2.circle(frame, (cx, cy), 3, color, -1)

            if len(player.trajectory) > 1:
                for i in range(1, len(player.trajectory)):
                    p1 = tuple(map(int, player.trajectory[i - 1]))
                    p2 = tuple(map(int, player.trajectory[i]))
                    cv2.line(frame, p1, p2, color, 2)

    def save_results(self, tracking_data, output_video_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_path = os.path.join(self.output_dir, "data", f"tracking_{timestamp}.json")
        with open(data_path, 'w') as f:
            json.dump(tracking_data, f, indent=2)

        stats_path = os.path.join(self.output_dir, "data", f"stats_{timestamp}.json")
        stats_data = {
            'video_path': self.video_path,
            'output_video_path': output_video_path,
            'statistics': self.stats
        }
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)

        logger.info(f"Results saved: {data_path} and {stats_path}")

    def print_statistics(self):
        print("=" * 50)
        print("PROCESSING STATISTICS")
        print("=" * 50)
        for k, v in self.stats.items():
            print(f"{k}: {v}")
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Player Re-ID Demo")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    parser.add_argument("--no-save", action="store_true", help="Do not save output video")

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        logger.error(f"Video not found: {args.video_path}")
        return

    demo = PlayerReIDDemo(args.video_path, args.output_dir)
    demo.process_video(show_video=not args.no_display, save_video=not args.no_save)

if __name__ == "__main__":
    main()
