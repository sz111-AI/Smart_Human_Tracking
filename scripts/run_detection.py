from src.Human_Tracking.pipeline.process_video import process_video

if __name__ == "__main__":
    process_video(
        camera_ip="rtsp://admin:Think22wise@192.168.15.31/video",
        min_quality=0.65,
        quality_threshold=0.1
    )