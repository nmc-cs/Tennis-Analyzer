from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker

def main():
    #reads the video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    #detects players & ball
    player_tracker = PlayerTracker(model_path = 'yolov8x')
    player_detections = player_tracker.detect_frames(video_frames, read_from_remain=True, remain_path="tracker_remains/player_detections.pkl")

    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_remain=True, remain_path="tracker_remains/ball_detections.pkl")

    #drawing output of player bounding boxes & ball bounding boxes
    output_video_frames = player_tracker.draw_boundingboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_boundingboxes(video_frames, ball_detections)

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
