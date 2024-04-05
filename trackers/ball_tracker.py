from ultralytics import YOLO
import cv2
import pickle

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_remain=False, remain_path=None):
        ball_detections = []

        if read_from_remain and remain_path is not None:
            with open(remain_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if remain_path is not None:
            with open(remain_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def draw_boundingboxes(self, video_frames, player_detections):
        output_video_frames = []
        #zip allows us to loop over 2 lists at the same time
        for frame, ball_dict in zip(video_frames, player_detections):
            #draw bounding boxes
            for track_id, boundingbox in ball_dict.items():
                x1, y1, x2, y2 = boundingbox
                #writes text ontop of frame
                cv2.putText(frame, f"Ball ID: {track_id}", (int(boundingbox[0]), int(boundingbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                #corners of the frame w/ RGB color border
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames
