from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        # creates a list of bounding boxes that are empty, meaning with no detection happening
        ball_positions = [x.get(1,[]) for x in ball_positions]
        #converts the list into a pandas data frame
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # implementing the interpolate to fill in missing position values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        #returns a list of dictionaries where 1 is track id and x is bounding box
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

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
