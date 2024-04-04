from ultralytics import YOLO
import cv2

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        player_detections = []

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        return player_detections



    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_boundingboxes(self, video_frames, player_detections):
        output_video_frames = []
        #zip allows us to loop over 2 lists at the same time
        for frame, player_dict in zip(video_frames, player_detections):
            #draw bounding boxes
            for track_id, boundingbox in player_dict.items():
                x1, y1, x2, y2 = boundingbox
                #writes text ontop of frame
                cv2.putText(frame, f"Player ID: {track_id}", (int(boundingbox[0], int(boundingbox[1] - 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                #corners of the frame w/ RGB color border
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames
