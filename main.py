import cv2
import numpy as np
import supervision as sv
from supervision import LabelAnnotator, RoundBoxAnnotator
from ultralytics import YOLOE

video_path = "videos/Intersection.mp4"

frame_generator = sv.get_video_frames_generator(video_path)
video_info = sv.VideoInfo.from_video_path(video_path)

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg.pt")
# Set text prompt to detect specific classes
names = ["car", "motorcycle", "bus", "truck"]
model.set_classes(names, model.get_text_pe(names))
# Load the model to GPU
model = model.to("cuda:0")

colors = sv.ColorPalette.DEFAULT

# Define the polygon zone
polygons = [
    np.array([[1, 333], [499, 339], [482, 818], [-2, 804]]),
    np.array([[1917, 267], [1917, 853], [1216, 837], [1261, 259]]),
    np.array([[1197, 238], [651, 314], [646, 2], [1111, 2], [1121, 143]]),
    np.array([[613, 965], [604, 889], [556, 834], [1145, 843], [1083, 900], [1059, 974], [1056, 1077], [611, 1077]]),
]

zones = [sv.PolygonZone(polygon=polygon) for polygon in polygons]

# Define the annotators
round_box_annotators = []
label_annotators = []
zone_annotators = []

for index, zone in enumerate(zones):
    round_box_annotators.append(RoundBoxAnnotator(color=colors.by_idx(index)))
    label_annotators.append(LabelAnnotator(text_position=sv.Position.TOP_CENTER, color=colors.by_idx(index)))
    zone_annotators.append(
        sv.PolygonZoneAnnotator(zone=zone, color=colors.by_idx(index), thickness=4, text_thickness=8, text_scale=4)
    )


for frame_index, frame in enumerate(frame_generator):
    # Resize the frame
    # frame = cv2.resize(frame, (640, 384))

    if frame_index == 0:
        cv2.imwrite("PolygonZone/first_frame.jpg", frame)
        print(
            "saved first frame as 'first_frame.jpg' in the PolygonZone folder.\n"
            "you can use it to create a polygon zones using https://polygonzone.roboflow.com/"
        )

    results = model.predict(frame, agnostic_nms=True, verbose=False)

    detections = sv.Detections.from_ultralytics(results[0])
    annotated_frame = frame.copy()

    for zone, round_box_annotator, label_annotator, zone_annotator in zip(
        zones, round_box_annotators, label_annotators, zone_annotators
    ):
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        annotated_frame = round_box_annotator.annotate(scene=annotated_frame, detections=detections_filtered)

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(detections_filtered["class_name"], detections_filtered.confidence)
        ]

        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections_filtered, labels=labels)

        annotated_frame = zone_annotator.annotate(scene=annotated_frame)

    annotated_frame = cv2.resize(annotated_frame, (1530, 780))

    # Display the annotated frame using OpenCV
    cv2.imshow("ZoneVision", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting...")
        break

cv2.destroyAllWindows()  # Close the OpenCV window
