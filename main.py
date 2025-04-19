"""
ZoneVision: Detect objects within custom polygonal zones in a video stream.
"""
import cv2
import numpy as np
import supervision as sv
from supervision import LabelAnnotator, RoundBoxAnnotator
from ultralytics import YOLOE

# --- Configuration ---
VIDEO_PATH = "videos/Intersection.mp4"
MODEL_PATH = "yoloe-11l-seg.pt"
DEVICE = "cuda:0"
DETECT_CLASSES = ["car", "motorcycle", "bus", "truck"]
RESIZE_OUTPUT = (1530, 780)
FIRST_FRAME_PATH = "PolygonZone/first_frame.jpg"
POLYGONS = [
    np.array([[1, 333], [499, 339], [482, 818], [-2, 804]]),
    np.array([[1917, 267], [1917, 853], [1216, 837], [1261, 259]]),
    np.array([[1197, 238], [651, 314], [646, 2], [1111, 2], [1121, 143]]),
    np.array([[613, 965], [604, 889], [556, 834], [1145, 843], [1083, 900], [1059, 974], [1056, 1077], [611, 1077]]),
]


def setup_model(model_path: str, device: str, class_names: list) -> YOLOE:
    """Load and configure the YOLOE model."""
    model = YOLOE(model_path)
    model.set_classes(class_names, model.get_text_pe(class_names))
    return model.to(device)


def create_zones_and_annotators(polygons, color_palette):
    """Create zones and their corresponding annotators."""
    zones = [sv.PolygonZone(polygon=polygon) for polygon in polygons]
    round_box_annotators = []
    label_annotators = []
    zone_annotators = []
    for idx, zone in enumerate(zones):
        color = color_palette.by_idx(idx)
        round_box_annotators.append(RoundBoxAnnotator(color=color))
        label_annotators.append(LabelAnnotator(text_position=sv.Position.TOP_CENTER, color=color))
        zone_annotators.append(
            sv.PolygonZoneAnnotator(zone=zone, color=color, thickness=4, text_thickness=8, text_scale=4)
        )
    return zones, round_box_annotators, label_annotators, zone_annotators


def process_frame(frame, model, zones, round_box_annotators, label_annotators, zone_annotators):
    """Run detection and annotate the frame for all zones."""
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
    return cv2.resize(annotated_frame, RESIZE_OUTPUT)


def main():
    frame_generator = sv.get_video_frames_generator(VIDEO_PATH)
    model = setup_model(MODEL_PATH, DEVICE, DETECT_CLASSES)
    colors = sv.ColorPalette.DEFAULT
    zones, round_box_annotators, label_annotators, zone_annotators = create_zones_and_annotators(POLYGONS, colors)

    for frame_index, frame in enumerate(frame_generator):
        if frame_index == 0:
            cv2.imwrite(FIRST_FRAME_PATH, frame)
            print(
                f"saved first frame as '{FIRST_FRAME_PATH}' in the PolygonZone folder.\n"
                "you can use it to create polygon zones using https://polygonzone.roboflow.com/"
            )
        annotated_frame = process_frame(frame, model, zones, round_box_annotators, label_annotators, zone_annotators)
        cv2.imshow("ZoneVision", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting...")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
