from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import supervision as sv
from tqdm.auto import tqdm

from logger import get_logger

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()
DEFAULT_MODEL_PATH = Path(__file__).parent.joinpath("./models/rev0/yolow-l.onnx")

logger = get_logger(__name__)


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4, text_scale=0.5, text_thickness=1)


def visualize(image, recognitions):
    labels = [
        f"{recognition['label']} {recognition['confidence']:0.2f}"
        for recognition in recognitions
    ]
    bboxes = [recognition["bbox"] for recognition in recognitions]
    scores = [recognition["confidence"] for recognition in recognitions]
    orig_classes = [text.rsplit(" ", 1)[0] for text in labels]
    labels_set = set(orig_classes)
    labels_nx = {label: i for i, label in enumerate(labels_set)}
    class_ids = np.asarray([labels_nx[class_] for class_ in orig_classes])
    detections = sv.Detections(
        xyxy=np.asarray(bboxes), class_id=class_ids, confidence=np.asarray(scores)
    )

    image_ = image.copy()
    image_ = BOUNDING_BOX_ANNOTATOR.annotate(image_, detections)
    image_ = LABEL_ANNOTATOR.annotate(image_, detections, labels=labels)
    return image_


def preprocess(image, size=(640, 640)):
    h, w = image.shape[:2]
    max_size = max(h, w)
    scale_factor = size[0] / max_size
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    pad_image = np.zeros((max_size, max_size, 3), dtype=image.dtype)
    pad_image[pad_h : h + pad_h, pad_w : w + pad_w] = image
    image = cv2.resize(pad_image, size, interpolation=cv2.INTER_LINEAR).astype(
        "float32"
    )
    image /= 255.0
    image = image[None]
    return image, scale_factor, (pad_h, pad_w)


def _load_labels(labels_file: Path) -> list[str]:
    """Read labels from file

    Args:
        labels_file: Path to labels file. File is a text file and should contain a single line of text in which labels
            are separated by commas. There can be additional whitespace between commas and text lines which will be
            stripped.
    """
    with open(labels_file, "r") as f:
        labels = f.read().strip().split(",")
    labels = [label.strip() for label in labels]
    return labels


class Model:
    def __init__(self, onnx_file: Optional[Path] = None):
        # Initialize model session
        if onnx_file is None:
            onnx_file = DEFAULT_MODEL_PATH
        self.ort_session = ort.InferenceSession(onnx_file)
        self.labels = _load_labels(onnx_file.parent.joinpath("labels.txt"))
        logger.info("Loaded model with labels:", str(self.labels))

    def predict(self, image: np.ndarray, confidence_threshold: float = 0.05):
        """Predict labels for image

        Returns: List of dictionaries with recognitions. Each dictionary has the following entries: label,
            bounding box, bounding box centroid, confidence.
        """
        h, w = image.shape[:2]
        image, scale_factor, pad_param = preprocess(image[:, :, [2, 1, 0]])
        results = self.ort_session.run(
            ["num_dets", "labels", "scores", "boxes"],
            {"images": image.transpose((0, 3, 1, 2))},
        )
        num_dets, labels, scores, bboxes = results
        num_dets = num_dets[0][0]
        labels = labels[0, :num_dets]
        scores = scores[0, :num_dets]
        bboxes = bboxes[0, :num_dets]
        bboxes_orig = bboxes.copy().round().astype("int")
        # Clip bounding boxes to image size
        bboxes_orig[:, 0::2] = np.clip(bboxes_orig[:, 0::2], 0, w)
        bboxes_orig[:, 1::2] = np.clip(bboxes_orig[:, 1::2], 0, h)

        bboxes /= scale_factor
        bboxes -= np.array([pad_param[1], pad_param[0], pad_param[1], pad_param[0]])
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
        bboxes = bboxes.round().astype("int")

        recognitions = []
        for label, score, bbox, bbox_orig in zip(labels, scores, bboxes, bboxes_orig):
            if score < confidence_threshold:
                continue
            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            average_color = (
                image[0][bbox_orig[1] : bbox_orig[3], bbox_orig[0] : bbox_orig[2]]
                .mean(axis=(0, 1))
                .tolist()
            )
            recognitions.append(
                {
                    "label": self.labels[label],
                    "bbox": bbox,
                    "centroid": centroid,
                    "confidence": score,
                    "average_color": average_color,
                }
            )
        return recognitions

    def visualize(self, image, recognitions):
        return visualize(image, recognitions)


if __name__ == "__main__":
    model = Model()

    output_dir = Path(__file__).parent.joinpath("./data/processed")
    if not Path(output_dir).exists:
        Path(output_dir).mkdir()

    images = list(Path(__file__).parent.joinpath("./data/raw").glob("*.png"))

    print("Start to inference.")
    progress_bar = tqdm(total=len(images))

    for img in images:
        image = cv2.imread(str(img))
        recognitions = model.predict(image)
        recognitions_table = []
        for nx, recognition in enumerate(recognitions):
            recognitions_table.append(
                {
                    "id": nx,
                    "label": recognition["label"],
                    "position": recognition["centroid"],
                    "confidence": round(recognition["confidence"], 2),
                    "color_rgb": [
                        np.clip(int(x * 255), 0, 255)
                        for x in recognition["average_color"]
                    ],
                }
            )
        robot_vacuum_position = (799.0, 33.5)
        # Find the row with the robot vacuum, place it on top of the list, and disregard all other rows with the label 'robot vacuum'
        robot_vacuum_row = None
        for nx, recognition in enumerate(recognitions_table):
            if recognition["label"] == "robot vacuum":
                robot_vacuum_row = recognition
                break
        if robot_vacuum_row is not None:
            recognitions_table.remove(robot_vacuum_row)
            recognitions_table.insert(0, robot_vacuum_row)

        recognitions_table = pd.DataFrame(recognitions_table)
        distance_table = pd.DataFrame(
            np.nan, index=recognitions_table["id"], columns=recognitions_table["id"]
        )
        # add distance between recognitions
        for nx1, recognition1 in recognitions_table.iterrows():
            for nx2, recognition2 in recognitions_table.iterrows():
                distance = np.linalg.norm(
                    np.array(recognition1["position"])
                    - np.array(recognition2["position"])
                )
                distance_table.loc[nx1, nx2] = distance
        distance_table = distance_table.astype(int)

        # Create DataFrame
        recognitions_table_markdown = recognitions_table.to_markdown(index=False)
        distance_table_markdown = distance_table.to_markdown()

        print(f"Recognitions: {recognitions}")

        image_out = visualize(image, recognitions)
        cv2.imwrite(str(Path(output_dir).joinpath(Path(img).name)), image_out)
        progress_bar.update()
    print("Finish inference")
