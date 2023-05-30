from typing import Iterable, Sequence
from types import NoneType
import os
import cv2
import numpy as np


IMAGES_DIR = 'input'
MODEL_PATH = 'model'
WEIGHTS_PATH = f'{MODEL_PATH}/yolov3_training_2000.weights'
CONFIG_PATH = f'{MODEL_PATH}/yolov3_testing.cfg'


def load_net() -> cv2.dnn.Net:
	net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
	return net


def gather_detection_data(outs: Iterable[Iterable[Sequence[int]]], size: tuple[int, int], confidence_threshold: float = .5) -> tuple[tuple[tuple[int, int, int, int], ...], tuple[float, ...], tuple[np.intp, ...]]:
	width, height = size
	class_ids = list()
	confidences = list()
	boxes = list()
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence <= confidence_threshold: continue
			# Object detected
			center_x = int(detection[0] * width)
			center_y = int(detection[1] * height)
			w = int(detection[2] * width)
			h = int(detection[3] * height)
			# Rectangle coordinates
			x = int(center_x - w / 2)
			y = int(center_y - h / 2)
			boxes.append([x, y, w, h])
			confidences.append(float(confidence))
			class_ids.append(class_id)
	return tuple(boxes), tuple(confidences), tuple(class_ids)


def mark_detected(image: cv2.Mat, boxes: tuple[tuple[int, int, int, int], ...], indices: Sequence[int], color: tuple[int, int, int], thickness: int = 2):
	for i, box in enumerate(boxes):
		if i not in indices: continue
		x, y, w, h = box
		vertex1 = x, y
		vertex2 = x + w, y + h
		cv2.rectangle(image, vertex1, vertex2, color, thickness)


def detect_weapons(net: cv2.dnn.Net, image: cv2.Mat, resize: (tuple[int, int] | NoneType) = None) -> tuple[cv2.Mat, tuple[float], int]:
	height, width, _ = image.shape
	if resize is not None:
		width, height = resize
		image = cv2.resize(image, (width, height))
	# Detecting objects
	blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
	net.setInput(blob)
	layer_names = net.getLayerNames()
	output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
	outs = net.forward(output_layers)
	boxes, confidences, class_ids = gather_detection_data(outs, size=(width, height), confidence_threshold=.5)
	indices = cv2.dnn.NMSBoxes(tuple(boxes), tuple(confidences), 0.5, 0.4)
	detected_count = len(indices)
	red_border_color = (0, 0, 255)
	mark_detected(image, boxes, indices, red_border_color, thickness=2)
	return image, confidences, detected_count


def open_image_window(image: cv2.Mat, label: str):
	cv2.imshow(label, image)
	exit_key = 27
	while cv2.getWindowProperty(label, cv2.WND_PROP_VISIBLE) >= 1:
		if cv2.waitKey(delay=1) == exit_key: return


def run_weapon_detection():
	net = load_net()
	resize_to = (850, 540)
	filenames = sorted(os.listdir(IMAGES_DIR))
	for filename in filenames:
		image_raw = cv2.imread(os.path.join(IMAGES_DIR, filename))
		if image_raw is None: continue
		image_detected, confidences, detected_count = detect_weapons(net, image_raw, resize_to)
		print(f'Detected {detected_count} items at "{filename}"')
		print(f'Configences for found in "{filename}":', confidences)
		print(f'Original "{filename}":\n', image_raw)
		print(f'With detections "{filename}":\n', filename, image_raw)
		open_image_window(image_detected, filename)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	try:
		run_weapon_detection()
	except (KeyboardInterrupt, EOFError):
		print(' exiting...')
		cv2.destroyAllWindows()
