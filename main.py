# -*- coding:utf-8 -*-
import cv2
import time
import argparse

from datetime import timezone
import datetime

import os.path

import numpy as np
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.tensorflow_loader import load_tf_model, tf_inference

sess, graph = load_tf_model('models/face_mask_detection.pb')
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
id2color = {0: (0, 255, 0), 1: (0, 0, 255)}


def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :return:
    '''

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Converto to RGB

    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)
    y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)

    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    # FIXME: Quick & dirty solution
    use_detected_class = os.path.exists("models/new_face_mask_detection.pb")

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx] if use_detected_class else 1 # FIXME: Quick & dirty solution
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    return output_info

def post_data(outdata:dict, url:str, user:str, passw:str):
    '''
    Function responsible for posting the detection data on the RESTful API
    :param outdata: Dictionary representing the data
    :param url: Address of the RESTful API server
    :param user: Username for Basic Access Authentication
    :param passw: Password for Basic Access Authentication
    :return:
    '''
    import requests
    from requests.auth import HTTPBasicAuth

    try:
        requests.post(url, json=outdata, auth=HTTPBasicAuth(user, passw), timeout=0.05)
    except Exception as e:
        print(str(e))

if __name__ == "__main__":
    # Example usage: python3 main.py --url "https://127.0.0.1/app/v1/core" --auth_username "admin" --auth_password "password"
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--camera', type=int, default=0, help='Set input camera device')
    parser.add_argument('--frame_interval', type=int, default=30, help='Set the frame interval for post data request')
    parser.add_argument('--url', type=str, help='Url of the RESTful API target for the post data request')
    parser.add_argument('--auth_username', type=str, help='Username for Basic Access Authentication required by the RESTful API')
    parser.add_argument('--auth_password', type=str, help='Password for Basic Access Authentication required by the RESTful API')
    parser.add_argument('--no_show', default=False, action='store_true', help='Disable video output')
    args = parser.parse_args()

    show_results = not args.no_show

    # Open camera device
    cap = cv2.VideoCapture(args.camera)
    # cap = cv2.VideoCapture("nvarguscamerasrc ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER) # For Raspberry Pi Camera Module v2
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Main inference loop
    frame_nr = 0
    while True:
        start_stamp = time.time()

        # Grab frames
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_nr += 1

        read_frame_stamp = time.time()

        # Fix frame aspect ratio for Raspberry Pi camera
        # 1920x1080(16:9) -> 1440x1080(4:3)
        # frame = frame[0:1080, 240:1920-240, :]

        # Run inference
        detections = inference(frame,
                               conf_thresh=0.5,
                               iou_thresh=0.5,
                               target_shape=(260, 260),
                               draw_result=True)

        inference_stamp = time.time()

        # Post detection results
        if frame_nr % args.frame_interval == 0:
            detected_classes = [0 , 0]
            for detection in detections:
                class_id = detection[0]
                detected_classes[class_id] += 1

            outdict = {
                "mask": detected_classes[0],
                "no_mask": detected_classes[1],
                "timestamp": datetime.datetime.now(timezone.utc).timestamp()
            }

            post_data(outdict, args.url, args.auth_username, args.auth_password)

        # Draw results
        if show_results:
            for detection in detections:
                # Classification information
                class_id, conf = detection[0], detection[1]
                # Detection information
                (xmin, ymin, xmax, ymax) = (detection[2], detection[3], detection[4], detection[5])

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), id2color[class_id], 2)
                cv2.putText(frame, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, id2color[class_id])

            # Show results
            cv2.putText(frame, "FPS: %.2f" % (1/(inference_stamp - start_stamp)), (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, "Inference time: %.4f" % (inference_stamp - read_frame_stamp), (10, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            # frame = cv2.resize(frame, (640, 480))
            cv2.imshow('image', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    # Clean exit
    cap.release()
    if show_results:
        cv2.destroyAllWindows()
