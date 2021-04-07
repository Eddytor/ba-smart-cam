"""yolov4 tiny

This module handles the ml model. Also objects are handled.
"""
import statistics
import time
import cv2
import numpy as np
import tensorflow as tf
from absl import flags

import utility

FLAGS = flags.FLAGS
flags.DEFINE_integer('input_size', 416, 'size of img in height/width')
flags.DEFINE_string('classes_file', './data/classes.names', 'path to the .names file containing all classes')
flags.DEFINE_string('color_file', './data/color.txt', 'path to the color information for each class')
flags.DEFINE_string('model_path', './models/model_person_chicken_cat_car.tflite', 'path to the tflite model used')
flags.DEFINE_boolean('test', False, 'create testfiles as text and image')


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


class TfLiteInterpreter:
    obj_thresh = 0.5
    class_threshold = 0.5
    anchors = [[81, 82, 135, 169, 344, 319], [10, 14, 23, 27, 37, 58]]
    scales_x_y = [1.05, 1.05]

    # Start the Tf Lite Interpreter
    def __init__(self):
        self.obj_handler = ObjectsHandler()
        self.interpreter = tf.lite.Interpreter(model_path=FLAGS.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    # All steps for one frame
    def iteration_step(self, frame, image_data, cam, publish=True):
        image_data = image_data / 255.  # Int -> Float64
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], image_data)
        self.interpreter.invoke()
        class_pred = self.interpreter.get_tensor(self.output_details[0]['index'])
        box_pred = self.interpreter.get_tensor(self.output_details[1]['index'])
        pred = [class_pred, box_pred]

        # Compute the Yolo layers
        boxes = list()
        for i in range(len(TfLiteInterpreter.anchors)):
            # decode the output of the network
            boxes += self.decode_netout(pred[i][0], TfLiteInterpreter.anchors[i], TfLiteInterpreter.obj_thresh,
                                        FLAGS.input_size, 3, TfLiteInterpreter.scales_x_y[i])

        self.correct_yolo_boxes(boxes, cam.img_height, cam.img_width)
        self.do_nms(boxes, 0.5)
        if FLAGS.test:
            self.create_test(boxes, frame, cam.img_height, cam.img_width)
        v_boxes, v_labels, v_scores, v_colors, v_label_nums = self.get_boxes(boxes,
                                                                             TfLiteInterpreter.class_threshold)

        bboxes = v_boxes, v_scores, v_label_nums, len(v_boxes), v_colors
        self.obj_handler.append_object(bboxes=bboxes)
        image = self.draw_bbox(frame, bboxes)
        cam.last_img = image
        obj_found = self.obj_handler.object_iteration(cam, publish)
        return image, obj_found

    def decode_netout(self, netout, anchors, obj_thresh, net_size, nb_box, scales_x_y):
        grid_h, grid_w = netout.shape[:2]
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))

        boxes = []
        netout[..., :2] = self._sigmoid(netout[..., :2])  # x, y
        netout[..., :2] = netout[..., :2] * scales_x_y - 0.5 * (scales_x_y - 1.0)  # scale x, y

        netout[..., 4:] = self._sigmoid(netout[..., 4:])  # objectness + classes probabilities

        for i in range(grid_h * grid_w):

            row = i / grid_w
            col = i % grid_w

            for b in range(nb_box):
                # 4th element is objectness
                objectness = netout[int(row)][int(col)][b][4]

                if objectness > obj_thresh:
                    # print("objectness: ", objectness)

                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[int(row)][int(col)][b][:4]
                    x = (col + x) / grid_w  # center position, unit: image width
                    y = (row + y) / grid_h  # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / net_size  # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / net_size  # unit: image height

                    # last elements are class probabilities
                    classes = objectness * netout[int(row)][col][b][5:]
                    classes *= classes > obj_thresh
                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
                    boxes.append(box)
        return boxes

    @staticmethod
    def correct_yolo_boxes(boxes, image_h, image_w):
        for i in range(len(boxes)):
            # times 1.0 because of division with zero
            # images all the same dim
            boxes[i].xmin = int(boxes[i].xmin / 1.0 * image_w)
            boxes[i].xmax = int(boxes[i].xmax / 1.0 * image_w)
            boxes[i].ymin = int(boxes[i].ymin / 1.0 * image_h)
            boxes[i].ymax = int(boxes[i].ymax / 1.0 * image_h)

    @staticmethod
    def _interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    @staticmethod
    def _sigmoid(x):
        return 1. / (1. + np.exp(-x))

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
        union = w1 * h1 + w2 * h2 - intersect
        return float(intersect) / union

    def do_nms(self, boxes, nms_thresh):
        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return
        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])
            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]
                if boxes[index_i].classes[c] == 0:
                    continue
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                        boxes[index_j].classes[c] = 0

    # get all of the results above a threshold
    def get_boxes(self, boxes, thresh):
        v_boxes, v_labels, v_scores, v_colors = list(), list(), list(), list()
        v_label_nums = []
        # enumerate all boxes
        for box in boxes:
            # enumerate all possible labels
            for i in range(self.obj_handler.num_classes):

                # check if the threshold for this label is high enough
                if box.classes[i] > thresh:
                    v_boxes.append(box)
                    v_labels.append(self.obj_handler.classes[i])
                    v_label_nums.append(i)
                    v_scores.append(box.classes[i] * 100)
                    v_colors.append(self.obj_handler.colors[i])
                    # don't break, many labels may trigger for one box
        return v_boxes, v_labels, v_scores, v_colors, v_label_nums

    def draw_bbox(self, image, bboxes):
        image_h, image_w, _ = image.shape

        out_boxes, out_scores, out_classes, num_boxes, v_colors = bboxes
        for i in range(num_boxes):
            if int(out_classes[i]) < 0 or int(out_classes[i]) > self.obj_handler.num_classes:
                continue
            coor = [out_boxes[i].ymin, out_boxes[i].xmin, out_boxes[i].ymax, out_boxes[i].xmax]
            font_scale = 0.5
            score = out_scores[i]
            class_ind = int(out_classes[i])
            bbox_thick = int(0.5 * (image_h + image_w) / 600)
            top_left, bottom_right = (coor[1], coor[0]), (coor[3], coor[2])

            cv2.rectangle(image, top_left, bottom_right, v_colors[i], bbox_thick)

            label_txt = '%s: %.2f' % (self.obj_handler.classes[class_ind], score)
            t_size = cv2.getTextSize(label_txt, 0, font_scale, thickness=bbox_thick // 2)[0]
            coor_text_box_end = (top_left[0] + t_size[0], top_left[1] - t_size[1] - 5)
            coor_text_start = (top_left[0], (top_left[1] - 2))
            cv2.rectangle(image, top_left, coor_text_box_end, v_colors[i], -1)  # filled
            cv2.putText(image, label_txt, coor_text_start, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
        return image

    def create_test(self, boxes, frame, image_h, image_w):
        re_boxes, re_labels, = list(), list()
        re_label_nums = []
        labels = ""
        for j, box in enumerate(boxes):
            for i in range(self.obj_handler.num_classes):
                if box.classes[i] > TfLiteInterpreter.class_threshold:
                    re_boxes.append(box)
                    re_labels.append(i)
                    re_label_nums.append(i)
        for i in range(len(re_boxes)):
            center_x = ((boxes[i].xmax / image_w) + (boxes[i].xmin / image_w)) / 2
            center_y = ((boxes[i].ymax / image_h) + (boxes[i].ymin / image_h)) / 2
            width = (boxes[i].xmax / image_w) - (boxes[i].xmin / image_w)
            height = (boxes[i].ymax / image_h) - (boxes[i].ymin / image_h)
            labels += str(re_labels[i]) + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " +\
                      str(height) + "\n"
        path = "./test/image_" + str(time.time_ns())
        file_name_img = path + ".jpg"
        file_name_txt = path + ".txt"
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_name_img, frame)
        f = open(file_name_txt, "a")
        f.write(labels)
        f.close()



class ObjectsHandler:
    def __init__(self):
        self.detected_objects = []
        self.classes = utility.read_info(FLAGS.classes_file)
        colors_str = utility.read_info(FLAGS.color_file)
        self.colors = [tuple(map(int, colors_str[i].split(','))) for i in colors_str]
        self.num_classes = len(self.classes)

    def append_object(self, bboxes):
        classes_already_checked = []

        out_boxes, out_scores, out_classes, num_boxes, _ = bboxes
        for i in range(num_boxes):

            score = out_scores[i]
            class_ind = int(out_classes[i])

            checked = False
            for added_class in classes_already_checked:
                if added_class[0] == class_ind:
                    checked = True
                    if added_class[1] < score:
                        added_class[1] = score
            if not checked:
                classes_already_checked.append([class_ind, score])

        for found_classes in classes_already_checked:
            exists = False
            for objects in self.detected_objects:
                if found_classes[0] == objects.id:
                    exists = True
                    objects.count_recent = objects.count_recent + 1
                    objects.add_prob(found_classes[1])
                    objects.just_added = True
                    break
            if not exists:
                new_class = DetectedObject(self.classes[found_classes[0]], found_classes[0])
                new_class.add_prob(found_classes[1])
                self.detected_objects.append(new_class)

    def object_iteration(self, cam, publish=True):
        all_detec_data = []
        for obj in self.detected_objects:
            if obj.count_recent >= DetectedObject.recent:
                cam.last_img = cv2.cvtColor(cam.last_img, cv2.COLOR_RGB2BGR)
                obj.img_name = f"{cam.name}_{obj.name}_{utility.get_datetime(file_format=True)}.jpg"
                if publish:
                    obj.publish_mqtt(cam)
                else:  # currently the case, if on Google Cloud
                    obj.img_name = f"{obj.name}_{utility.get_datetime(file_format=True)}_{time.time_ns()}.jpg"
                    all_detec_data.append(obj.return_detected_obj())
            else:
                if not obj.just_added:
                    obj.decr_stats()
            obj.just_added = False
        if all_detec_data:
            return all_detec_data
        else:
            return None


class DetectedObject:
    recent = 5
    min_time_detected = 1  # in seconds

    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.count_recent = 1
        self.just_added = True
        self.prob_med = 0
        self.probs = []
        self.time = utility.get_datetime()
        self.img_name = None

    def publish_mqtt(self, cam):
        _, img_encode = cv2.imencode('.jpg', cam.last_img)
        data_encode = np.array(img_encode)
        str_encode = data_encode.tobytes()

        print(f"Cam {cam.id}: publish: {self.id}<:>{self.prob_med}<:>{self.img_name}<:>{self.time}")

        cam.client.publish(f"{cam.topic}/{cam.mqtt_topics['detection_info']}/{self.name}",
                           f"{self.id}<:>{self.prob_med}<:>{self.img_name}<:>{self.time}",
                           retain=False, qos=1)
        cam.client.publish(f"{cam.topic}/{cam.mqtt_topics['image']}/{self.img_name}",
                           str_encode,
                           retain=False, qos=1)
        self.reset_obj()
        return

    def return_detected_obj(self):
        data = [self.name, self.id, self.prob_med, self.time, self.img_name]
        self.reset_obj()
        return data

    def add_prob(self, prob):
        if len(self.probs) >= DetectedObject.recent:
            self.probs.pop(0)
        self.probs.append(prob)
        self.prob_med = statistics.median(self.probs)
        self.time = utility.get_datetime()

    def decr_stats(self):
        if self.count_recent > 0:
            self.count_recent = self.count_recent - 1
        if len(self.probs) > 0:
            self.probs.pop(0)
            if len(self.probs) == 0:
                self.prob_med = 0
            else:
                self.prob_med = statistics.median(self.probs)

    def reset_obj(self):
        self.count_recent = 0
        self.probs = []
        self.prob_med = 0
        self.just_added = False
