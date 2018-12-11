# https://www.shiftedup.com/2018/10/10/confusion-matrix-in-object-detection-api-with-tensorflow
# https://github.com/svpino/tf_object_detection_cm
# Ejemplo:
# # Obtener detection record
# python object_detection/inference/infer_detections.py \
#    --input_tfrecord_paths=/path/to/input/tfrecord1,/path/to/input/tfrecord2 \
#    --output_tfrecord_path_prefix=/path/to/output/detections.tfrecord \
#    --inference_graph=/path/to/frozen_weights_inference_graph.pb
# # Correr el script
# python confusion_matrix.py --detections_record=testing_detections.record --label_map=label_map.pbtxt --output_dir=./bad_examples
import numpy as np
import tensorflow as tf
import io
from PIL import Image, ImageDraw, ImageFont

from object_detection.core import standard_fields
from object_detection.metrics import tf_example_parser
from object_detection.utils import label_map_util

flags = tf.app.flags

flags.DEFINE_string('label_map', None, 'Path to the label map')
flags.DEFINE_string('detections_record', None, 'Path to the detections record file')
flags.DEFINE_string('output_dir', None, 'Path to ouput dir to write incorrect detections')

FLAGS = flags.FLAGS

IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5


def compute_iou(groundtruth_box, detection_box):
    g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box.tolist())
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box.tolist())

    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)


def draw_box(draw, groundtruth_box, label, w, h, line_color, text_color, text_position):
    box = groundtruth_box * np.array([h, w, h, w])
    y0, x0, y1, x1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    yt = y0+1 if text_position == 'top' else y1-12
    draw.rectangle(((x0, y0), (x1, y1)), outline=line_color)
    draw.rectangle(((x0+1, y0+1), (x1-1, y1-1)), outline=line_color)
    draw.rectangle(((x0-1, y0-1), (x1+1, y1+1)), outline=line_color)
    draw.text((x0, yt), label, font=ImageFont.truetype("arial", 11), fill=text_color)


def save_incorrect_example(example, groundtruth_boxes, groundtruth_classes, detection_boxes, detection_classes, categories, errors):
    img_filename = (example.features.feature['image/filename'].bytes_list.value[0]).decode("utf-8")
    img_bytes = (example.features.feature['image/encoded'].bytes_list.value[0])
    w = int(example.features.feature['image/width'].int64_list.value[0])
    h = int(example.features.feature['image/height'].int64_list.value[0])

    # with open(img_filename, 'wb') as output:
    #    output.write(img_bytes)

    img = Image.open(io.BytesIO(img_bytes))
    draw = ImageDraw.Draw(img)
    for i in range(len(groundtruth_boxes)):
        box = groundtruth_boxes[i]
        label = [c['name'] for c in categories if c['id'] == groundtruth_classes[i]]
        draw_box(draw, box, label[0], w, h, "green", "white", "top")

    # for i in range(len(detection_boxes)):
    #    box = detection_boxes[i]
    #    label = [c['name'] for c in categories if c['id'] == detection_classes[i]]
    #    draw_box(draw, box, label[0], w, h, "blue", "blue", "bottom")

    for obj in errors['wrong_class']:
        box = obj['box']
        label = [c['name'] for c in categories if c['id'] == obj['class']]
        draw_box(draw, box, label[0], w, h, "red", "red", "bottom")

    for obj in errors['false_detected']:
        box = obj['box']
        label = [c['name'] for c in categories if c['id'] == obj['class']]
        draw_box(draw, box, label[0], w, h, "orange", "orange", "bottom")

    for obj in errors['not_detected']:
        box = obj['box']
        label = [c['name'] for c in categories if c['id'] == obj['class']]
        draw_box(draw, box, label[0], w, h, "yellow", "yellow", "top")

    img.save(FLAGS.output_dir+'/'+img_filename, "JPEG")


def process_detections(detections_record, categories):
    record_iterator = tf.python_io.tf_record_iterator(path=detections_record)
    data_parser = tf_example_parser.TfExampleDetectionAndGTParser()

    confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))

    image_index = 0
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        decoded_dict = data_parser.parse(example)

        image_index += 1

        if decoded_dict:
            groundtruth_boxes = decoded_dict[standard_fields.InputDataFields.groundtruth_boxes]
            groundtruth_classes = decoded_dict[standard_fields.InputDataFields.groundtruth_classes]

            detection_scores = decoded_dict[standard_fields.DetectionResultFields.detection_scores]
            detection_classes = decoded_dict[standard_fields.DetectionResultFields.detection_classes][detection_scores >= CONFIDENCE_THRESHOLD]
            detection_boxes = decoded_dict[standard_fields.DetectionResultFields.detection_boxes][detection_scores >= CONFIDENCE_THRESHOLD]

            matches = []

            if image_index % 100 == 0:
                print("Processed %d images" % (image_index))

            for i in range(len(groundtruth_boxes)):
                for j in range(len(detection_boxes)):
                    iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])

                    if iou > IOU_THRESHOLD:
                        matches.append([i, j, iou])

            matches = np.array(matches)
            if matches.shape[0] > 0:
                # Sort list of matches by descending IOU so we can remove duplicate detections
                # while keeping the highest IOU entry.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

                # Remove duplicate detections from the list.
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

                # Sort the list again by descending IOU. Removing duplicates doesn't preserve
                # our previous sort.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

                # Remove duplicate ground truths from the list.
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            errors = {"wrong_class": [], "not_detected": [], "false_detected": []}
            has_errors = False
            for i in range(len(groundtruth_boxes)):
                if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
                    confusion_matrix[groundtruth_classes[i] - 1][detection_classes[int(matches[matches[:, 0] == i, 1][0])] - 1] += 1
                    if groundtruth_classes[i] != detection_classes[int(matches[matches[:, 0] == i, 1][0])]:
                        # Clase equivocada
                        errors["wrong_class"].append({"box": detection_boxes[int(matches[matches[:, 0] == i, 1][0])],
                                                      "class": detection_classes[int(matches[matches[:, 0] == i, 1][0])]})
                        has_errors = True
                else:
                    confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1
                    # Objeto no detectado
                    errors["not_detected"].append({"box": groundtruth_boxes[i], "class": groundtruth_classes[i]})
                    has_errors = True

            for i in range(len(detection_boxes)):
                if matches.shape[0] > 0 and matches[matches[:, 1] == i].shape[0] == 0:
                    confusion_matrix[confusion_matrix.shape[0] - 1][detection_classes[i] - 1] += 1
                    # Objeto detectado incorrectamente
                    errors["false_detected"].append({"box": detection_boxes[i], "class": detection_classes[i]})

            if has_errors and FLAGS.output_dir is not None:
                save_incorrect_example(example, groundtruth_boxes, groundtruth_classes, detection_boxes, detection_classes, categories, errors)

        else:
            print("Skipped image %d" % (image_index))

    print("Processed %d images" % (image_index))

    return confusion_matrix


def display(confusion_matrix, categories):
    print("\nConfusion Matrix:")
    print(confusion_matrix, "\n")

    for i in range(len(categories)):
        id = categories[i]["id"] - 1
        name = categories[i]["name"]

        total_target = np.sum(confusion_matrix[id, :])
        total_predicted = np.sum(confusion_matrix[:, id])

        precision = float(confusion_matrix[id, id] / total_predicted)
        recall = float(confusion_matrix[id, id] / total_target)

        print('precision_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, precision))
        print('recall_{}@{}IOU: {:.2f}'.format(name, IOU_THRESHOLD, recall))


def main(argv):
    del argv
    required_flags = ['detections_record', 'label_map']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    label_map = label_map_util.load_labelmap(FLAGS.label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)

    confusion_matrix = process_detections(FLAGS.detections_record, categories)

    display(confusion_matrix, categories)


if __name__ == '__main__':
    tf.app.run(main)
