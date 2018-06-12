from __future__ import print_function
import argparse
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageColor
import collections
import matplotlib.pyplot as plt
from lxml import etree, objectify
# This is needed since this file is stored in the object_detection folder.

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

'''
EJEMPLO
find -L /root/databases/temp -type f -name "*.jpg" | sort -t '\0' -n > images_vp-camara-019_2018-05-02.txt

python images_test.py \
    -m /root/models/tf-object-detection-api/train/Trucks-detector/faster_rcnn_resnet101/train/inference_graph/frozen_inference_graph.pb \
    -f images_vp-camara-019_2018-05-02.txt \
    -l data/trucks5_label_map.pbtxt \
    -r images_vp-camara-019_2018-05-02_faster_rcnn_resnet101_detections.txt
'''


def parse_args():
    """Handle the command line arguments.    
    Returns:
        Output of argparse.ArgumentParser.parse_args.
    """
    parser = argparse.ArgumentParser(
        description='Batch image detection using freezed graph')
    parser.add_argument('-m', '--frozen_model_filename',
                        help="Frozen model file to import", type=str, required=True)
    parser.add_argument('-f', '--input_file', type=str,
                        help="Input text file, one image per line", required=True)
    parser.add_argument('-l', '--labels_file', type=str,
                        help="Class labels pbtxt file path", required=True)
    parser.add_argument('-r', '--output_file', default=None, type=str,
                        help="Output file name for prediction probabilities")
    parser.add_argument('-d', '--output_dir', default=None, type=str,
                        help="Output dir to move images with detected objects drawn")
    parser.add_argument('-t', '--thresholds', default=None, type=str,
                        help="Thresholds to use to detect objects, defaults to 0.5 . One per class")
    args = parser.parse_args()
    return args


# Loading label map
# Label maps map indices to category names, so that when our convolution
# network predicts 5, we know that this corresponds to airplane.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
def load_category_index(path_to_labels):
    '''label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, len(label_map.item), use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index'''
    items = []
    with open(path_to_labels) as f:
        line = f.readline()
        while line:
            data = line.strip()
            if data == "item {":
                item = {}
                line = f.readline()
                while line:
                    data = line.strip()
                    if data == "}":
                        items.append(item)
                        break
                    data = data.split(":")
                    item[data[0].strip()] = data[1].strip().replace('"', '')
                    line = f.readline()
            line = f.readline()

    category_index = {}
    for item in items:
        cat_id = int(item['id'])
        category_index[cat_id] = {}
        for key, val in item.items():
            category_index[cat_id][key] = val
    return category_index


# Load a (frozen) Tensorflow model into memory.
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="")
    return graph


def get_input_output_tensors(graph):
    # Get handles to input and output tensors
    ops = graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}

    for key in ['num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = graph.get_tensor_by_name(tensor_name)

    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates 
        # to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    return image_tensor, tensor_dict


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def reframe_box_masks_to_image_masks(box_masks,
                                     boxes, image_height,
                                     image_width):
    """Transforms the box masks back to full image masks.

    Embeds masks in bounding boxes of larger masks whose shapes correspond to
    image shape.

    Args:
      box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
      boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
             corners. Row i contains [ymin, xmin, ymax, xmax] of the box
             corresponding to mask i. Note that the box corners are in
             normalized coordinates.
      image_height: Image height. The output mask will have the same height as
                    the image height.
      image_width: Image width. The output mask will have the same width as the
                   image width.

    Returns:
      A tf.float32 tensor of size [num_masks, image_height, image_width].
    """
    # TODO(rathodv): Make this a public function.
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
        boxes = tf.reshape(boxes, [-1, 2, 2])
        min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
        max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
        transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
        return tf.reshape(transformed_boxes, [-1, 4])

    box_masks = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks)[0]
    unit_boxes = tf.concat([tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
    image_masks = tf.image.crop_and_resize(image=box_masks,
                                           boxes=reverse_boxes,
                                           box_ind=tf.range(num_boxes),
                                           crop_size=[image_height, image_width],
                                           extrapolation_value=0.0)
    return tf.squeeze(image_masks, axis=3)


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box
                                            (each to be shown on its own line).
        use_normalized_coordinates: If True (default), treat coordinates
            ymin, xmin, ymax, xmax as relative to the image.    Otherwise treat
            coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin*im_width, xmax*im_width,
                                      ymin*im_height, ymax*im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
              (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', int(im_height*0.04))
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
                [(left, text_bottom - text_height - 2 * margin),
                 (left + text_width, text_bottom)], fill=color)
        draw.text(
                (left + margin, text_bottom - text_height - margin),
                display_str, fill='black', font=font)
        text_bottom -= text_height - 2 * margin


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
    """Draws keypoints on an image.

    Args:
        image: a PIL.Image object.
        keypoints: a numpy array with shape [num_keypoints, 2].
        color: color to draw the keypoints with. Default is red.
        radius: keypoint radius. Default value is 2.
        use_normalized_coordinates: if True (default), treat keypoint values as
            relative to the image.    Otherwise treat them as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    keypoints_x = [k[1] for k in keypoints]
    keypoints_y = [k[0] for k in keypoints]
    if use_normalized_coordinates:
        keypoints_x = tuple([im_width * x for x in keypoints_x])
        keypoints_y = tuple([im_height * y for y in keypoints_y])
    for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
        draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                      (keypoint_x + radius, keypoint_y + radius)],
                     outline=color, fill=color)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
    """Draws mask on an image.

    Args:
        image: uint8 numpy array with shape (img_height, img_height, 3)
        mask: a uint8 numpy array of shape (img_height, img_height) with
            values between either 0 or 1.
        color: color to draw the keypoints with. Default is red.
        alpha: transparency value between 0 and 1. (default: 0.4)

    Raises:
        ValueError: On incorrect data type for image or masks.
    """
    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')
    if mask.dtype != np.uint8:
        raise ValueError('`mask` not of type np.uint8')
    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError('`mask` elements should be in [0, 1]')
    if image.shape[:2] != mask.shape:
        raise ValueError('The image has spatial dimensions %s but the mask has '
                         'dimensions %s' % (image.shape[:2], mask.shape))
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)

    solid_color = np.expand_dims(
                      np.ones_like(mask), axis=2)*np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
    pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert('RGB')))


def visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        instance_boundaries=None,
        keypoints=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        agnostic_mode=False,
        line_thickness=4,
        groundtruth_box_visualization_color='black',
        skip_scores=False,
        skip_labels=False):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and 
    returns  that same image.

    Args:
        image: uint8 numpy array with shape (img_height, img_width, 3)
        boxes: a numpy array of shape [N, 4]
        classes: a numpy array of shape [N]. Note that class indices are 1-based,
            and match the keys in the label map.
        scores: a numpy array of shape [N] or None.    If scores=None, then
            this function assumes that the boxes to be plotted are groundtruth
            boxes and plot all boxes as black with no classes or scores.
        category_index: a dict containing category dictionaries (each holding
            category index `id` and category name `name`) keyed by category indices.
        instance_masks: a numpy array of shape [N, image_height, image_width] with
            values ranging between 0 and 1, can be None.
        instance_boundaries: a numpy array of shape [N, image_height, image_width]
            with values ranging between 0 and 1, can be None.
        keypoints: a numpy array of shape [N, num_keypoints, 2], can
            be None
        use_normalized_coordinates: whether boxes is to be interpreted as
            normalized coordinates or not.
        max_boxes_to_draw: maximum number of boxes to visualize.    If None, draw
            all boxes.
        min_score_thresh: minimum score threshold for a box to be visualized
        agnostic_mode: boolean (default: False) controlling whether to evaluate in
            class-agnostic mode or not.    This mode will display scores but ignore
            classes.
        line_thickness: integer (default: 4) controlling line width of the boxes.
        groundtruth_box_visualization_color: box color for visualizing groundtruth
            boxes
        skip_scores: whether to skip score when drawing a single detection
        skip_labels: whether to skip label when drawing a single detection

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in category_index.keys():
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = '{}%'.format(int(100*scores[i]))
                    else:
                        display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                else:
                    box_to_color_map[box] = STANDARD_COLORS[
                            classes[i] % len(STANDARD_COLORS)]

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        if instance_masks is not None:
            draw_mask_on_image_array(
                    image,
                    box_to_instance_masks_map[box],
                    color=color
            )
        if instance_boundaries is not None:
            draw_mask_on_image_array(
                    image,
                    box_to_instance_boundaries_map[box],
                    color='red',
                    alpha=1.0
            )
        draw_bounding_box_on_image(
                image,
                ymin,
                xmin,
                ymax,
                xmax,
                color=color,
                thickness=line_thickness,
                display_str_list=box_to_display_str_map[box],
                use_normalized_coordinates=use_normalized_coordinates)
        if keypoints is not None:
            draw_keypoints_on_image(
                    image,
                    box_to_keypoints_map[box],
                    color=color,
                    radius=line_thickness / 2,
                    use_normalized_coordinates=use_normalized_coordinates)

    return image


# Print iterations progress
def printProgressBar(iteration,
                     total,
                     prefix='',
                     suffix='',
                     decimals=1,
                     length=100,
                     fill='#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def detections_to_voc_annotation(image_path,
                                 image_width,
                                 image_height,
                                 detection_boxes,
                                 detection_classes,
                                 detection_scores,
                                 category_index):
    image_name = os.path.basename(image_path)

    E = objectify.ElementMaker(annotate=False)
    annotation = E.annotation(
            E.folder('JPEGImages'),
            E.filename(image_name),
            E.path(image_path),
            E.source(
                E.database('Unknown')
                ),
            E.size(
                E.width(image_width),
                E.height(image_height),
                E.depth('3'),
                ),
            E.segmented(0)
            )            
    detecciones_idx = [i for i, s in enumerate(detection_scores) if s >= 0.5]
    for idx in detecciones_idx:
        E = objectify.ElementMaker(annotate=False)
        instance = E.object(
            E.name(category_index[detection_classes[idx]]['name']),
            E.pose('Unspecified'),
            E.truncated(0),
            E.difficult(0),
            E.bndbox(
                E.xmin(int(image_width*detection_boxes[idx][1])),
                E.ymin(int(image_height*detection_boxes[idx][0])),
                E.xmax(int(image_width*detection_boxes[idx][3])),
                E.ymax(int(image_height*detection_boxes[idx][2])),
                ),
            )
        annotation.append(instance)
    return annotation


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    args = parse_args()

    # Cargamos nuestro diccionario de ids - clases
    category_index = load_category_index(args.labels_file)

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We access the input and output nodes
    image_tensor, tensor_dict = get_input_output_tensors(graph)

    # Obtenemos los paths de las imagenes a procesar
    image_paths = [s.strip() for s in open(args.input_file)]
    nimages = len(image_paths)

    # Verificamos donde escribir los resultados
    fout = sys.stdout
    if args.output_file is not None:
        fout = open(args.output_file, 'w')
        print('')
        printProgressBar(0, nimages, prefix='Progress:', suffix='Complete', length=50)
    write_header = True

    if args.output_dir != None and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        for progress_idx, image_path in enumerate(image_paths):
            # Cargamos la imagen
            image = Image.open(image_path)
            # Actual detection.
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]        
            # Obtenemos las detecciones relevantes
            detecciones_idx = [i for i, s in enumerate(output_dict['detection_scores']) if s >= 0.5]
            image_name = os.path.basename(image_path)
            image_width, image_height = image.size
            row = [image_path, image_name, image_width, image_height, len(detecciones_idx)]
            for idx in detecciones_idx:
                row.append(output_dict['detection_scores'][idx])                           # score
                row.append(category_index[output_dict['detection_classes'][idx]]['name'])  # class
                row.append(output_dict['detection_classes'][idx])                          # class id
                row.extend(output_dict['detection_boxes'][idx])

            # Escribimos en el archivo de salida
            if write_header:
                header = ['image_path',
                          'image_name',
                          'image_width',
                          'image_height',
                          'n_detections',
                          'obj_score_0',
                          'obj_class_0',
                          'obj_class_id_0',
                          'obj_ymin_0',
                          'obj_xmin_0',
                          'obj_ymax_0',
                          'obj_xmax_0']
                fout.write('\t'.join(header) + "\n")
                write_header = False
            fout.write('\t'.join([str(e) for e in row]) + "\n")

            if args.output_file is not None:
                printProgressBar(progress_idx+1, nimages, 
                                 prefix='Progress:',
                                 suffix='Complete',
                                 length=50)

            if args.output_dir is not None and len(detecciones_idx) > 0:
                # Visualization of the results of a detection.
                visualize_boxes_and_labels_on_image_array(
                        image,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=8)
                image.save(os.path.join(args.output_dir, image_name))
                annotation = detections_to_voc_annotation(
                        image_path,
                        image_width,
                        image_height,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index
                )
                etree.ElementTree(annotation).write(
                        os.path.join(
                            args.output_dir,
                            image_name[:-3]+'xml'), pretty_print=True)
            #plt.imshow(image)
            #plt.show()
