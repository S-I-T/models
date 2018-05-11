from __future__ import print_function
import argparse 
import numpy as np
import os
import sys
import tensorflow as tf

from io import StringIO
from PIL import Image

if tf.__version__ < '1.4.0':
	raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# This is needed since this file is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util

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
	parser = argparse.ArgumentParser(description='Batch image classification using VP2 model freezed graph')
	parser.add_argument('-m', '--frozen_model_filename', help="Frozen model file to import", type=str, required=True)
	parser.add_argument('-f','--input_file', type=str, help="Input text file, one image per line", required=True)
	parser.add_argument('-l','--labels_file', type=str, help="Class labels pbtxt file path", required=True)
	
	parser.add_argument('-r','--output_file', default=None, type=str, help="Output file name for prediction probabilities")
	parser.add_argument('-d','--output_dir', default=None, type=str, help="Output dir to move images with detected objects drawn")
	parser.add_argument('-t','--thresholds', default=None, type=str, help="Thresholds to use to detect objects, defaults to 0.5 . One per class")
	args = parser.parse_args()
	return args

#Loading label map
#Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. 
#Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
def load_category_index(path_to_labels):
	label_map = label_map_util.load_labelmap(path_to_labels)
	categories = label_map_util.convert_label_map_to_categories(label_map, len(label_map.item), use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	return category_index

#Load a (frozen) Tensorflow model into memory.
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

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

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
			ymin, xmin, ymax, xmax as relative to the image.	Otherwise treat
			coordinates as absolute.
	"""
	draw = ImageDraw.Draw(image)
	im_width, im_height = image.size
	if use_normalized_coordinates:
		(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
																	ymin * im_height, ymax * im_height)
	else:
		(left, right, top, bottom) = (xmin, xmax, ymin, ymax)
	draw.line([(left, top), (left, bottom), (right, bottom),
						 (right, top), (left, top)], width=thickness, fill=color)
	try:
		font = ImageFont.truetype('arial.ttf', 24)
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
				(left + text_width,text_bottom)],
				fill=color)
		draw.text(
				(left + margin, text_bottom - text_height - margin),
				display_str,
				fill='black',
				font=font)
		text_bottom -= text_height - 2 * margin

		
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
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

		
def get_input_output_tensors(graph):
	# Get handles to input and output tensors
	ops = graph.get_operations()
	all_tensor_names = {output.name for op in ops for output in op.outputs}
	tensor_dict = {}
	
	for key in ['num_detections', 'detection_boxes', 'detection_scores',
				'detection_classes', 'detection_masks' ]:
		tensor_name = key + ':0'
		if tensor_name in all_tensor_names:
			tensor_dict[key] = graph.get_tensor_by_name(tensor_name)
	
	if 'detection_masks' in tensor_dict:
		# The following processing is only for single image
		detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
		detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
		# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
		real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
		detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
		detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
		detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
		detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
		# Follow the convention by adding back the batch dimension
		tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
		
	image_tensor = graph.get_tensor_by_name('image_tensor:0')
	
	return image_tensor, tensor_dict


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
		printProgressBar(0, nimages, prefix = 'Progress:', suffix = 'Complete', length = 50)
	write_header = True
	
	# We launch a Session
	with tf.Session(graph=graph) as sess:	
		for progress_idx,image_path in enumerate(image_paths):			
					
			# Cargamos la imagen		
			image = Image.open(image_path)
			# the array based representation of the image will be used later in order to prepare the
			# result image with boxes and labels on it.
			image_np = load_image_into_numpy_array(image)
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)	
			# Actual detection.		
			output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
			# all outputs are float32 numpy arrays, so convert types as appropriate
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
			if 'detection_masks' in output_dict:
				output_dict['detection_masks'] = output_dict['detection_masks'][0]		
			#Obtenemos las detecciones relevantes
			detecciones_idx = [ i for i,s in enumerate(output_dict['detection_scores']) if s >= 0.5]
			image_name = os.path.basename(image_path)
			image_width, image_height = image.size
			row = [image_path,image_name,image_width,image_height,len(detecciones_idx)]
			for idx in detecciones_idx:
				row.append(output_dict['detection_scores'][idx])                          #score
				row.append(category_index[output_dict['detection_classes'][idx]]['name']) #class
				row.append(output_dict['detection_classes'][idx])                         #class id
				row.extend(output_dict['detection_boxes'][idx])		
			
			#Escribimos en el archivo de salida
			if write_header:
				header = ['image_path','image_name','image_width','image_height','n_detections','obj_score_0','obj_class_0','obj_class_id_0','obj_ymin_0','obj_xmin_0','obj_ymax_0','obj_xmax_0']
				fout.write('\t'.join(header) + "\n")
				write_header = False
			fout.write('\t'.join([str(e) for e in row]) + "\n")	
			
			if args.output_file is not None:
				printProgressBar(progress_idx+1, nimages, prefix = 'Progress:', suffix = 'Complete', length = 50)
			
			## Visualization of the results of a detection.
			#vis_util.visualize_boxes_and_labels_on_image_array(
			#		image_np,
			#		output_dict['detection_boxes'],
			#		output_dict['detection_classes'],
			#		output_dict['detection_scores'],
			#		category_index,
			#		instance_masks=output_dict.get('detection_masks'),
			#		use_normalized_coordinates=True,
			#		line_thickness=8)
			#plt.figure(figsize=IMAGE_SIZE)
			#plt.imshow(image_np)