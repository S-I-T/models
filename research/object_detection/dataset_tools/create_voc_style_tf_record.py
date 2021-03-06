r"""Convert raw PASCAL VOC style dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit/VOC2012 \
		--set=trainval
        --output_path=/home/user/voc_trainval.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import numpy as np

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.compat.v1.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC style dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set (train, val, trainval, test).')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
flags.DEFINE_integer('n_shards', 5, 'Number of output file divisions (shards)')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test', 'all']


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    print(full_path)
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].replace(' ', '_').encode('utf8'))
      classes.append(label_map_dict[obj['name'].replace(' ', '_')])
      truncated.append(int(obj['truncated']) if 'truncated' in obj else 0)
      poses.append(obj['pose'].encode('utf8') if 'pose' in obj else 'Unspecified'.encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example

def writing_loop(FLAGS, writer, data_batch, data_dir):
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  for idx, example in enumerate(data_batch):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(data_batch))
    annotations_dir = os.path.join(data_dir[idx], FLAGS.annotations_dir)
    path = os.path.join(annotations_dir, example + '.xml')
    # print("Procesando:"+path)
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(bytes(xml_str, encoding='utf-8'))
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, data_dir[idx], label_map_dict,
                                  FLAGS.ignore_difficult_instances)
    writer.write(tf_example.SerializeToString())

def batch(iterable, n=2):
  l = len(iterable)
  for ndx in range(0, l, n):
      yield iterable[ndx:min(ndx + n, l)]

def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {} '.format(SETS))
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  data_dirs = FLAGS.data_dir.split(';')
  output_path = FLAGS.output_path
  n_shards = FLAGS.n_shards
  total_examples = []
  dirs = []
  # Loop for merging the paths
  for data_dir in data_dirs:
    #logging.info('Reading from dataset in: ', data_dir)
    examples_path = os.path.join(data_dir, 'ImageSets', 'Main', FLAGS.set + '.txt')
    examples_list = dataset_util.read_examples_list(examples_path)
    for example in examples_list:
      total_examples.append(example)
      dirs.append(data_dir)
  #Determining the batch size for the tfrecords
  batch_size = len(total_examples)//n_shards
  rest = len(total_examples)%n_shards

  #Writing the tfrecords
  part_num=0
  dirs_iter = batch(dirs, batch_size)
  for data_batch in batch(total_examples, batch_size):
    dirs_batch = next(dirs_iter)
    if rest==0:  
      writer = tf.python_io.TFRecordWriter(output_path + '-{:05d}-of-{:05d}'.format(part_num, n_shards))
    else:
      writer = tf.python_io.TFRecordWriter(output_path + '-{:05d}-of-{:05d}'.format(part_num, n_shards+1))
    writing_loop(FLAGS, writer, data_batch, dirs_batch)
    writer.close()
    part_num=part_num+1
    

if __name__ == '__main__':
  tf.compat.v1.app.run()
