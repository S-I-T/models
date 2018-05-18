# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the flowers dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils
from datasets.download_and_convert_trucks import _get_filenames_and_classes

slim = tf.contrib.slim


# The database prefix
_DB_PREFIX = 'trucks' 

# The percent of samples to keep for validation
_PERC_VALIDATION = 20

_FILE_PATTERN = _DB_PREFIX + '_%s_*.tfrecord'

def _get_filenames_and_classes_count(dataset_dir):
	"""Returns a count of filenames and inferred class names.
	
	Args:
		dataset_dir: A directory containing a set of subdirectories representing
			class names. Each subdirectory should contain PNG or JPG encoded images.

	Returns:
		A list of image file paths, relative to `dataset_dir` and the list of
		subdirectories, representing class names.
	"""
	dir_root = os.path.join(dataset_dir, 'images')
	directories = []
	class_count = 0
	for filename in os.listdir(dir_root):
		path = os.path.join(dir_root, filename)
		if os.path.isdir(path):
			directories.append(path)
			class_count += 1
	
	images_count = 0
	for directory in directories:
		for filename in os.listdir(directory):
			images_count += 1
	
	return images_count, class_count


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
	"""Gets a dataset tuple with instructions for reading database.
	
	Args:
		split_name: A train/validation split name.
		dataset_dir: The base directory of the dataset sources.
		file_pattern: The file pattern to use when matching the dataset sources.
			It is assumed that the pattern contains a '%s' string so that the split
			name can be inserted.
		reader: The TensorFlow reader type.
	
	Returns:
		A `Dataset` namedtuple.
	
	Raises:
		ValueError: if `split_name` is not a valid train/validation split.
	"""
	
	images_count, class_count = _get_filenames_and_classes_count(dataset_dir)
	num_validation = images_count*_PERC_VALIDATION//100
	
	splits_to_sizes = {'train': (images_count-num_validation), 'validation': num_validation}
	
	_items_to_descriptions = {
		'image': 'A color image of varying size.',
		'label': 'A single integer between 0 and ' + str(class_count),
	}
	
	
	if split_name not in splits_to_sizes:
		raise ValueError('split name %s was not recognized.' % split_name)
	
	if not file_pattern:
		file_pattern = _FILE_PATTERN
	file_pattern = os.path.join(dataset_dir + '/tfrecords', file_pattern % split_name)
	
	# Allowing None in the signature so that dataset_factory can use the default.
	if reader is None:
		reader = tf.TFRecordReader
	
	keys_to_features = {
			'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
			'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
			'image/class/label': tf.FixedLenFeature(
					[], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
	}
	
	items_to_handlers = {
			'image': slim.tfexample_decoder.Image(),
			'label': slim.tfexample_decoder.Tensor('image/class/label'),
	}
	
	decoder = slim.tfexample_decoder.TFExampleDecoder(
			keys_to_features, items_to_handlers)
	
	labels_to_names = None
	if dataset_utils.has_labels(dataset_dir + '/tfrecords'):
		labels_to_names = dataset_utils.read_label_file(dataset_dir + '/tfrecords')
	
	return slim.dataset.Dataset(
			data_sources=file_pattern,
			reader=reader,
			decoder=decoder,
			num_samples=splits_to_sizes[split_name],
			items_to_descriptions=_items_to_descriptions,
			num_classes=class_count,
			labels_to_names=labels_to_names)
