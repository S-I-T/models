# Basado en https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/flowers
# ==============================================================================
"""Saves out a GraphDef containing the architecture of the model.

To use it, run something like this, with a model name defined by slim:

bazel build tensorflow_models/research/slim:export_inference_graph
bazel-bin/tensorflow_models/research/slim/export_inference_graph \
--model_name=inception_v3 --output_file=/tmp/inception_v3_inf_graph.pb

If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:

bazel build tensorflow/python/tools:freeze_graph
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/tmp/inception_v3_inf_graph.pb \
--input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
--input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
--output_node_names=InceptionV3/Predictions/Reshape_1

The output node names will vary depending on the model, but you can inspect and
estimate them using the summarize_graph tool:

bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=/tmp/inception_v3_inf_graph.pb

To run the resulting graph in C++, you can look at the label_image sample code:

bazel build tensorflow/examples/label_image:label_image
bazel-bin/tensorflow/examples/label_image/label_image \
--image=${HOME}/Pictures/flowers.jpg \
--input_layer=input \
--output_layer=InceptionV3/Predictions/Reshape_1 \
--graph=/tmp/frozen_inception_v3.pb \
--labels=/tmp/imagenet_slim_labels.txt \
--input_mean=0 \
--input_std=255

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import gfile
from datasets import dataset_factory
from nets import nets_factory


slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to save.')

tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

tf.app.flags.DEFINE_integer(
    'image_size', None,
    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

tf.app.flags.DEFINE_string('dataset_name', 'imagenet',
                           'The name of the dataset to use with the model.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'output_file', '', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'Directory to save intermediate dataset files to')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.output_file:
    raise ValueError('You must supply the path to save to with --output_file')
  tf.logging.set_verbosity(tf.logging.INFO)
  
  with tf.Graph().as_default() as graph:
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
                                          FLAGS.dataset_dir)
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=FLAGS.is_training)
    image_size = FLAGS.image_size or network_fn.default_image_size
    
    IMAGE_WIDTH = image_size
    IMAGE_HEIGHT = image_size
    IMAGE_CHANNELS = 3
    
    # La imagen ya viene con shape que incluye el batch size
    input_image = tf.placeholder(name='input_image', dtype=tf.uint8, shape=[FLAGS.batch_size,None,None,IMAGE_CHANNELS])
    
    #TODO: Esta solo para inferencia, no para entrenamiento
    
    #Redimensionamos la imagen
    image = tf.image.resize_bilinear(input_image, [IMAGE_HEIGHT, IMAGE_WIDTH], align_corners=False)
    image = tf.cast(image, dtype=tf.uint8)
    
    # TODO: Generalizar
    # dims_expander = tf.expand_dims(float_caster, 0)
    # resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    # normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    
    # TODO: Esta parte es particular para inception_v3, dependiendo del modelo deberia cambiar
    # convert_image_dtype, also scales [0, uint8_max] -> [0 ,1).
    
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Then shift images to [-1, 1) for Inception.
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    
    # Aplicamos la red a la imagen de entrada
    network_fn(image)
    
    # Guardamos la definicion del grafo
    graph_def = graph.as_graph_def()
    with gfile.GFile(FLAGS.output_file, 'wb') as f:
      f.write(graph_def.SerializeToString())


if __name__ == '__main__':
  tf.app.run()

