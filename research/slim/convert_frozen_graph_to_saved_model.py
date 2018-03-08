#cd /root/models/tf-slim/train/Trucks/inception_v3/all
#--output_node_names=InceptionV3/Predictions/Reshape_1

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


tf.app.flags.DEFINE_string(
    'input_graph', '', 'Path al modelo congelado de entrada.')

tf.app.flags.DEFINE_string('input_node_name', 'input_jpeg',
                           'Nombre del nodo de entrada de la red.')

tf.app.flags.DEFINE_string('output_node_name', 'InceptionV3/Predictions/Reshape_1',
                           'Nombre del nodo de salida de la red.')			   

tf.app.flags.DEFINE_string(
    'output_dir', './saved_model', 'Directorio en donde guardar el modelo')

FLAGS = tf.app.flags.FLAGS

def main(_):
	
	input_graph = FLAGS.input_graph
	input_node_name = FLAGS.input_node_name + ":0"
	output_node_name = FLAGS.output_node_name + ":0"
	output_dir = FLAGS.output_dir
	
	builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
	
	with tf.gfile.GFile(input_graph, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	
	sigs = {}
	
	with tf.Session(graph=tf.Graph()) as sess:
		# name="" is important to ensure we don't get spurious prefixing
		tf.import_graph_def(graph_def, name="")
		g = tf.get_default_graph()
		input_jpeg = g.get_tensor_by_name(input_node_name)
		scores = g.get_tensor_by_name(output_node_name)
		
		keys_placeholder = tf.placeholder(tf.string, shape=[None])
		inputs = {
			'key': keys_placeholder,
			'image_bytes': input_jpeg
		}
		
		# To extract the id, we need to add the identity function.
		keys = tf.identity(keys_placeholder)
		outputs = {
			'key': keys,
			#'prediction': tensors.predictions[0],
			'scores': scores
		}
		
		sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = tf.saved_model.signature_def_utils.predict_signature_def(inputs,outputs)
		
		builder.add_meta_graph_and_variables(sess,
											[tag_constants.SERVING],
											signature_def_map=sigs)
	
	builder.save()

if __name__ == '__main__':
	tf.app.run()

'''
YOUR_LOCAL_EXPORT_DIR=/root/models/tf-slim/train/Trucks/inception_v3/all
saved_model_cli show --dir ${YOUR_LOCAL_EXPORT_DIR}/saved_model --all
saved_model_cli show --dir ${YOUR_LOCAL_EXPORT_DIR}/saved_model --tag_set serve --signature_def serving_default
saved_model_cli run  --dir ${YOUR_LOCAL_EXPORT_DIR}/saved_model --tag_set serve --signature_def serving_default --inputs image=image.npy --outdir /tmp/out --overwrite



# Since the image is passed via JSON, we have to encode the JPEG string first.
python -c 'import base64, sys, json; img = base64.b64encode(open(sys.argv[1], "rb").read()); print json.dumps({"key":"0", "image_bytes": {"b64": img}})' image.jpg &> request.json
saved_model_cli run  --dir ${YOUR_LOCAL_EXPORT_DIR}/saved_model --tag_set serve --signature_def serving_default --json-instances=request.json --outdir /tmp/out --overwrite


'''
