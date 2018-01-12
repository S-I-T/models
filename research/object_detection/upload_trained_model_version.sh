#Para exportar un modelo para servir de los ya preentrenados de object detection api
#https://cloud.google.com/blog/big-data/2017/09/performing-prediction-with-tensorflow-object-detection-models-on-google-cloud-machine-learning-engine

##Si no se ha echo: instalar object_detection https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
#apt-get install protobuf-compiler python-pil python-lxml python-tk -y
## From tensorflow/models/research/
#protoc object_detection/protos/*.proto --python_out=.
## Add Libraries to PYTHONPATH
## When running locally, the tensorflow/models/research/ and slim directories should be appended to PYTHONPATH. 
## This can be done by running the following from tensorflow/models/research/:
#export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim


BASE_DIR=/root/src/models/research
#VERSION_NAME=faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08
VERSION_NAME=faster_rcnn_resnet101_coco_2017_11_08
MODEL_DIR=/root/models/tf-object-detection-api/public/${VERSION_NAME}
OBJECT_DETECTION_CONFIG=/root/src/models/research/object_detection/samples/configs/faster_rcnn_resnet101_coco.config

#Exportar el modelo que reciba una imagen serializada en un string. Por defecto recibe imagenes
cd ${BASE_DIR}
python object_detection/export_inference_graph.py \
	--input_type encoded_image_string_tensor \
	--pipeline_config_path ${OBJECT_DETECTION_CONFIG} \
	--trained_checkpoint_prefix ${MODEL_DIR}/model.ckpt \
	--output_directory ${MODEL_DIR}/saved_model_tmp

#	--trained_checkpoint_prefix ${YOUR_LOCAL_CHK_DIR}/model.ckpt-${CHECKPOINT_NUMBER} \

#Eliminamos los archivos innecesarios
cd ${MODEL_DIR}/saved_model_tmp
rm -rf checkpoint frozen_inference_graph.pb model.ckpt.*
cd ..
mv saved_model_tmp/saved_model saved_model_serve
rm saved_model_tmp -rf


#Revisar el modelo exportado
ls -alh ${MODEL_DIR}/saved_model_serve
saved_model_cli show --dir ${MODEL_DIR}/saved_model_serve --all

#MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
#
#signature_def['serving_default']:
#The given SavedModel SignatureDef contains the following input(s):
#inputs['inputs'] tensor_info:
#    dtype: DT_STRING         <----------------------------------------ESTO ES LO QUE CAMBIA!
#    shape: (-1)
#    name: encoded_image_string_tensor:0
#The given SavedModel SignatureDef contains the following output(s):
#outputs['detection_boxes'] tensor_info:
#    dtype: DT_FLOAT
#    shape: (-1, 100, 4)
#    name: detection_boxes:0
#outputs['detection_classes'] tensor_info:
#    dtype: DT_FLOAT
#    shape: (-1, 100)
#    name: detection_classes:0
#outputs['detection_scores'] tensor_info:
#    dtype: DT_FLOAT
#    shape: (-1, 100)
#    name: detection_scores:0
#outputs['num_detections'] tensor_info:
#    dtype: DT_FLOAT
#    shape: (-1)
#    name: num_detections:0
#Method name is: tensorflow/serving/predict

#Servir el modelo localmente
#Si la imagen debe tener un tamaño específico o se quiere reducir el consumo de red, 
#se puede redimensiona la imagen de entrada antes de pasarla a json
TEST_IMAGE=/root/src/models/research/object_detection/test_images/image1.jpg

python -c 'import base64, sys, json; img = base64.b64encode(open(sys.argv[1], "rb").read()); print json.dumps({"inputs": {"b64": img}})' ${TEST_IMAGE} &> inputs.json
time gcloud ml-engine local predict --model-dir ${MODEL_DIR}/saved_model_serve --json-instances=inputs.json
rm inputs.json

#Servir el modelo en la nube
#Para subir el modelo a Google ML Engine, se debe subir la carpeta saved_model 
#a un bucket, el cual se define aca como GCS_BUCKET. Se puede subir usando --staging-bucket
#pero mejor dejarlos ordenados
#sudo ./sit-machine-learning_upload.sh
MODEL_NAME=COCO
REGION=us-central1
VERSION_NAME=${VERSION_NAME}
PROJECT_NAME=visualprogress2
VERSION_DESCRIPTION="Clases de Microsoft COCO database"
GCS_BUCKET=gs://sit-machine-learning/data/models/detector_classificator/tf-object-detection-api/public/${VERSION_NAME}/saved_model_serve

gcloud ml-engine models create "$MODEL_NAME" --regions "$REGION"
gcloud ml-engine versions create "$VERSION_NAME" \
  --model "$MODEL_NAME" \
  --origin "${GCS_BUCKET}" \
  --runtime-version=1.4

#Probamos el modelo subido con una imagen
python -c 'import base64, sys, json; img = base64.b64encode(open(sys.argv[1], "rb").read()); print json.dumps({"inputs": {"b64": img}})' ${TEST_IMAGE} &> inputs.json
time gcloud ml-engine predict --model ${MODEL_NAME} --json-instances inputs.json
rm inputs.json

#Probamos con curl
python -c 'import base64, sys, json; img = base64.b64encode(open(sys.argv[1], "rb").read()); print json.dumps({"instances": [{"inputs": {"b64": img}}]})' ${TEST_IMAGE} &> payload.json
time curl -m 180 -X POST -v -k -H "Content-Type: application/json" \
    -d @payload.json  \
    -H "Authorization: Bearer `gcloud auth print-access-token`" \
https://ml.googleapis.com/v1/projects/${PROJECT_NAME}/models/${MODEL_NAME}/versions/${VERSION_NAME}:predict
rm payload.json
