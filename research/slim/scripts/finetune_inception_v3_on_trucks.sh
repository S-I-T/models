#Limpiar modelos previos
#rm -rf /root/databases/Trucks/tfrecords
#rm -rf /root/models/tf-slim/train/Trucks/inception_v3 


# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/root/models/tf-slim/public

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/root/models/tf-slim/train/Trucks/inception_v3

# Where the dataset is saved to.
DATASET_DIR=/root/databases/Trucks

# Otras definiciones
TENSORFLOWDIR=/tensorflow
BAZEL=bazel
SLIM_DIR=/root/src/models/research/slim


cd ${SLIM_DIR}

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=trucks \
  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 1000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=trucks \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=2000 \
  --batch_size=64 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
  
  
# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=trucks \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3


# Fine-tune all the new layers for 500 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=trucks \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=500 \
  --batch_size=64 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
 
# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=trucks \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3

#Para exportar el grafo 
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --dataset_name=trucks \
  --dataset_dir=${DATASET_DIR} \
  --output_file=${TRAIN_DIR}/inception_v3_inf_graph.pb
  

#Para revisar el grafo creado, y ver los nombres de los nodos de salida
cd $TENSORFLOWDIR
#$BAZEL build tensorflow/tools/graph_transforms:summarize_graph

bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
  --in_graph=${TRAIN_DIR}/inception_v3_inf_graph.pb
  
  
#Para congelar un grafo con los parametros entrenados
cd $TENSORFLOWDIR
#$BAZEL build tensorflow/python/tools:freeze_graph

bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=${TRAIN_DIR}/inception_v3_inf_graph.pb \
  --input_checkpoint=${TRAIN_DIR}/all/model.ckpt-500 \
  --input_binary=true --output_graph=${TRAIN_DIR}/all/frozen_inception_v3.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1


#Para evaluar un grafo con una imagen
cd $TENSORFLOWDIR
#$BAZEL build tensorflow/examples/label_image:label_image

bazel-bin/tensorflow/examples/label_image/label_image \
  --input_layer=input \
  --output_layer=InceptionV3/Predictions/Reshape_1 \
  --graph=${TRAIN_DIR}/all/frozen_inception_v3.pb \
  --labels=${DATASET_DIR}/tfrecords/labels.txt \
  --input_mean=0 \
  --input_std=255 \
  --image=${DATASET_DIR}/images/concrete-mixer/00000000f5967924_150505_093504_-1.jpg_crop0.jpg
  
  
#Para subir y servir el grafo en Google ML Engine__________________________________
#Ejemplos sacados de https://github.com/GoogleCloudPlatform/cloudml-samples/flowers

cd ${SLIM_DIR}

#Para exportar el grafo 
python export_inference_graph_serve.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --dataset_name=trucks \
  --dataset_dir=${DATASET_DIR} \
  --output_file=${TRAIN_DIR}/inception_v3_inf_graph_serve.pb

  
#Para revisar el grafo creado, y ver los nombres de los nodos de salida
cd $TENSORFLOWDIR
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
  --in_graph=${TRAIN_DIR}/inception_v3_inf_graph_serve.pb

  
#Para congelar un grafo con los parametros entrenados
cd $TENSORFLOWDIR
bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=${TRAIN_DIR}/inception_v3_inf_graph_serve.pb \
  --input_checkpoint=${TRAIN_DIR}/all/model.ckpt-1000 \
  --input_binary=true --output_graph=${TRAIN_DIR}/all/frozen_inception_v3_serve.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1

  
#Para transformar ese grafo a formato saved_model
cd ${SLIM_DIR}
python convert_frozen_graph_to_saved_model.py \
  --input_graph=${TRAIN_DIR}/all/frozen_inception_v3_serve.pb \
  --input_node_name=input_jpeg \
  --output_node_name=InceptionV3/Predictions/Reshape_1
  --output_dir=${TRAIN_DIR}/all/saved_model


#Para subir el modelo a Google ML Engine, se debe subir la carpeta saved_model 
#a un bucket, el cual se define aca como GCS_BUCKET
#sudo ./sit-machine-learning_upload.sh
MODEL_NAME=Trucks_classificator
REGION=us-central1
VERSION_NAME=inceptionv3_v0
VERSION_DESCRIPTION="Camiones descargados de internet. Tipos de camiones: 0:concrete-mixer, 1:crane-truck, 2:dump-truck, 3:other-truck, 4:water-tank-truck"
GCS_BUCKET=gs://sit-machine-learning/data/models/detector_classificator/tf-slim/train/Trucks/inception_v3/all/saved_model

gcloud ml-engine models create "$MODEL_NAME" --regions "$REGION"
# --enable-logging --> log a stackdrive
gcloud ml-engine versions create "$VERSION_NAME" \
  --model "$MODEL_NAME" \
  --origin "${GCS_BUCKET}" \
  --runtime-version=1.4

#Probamos el modelo subido con una imagen
TEST_IMAGE=https://i.pinimg.com/736x/f4/2d/7e/f42d7eb4f4aa9318bfbeab73285ed2fc--used-trucks-box-sets.jpg
wget ${TEST_IMAGE} -O image.jpg
python -c 'import base64, sys, json; img = base64.b64encode(open(sys.argv[1], "rb").read()); print json.dumps({"key":"0", "image_bytes": {"b64": img}})' image.jpg &> request.json
gcloud ml-engine predict --model ${MODEL_NAME} --json-instances request.json
rm -f image.jpg request.json


#POST https://ml.googleapis.com/v1/projects/my-project/models/my-model/versions/my-version:predict
#POST https://ml.googleapis.com/v1/projects/visualprogress2/models/Truck_clasificator_inet/versions/inceptionv3_v0:predict
