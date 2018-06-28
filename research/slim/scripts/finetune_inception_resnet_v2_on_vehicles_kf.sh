#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an Inception Resnet V2 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_resnet_v2_on_flowers.sh
set -e

# Model to train
MODEL_NAME=inception_resnet_v2

# Where the pre-trained InceptionResnetV2 checkpoint is saved to.
# O bien el directorio donde estan los archivos de checkpoint pre-entrenados
PRETRAINED_CHECKPOINT_DIR=/root/models/tf-slim/public/${MODEL_NAME}.ckpt

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/root/models/tf-slim/train/Vehicles-KF-CLA/${MODEL_NAME}

# Where the dataset is saved to.
DATASET_DIR=/root/databases/Vehicles-KF-CLA

# Cuantos pasos entrenar cada fase del modelo
FINETUNE_STEPS=1000
FINETUNE_ALL_LAYERS_STEPS=1000

# Otras definiciones
TENSORFLOWDIR=/tensorflow
BAZEL=bazel
SLIM_DIR=/root/src/models/research/slim


cd ${SLIM_DIR}

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/${MODEL_NAME}.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
  tar -xvf inception_resnet_v2_2016_08_30.tar.gz
  mv inception_resnet_v2_2016_08_30.ckpt ${PRETRAINED_CHECKPOINT_DIR}
  rm inception_resnet_v2_2016_08_30.tar.gz
fi

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=vehicles_kf \
  --dataset_dir=${DATASET_DIR}


# Fine-tune only the new layers for N steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=vehicles_kf \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
  --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
  --max_number_of_steps=${FINETUNE_STEPS} \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
  
# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=vehicles_kf \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --eval_image_size=299


# Fine-tune all the new layers for N steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=vehicles_kf \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=${FINETUNE_ALL_LAYERS_STEPS} \
  --batch_size=32 \
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
  --dataset_name=vehicles_kf \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --eval_image_size=299


# Matriz de confusion
NUM_CLASSES=$(wc -l < ${DATASET_DIR}/tfrecords/labels.txt)
find -L ${DATASET_DIR}/images/ -maxdepth 2 -type f > images.txt  
python classify_image.py \
  --num_classes ${NUM_CLASSES} \
  --infile images.txt \
  --model_name ${MODEL_NAME} \
  --checkpoint_path ${TRAIN_DIR}/all \
  --outfile predictions.txt
gsutil mv images.txt gs://sit-temp && gsutil mv predictions.txt gs://sit-temp

python eval_image_classifier_cm.py  \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=vehicles_kf \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --eval_image_size=299





#Para exportar el grafo 
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=${MODEL_NAME} \
  --dataset_name=vehicles_kf \
  --dataset_dir=${DATASET_DIR} \
  --output_file=${TRAIN_DIR}/${MODEL_NAME}_inf_graph.pb \
  --image_size=299
  

#Para revisar el grafo creado, y ver los nombres de los nodos de salida
cd $TENSORFLOWDIR
#$BAZEL build tensorflow/tools/graph_transforms:summarize_graph

bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
  --in_graph=${TRAIN_DIR}/${MODEL_NAME}_inf_graph.pb
  
  
#Para congelar un grafo con los parametros entrenados
cd $TENSORFLOWDIR
#$BAZEL build tensorflow/python/tools:freeze_graph

bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=${TRAIN_DIR}/${MODEL_NAME}_inf_graph.pb \
  --input_checkpoint=${TRAIN_DIR}/all/model.ckpt-${FINETUNE_ALL_LAYERS_STEPS} \
  --input_binary=true --output_graph=${TRAIN_DIR}/all/frozen_${MODEL_NAME}.pb \
  --output_node_names=InceptionResnetV2/Logits/Predictions


#Para evaluar un grafo con una imagen
cd $TENSORFLOWDIR
#$BAZEL build tensorflow/examples/label_image:label_image

bazel-bin/tensorflow/examples/label_image/label_image \
  --input_layer=input \
  --output_layer=InceptionResnetV2/Logits/Predictions \
  --graph=${TRAIN_DIR}/all/frozen_${MODEL_NAME}.pb \
  --labels=${DATASET_DIR}/tfrecords/labels.txt \
  --input_width=299 \
  --input_height=299 \
  --input_mean=0 \
  --input_std=255 \
  --image=${DATASET_DIR}/images/concrete-mixer/111_0_camion-hormigonerorevolvedor-de-concreto-2009-8600-1658_n.jpg

 
#Para servir el grafo en plataforma VP2____________________________________________ 
cd ${SLIM_DIR}
python export_inference_graph_vp2.py \
  --alsologtostderr \
  --model_name=${MODEL_NAME} \
  --dataset_name=vehicles_kf \
  --dataset_dir=${DATASET_DIR} \
  --output_file=${TRAIN_DIR}/${MODEL_NAME}_inf_graph_vp2.pb \
  --image_size=299


#Para congelar un grafo con los parametros entrenados
cd $TENSORFLOWDIR
bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=${TRAIN_DIR}/${MODEL_NAME}_inf_graph_vp2.pb \
  --input_checkpoint=${TRAIN_DIR}/all/model.ckpt-${FINETUNE_ALL_LAYERS_STEPS} \
  --input_binary=true \
  --output_graph=${TRAIN_DIR}/all/frozen_${MODEL_NAME}_vp2.pb \
  --output_node_names=InceptionResnetV2/Logits/Predictions
  
  
#Para subir y servir el grafo en Google ML Engine__________________________________
#Ejemplos sacados de https://github.com/GoogleCloudPlatform/cloudml-samples/flowers

cd ${SLIM_DIR}

#Para exportar el grafo 
python export_inference_graph_serve.py \
  --alsologtostderr \
  --model_name=${MODEL_NAME} \
  --dataset_name=vehicles_kf \
  --dataset_dir=${DATASET_DIR} \
  --output_file=${TRAIN_DIR}/${MODEL_NAME}_inf_graph_serve.pb

  
#Para revisar el grafo creado, y ver los nombres de los nodos de salida
cd $TENSORFLOWDIR
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
  --in_graph=${TRAIN_DIR}/${MODEL_NAME}_inf_graph_serve.pb

  
#Para congelar un grafo con los parametros entrenados
cd $TENSORFLOWDIR
bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=${TRAIN_DIR}/${MODEL_NAME}_inf_graph_serve.pb \
  --input_checkpoint=${TRAIN_DIR}/all/model.ckpt-${FINETUNE_ALL_LAYERS_STEPS} \
  --input_binary=true --output_graph=${TRAIN_DIR}/all/frozen_${MODEL_NAME}_serve.pb \
  --output_node_names=InceptionResnetV2/Predictions/Reshape_1

  
#Para transformar ese grafo a formato saved_model
cd ${SLIM_DIR}
python convert_frozen_graph_to_saved_model.py \
  --input_graph=${TRAIN_DIR}/all/frozen_${MODEL_NAME}_serve.pb \
  --input_node_name=input_jpeg \
  --output_node_name=InceptionResnetV2/Predictions/Reshape_1 \
  --output_dir=${TRAIN_DIR}/all/saved_model


#Para subir el modelo a Google ML Engine, se debe subir la carpeta saved_model 
#a un bucket, el cual se define aca como GCS_BUCKET
#sudo ./sit-machine-learning_upload.sh
MODEL_NAME=Vehicles-KF-CLA_classificator
REGION=us-central1
VERSION_NAME=inception_resnetv2_v0
VERSION_DESCRIPTION="Camiones descargados de internet. Tipos de camiones: 0:concrete-mixer, 1:crane-truck, 2:dump-truck, 3:other-truck, 4:water-tank-truck"
GCS_BUCKET=gs://sit-machine-learning/data/models/detector_classificator/tf-slim/train/Vehicles-KF-CLA/${MODEL_NAME}/all/saved_model

gcloud ml-engine models create "$MODEL_NAME" --regions "$REGION"
# --enable-logging --> log a stackdrive
gcloud ml-engine versions create "$VERSION_NAME" \
  --model "$MODEL_NAME" \
  --origin "${GCS_BUCKET}" \
  --runtime-version=1.4

#Probamos el modelo subido con una imagen
TEST_IMAGE=http://www.lambocars.com/images/lamboevents/spaitalia0920.jpg
wget ${TEST_IMAGE} -O image.jpg
python -c 'import base64, sys, json; img = base64.b64encode(open(sys.argv[1], "rb").read()); print json.dumps({"key":"0", "image_bytes": {"b64": img}})' image.jpg &> request.json
gcloud ml-engine predict --model ${MODEL_NAME} --json-instances request.json
#gcloud ml-engine local predict --model-dir ${TRAIN_DIR}/all/saved_model --json-instances=request.json
rm -f image.jpg request.json


#POST https://ml.googleapis.com/v1/projects/my-project/models/my-model/versions/my-version:predict
#POST https://ml.googleapis.com/v1/projects/visualprogress2/models/Truck_clasificator_inet/versions/inceptionv3_v0:predict


#_slim
#python export_inference_graph_vp2.py \
#  --alsologtostderr \
#  --model_name=inception_resnet_v2 \
#  --dataset_name=imagenet \
#  --dataset_dir=/root/databases \
#  --output_file=inception_resnet_v2_inf_graph_vp2.pb \
#  --image_size=299
#
#bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=/root/src/models/research/slim/inception_resnet_v2_inf_graph_vp2.pb > eliminar.txt
#rm eliminar.txt 
#
#_tf
#bazel-bin/tensorflow/python/tools/freeze_graph \
#  --input_graph=/root/src/models/research/slim/inception_resnet_v2_inf_graph_vp2.pb \
#  --input_checkpoint=/root/models/tf-slim/public/inception_resnet_v2.ckpt \
#  --input_binary=true --output_graph=/root/src/models/research/slim/frozen_inception_resnet_v2.pb \
#  --output_node_names=InceptionResnetV2/Logits/Predictions
#  
#gsutil cp /root/src/models/research/slim/frozen_inception_resnet_v2.pb gs://sit-temp/
#mv /root/src/models/research/slim/frozen_inception_resnet_v2.pb /root/models/tf-slim/public/
