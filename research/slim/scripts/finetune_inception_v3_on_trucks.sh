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