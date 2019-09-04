import argparse 
import tensorflow as tf
import imageio
import numpy as np
import sys
import os
import shutil

## Ejemplos
## Eliminar imagenes erroneas
# find -L $DIRECTORY -type f -name "*.jpg" -size -1k -delete
## Crear archivo images 
# find -L $DIRECTORY -type f -name "*.jpg" | sort -t '\0' -n > images.txt
# python classify_images.py -m models/Trucks_c_4c_inceptionv3_v0.pb -i input_image -o InceptionV3/Predictions/Reshape_1 -f images.txt -r predictions.txt
# python classify_images.py -m models/Trucks_c_5c_inceptionv3_v0.pb -i input_image -o InceptionV3/Predictions/Reshape_1 -f images2.txt -r predictions_test.txt -d eliminar -l models/Trucks_c_5c_inceptionv3_v0.txt
#
#MODEL=/root/models/tf-slim/train/Trucks-concrete/inception_v3/all/frozen_inception_v3_vp2.pb
#python classify_images.py \
#    -m $MODEL \
#    -i input_image \
#    -o InceptionV3/Predictions/Reshape_1 \
#    -f images.txt \
#    -r predictions.txt 



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
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def parse_args():
    """Handle the command line arguments.
    
    Returns:
        Output of argparse.ArgumentParser.parse_args.
    """
    parser = argparse.ArgumentParser(description='Batch image classification using VP2 model freezed graph')
    parser.add_argument('-m', '--frozen_model_filename', help="Frozen model file to import", type=str, required=True)
    parser.add_argument('-i','--input_tensor', type=str, help="Input tensor name", required=True)
    parser.add_argument('-o','--output_tensor', type=str, help="Output tensor name", required=True)
    
    parser.add_argument('-f','--input_file', type=str, help="Input text file, one image per line", required=True)
    parser.add_argument('-r','--output_file', default=None, type=str, help="Output file name for prediction probabilities")
    parser.add_argument('-d','--output_dir', default=None, type=str, help="Output dir to move classified images")
    parser.add_argument('-t','--thresholds', default=None, type=str, help="Thresholds to use to classify images, defaults to max score. One per class")
    parser.add_argument('-l','--labels', default=None, type=str, help="Labels to use when moving images")
    args = parser.parse_args()
    return args

def read_labels(labels_file):
    with open(labels_file) as f:
        content = f.readlines()
        labels = [x.strip().split(':')[1] for x in content] 
        return labels

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

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    args = parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    ## We can verify that we can access the list of operations in the graph
    #for op in graph.get_operations():
    #    print(op.name)
    #    # prefix/input_image
    #    # ...
    #    # prefix/InceptionV3/Predictions/Reshape_1
        
    # We access the input and output nodes 
    x = graph.get_tensor_by_name('prefix/' + args.input_tensor + ':0')
    y = graph.get_tensor_by_name('prefix/' + args.output_tensor + ':0')
    
    # Obtenemos los paths de las imagenes a procesar
    image_paths = [s.strip() for s in open(args.input_file)]
    nimages = len(image_paths)
    
    # Verificamos donde escribir los resultados
    fout = sys.stdout
    if args.output_file is not None:
        fout = open(args.output_file, 'w')
        print('')
        printProgressBar(0, nimages, prefix='Progress:', suffix='Complete', length=50)
    
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)        
    
        if args.labels is not None:            
            labels = read_labels(args.labels)
            for label in labels:
                if not os.path.exists(args.output_dir + '/' + label):
                    os.makedirs(args.output_dir + '/' + label)
            if not os.path.exists(args.output_dir + '/unknown' ):
                os.makedirs(args.output_dir + '/unknown' )
    
    if args.thresholds is not None:
        thresholds = [ float(s) for s in args.thresholds.strip().split(',') ]
    
    num_classes = 0    
    
    # We launch a Session
    with tf.Session(graph=graph) as sess:    
        
        for progress_idx, image_path in enumerate(image_paths):  
            try:        
                
                #x = tf.gfile.FastGFile(fl).read() # You can also use x = open(fl).read()
                #image_name = os.path.basename(fl)
                
                # Cargamos la imagen        
                image = imageio.imread(image_path)
                image_np_expanded = np.expand_dims(image, axis=0)        
                
                # Corremos la session
                # Nota: no se necesita inicializar/restaurar nada, 
                # no hay variables en este grafo, solo constantes harcodeadas
                y_out = sess.run(y, feed_dict={ x: image_np_expanded })
                #print(y_out) 
                
                # Esto es particular para la clasificacion, en donde 
                # el tensor de salida son los scores de las clases
                if num_classes == 0:
                    num_classes = len( y_out[0, 0:] )
                    header = ['image']
                    if args.labels is not None:    
                        header.extend(['%s' % s for s in labels])
                    else:
                        header.extend(['class%s' % i for i in range(num_classes)])
                        labels = [str(i) for i in range(num_classes)]
                    header.append('max_score')
                    header.append('threshold')
                    header.append('predicted_class')
                    
                    fout.write('\t'.join(header) + "\n")
                
                image_name = os.path.basename(image_path)
                probs = y_out[0, 0:]
                max_i = np.argmax(probs)
                row = [image_name]
                row.extend(probs)
                row.append(probs[max_i])
                
                if args.thresholds is not None:
                    row.append(thresholds[max_i])
                    if probs[max_i] >= thresholds[max_i]:
                        row.append(labels[max_i])
                    else:
                        row.append('unknown')
                else:
                    row.append(0)
                    row.append(labels[max_i])
                
                fout.write('\t'.join([str(e) for e in row]) + "\n")
                
                if args.output_dir is not None:
                    shutil.move(image_path, args.output_dir + '/' + row[-1] + '/' + image_name)
                    #shutil.copy2(input_dir + '/' + image_name, output_dir + '/' + label + '/' + image_name.replace('.jpg','--({})-({:.5f}).jpg'.format(max_index,max_score)))
                
                if args.output_file is not None:
                    printProgressBar(progress_idx+1, nimages, 
                                     prefix='Progress:',
                                     suffix='Complete',
                                     length=50)
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                print("error procesando:" + image_path)
            
        