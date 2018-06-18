import glob
import os
from PIL import Image
import argparse
import concurrent.futures
import time
# import json
# import base64
# from io import BytesIO


class ImageProcessor:
    def __init__(self, roi, resize, keep_ar, output_dir):
        self.roi = roi
        self.resize = resize
        self.keep_ar = keep_ar
        self.output_dir = output_dir

    def process_image(self, image_path):

        # print("Procesando: {}".format(image_file))
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)
        image_name_out = None

        if self.roi is not None:
            image = image.crop((self.roi))

        if self.resize is not None:
            imageW, imageH = image.size
            resizeW, resizeH = self.resize
            if not((imageW <= resizeW and imageH <= resizeH) or
                   (imageW <= resizeH and imageH <= resizeW)):
                if self.keep_ar:
                    imageRatio = imageW / imageH
                    resizeRatio = resizeW / resizeH
                    if imageRatio < resizeRatio:
                        scale = imageW / resizeW
                    else:
                        scale = imageH / resizeH

                    resizeW = (imageW / scale)
                    resizeH = (imageH / scale)
                image = image.resize((int(resizeW), int(resizeH)), Image.BILINEAR)

        if self.output_dir is not None:
            image_name_out = os.path.join(self.output_dir, image_name)
            image.save(image_name_out, format='JPEG')

        row = None
        '''if None is not None:
            resized_handle = BytesIO()
            image.save(resized_handle, format='JPEG')
            encoded_contents = base64.b64encode(
                resized_handle.getvalue()).decode('ascii')
            # key can be any UTF-8 string, since it goes in a HTTP request.
            row = json.dumps({'key': image_name, 'image_bytes': {'b64': encoded_contents}})
            # self.output_file.write(row)
            # self.output_file.write('\n')'''
        return row


def parse_args():
    """Handle the command line arguments.
    Returns:
        Output of argparse.ArgumentParser.parse_args.
    """
    parser = argparse.ArgumentParser(description='Preprocesamiento de imagenes para inferencia')
    # parser.add_argument('-f', '--output', default=None, help='Archivo de salida', type=str)
    parser.add_argument('-t', '--threads', default=1, help='Nro de hilos a ejecutar (1 por defecto)', type=int)
    parser.add_argument('-r', '--resize', help='Redimensionar las imagenes localmente a tamano width,height', type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-c', '--crop', help='ROI para recortar la imagen, en formato left,top,right,bottom', type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('-o', '--output_dir', help='Directorio donde guardar las imagenes de salida, si se especifica se guardaran las imagenes', type=str)
    parser.add_argument('-a', '--keep_ar', nargs='?', default=False, help='Flag para guardar las imagenes de salida con su aspect ratio original. Debe ponerse -o o --output_dir para que funcione', type=str2bool, const=True)
    requiredNamed = parser.add_argument_group('Parametros obligatorios')
    requiredNamed.add_argument('-i', '--input_dir', help='Directorio con imagenes de entrada', type=str, required=True)
    args = parser.parse_args()
    return args


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Se espera un valor booleano')


def preprocess(input_dir, output_dir, roi, resize, keep_ar, workers):

    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a pool of processes. By default, one is created
    # for each CPU in your machine.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:

        ip = ImageProcessor(roi, resize, keep_ar, output_dir)
        image_paths = glob.glob(input_dir + '/*.jpg')
        total_files = len(image_paths)
        counter = 0
        for image_path, image_json_row in zip(image_paths, executor.map(ip.process_image, image_paths)):
            # print( "Procesada: {}".format(image_file) )
            counter += 1
            if not counter % 100:
                print("Procesadas: {} de {}".format(counter, total_files))    


if __name__ == '__main__':
    args = parse_args()
    print(args)
    start_time = time.time()
    preprocess(args.input_dir, args.output_dir, args.crop, args.resize, args.keep_ar, args.threads)
    print("--- %s seconds ---" % (time.time() - start_time))
