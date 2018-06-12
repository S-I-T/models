import os
import base64
import json
from io import BytesIO
from PIL import Image
import argparse
import time

def parse_args():
	"""Handle the command line arguments.
	
	Returns:
		Output of argparse.ArgumentParser.parse_args.
	"""
	parser = argparse.ArgumentParser(description='Preprocesamiento de imagenes para inferencia')
	parser.add_argument('-f', '--output', default='request.json', help='Archivo de salida', type=str)
	parser.add_argument('-r', '--resize', help='Redimensionar las imagenes localmente a tamano width,height', type=lambda s: [int(item) for item in s.split(',')])
	parser.add_argument('-c', '--crop', help='ROI para recortar la imagen, en formato top,left,bottom,right', type=lambda s: [int(item) for item in s.split(',')])
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


def preprocess(dir, dir_out, output_json, roi, resize, keep_ar):
	
	do_save = (dir_out != None)
		
	if do_save and not os.path.exists(dir_out):
		os.makedirs(dir_out)
	
	with open(output_json, 'w') as ff:
		for file in os.listdir(dir):
			if file.endswith(".jpg"):
				image_name = os.path.join(dir, file)
			
				#print("Procesando: {}".format(image_name))
				aux = image_name[image_name.rfind('/')+1:]
				aux = aux[aux.rfind('\\')+1:]		
								
				image = Image.open(image_name)

				if roi != None:
					image = image.crop((roi))
				
				if do_save and keep_ar:
					image_name_out = dir_out + '/' + aux
					image.save(image_name_out, format='JPEG')
				
				if resize != None:
					image = image.resize((resize[0], resize[1]), Image.BILINEAR)
				
				if do_save and not keep_ar:
					image_name_out = dir_out + '/' + aux
					image.save(image_name_out, format='JPEG')
					
				resized_handle = BytesIO()
				image.save(resized_handle, format='JPEG')
				encoded_contents = base64.b64encode(resized_handle.getvalue()).decode('ascii')
				
				# key can be any UTF-8 string, since it goes in a HTTP request.
				row = json.dumps({'key': aux, 'image_bytes': {'b64': encoded_contents}})
				
				ff.write(row)
				ff.write('\n')


if __name__ == '__main__':
	args = parse_args()
	print(args)
	start_time = time.time()
	preprocess(args.input_dir, args.output_dir, args.output, args.crop, args.resize, args.keep_ar)
	print("--- %s seconds ---" % (time.time() - start_time))