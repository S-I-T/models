import glob
import os
import base64
import json
from io import BytesIO
from PIL import Image
import argparse
import concurrent.futures
import time

class ImageProcessor:
	def __init__(self, roi, resize, keep_ar, output_dir):
		self.roi = roi
		self.resize = resize
		self.keep_ar = keep_ar
		self.output_dir = output_dir

	def process_image(self, image_file):
		
		#print("Procesando: {}".format(image_file))
		aux = image_file[image_file.rfind('/')+1:]
		aux = aux[aux.rfind('\\')+1:]		
						
		image = Image.open(image_file)
		image_name_out = None
	    
		if self.roi != None:
			image = image.crop((self.roi))
		
		if self.output_dir != None and self.keep_ar:			
			image_name_out = os.path.join(self.output_dir, aux)
			image.save(image_name_out, format='JPEG')
		
		if self.resize != None:
			image = image.resize((self.resize[0], self.resize[1]), Image.BILINEAR)
		
		if self.output_dir != None and not self.keep_ar:
			image_name_out = os.path.join(self.output_dir, aux)
			image.save(image_name_out, format='JPEG')
			
		resized_handle = BytesIO()
		image.save(resized_handle, format='JPEG')
		encoded_contents = base64.b64encode(resized_handle.getvalue()).decode('ascii')
		
		# key can be any UTF-8 string, since it goes in a HTTP request.
		row = json.dumps({'key': aux, 'image_bytes': {'b64': encoded_contents}})
		#
		#self.output_file.write(row)
		#self.output_file.write('\n')	
		return row


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


def preprocess(input_dir, output_dir, output_json, roi, resize, keep_ar):
		
	if output_dir != None and not os.path.exists(output_dir):
		os.makedirs(output_dir)
		
	# Create a pool of processes. By default, one is created for each CPU in your machine.
	workers = 3
	with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
		
		ip = ImageProcessor( roi, resize, keep_ar, output_dir)
		image_files = glob.glob( input_dir + '/*.jpg' )
		total_files = len(image_files)
		counter = 0
		for image_file, image_json_row in zip(image_files, executor.map(ip.process_image, image_files)):
			#print( "Procesada: {}".format(image_file) )
			counter += 1
			if not counter % 100:
				print( "Procesadas: {} de {}".format(counter,total_files) )
			


if __name__ == '__main__':
	args = parse_args()
	print(args)
	start_time = time.time()
	preprocess(args.input_dir, args.output_dir, args.output, args.crop, args.resize, args.keep_ar)
	print("--- %s seconds ---" % (time.time() - start_time))