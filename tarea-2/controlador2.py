import subprocess
import threading
import multiprocessing
import random
import time
import sys

## TAREA 2:
## Controlador modificado respecto de la tarea 1.
## Esta vez se llama al programa Kvazaar, utilizando los parámetros necesarios para su ejecución.
## Además, el programa controlador permite comunicarse con el subproceso de Kvazaar generado: 
## 		1. Este programa recoge un número entero por entrada estándar, que representa el número de cores que Kvazaar va a utilizar
## 		2. Se comunica este dato a Kvazaar.
# 		3. Kvazaar realiza la operación de procesado de un bloque con la entrada anterior y 
# 		   el controlador recoge la salida mostrándola por pantalla.
## 		4. Se repite el proceso desde el punto 1 hasta que el vídeo se codifica por completo

## Esta implementación es la base del funcionamiento del entorno de GYM.

nCores = multiprocessing.cpu_count()
kvazaar_path = "/home/pedro/malleable_kvazaar/bin/./kvazaar"
vid_path = "/home/pedro/Descargas/E_KristenAndSara_1280x720_60p.yuv"

def main():
	
	comando = [kvazaar_path, "--input", vid_path, "--output", "/dev/null", "--preset=ultrafast", "--qp=22", "--owf=0", "--threads=" + str(nCores)]
	# creamos subproceso
	proc = subprocess.Popen(comando, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, bufsize=1, env={'NUM_FRAMES_PER_BATCH': '24'})
	
	# thread que maneja la entrada del subproceso 
	t_input = threading.Thread(target=input_handler, args=[proc.stdin, proc.stdout], daemon=True)
	t_input.start()
	
	try: 
		#mantenemos vivo thread principal
		proc.wait()
	except KeyboardInterrupt:
		print ("\nKeyboard Interrupt")
		sys.exit(0)
	

# manejador de la entrada del subproceso: genera numeros aleatorios periodicamente 
# y los pasa al subproceso escribiendolos por entrada estandar
def input_handler(stdin, stdout):
	random.seed() #generamos la semilla para randoms
	while(True):
		s = "nThs:" + str(random.randint(1, nCores)) 
		print(s)
		stdin.write(s + "\n")
		output = stdout.readline()
		print(output.strip())
		if(output.strip() == "END") :
			break

		   
if __name__ == "__main__": 
    main()




