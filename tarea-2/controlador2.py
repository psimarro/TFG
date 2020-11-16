import subprocess
import threading
import multiprocessing
import random
import time
import sys

nCores = multiprocessing.cpu_count()
kvazaar_path = "/home/pedro/malleable_kvazaar/bin/./kvazaar"
vid_path = "/home/pedro/Descargas/E_KristenAndSara_1280x720_60p.yuv"

def main():
	#subprocess.run(["g++", "controlado.cpp", "-o", "controlado"]) # compilamos controlado.cpp
	
	comando = [kvazaar_path, "--input", vid_path, "--output", "/dev/null", "--preset=ultrafast", "--qp=22", "--owf=0", "--threads=" + str(nCores)]
	# creamos subproceso
	proc = subprocess.Popen(comando, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, bufsize=1, env={'NUM_FRAMES_PER_BATCH': '24'})
	
	# thread que maneja la entrada del subproceso 
	t_input = threading.Thread(target=input_handler, args=[proc.stdin, proc.stdout], daemon=True)
	t_input.start()
	
	# thread que maneja la salida del subproceso
	#t_output = threading.Thread(target=output_handler, args=[proc.stdout], daemon=True)
	#t_output.start()
	
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
		

# # manejador de la salida: comprueba constantemente la salida del subproceso y la muestra si existe.
# def output_handler(stdout):
	# while True:
		# output = stdout.readline() #bloqueante
		# #if proc.poll() is not None: # no necesario ya que subproceso es un bucle infinito y nunca termina
			# #break
		# if output:
			# print (output.strip())
		# else:
			# break
		
		   
if __name__ == "__main__": 
    main()




