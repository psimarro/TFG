import subprocess
import threading
import random
import time
import sys

def main():
	#subprocess.run(["g++", "controlado.cpp", "-o", "controlado"]) # compilamos controlado.cpp
	
	# creamos subproceso
	proc = subprocess.Popen(["./controlado"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True, bufsize=1)
	
	# thread que maneja la entrada la entrada del subproceso 
	t_input = threading.Thread(target=input_handler, args=[proc.stdin], daemon=True)
	t_input.start()
	
	# thread que maneja la salida del subproceso
	t_output = threading.Thread(target=output_handler, args=[proc], daemon=True)
	t_output.start()
	
	try: 
		#mantenemos vivo thread principal
		while(True):
			time.sleep(1)
	except KeyboardInterrupt:
		print ("\nKeyboard Interrupt")
		sys.exit(0)
	

# manejador de la entrada del subproceso: genera numeros aleatorios periodicamente 
# y los pasa al subproceso escribiendolos por entrada estandar
def input_handler(stdin):
	random.seed() #generamos la semilla para randoms
	while(True):
		stdin.write(str(random.randint(1,1000000)) + "\n")
		time.sleep(1)

# manejador de la salida: comprueba constantemente la salida del subproceso y la muestra si existe.
def output_handler(proc):
	while True:
		output = proc.stdout.readline()
		if proc.poll() is not None: # no necesario ya que subproceso es un bucle infinito y nunca termina
			break
		if output:
			print (output.strip())
		
		   
if __name__ == "__main__": 
    main()

