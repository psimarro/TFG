import subprocess
import multiprocessing
# import threading
# import random
# import time
import sys
import gym
import gym_example

nCores = multiprocessing.cpu_count()
kvazaar_path = "/home/pedro/malleable_kvazaar/bin/./kvazaar"
vid_path = "/home/pedro/Descargas/E_KristenAndSara_1280x720_60p.yuv"

def main():
	
	
	# thread que maneja la entrada del subproceso 
	# t_input = threading.Thread(target=input_handler, args=[proc.stdin, proc.stdout], daemon=True)
	# t_input.start()
	
	# thread que maneja la salida del subproceso
	#t_output = threading.Thread(target=output_handler, args=[proc.stdout], daemon=True)
	#t_output.start()
	
	try: 
		#mantenemos vivo thread principal
		# proc.wait()

		env = gym.make("kvazaar-v0", 
						kvazaar_path=kvazaar_path, 
						vid_path=vid_path, 
						nCores=multiprocessing.cpu_count(),
						intervalos=[25, 50, 100, 150])
		run_episode(env, verbose=True)

	except KeyboardInterrupt:
		print ("\nKeyboard Interrupt")
		sys.exit(0)
	
def run_episode(env, verbose):
	env.reset()
	env.render()
	sum_reward = 0

	while env.done == False:
		action = env.action_space.sample()
		
		state, reward, done, info = env.step(action)
		sum_reward += reward

		if verbose and info["estado"] != 'END':
			print("action:", action+1, "core(s)")

		if verbose:
			env.render()
		
		if done and verbose:
			print("done @ step {}".format(env.count))
			break

	if verbose:
		print("cumulative reward", sum_reward)
	
	return sum_reward


# manejador de la entrada del subproceso: genera numeros aleatorios periodicamente 
# y los pasa al subproceso escribiendolos por entrada estandar
def input_handler(stdin, stdout):
	random.seed() #generamos la semilla para randoms
	while(True):
		s = "nThs:" + str(random.randint(1,6)) 
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




