# TRABAJO DE FIN DE GRADO - APRENDIZAJE POR REFUERZO PARA LA MEJORA DE CALIDAD DE SERVICIO EN PROCESOS DE CODIFICACIÓN DE VÍDEO
<p align="center">
   <img src="3-2016-07-21-Marca UCM logo negro.png" width=175>
   <img src="escudofdigrande.png" width=175>
</p>

## Descripción
Este repositorio engloba la implementación de un sistema de aprendizaje por refuerzo usando las librerías RLLIB y GYM para una tarea de codificación de vídeos usando Kvazaar, un codificador que implementa el estándar HVEC. Se ha diseñado un entorno para [una versión modificada de este programa](https://github.com/luismacostero/malleable_kvazaar) que permite seleccionar de manera dinánimca el numero de hilos que se ejecutan. El objetivo de este proyecto es generar un agente de aprendizaje por refuerzo capaz de aprender a utilizar este codificador de manera que el resultado de las codificaciones de vídeo se mantenga entre 20 y 30 FPS. 

## Instalación
Es necesario tener instalado Python en su versión 3.
Este proyecto es instalable en Linux de la siguiente manera:
   

### Instalación de Kvazaar
Es necesario tener instalados los siguientes paquetes: `git, automake, autoconf, libtool, m4, build-essential`
   
En una terminal hacemos
```
git clone https://github.com/luismacostero/malleable_kvazaar
cd malleable_kvazaar
autoreconf -fiv
./configure --prefix=$(pwd)
make
make install
```

Esto instala Kvazaart en la ruta actual, se puede modificar la ruta de instalación cambiando la opción `--prefix`

### Instalación del entorno GYM
Desde una terminal, clonamos este repositorio
```
git clone https://github.com/psimarro/TFG.git
```
Creamos un entorno virtual e instalamos dependencias
```
cd TFG
python3 -m venv --system-site-packages ./venv
source venv/bin/activate/
pip install --upgrade pip
pip install -r requirements.txt
```
Instalamos el entorno
```
pip install -e kvazaar_gym/
```

## Uso

### Archivo de configuración `src/config.ini`

Este archivo necesita ser modificado antes de utilizar los scripts del proyecto. Presenta dos secciones:
1. `[common]` : configura opciones para el entrenamiento, la ejecución de la política entrenada y los baselines
2. `[train]` : únicamente usado en el entrenamiento

#### Sección `[common]`
* `kvazaar` : ruta absoluta del ejecutable de kvazaar
* `rewards` : ruta del archivo de recompensas
* `cores` : dos enteros separados por una coma que indican los núclos que utliza Kvazaar

#### Sección `[train]`
* `batch`: tamaño de batch del entrenamiento
* `mini_batch` : minibatch del entrenaimento
* `videos` : ruta de los vídeos de entrenamiento
* `mode` : selección de vídeos. Puede ser `random` o `rotating`
* `iters` : número de iteraciones del entrenamiento (batch x iters pasos en total) 
* `name` : nombre de la ruta de de los resultados. Se generará una ruta desde `resultados`
* 
### Script de entrenamiento
Ejecuta un entrenamiento y guarda sus resultados en `resultados/<name de archivo config>`.
```
python3 src/train_kvazaar.py
```

### Script de ejecución de política aprendida

```
python3 src/learned_kvazaar.py -v <video> -p <path de checkpoints>
```
Ejecuta un agente con la política aprendida sobre un vídeo cuya ruta se indica en `-v <video>`.
La ruta de los checkpoints se encuentra en `resultados/<name de archivo config>/checkpoints`.
Se generan un archivo `.csv` con los resultados obtenidos que se guarda en `csv/`.

### Script de datos base
Este script ejecuta un agente con acciones base para Kvazaar
```
python3 src/baseline.py -v <video> [-r | -c <nº Cores>]
```
La ruta vídeo sobre el que se realiza el baseline se indica con `-v <video>`.
Se debe elegir entre acciones aleatorias con `-r` o un número fijo de núcleos para Kvazaar con `-c <nº Cores>`.
Se generan un archivo `.csv` con los resultados obtenidos que se guarda en `csv/`.





