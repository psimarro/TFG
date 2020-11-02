#include <iostream>
#include <unistd.h>
#include <chrono>

using namespace std;
void dummy(double);

int main(){
    double n;
    
    while(true){
		cin >> n;
    	// check tiempo inicio
    	auto start = chrono::high_resolution_clock::now();
    	dummy(n);
    	// check tiempo de finalizacion
    	auto finish = chrono::high_resolution_clock::now();
    	//Calcula tiempo trascurrido
    	chrono::duration<double> elapsed = finish-start;
    		
    	cout << "Tiempo transcurrido: " << elapsed.count()*1000 << " ms" << "\n------------\n";
    }
    
    return 0;
}

// Función de prueba que calcula la suma de los enteros de 0 hasta n y escribe el resultado por pantalla
void dummy(double n){
	
	double suma = 0;
	 for(double i = 1; i <= n; ++i){
	 	suma += i;
	 }
	 cout << "n: " << n << "\n" << "suma = " << suma << "\n";
}
