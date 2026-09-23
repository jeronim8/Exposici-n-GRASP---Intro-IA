# GRASP aplicado a las N-Reinas

Taller Investigativo — Introducción a la Inteligencia Artificial
Ing. de Sistemas, Pontificia Universidad Javeriana

Implementación del metaheurístico GRASP (Greedy Randomized Adaptive Search Procedure):
fase constructiva golosa-aleatoria con Lista Restringida de Candidatos (RCL) + fase de
búsqueda local con operador de swap, repetidas en un ciclo multi-arranque.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python grasp_nqueens.py
```

Caso base N=8. Para experimentar con escalabilidad y parámetros:

```bash
python grasp_nqueens.py --n 50 --alpha 0.3 --iteraciones 30
python grasp_nqueens.py --n 100 --alpha 0.2 --iteraciones 30 --semilla 42
```

Banderas disponibles: `--n`, `--alpha`, `--iteraciones`, `--semilla`,
`--vecindario` (`aleatorio` o `completo`), `--todas` (no detenerse al
encontrar costo 0) y `--sin-grafica`. Los parámetros también se pueden
editar directamente en el bloque `PARÁMETROS` al inicio del script.

## Salida

- Vector solución y tablero (el tablero se dibuja solo si N <= 20)
- Costo final (ataques diagonales; 0 = óptimo) y tiempo de ejecución
- Gráfica de convergencia guardada como `convergencia_grasp.png`, con tres
  series: costo tras la fase constructiva, costo tras la búsqueda local y
  mejor costo global acumulado

## Las dos estrategias de búsqueda local

El script incluye dos operadores de mejora, seleccionables con `--vecindario`:

- **`aleatorio`** (por defecto): swap entre dos posiciones al azar, se acepta
  solo si baja el costo (primera mejora). Es el operador base del enunciado.
- **`completo`**: revisa los N(N-1)/2 swaps posibles y se mueve al mejor
  (mejor mejora). Llega a soluciones igual de buenas, pero cada paso evalúa
  todo el vecindario, así que escala peor.

Tiempos medidos hasta llegar a costo 0 (semilla 42, alpha 0.3 salvo N=100
que usa 0.2). Los dos alcanzan costo 0 en todos los tamaños:

| N   | vecindario aleatorio | vecindario completo |
|-----|----------------------|---------------------|
| 8   | < 0.001 s            | < 0.001 s           |
| 50  | 0.004 s              | 0.084 s             |
| 100 | 0.142 s              | 0.609 s             |

## Nota para la gráfica

Como la fase constructiva ya entrega soluciones muy buenas, GRASP suele
llegar a costo 0 en la primera o segunda iteración y la curva queda con
muy pocos puntos. Para una gráfica más ilustrativa conviene subir alpha
(construcciones más aleatorias) y usar `--todas`:

```bash
python grasp_nqueens.py --n 30 --alpha 0.9 --iteraciones 20 --semilla 7 --todas
```
