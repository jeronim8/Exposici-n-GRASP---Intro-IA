# GRASP aplicado a las N-Reinas

Taller Investigativo — Introducción a la Inteligencia Artificial
Ing. de Sistemas, Pontificia Universidad Javeriana

Implementación del metaheurístico GRASP (Greedy Randomized Adaptive Search
Procedure) para el problema de las N-Reinas: una fase constructiva
golosa-aleatoria con Lista Restringida de Candidatos (RCL) y una fase de
búsqueda local con swaps, repetidas en un ciclo multi-arranque.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python grasp_nqueens.py
```

Corre el caso base N=8. Para otros tamaños, N se pasa como parámetro:

```bash
python grasp_nqueens.py 50
python grasp_nqueens.py 100
```

Alpha, el número de iteraciones y la semilla se editan en el bloque
`PARÁMETROS` al inicio del script.

## Qué imprime

- El vector solución y el tablero (el tablero solo si N <= 20)
- El costo final en ataques diagonales, donde 0 es el óptimo
- El tiempo de ejecución en segundos
- La gráfica de convergencia, que se abre en pantalla y se guarda como
  `convergencia_grasp.png`

## Resultados

Con alpha = 0.5 y 30 iteraciones, los tres tamaños llegan a costo 0:

| N   | tiempo  |
|-----|---------|
| 8   | 0.13 s  |
| 50  | 1.35 s  |
| 100 | 4.02 s  |
