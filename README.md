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

## Benchmark estadístico

`benchmark_grasp.py` ejecuta múltiples corridas de GRASP para cada tamaño
de tablero y reporta promedios:

| N   | Tasa de éxito (%) | Iteraciones hasta óptimo | Conflictos tras construcción | Tiempo (s) |
|-----|-------------------|--------------------------|------------------------------|------------|
| 8   | 100.0             | 2.25                     | 1.48                         | 0.0070     |
| 50  | 100.0             | 2.35                     | 3.72                         | 0.8552     |
| 100 | 100.0             | 2.30                     | 4.55                         | 3.6146     |

Parámetros: alpha=0.3, 30 iteraciones GRASP, 20 corridas por N, semilla=42.

### Uso

```bash
python benchmark_grasp.py                               # configuración por defecto (N=8,50,100; 20 corridas)
python benchmark_grasp.py --n 8,50,100 --runs 20
python benchmark_grasp.py --alpha 0.3 --iterations 30 --seed 42
python benchmark_grasp.py --n 8 --runs 50 --output results.csv    # también exporta CSV
```

Los parámetros por defecto también se pueden editar directamente en el bloque
`PARÁMETROS CONFIGURABLES` al inicio del script.
