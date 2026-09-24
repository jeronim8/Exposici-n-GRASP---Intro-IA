"""
Benchmark para GRASP en el problema de las N-Reinas.

Ejecuta múltiples corridas de GRASP, a partir de las funciones ya
elaboradas en el archivo grasp_nqueens, para diferentes tamaños de
tablero y recopila estadísticas:

  - Tasa de éxito: porcentaje de corridas que alcanzaron costo 0.
  - Iteraciones hasta el óptimo: número promedio de iteraciones GRASP
    necesarias para encontrar la solución óptima (0 conflictos).
  - Conflictos tras la construcción: costo promedio obtenido por la
    fase constructiva (antes de la búsqueda local), promediado sobre
    todas las iteraciones GRASP de todas las corridas.
  - Tiempo (s): tiempo promedio de ejecución de una corrida completa.

Uso:
    python benchmark_grasp.py                    # configuración por defecto
    python benchmark_grasp.py --n 8,50,100       # N values personalizados
    python benchmark_grasp.py --runs 20          # número de corridas
    python benchmark_grasp.py --alpha 0.3        # parámetro alpha
    python benchmark_grasp.py --iterations 30    # iteraciones GRASP
    python benchmark_grasp.py --seed 42          # semilla fija (reproducible)
    python benchmark_grasp.py --n 8,50,100 --runs 20 --alpha 0.3 --iterations 30 --seed 42
"""

import argparse
import random
import time

# ================================================================
# Importar funciones y parámetros del módulo principal de GRASP.
# ================================================================

from grasp_nqueens import (
    construct_greedy_randomized,
    local_search,
    cost,
    MAX_NEIGHBOR_EV,
)


# ================================================================
# PARÁMETROS CONFIGURABLES

# Declarados aquí para facilitar la experimentación.
# También pueden sobreescribirse desde la línea de comandos.
# ================================================================

N_VALUES = [8, 50, 100]          # Tamaños de tablero a evaluar
RUNS = 20                        # Número de corridas por cada N
ALPHA = 0.3                      # Umbral de calidad para la RCL
GRASP_ITERATIONS = 30            # Iteraciones constructivas + búsqueda local por corrida
MAX_NEIGHBOR_SWAP = MAX_NEIGHBOR_EV  # Máximo de swaps evaluados por iteración de local_search
SEED = None                      # None = aleatorio por corrida; entero = reproducible


# ================================================================
# FUNCIÓN: UNA CORRIDA INDIVIDUAL DE GRASP

# Parametro(s):
#   n: número de reinas.
#   alpha: nivel de aleatoriedad de la RCL.
#   iterations: número de repeticiones de construcción + búsqueda local,
#   al interior de cada GRASP.
#   max_neighbor_swap: límite de swaps evaluados en local_search.

# Retorno(s): diccionario con las siguientes claves:
#   - best_cost: costo final de la mejor solución encontrada.
#   - success: True si se alcanzó costo 0 (solución óptima).
#   - iterations_to_optimal: número de iteración GRASP en que se
#     alcanzó el óptimo por primera vez (o `iterations` si nunca).
#   - construction_cost: PROMEDIO del costo justo después de la
#     fase constructiva (antes de local_search), sobre todas las
#     iteraciones GRASP de esta corrida.
#   - elapsed: tiempo total de ejecución en segundos.

# Se ejecutan todas las iteraciones GRASP completas (no se corta
# al encontrar el óptimo) para que la métrica de tiempo sea
# comparable. La iteración en que apareció el óptimo se registra
# por separado.
# ================================================================

def grasp_single_run(n, alpha, iterations, max_neighbor_swap):

    start_time = time.perf_counter() #Contador para el tiempo de ejecución.

    best_state = None #Mejor solución
    best_cost = float("inf") #Mejor costo, inicialmente infinito para seleccionar la primera solución identificada.
    construction_costs = []
    iterations_to_optimal = None

    for it in range(iterations):
        # --- Fase constructiva ---
        candidate = construct_greedy_randomized(n, alpha)
        construction_costs.append(cost(candidate))

        # --- Fase de búsqueda local ---
        candidate, candidate_cost = local_search(candidate, max_swaps=max_neighbor_swap)

        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_state = candidate #Se reemplaza la solución, y su costo, siempre que se identifique una mejor alternativa en la búsqueda local.

        #Registrar cuándo se alcanzó el óptimo por primera vez
        if best_cost == 0 and iterations_to_optimal is None:
            iterations_to_optimal = it + 1

    elapsed = time.perf_counter() - start_time

    if iterations_to_optimal is None:
        iterations_to_optimal = iterations

    avg_construction_cost = sum(construction_costs) / len(construction_costs)

    return {
        "best_cost": best_cost,
        "success": best_cost == 0,
        "iterations_to_optimal": iterations_to_optimal,
        "construction_cost": avg_construction_cost,
        "elapsed": elapsed,
    }


# ================================================================
# FUNCIÓN: EJECUTAR TODAS LAS CORRIDAS PARA UN N

# Parametro(s):
#   n: número de reinas.
#   runs: número de corridas completas a ejecutar.
#   alpha: parámetro alpha de la RCL.
#   grasp_iterations: iteraciones GRASP por corrida.
#   max_neighbor_swap: límite de swaps en local_search.
#   seed: semilla base (None = totalmente aleatorio).

# Retorno(s): diccionario con los promedios de las métricas.
# ================================================================

def run_experiment(n, runs, alpha, grasp_iterations, max_neighbor_swap, seed):

    print(f"\n{'=' * 70}")
    print(f"  N = {n}   |   Corridas: {runs}   |   Alpha: {alpha}   "
          f"|   GRASP iter: {grasp_iterations}")
    print(f"{'=' * 70}") #Impresión de los parámetros del experimento.

    results = []

    for run in range(runs):
        run_seed = None
        if seed is not None:
            run_seed = seed + run #Si el usuario definió una semilla, se procura que, para cada GRASP sea distinto, de modo que los resultados sean diferentes, pero el proceso sea reproducible.
            random.seed(run_seed)

        result = grasp_single_run(n, alpha, grasp_iterations, max_neighbor_swap)
        results.append(result) #Se realizan los GRASP y se almacenan sus resultados.

        status = "[OK] optimo" if result["success"] else "[--] suboptimo"
        print(f"  Corrida {run + 1:2d}/{runs}: {status}  "
              f"costo={result['best_cost']:>4d}  "
              f"iters_óptimo={result['iterations_to_optimal']:>3d}  "
              f"conf_constructiva={result['construction_cost']:.2f}  "
              f"tiempo={result['elapsed']:.4f}s") #Se imprimen los resultados de los GRASP.

    #Obtención de promedios
    avg_success = sum(1 for r in results if r["success"]) / runs * 100
    avg_iters = sum(r["iterations_to_optimal"] for r in results) / runs
    avg_construction = sum(r["construction_cost"] for r in results) / runs
    avg_time = sum(r["elapsed"] for r in results) / runs

    return {
        "tasa_exito": avg_success,
        "iters_optimo": avg_iters,
        "conflictos_construccion": avg_construction,
        "tiempo": avg_time,
    }


# ================================================================
# FUNCIÓN: IMPRIMIR TABLA DE RESULTADOS

# Parametro(s):
#   averages: diccionario {N: {métrica: promedio, ...}, ...}
#   n_values: lista de valores de N evaluados.
#   alpha: parámetro alpha usado.
#   grasp_iterations: iteraciones GRASP usadas.
#   runs: número de corridas.

# Retorno(s): ninguno (únicamente se presenta por pantalla).
# ================================================================

def print_table(averages, n_values, alpha, grasp_iterations, runs):

    print(f"\n{'=' * 85}")
    print(f"RESULTADOS PROMEDIO — {runs} corridas por cada N")
    print(f"Parámetros: alpha={alpha}, GRASP_iteraciones={grasp_iterations}")
    print(f"{'=' * 85}")

    header = (
        f"{'N':>6} | "
        f"{'Tasa de éxito (%)':>18} | "
        f"{'Iteraciones':>12} | "
        f"{'Conflictos':>14} | "
        f"{'Tiempo (s)':>12}"
    )
    print(header)
    print("-" * 85)

    for n in n_values:
        avg = averages[n]
        print(
            f"{n:>6} | "
            f"{avg['tasa_exito']:>17.1f}% | "
            f"{avg['iters_optimo']:>12.2f} | "
            f"{avg['conflictos_construccion']:>14.2f} | "
            f"{avg['tiempo']:>12.4f}"
        )

    print("-" * 85)
    print("  Tasa de éxito: porcentaje de corridas con costo óptimo (0).")
    print("  Iteraciones: número promedio de iteraciones GRASP hasta el óptimo.")
    print("  Conflictos: costo promedio tras la fase constructiva (antes de local_search).")
    print("  Tiempo (s): tiempo promedio de una corrida completa.")
    print(f"{'=' * 85}\n")


# ================================================================
# FUNCIÓN PRINCIPAL

# Parsea argumentos de línea de comandos, ejecuta el experimento
# para cada valor de N y muestra la tabla de resultados.
# ================================================================

def main():

    #Todo lo relacionado con parser permite modificar parámetros desde línea de comandos, por parte del usuario
    parser = argparse.ArgumentParser(
        description="Benchmark estadístico para GRASP en el problema de las N-Reinas."
    )
    parser.add_argument(
        "--n",
        type=str,
        default=",".join(str(n) for n in N_VALUES),
        help=f"valores de N separados por coma (por defecto: {','.join(str(n) for n in N_VALUES)})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS,
        help=f"número de corridas por cada N (por defecto: {RUNS})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help=f"parámetro alpha de la RCL (por defecto: {ALPHA})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=GRASP_ITERATIONS,
        help=f"iteraciones GRASP por corrida (por defecto: {GRASP_ITERATIONS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="semilla aleatoria base (por defecto: None = aleatorio)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="archivo CSV para guardar los resultados promedio (ej: results.csv)",
    )
    parser.add_argument(
        "--max-swaps",
        type=int,
        default=MAX_NEIGHBOR_SWAP,
        help=f"máximo de swaps evaluados por iteración de local_search (por defecto: {MAX_NEIGHBOR_SWAP})",
    )

    args = parser.parse_args()

    # Parsear N values
    n_values = [int(x.strip()) for x in args.n.split(",")]
    for n in n_values:
        if n < 1:
            parser.error(f"N={n} debe ser un entero positivo")

    print(f"\n{'#' * 85}")
    print(f"#  BENCHMARK GRASP — Problema de las N-Reinas")
    print(f"#  N values: {n_values}")
    print(f"#  Corridas por N: {args.runs}")
    print(f"#  Alpha: {args.alpha}")
    print(f"#  Iteraciones GRASP: {args.iterations}")
    print(f"#  Semilla: {args.seed if args.seed is not None else 'aleatoria'}")
    print(f"{'#' * 85}")

    # Ejecutar experimento para cada N
    averages = {}
    for n in n_values:
        avg = run_experiment(
            n,
            args.runs,
            args.alpha,
            args.iterations,
            args.max_swaps,
            args.seed,
        )
        averages[n] = avg

    # Mostrar tabla de resultados
    print_table(averages, n_values, args.alpha, args.iterations, args.runs)

    # Guardar resultados en CSV si se especificó archivo
    if args.output:
        import csv
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["N", "Tasa de exito (%)", "Iteraciones hasta optimo",
                             "Conflictos tras construccion", "Tiempo (s)"])
            for n in n_values:
                avg = averages[n]
                writer.writerow([
                    n,
                    f"{avg['tasa_exito']:.1f}",
                    f"{avg['iters_optimo']:.2f}",
                    f"{avg['conflictos_construccion']:.2f}",
                    f"{avg['tiempo']:.4f}",
                ])
        print(f"Resultados guardados en: {args.output}")


if __name__ == "__main__":
    main()
