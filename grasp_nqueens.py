"""
GRASP (Greedy Randomized Adaptive Search Procedure) aplicado al
Problema de las N-Reinas Escalable.

Taller Investigativo - Introducción a la Inteligencia Artificial
Algoritmo asignado: GRASP

Representación del estado: vector de tamaño N, donde el índice es la
columna y el valor es la fila (ej. [2, 0, 3, 1]). Como cada columna
aparece una sola vez y el vector se mantiene como permutación (sin
valores repetidos), quedan descartados por construcción los ataques en
filas y en columnas: solo hay que revisar las diagonales.

Función de costo: número total de ataques en las diagonales (óptimo = 0).

Uso:
    python grasp_nqueens.py        # caso base, N = 8
    python grasp_nqueens.py 50     # recibe N como parámetro
    python grasp_nqueens.py 100
"""

import argparse
import random
import time
import matplotlib.pyplot as plt


# ============================================================
# PARÁMETROS (declarados aquí para facilitar la experimentación)
# ============================================================
N = 8                      # Número de reinas / tamaño del tablero
ALPHA = 0.3                # Umbral de calidad para la RCL (0 = greedy puro, 1 = aleatorio puro)
GRASP_ITERATIONS = 30      # Número de veces que se repite construcción + búsqueda local
MAX_NO_IMPROVE = 2000      # Intentos de swap seguidos sin mejora antes de declarar óptimo local
STOP_AT_ZERO = False       # True: cortar apenas se encuentre costo 0
SEED = None                # None = cada ejecución da un resultado distinto.
                           # Con un entero fijo (ej. 7) siempre sale lo mismo,
                           # útil para repetir una corrida durante la sustentación.
PLOT_FILE = "convergencia_grasp.png"
MAX_BOARD_PRINT = 20       # Si N es mayor que esto, se imprime solo el vector solución


# ============================================================
# FUNCIÓN DE COSTO
# ============================================================
def cost(state):
    """
    Cuenta el número de pares de reinas que se atacan en diagonal.

    state: lista de longitud N, state[col] = fila de la reina en esa columna.
    Devuelve: entero >= 0 (0 = solución óptima).

    En vez de comparar todos los pares de reinas (O(N^2)), agrupamos las
    reinas por diagonal: dos reinas comparten diagonal "\\" si col - fila
    es igual, y diagonal "/" si col + fila es igual. Si una diagonal tiene
    k reinas, aporta k*(k-1)/2 pares en conflicto. Así el costo se calcula
    en O(N), que es lo que permite escalar a N=50 y N=100.
    """
    main_diagonals = {}       # clave: col - fila
    secondary_diagonals = {}  # clave: col + fila

    for col, row in enumerate(state):
        d1 = col - row
        d2 = col + row
        main_diagonals[d1] = main_diagonals.get(d1, 0) + 1
        secondary_diagonals[d2] = secondary_diagonals.get(d2, 0) + 1

    attacks = 0
    for counts in (main_diagonals, secondary_diagonals):
        for k in counts.values():
            attacks += k * (k - 1) // 2
    return attacks


# ============================================================
# FASE CONSTRUCTIVA (Greedy Randomizada)
# ============================================================
def diagonal_attacks_for_row(partial_state, col, row):
    """
    Cuenta cuántos ataques diagonales generaría colocar una reina en
    (row, col) contra las reinas YA colocadas en partial_state
    (columnas 0 .. col-1).

    Dos reinas están en la misma diagonal cuando la distancia horizontal
    entre ellas es igual a la distancia vertical.
    """
    attacks = 0
    for c in range(col):
        if abs(row - partial_state[c]) == abs(col - c):
            attacks += 1
    return attacks


def construct_greedy_randomized(n, alpha):
    """
    Construye una solución columna por columna usando una Lista
    Restringida de Candidatos (RCL).

    Para cada columna:
    1. Se calcula, para cada fila todavía disponible, cuántos ataques
       diagonales generaría contra las reinas ya colocadas.
    2. Se arma la RCL con las filas cuyo costo no supera el umbral
       min + alpha * (max - min).
    3. Se elige una fila al azar dentro de la RCL y se quita de las
       disponibles (para no repetir fila).

    Con alpha = 0 el umbral es el mínimo, o sea greedy puro; con alpha = 1
    entran todas las filas libres, o sea aleatorio puro. Los valores
    intermedios dan la aleatoriedad controlada propia de GRASP: cada
    ejecución construye una solución distinta sin perder el criterio goloso.
    """
    state = []
    available_rows = list(range(n))

    for col in range(n):
        # Ataques diagonales que generaría cada fila disponible.
        candidates = [
            (row, diagonal_attacks_for_row(state, col, row))
            for row in available_rows
        ]

        min_attacks = min(attacks for _, attacks in candidates)
        max_attacks = max(attacks for _, attacks in candidates)

        # RCL: las filas suficientemente buenas en este paso.
        threshold = min_attacks + alpha * (max_attacks - min_attacks)
        rcl = [row for row, attacks in candidates if attacks <= threshold]

        chosen_row = random.choice(rcl)
        state.append(chosen_row)
        available_rows.remove(chosen_row)

    return state


# ============================================================
# FASE DE BÚSQUEDA LOCAL (mejora con operador de swap)
# ============================================================
def local_search(state):
    """
    Mejora la solución construida con swaps entre dos posiciones
    aleatorias del vector.

    Estrategia hill climbing: se eligen dos columnas al azar, se
    intercambian sus filas y el movimiento se acepta solo si el costo
    baja; si no, se deshace. Se detiene al llegar a costo 0 o tras
    MAX_NO_IMPROVE intentos seguidos sin mejora, que es la señal de haber
    llegado a un óptimo local.

    El swap es el operador natural aquí porque conserva la permutación:
    nunca repite una fila y por lo tanto nunca introduce ataques de fila.
    """
    current = state.copy()
    current_cost = cost(current)
    n = len(current)
    failures = 0

    while current_cost > 0 and failures < MAX_NO_IMPROVE:
        i, j = random.sample(range(n), 2)

        current[i], current[j] = current[j], current[i]
        neighbor_cost = cost(current)

        if neighbor_cost < current_cost:
            current_cost = neighbor_cost   # nos quedamos con el vecino
            failures = 0
        else:
            current[i], current[j] = current[j], current[i]  # deshacer el swap
            failures += 1

    return current, current_cost


# ============================================================
# CICLO PRINCIPAL DE GRASP
# ============================================================
def grasp(n, alpha, iterations):
    """
    Ejecuta el ciclo completo de GRASP: repetir (construcción + búsqueda
    local) 'iterations' veces, guardando siempre la mejor solución global.

    Lo que distingue a GRASP de Hill Climbing o Simulated Annealing es que
    cada iteración arranca de una solución NUEVA generada por la fase
    constructiva golosa-aleatoria, en lugar de arrastrar una sola solución
    inicial durante todo el proceso.

    Devuelve:
    - best_state: mejor vector solución encontrado
    - best_cost: su costo (ataques diagonales)
    - history: mejor costo alcanzado hasta cada iteración (para graficar
      la convergencia)
    - elapsed: tiempo de ejecución en segundos
    """
    start_time = time.perf_counter()

    best_state = None
    best_cost = float("inf")
    history = []

    for _ in range(iterations):
        candidate = construct_greedy_randomized(n, alpha)
        candidate, candidate_cost = local_search(candidate)

        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_state = candidate

        history.append(best_cost)

        if best_cost == 0 and STOP_AT_ZERO:
            break

    elapsed = time.perf_counter() - start_time
    return best_state, best_cost, history, elapsed


# ============================================================
# VISUALIZACIÓN Y MÉTRICAS
# ============================================================
def print_board(state):
    """Imprime el vector solución y, si N es pequeño, dibuja el tablero."""
    n = len(state)
    print(f"\nVector solución: {state}\n")

    if n > MAX_BOARD_PRINT:
        return

    for row in range(n):
        line = ""
        for col in range(n):
            line += " Q " if state[col] == row else " . "
        print(line)
    print()


def plot_convergence(history, n):
    """
    Grafica cómo disminuyó el costo de la mejor solución iteración tras
    iteración. Guarda la imagen y la muestra en pantalla.
    """
    plt.figure()
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("Iteración GRASP")
    plt.ylabel("Mejor costo encontrado (ataques diagonales)")
    plt.title(f"Convergencia de GRASP (N={n})")
    plt.grid(True)
    plt.savefig(PLOT_FILE)
    print(f"Gráfica guardada en: {PLOT_FILE}")
    plt.show()


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GRASP para el problema de las N-Reinas")
    parser.add_argument("n", nargs="?", type=int, default=N,
                        help=f"tamaño del tablero (por defecto {N})")
    args = parser.parse_args()

    if args.n < 1:
        parser.error("N debe ser un entero positivo")

    if SEED is not None:
        random.seed(SEED)

    print(f"Ejecutando GRASP para N={args.n} con {GRASP_ITERATIONS} iteraciones "
          f"y alpha={ALPHA}...")

    best_state, best_cost, history, elapsed = grasp(args.n, ALPHA, GRASP_ITERATIONS)

    print_board(best_state)
    print(f"Costo final (ataques diagonales): {best_cost}")
    print(f"Tiempo de ejecución: {elapsed:.4f} segundos")
    print(f"Iteraciones realizadas: {len(history)}")

    plot_convergence(history, args.n)


if __name__ == "__main__":
    main()
