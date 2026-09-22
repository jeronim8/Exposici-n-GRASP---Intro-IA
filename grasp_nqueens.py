"""
GRASP (Greedy Randomized Adaptive Search Procedure) aplicado al
Problema de las N-Reinas Escalable.

Taller Investigativo - Introducción a la Inteligencia Artificial
Algoritmo asignado: GRASP

Representación del estado: vector de tamaño N, donde el índice es la
columna y el valor es la fila (ej. [2, 0, 3, 1]).
Función de costo: número total de ataques en las diagonales (óptimo = 0).
"""

import random
import time
import matplotlib.pyplot as plt


# ============================================================
# PARÁMETROS (declarados aquí para facilitar la experimentación)
# ============================================================
N = 8                     # Número de reinas / tamaño del tablero
GRASP_ITERATIONS = 100    # Número de veces que se repite construcción + mejora
ALPHA = 0.3                # Umbral de calidad para la RCL (0 = greedy puro, 1 = aleatorio puro)
# Alternativa a ALPHA: usar un tamaño fijo de RCL en lugar de un umbral.
RCL_SIZE = 3               # Úsenlo si deciden construir la RCL por tamaño fijo en vez de alpha
MAX_LOCAL_SEARCH_ITER = 200  # Máximo de iteraciones sin mejora antes de parar la búsqueda local
SEED = None                 # Fijar un entero aquí (ej. 42) para resultados reproducibles


# ============================================================
# FUNCIÓN DE COSTO
# ============================================================
def cost(state):
    """
    Cuenta el número de pares de reinas que se atacan en diagonal.

    state: lista de longitud N, state[col] = fila de la reina en esa columna.
    Devuelve: entero >= 0 (0 = solución óptima).

    """
    n = len(state)
    attacks = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(state[i] - state[j]) == abs(i - j):
                attacks += 1
    return attacks


def diagonal_attacks_for_row(partial_state, col, row):
    """
    Cuenta cuántos ataques diagonales generaría colocar una reina en
    (row, col) contra las reinas YA colocadas en partial_state
    (columnas 0 .. col-1).

    Se usa en la fase constructiva de GRASP para evaluar candidatos.

    """
    attacks = 0
    for c in range(col):
        if abs(row - partial_state[c]) == abs(col - c):
            attacks += 1
    return attacks


# ============================================================
# FASE CONSTRUCTIVA (Greedy Randomizada)
# ============================================================
def construct_greedy_randomized(n):
    """
    Construye una solución columna por columna usando una Lista
    Restringida de Candidatos (RCL).

    Pasos (según el enunciado):
    1. Empezar con un vector vacío / lista de filas disponibles = [0..n-1].
    2. Para cada columna (de 0 a n-1):
        a. Para cada fila disponible, calcular cuántos ataques diagonales
           generaría contra las reinas ya colocadas
           (usar diagonal_attacks_for_row).
        b. Armar la RCL con las mejores filas (menor número de ataques),
           usando ALPHA o RCL_SIZE para decidir cuántas/cuáles entran.
        c. Elegir aleatoriamente una fila dentro de la RCL.
        d. Colocarla en esa columna y quitarla de las filas disponibles
           (para no repetir fila, ya que el vector no puede tener valores
           repetidos).
    3. Devolver el vector completo de tamaño n.

    """
    state = []
    available_rows = list(range(n))

    for col in range(n):
        # Calcular ataques diagonales para cada fila disponible
        candidates = [
            (row, diagonal_attacks_for_row(state, col, row))
            for row in available_rows
        ]

        min_attacks = min(a for _, a in candidates)
        max_attacks = max(a for _, a in candidates)

        # Construir RCL: filas cuyo costo está dentro del umbral ALPHA
        threshold = min_attacks + ALPHA * (max_attacks - min_attacks)
        rcl = [row for row, attacks in candidates if attacks <= threshold]

        # Elegir aleatoriamente una fila de la RCL
        chosen_row = random.choice(rcl)
        state.append(chosen_row)
        available_rows.remove(chosen_row)

    return state


# ============================================================
# FASE DE BÚSQUEDA LOCAL (mejora con operador de swap)
# ============================================================
def get_neighbors(state):
    """
    Genera vecinos intercambiando (swap) los valores de dos posiciones
    del vector.

    """
    n = len(state)
    neighbors = []
    for i in range(n):
        for j in range(i + 1, n):
            neighbor = state.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors


def local_search(state):
    """
    Aplica hill climbing sobre 'state' usando get_neighbors(), hasta que
    no haya ningún vecino que mejore el costo actual (óptimo local) o se
    alcance MAX_LOCAL_SEARCH_ITER.

    """
    current = state.copy()
    current_cost = cost(current)

    for _ in range(MAX_LOCAL_SEARCH_ITER):
        neighbors = get_neighbors(current)
        best_neighbor = None
        best_neighbor_cost = current_cost

        for neighbor in neighbors:
            neighbor_cost = cost(neighbor)
            if neighbor_cost < best_neighbor_cost:
                best_neighbor_cost = neighbor_cost
                best_neighbor = neighbor

        # Si ningún vecino mejora, hemos alcanzado un óptimo local
        if best_neighbor is None:
            break

        current = best_neighbor
        current_cost = best_neighbor_cost

    return current


# ============================================================
# CICLO PRINCIPAL DE GRASP
# ============================================================
def grasp(n, iterations):
    """
    Ejecuta el ciclo completo de GRASP:
    repetir (construcción + búsqueda local) 'iterations' veces,
    guardando siempre la mejor solución encontrada.

    Devuelve:
    - best_state: mejor vector solución encontrado
    - best_cost: su costo (ataques diagonales)
    - history: lista con el mejor costo encontrado hasta cada iteración
               (para la gráfica de convergencia)
    """
    best_state = None
    best_cost = float("inf")
    history = []

    for it in range(iterations):
        candidate = construct_greedy_randomized(n)
        candidate = local_search(candidate)
        candidate_cost = cost(candidate)

        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_state = candidate

        history.append(best_cost)

        # Corte anticipado: ya encontramos una solución óptima
        if best_cost == 0:
            break

    return best_state, best_cost, history


# ============================================================
# VISUALIZACIÓN Y MÉTRICAS
# ============================================================
def print_board(state):
    """Imprime el tablero en consola usando el vector solución."""
    n = len(state)
    print(f"\nVector solución: {state}")
    for row in range(n):
        line = ""
        for col in range(n):
            line += " Q " if state[col] == row else " . "
        print(line)
    print()


def plot_convergence(history):
    """Grafica cómo disminuyó el mejor costo encontrado por iteración."""
    plt.figure()
    plt.plot(history)
    plt.xlabel("Iteración GRASP")
    plt.ylabel("Mejor costo encontrado (ataques diagonales)")
    plt.title(f"Convergencia de GRASP (N={N})")
    plt.grid(True)
    plt.savefig("convergencia_grasp.png")
    plt.show()


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================
def main():
    if SEED is not None:
        random.seed(SEED)

    print(f"Ejecutando GRASP para N={N} con {GRASP_ITERATIONS} iteraciones...")

    start_time = time.time()
    best_state, best_cost, history = grasp(N, GRASP_ITERATIONS)
    elapsed_time = time.time() - start_time

    print_board(best_state)
    print(f"Costo final (ataques diagonales): {best_cost}")
    print(f"Tiempo de ejecución: {elapsed_time:.4f} segundos")
    print(f"Iteraciones realizadas: {len(history)}")

    plot_convergence(history)


if __name__ == "__main__":
    main()