"""
GRASP (Greedy Randomized Adaptive Search Procedure) aplicado al
Problema de las N-Reinas Escalable.

Taller Investigativo - Introducción a la Inteligencia Artificial
Algoritmo asignado: GRASP

Representación del estado: vector de tamaño N, donde el índice es la
columna y el valor es la fila (ej. [2, 0, 3, 1]).
Función de costo: número total de ataques en las diagonales (óptimo = 0).

Uso:
    python grasp_nqueens.py                    # caso base con los parámetros de abajo
    python grasp_nqueens.py --n 50             # cambiar N por línea de comandos
    python grasp_nqueens.py --n 100 --alpha 0.2 --iteraciones 30
"""

import argparse
import random
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ============================================================
# PARÁMETROS (declarados aquí para facilitar la experimentación)
# También se pueden sobrescribir por línea de comandos.
# ============================================================
N = 8                     # Número de reinas / tamaño del tablero
GRASP_ITERATIONS = 30     # Número de veces que se repite construcción + mejora
ALPHA = 0.3                # Umbral de calidad para la RCL (0 = greedy puro, 1 = aleatorio puro)
MAX_NO_IMPROVE = 2000      # Intentos de swap seguidos sin mejora antes de declarar óptimo local
NEIGHBORHOOD = "aleatorio"  # "aleatorio" (swap entre dos posiciones al azar) o "completo"
SEED = None                 # Fijar un entero aquí (ej. 42) para resultados reproducibles
PLOT_FILE = "convergencia_grasp.png"
MAX_BOARD_PRINT = 20        # Si N es mayor que esto, solo se imprime el vector


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
def construct_greedy_randomized(n, alpha):
    """
    Construye una solución columna por columna usando una Lista
    Restringida de Candidatos (RCL).

    Pasos (según el enunciado):
    1. Empezar con un vector vacío / lista de filas disponibles = [0..n-1].
    2. Para cada columna (de 0 a n-1):
        a. Para cada fila disponible, calcular cuántos ataques diagonales
           generaría contra las reinas ya colocadas.
        b. Armar la RCL con las mejores filas (menor número de ataques),
           usando el umbral   min + alpha * (max - min).
        c. Elegir aleatoriamente una fila dentro de la RCL.
        d. Colocarla en esa columna y quitarla de las filas disponibles
           (para no repetir fila, ya que el vector no puede tener valores
           repetidos).
    3. Devolver el vector completo de tamaño n.

    Con alpha = 0 el umbral es el mínimo, o sea greedy puro; con alpha = 1
    entran todas las filas libres, o sea aleatorio puro. Los valores
    intermedios dan la aleatoriedad controlada propia de GRASP.

    Para saber cuántos ataques genera una fila candidata NO recorremos las
    columnas ya colocadas: llevamos contadores de cuántas reinas hay en
    cada diagonal, así la evaluación de cada candidato es O(1).
    """
    state = []
    available_rows = list(range(n))

    # Los índices se desplazan en n - 1 para que nunca sean negativos.
    occupied_d1 = [0] * (2 * n - 1)  # col - fila + (n - 1)
    occupied_d2 = [0] * (2 * n - 1)  # col + fila

    for col in range(n):
        # Ataques diagonales que generaría cada fila disponible.
        attacks_per_row = [occupied_d1[col - row + n - 1] + occupied_d2[col + row]
                           for row in available_rows]

        min_attacks = min(attacks_per_row)
        max_attacks = max(attacks_per_row)

        # Construir RCL: filas cuyo costo está dentro del umbral alpha.
        threshold = min_attacks + alpha * (max_attacks - min_attacks)
        rcl = [row for row, attacks in zip(available_rows, attacks_per_row)
               if attacks <= threshold]

        # Elegir aleatoriamente una fila de la RCL.
        chosen_row = random.choice(rcl)
        state.append(chosen_row)
        available_rows.remove(chosen_row)

        occupied_d1[col - chosen_row + n - 1] += 1
        occupied_d2[col + chosen_row] += 1

    return state


# ============================================================
# FASE DE BÚSQUEDA LOCAL (mejora con operador de swap)
# ============================================================
def local_search_random_swap(state, max_no_improve=MAX_NO_IMPROVE):
    """
    Búsqueda local con el operador base del enunciado: swap entre dos
    posiciones ALEATORIAS del vector.

    Estrategia hill climbing de primera mejora: se eligen dos columnas al
    azar, se intercambian sus filas y el movimiento se acepta solo si el
    costo baja; si no, se deshace. Se detiene al llegar a costo 0 o tras
    max_no_improve intentos seguidos sin mejora, lo que se interpreta como
    haber llegado a un óptimo local.

    El swap es el operador natural aquí porque conserva la permutación:
    nunca repite una fila y por lo tanto nunca introduce ataques de fila.
    """
    current = state.copy()
    current_cost = cost(current)
    n = len(current)
    failures = 0

    while current_cost > 0 and failures < max_no_improve:
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


def local_search_full_neighborhood(state, max_iterations=200):
    """
    Variante de búsqueda local: en vez de swaps al azar, revisa TODOS los
    vecinos posibles (los N*(N-1)/2 swaps) y se mueve al mejor de ellos
    (hill climbing de mejor mejora), hasta que ningún vecino mejore.

    Encuentra soluciones igual de buenas, pero cada paso revisa N(N-1)/2
    vecinos, así que escala peor: en N=100 tarda ~0.6 s frente a ~0.14 s de
    la versión aleatoria. Se deja implementada para poder comparar las dos
    estrategias con la bandera --vecindario.
    """
    current = state.copy()
    current_cost = cost(current)
    n = len(current)

    for _ in range(max_iterations):
        best_neighbor = None
        best_neighbor_cost = current_cost

        for i in range(n):
            for j in range(i + 1, n):
                neighbor = current.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbor_cost = cost(neighbor)
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor_cost = neighbor_cost
                    best_neighbor = neighbor

        # Si ningún vecino mejora, hemos alcanzado un óptimo local.
        if best_neighbor is None:
            break

        current = best_neighbor
        current_cost = best_neighbor_cost

    return current, current_cost


def local_search(state, neighborhood=NEIGHBORHOOD):
    """Despacha a la estrategia de búsqueda local elegida."""
    if neighborhood == "completo":
        return local_search_full_neighborhood(state)
    return local_search_random_swap(state)


# ============================================================
# CICLO PRINCIPAL DE GRASP
# ============================================================
def grasp(n, iterations, alpha=ALPHA, neighborhood=NEIGHBORHOOD, stop_at_zero=True):
    """
    Ejecuta el ciclo completo de GRASP:
    repetir (construcción + búsqueda local) 'iterations' veces,
    guardando siempre la mejor solución encontrada.

    Lo que distingue a GRASP de Hill Climbing o Simulated Annealing es que
    cada iteración arranca de una solución NUEVA generada por la fase
    constructiva golosa-aleatoria, en lugar de arrastrar una sola solución
    inicial durante todo el proceso.

    Devuelve:
    - best_state: mejor vector solución encontrado
    - best_cost: su costo (ataques diagonales)
    - history: diccionario con tres listas por iteración
        "construction" -> costo con el que salió la fase constructiva
        "iteration"    -> costo después de la búsqueda local
        "best"         -> mejor costo global acumulado (curva de convergencia)
    """
    best_state = None
    best_cost = float("inf")
    history = {"construction": [], "iteration": [], "best": []}

    for it in range(1, iterations + 1):
        # Fase 1: construcción greedy randomizada.
        candidate = construct_greedy_randomized(n, alpha)
        construction_cost = cost(candidate)

        # Fase 2: búsqueda local sobre lo construido.
        candidate, candidate_cost = local_search(candidate, neighborhood)

        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_state = candidate

        history["construction"].append(construction_cost)
        history["iteration"].append(candidate_cost)
        history["best"].append(best_cost)

        print(f"  iteración {it:3d}/{iterations}  construcción = {construction_cost:4d}"
              f"   tras búsqueda local = {candidate_cost:4d}"
              f"   mejor global = {best_cost:4d}")

        # Corte anticipado: ya encontramos una solución óptima.
        if best_cost == 0 and stop_at_zero:
            print("  -> se encontró una solución óptima, se detiene el ciclo")
            break

    return best_state, best_cost, history


# ============================================================
# VISUALIZACIÓN Y MÉTRICAS
# ============================================================
def print_board(state):
    """Imprime el tablero en consola usando el vector solución."""
    n = len(state)
    print(f"\nVector solución: {state}")
    if n > MAX_BOARD_PRINT:
        print(f"(tablero omitido por tamaño N={n}; se muestra solo el vector)\n")
        return
    for row in range(n):
        line = ""
        for col in range(n):
            line += " Q " if state[col] == row else " . "
        print(line)
    print()


def plot_convergence(history, n, alpha, filename=PLOT_FILE):
    """
    Grafica cómo disminuyó el mejor costo encontrado por iteración.

    Se dibujan tres series para poder explicar el aporte de cada fase de
    GRASP: con qué costo sale la construcción, en cuánto lo deja la
    búsqueda local y cómo evoluciona el mejor costo global.
    """
    iterations = range(1, len(history["best"]) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(iterations, history["construction"], marker="^", linestyle=":",
             color="tab:green", alpha=0.7, label="Tras fase constructiva (RCL)")
    plt.plot(iterations, history["iteration"], marker="o", linestyle="--",
             color="tab:orange", alpha=0.7, label="Tras búsqueda local")
    plt.plot(iterations, history["best"], marker="s", color="tab:blue",
             label="Mejor costo global (convergencia)")
    plt.xlabel("Iteración GRASP")
    plt.ylabel("Costo (ataques diagonales)")
    plt.title(f"Convergencia de GRASP - N-Reinas (N={n}, alpha={alpha})")
    # Iteraciones y costos son números enteros, evitamos marcas con decimales.
    axes = plt.gca()
    axes.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\nGráfica guardada en: {filename}")
    plt.show()


# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GRASP para el problema de las N-Reinas")
    parser.add_argument("--n", type=int, default=N,
                        help=f"tamaño del tablero (default {N})")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help=f"tamaño relativo de la RCL, 0=greedy 1=aleatorio (default {ALPHA})")
    parser.add_argument("--iteraciones", type=int, default=GRASP_ITERATIONS,
                        help=f"iteraciones de GRASP (default {GRASP_ITERATIONS})")
    parser.add_argument("--semilla", type=int, default=SEED,
                        help="semilla aleatoria para reproducir resultados")
    parser.add_argument("--vecindario", choices=["aleatorio", "completo"], default=NEIGHBORHOOD,
                        help="estrategia de búsqueda local (default aleatorio)")
    parser.add_argument("--todas", action="store_true",
                        help="ejecutar todas las iteraciones aunque ya se encuentre costo 0 "
                             "(útil para que la gráfica de convergencia sea más ilustrativa)")
    parser.add_argument("--sin-grafica", action="store_true",
                        help="no generar la gráfica de convergencia")
    args = parser.parse_args()

    # Validación de los parámetros: con alpha fuera de [0, 1] el umbral de la
    # RCL deja de tener sentido (y con alpha negativo la RCL queda vacía).
    if args.n < 1:
        parser.error("N debe ser un entero positivo")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("alpha debe estar entre 0 y 1")
    if args.iteraciones < 1:
        parser.error("el número de iteraciones debe ser al menos 1")

    if args.semilla is not None:
        random.seed(args.semilla)

    print("=" * 60)
    print(f"Ejecutando GRASP para N={args.n} con {args.iteraciones} iteraciones")
    print(f"alpha = {args.alpha} | búsqueda local = {args.vecindario}")
    print("=" * 60)

    # Caso conocido del problema: para N=2 y N=3 no existe ninguna solución
    # sin ataques, así que el costo nunca podrá llegar a 0.
    if args.n in (2, 3):
        print(f"Aviso: para N={args.n} el problema no tiene solución, "
              f"el costo mínimo alcanzable es mayor que 0.\n")

    start_time = time.perf_counter()
    best_state, best_cost, history = grasp(args.n, args.iteraciones, args.alpha,
                                           args.vecindario, stop_at_zero=not args.todas)
    elapsed_time = time.perf_counter() - start_time

    print("\n" + "-" * 60)
    print("RESULTADOS")
    print("-" * 60)
    print_board(best_state)
    print(f"Costo final (ataques diagonales): {best_cost}")
    print(f"Tiempo de ejecución: {elapsed_time:.4f} segundos")
    print(f"Iteraciones realizadas: {len(history['best'])}")
    if best_cost == 0:
        print("Estado: solución ÓPTIMA encontrada")
    else:
        print("Estado: no se alcanzó el óptimo, la mejor solución aún tiene ataques")

    if not args.sin_grafica:
        plot_convergence(history, args.n, args.alpha)


if __name__ == "__main__":
    main()
