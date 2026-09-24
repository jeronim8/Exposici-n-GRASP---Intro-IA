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


# ================================================================
# PARÁMETROS

# Declarados aquí para facilitar la experimentación.
# ================================================================

N = 8                      # Número de reinas / tamaño del tablero
ALPHA = 0.3                # Umbral de calidad para la RCL (0 = greedy puro, 1 = aleatorio puro)
GRASP_ITERATIONS = 30      # Número de veces que se repite construcción + búsqueda local
MAX_NEIGHBOR_EV = 2000     # Intentos de swap seguidos sin mejora antes de declarar óptimo local. Criterio de detenimiento.
STOP_AT_ZERO = False       # True: cortar apenas se encuentre costo 0 (la primera solución óptima).
SEED = None                # None = cada ejecución da un resultado distinto.
                           # Con un entero fijo (ej. 7) siempre sale lo mismo,
                           # útil para repetir una corrida durante la sustentación.
PLOT_FILE = "convergencia_grasp.png"
MAX_BOARD_PRINT = 50       # Si N es mayor que esto, se imprime solo el vector solución


# ================================================================
# FUNCIÓN DE COSTO

# Parámetro(s): state: lista de longitud N, que contiene la
# información sobre el posicionamiento de las reinas. El índice 
# representa la columna, mientras que el valor presenta la fila.

# Retorno(s): entero >= 0 (0 = solución óptima).

# Cuenta el número de pares de reinas que se atacan en diagonal.

# En vez de comparar todos los pares de reinas ( complejidad
# temporal de O(N^2)), agrupamos las reinas por diagonal: dos 
# reinas comparten diagonal "\\" si col - fila es igual, y
# diagonal "//" si col + fila es igual. Si una diagonal tiene
# k reinas, aporta k*(k-1)/2 pares en conflicto. Así el costo
# se calculaen O(N), que es lo que permite escalar a N=50 y N=100.
# ================================================================

def cost(state):
    
    main_diagonals = {}       #clave: col - fila
    secondary_diagonals = {}  #clave: col + fila

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

# ================================================================
# FUNCIÓN ATAQUES NUEVA REINA

# Parámetro(s): partial_state: lista temporal de longitud col-1,
# que alberga las locaciones de las reinas posicionadas hasta
# cierto instante. El índice representa la columna, mientras que
# el valor presenta la fila.
# col: siguiente columna disponible para ubicar una reina.
# row: siguiente fila válida para establecer una reina.

# Retorno(s): entero (costo de la nueva reina, en la ubicación
# ingresada) >= 0 (0 = solución óptima).

# Enumera cuántos ataques diagonales propiciaría ubicar una reina
# en (row, col), de acuerdo con las piezas ya establecidas en
# partial_state. 
# ================================================================

def diagonal_attacks_for_row(partial_state, col, row):
    
    attacks = 0
    for c in range(col):
        if abs(row - partial_state[c]) == abs(col - c): #Dos reinas están en la misma diagonal cuando la distancia horizontal
            #entre ellas es igual a la distancia vertical.
            attacks += 1
    return attacks

# ================================================================
# FUNCIÓN CONSTRUCTIVA DE GRASP

# Parámetro(s): n: número de reinas.
# alpha: nivel de aleatoriedad de la Lista Restringida de
# Candidatos (proporción de alternativas a incluir, según
# el margen entre costo mínimimo y máximo).

# Retorno(s): lista solución con las posiciones de las reinas.

# Construye una solución columna por columna usando una Lista
# Restringida de Candidatos (RCL).

# Para cada columna:

# 1. Se calcula, para cada fila todavía disponible, cuántos
# ataques diagonales generaría contra las reinas ya colocadas.
# 2. Se construye la RCL con las filas cuyo costo no supera el
# umbral min + alpha * (max - min).
# 3. Se elige una fila al azar dentro de la RCL y se quita de las
# disponibles (para no repetir fila).

# Con alpha = 0, el umbral es el mínimo, o sea greedy puro; con
# alpha = 1, entran todas las filas libres, o sea aleatorio puro.
# Los valores intermedios dan la aleatoriedad controlada propia
# de GRASP, por esa razón se prefirió un alpha de 0.3.
# ================================================================

def construct_greedy_randomized(n, alpha):
    
    state = []
    available_rows = list(range(n))

    for col in range(n):
        #Ataques diagonales que generaría cada fila disponible.
        candidates = [
            (row, diagonal_attacks_for_row(state, col, row))
            for row in available_rows
        ]

        min_attacks = min(attacks for _, attacks in candidates)
        max_attacks = max(attacks for _, attacks in candidates)

        #RCL: las filas suficientemente buenas en este paso.
        lim = min_attacks + alpha * (max_attacks - min_attacks)
        rcl = [row for row, attacks in candidates if attacks <= lim]

        chosen_row = random.choice(rcl)
        state.append(chosen_row)
        available_rows.remove(chosen_row)

    return state

# ================================================================
# FASE DE BÚSQUEDA LOCAL (mejora con operador de swap)

# Mejora la solución construida con swaps entre dos posiciones
# aleatorias del vector. 

# Estrategia hill climbing: se eligen dos columnas al azar, se
# intercambian sus filas y el movimiento se acepta solo si el
# costo baja; si no, se deshace. Para la selección de las
# posiciones a alternar, se consideran todos los posibles vecinos
# de una distribución (todas sus modificaciones de pares) y se
# evalúan aleatoriamente. Ahora bien, si el número de vecinos
# disponibles supera a MAX_NEIGHBORS_EV (2000), se analizan
# únicamente MAX_NEIGHBORS_EV swaps, para que el programa sea
# escalable a más tableros de N = 100 o superiores. Si se obtiene
# una solución óptima (costo 0), o se evalúan todas las
# posibilidades sin mejora alguna, se detiene el programa.

# El swap es el operador natural aquí porque conserva la
# permutación: nunca repite una fila y por lo tanto nunca
# introduce ataques de fila.
# ================================================================

def local_search(state, max_swaps = MAX_NEIGHBOR_EV):

    current = state.copy()
    current_cost = cost(current)
    n = len(current)

    while current_cost > 0:

        swaps = [
            (i,j)
            for i in range(n)
            for j in range(i+1, n)
        ]

        random.shuffle(swaps)

        limit = min(len(swaps), max_swaps) #se elige entre un máximo de iteraciones definido por nosotros y el número total posible de swaps, para mejorar la eficiencia del código.
        improved = False

        for i, j in swaps[:limit]:
            current[i], current[j] = current[j], current[i]
            neighbor_cost = cost(current)

            if neighbor_cost < current_cost:
                current_cost = neighbor_cost   #nos quedamos con el vecino
                improved = True
                break

            current[i], current[j] = current[j], current[i]  # deshacer el swap

        if not improved:
            break

    return current, current_cost

# ================================================================
# CICLO PRINCIPAL DE GRASP

# Parametro(s): n: número de reinas.
# alpha: nivel de aleatoriedad de la Lista Restringida de
# Candidatos (proporción de alternativas a incluir, según
# el margen entre costo mínimimo y máximo).
# iterations: cantidad de oportunidades en que se ejecuta
# GRASP para obtener distintas soluciones.

# Retorno(s): mejor vector solución identificado, además de
# su costo; el historial de los mejores costos logrados en
# cada iteración de GRASP (para el gráfico de convergencia),
# y el tiempo de ejecución, en segundos.

# Ejecuta el ciclo completo de GRASP: repetir (construcción
# + búsqueda local) 'iterations' veces, guardando siempre la
# mejor solución global. Lo que distingue a GRASP de Hill 
# Climbing o Simulated Annealing es que cada iteración arranca
# de una solución nueva generada por la fase constructiva
# aleatoria, en lugar de arrastrar una sola solución inicial
#  durante todo el proceso.
# ================================================================

def grasp(n, alpha, iterations):
    
    start_time = time.perf_counter()

    best_state = None
    best_cost = float("inf") #Costo inicializado en infinito para que cualquier solución generada sea preferida, para comenzar.
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

# ================================================================
# IMPRESIÓN DE LA SOLUCIÓN

# Parametro(s): state: lista solución.

# Retorno(s): ninguno (únicamente se presenta por pantalla).

# Se imprime el vector solución, resultado del proceso de
# GRASP, y se dibuja el tablero con las N reinas, si N es
# inferior a MAX_BOARD_PRINT.
# ================================================================

def print_board(state):

    n = len(state)
    print(f"\nVector solución: {state}\n") #Se imprime el vector solución,
    # en el que cada posición representa la columna de cada reina, mientras
    # que el valor asociado a esa ubicación refleja la fila a la que pertenece
    # la pieza.
    
    if n > MAX_BOARD_PRINT: #Si el número de reinas es superior a un límite (inicialmente
                            # definido como 50), no se genera la visualización del tablero, porque, dadas las
                            # dimensiones, no se logran distinguir adecuadamente las reinas en la cuadrícula.
        return

    from matplotlib.patches import Rectangle #Importa la clase Rectangle desde matplotlib.patches,
                                             # para poder generar las casillas del tablero de ajedrez.

    # Tamaño adaptable de la figura
    fig_size = min(12, max(6, n / 5)) #Se produce una figura con tamaño adaptable, que crece si se incrementa
                                      # el número de reinas, pero se mantiene en el rango entre 6 y 12.
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size)) #Se genera el marco de la figura y el espacio de la cuadrícula (ax).

    #Dibujar tablero
    for row in range(n):
        for col in range(n):

            if (row + col) % 2 == 0:
                color = "wheat"
            else:
                color = "saddlebrown"

            square = Rectangle(
                (col, row),
                1,
                1,
                facecolor=color
            )

            ax.add_patch(square)

    #Tamaño adaptable de las reinas (para evitar que, en tableros muy grandes, con cuadrados muy pequeños,
    # se ubiquen fichas con tamaños desporporcionados)
    queen_size = max(4, min(40, 300 // n))

    for col, row in enumerate(state):
        ax.text(
            col + 0.5,
            row + 0.5,
            "♛",
            fontsize=queen_size,
            ha="center",
            va="center",
            color="black"
        )

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    #Etiquetas adaptables (de modo que no se sobrepongan en tableros muy grandes)
    if n <= 20:
        step = 1
    elif n <= 50:
        step = 5
    else:
        step = 10

    ticks = list(range(0, n, step))

    ax.set_xticks(
        [i + 0.5 for i in ticks],
        labels=ticks
    )

    ax.set_yticks(
        [i + 0.5 for i in ticks],
        labels=ticks
    )

    ax.set_xlabel("Columnas")
    ax.set_ylabel("Filas")
    ax.set_title(f"N-Reinas — N={n} — Costo: {cost(state)}")

    plt.tight_layout()
    plt.show()

# ================================================================
# GENERACIÓN DE GRÁFICA DE CONVERGENCIA

# Parametro(s): history: serie de costos generados por las
# soluciones en cada iteración.
# n: número de reinas.

# Retorno(s): ninguno (únicamente se presenta por pantalla).

# Se genera una gráfica con la secuencia de costos de las
# soluciones resultantes.
# ================================================================

def plot_convergence(history, n):
    
    plt.figure()
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.xlabel("Iteración GRASP")
    plt.ylabel("Mejor costo encontrado (ataques diagonales)")
    plt.title(f"Convergencia de GRASP (N={n})")
    plt.grid(True)
    plt.savefig(PLOT_FILE)
    print(f"Gráfica guardada en: {PLOT_FILE}\n")
    plt.show()


# ================================================================
# EJECUCIÓN PRINCIPAL

# Parametro(s): ninguno.
# Retorno(s): ninguno (exclusivamente se presenta por pantalla)

# Se condensan todas las funciones previamente definidas para
# generar el flujo del algoritmo GRASP.
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="GRASP para el problema de las N-Reinas")
    parser.add_argument("n", nargs="?", type=int, default=N,
                        help=f"tamaño del tablero (por defecto {N})")
    args = parser.parse_args()

    if args.n < 1:
        parser.error("N debe ser un entero positivo")

    if SEED is not None:
        random.seed(SEED)

    print(f"\nEjecutando GRASP para N={args.n} con {GRASP_ITERATIONS} iteraciones "
          f"y alpha={ALPHA}...")

    best_state, best_cost, history, elapsed = grasp(args.n, ALPHA, GRASP_ITERATIONS)

    print_board(best_state)
    print(f"Costo final (ataques diagonales): {best_cost}")
    print(f"Tiempo de ejecución: {elapsed:.4f} segundos")
    print(f"Iteraciones realizadas: {len(history)}")

    plot_convergence(history, args.n)


if __name__ == "__main__":
    main()