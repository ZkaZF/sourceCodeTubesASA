"""
============================================================
SIMULASI PERBANDINGAN ALGORITMA PATHFINDING
============================================================
Evaluasi Adaptabilitas pada Pencarian Rute Pesan-Antar Makanan
Terhadap Penutupan Portal Jalan Dinamis di Kawasan Tembalang

Algoritma yang diuji:
  [Uninformed]  1. BFS   (Breadth-First Search)
                2. DFS   (Depth-First Search)
                3. UCS   (Uniform Cost Search)
  [Informed]    4. Greedy Search (Pure Greedy, tanpa backtracking)
                5. GBFS  (Greedy Best-First Search)
                6. A*    (A-Star Search)
  [Swarm]       7. ACO   (Ant Colony Optimization)

Skenario:
  Phase 1 — Routing statis pada graf normal
  Phase 2 — Rerouting setelah penutupan jalan dinamis

Matkul  : Analisis & Strategi Algoritma
Kampus  : Universitas Diponegoro
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================
import collections  # deque untuk BFS
import heapq        # Priority queue untuk UCS, GBFS, A*
import time         # High-precision timer
import random       # Probabilistic selection di ACO
import copy         # Deep copy graf untuk Phase 2
import math         # Euclidean distance


# ============================================================
# SECTION 1: GRAPH DATA — Peta Jalan Kawasan Tembalang
# ============================================================
# Koordinat node (x, y) diestimasi dari layout grafMapTembalang.jpg
# Digunakan untuk:
#   1. Menghitung bobot edge (jarak Euclidean antar node)
#   2. Menghitung heuristik SLD (Straight-Line Distance ke Goal)

NODE_COORDS = {
    'Start': (80, 850),    # Titik awal pengantaran (biru, kiri-bawah)
    'N1':    (160, 780),   # Persimpangan dekat Start
    'N2':    (350, 880),   # Persimpangan bawah-tengah
    'N3':    (200, 660),   # Persimpangan kiri-tengah bawah
    'N4':    (50, 540),    # Persimpangan paling kiri
    'N5':    (200, 560),   # Persimpangan kiri-tengah
    'N6':    (310, 700),   # Persimpangan tengah-bawah
    'N7':    (320, 620),   # Persimpangan tengah
    'N8':    (470, 580),   # Persimpangan tengah
    'N9':    (350, 460),   # Persimpangan tengah-atas
    'N10':   (380, 310),   # Persimpangan atas-tengah kiri
    'N11':   (450, 310),   # Persimpangan atas-tengah
    'N12':   (640, 440),   # Persimpangan kanan-tengah
    'N13':   (700, 340),   # Persimpangan kanan-atas
    'N14':   (850, 440),   # Persimpangan paling kanan
    'N15':   (870, 530),   # Persimpangan kanan-bawah
    'N16':   (750, 630),   # Persimpangan kanan-bawah jauh
    'N17':   (310, 60),    # Persimpangan atas-kiri
    'N18':   (420, 80),    # Persimpangan atas-tengah kiri
    'N19':   (600, 60),    # Persimpangan atas-tengah kanan
    'N20':   (800, 90),    # Persimpangan atas-kanan (dekat Goal)
    'Goal':  (940, 20),    # Titik tujuan pengantaran (hijau, kanan-atas)
}

# Daftar edge (segmen jalan) — Undirected Graph
# Bobot dihitung otomatis dari jarak Euclidean antar koordinat node
EDGES = [
    # --- Cluster Bawah-Kiri (dekat Start) ---
    ('Start', 'N1'),     # Jalan keluar dari Start ke persimpangan terdekat
    ('Start', 'N3'),     # Jalan alternatif dari Start menuju kiri-tengah
    ('N1', 'N2'),        # Jalan ke persimpangan bawah-tengah
    ('N1', 'N6'),        # Jalan ke persimpangan tengah-bawah
    ('N2', 'N8'),        # Jalan naik dari bawah ke tengah

    # --- Cluster Kiri (N3, N4, N5) ---
    ('N3', 'N4'),        # Jalan ke persimpangan paling kiri
    ('N3', 'N5'),        # Jalan pendek ke kiri-tengah
    ('N3', 'N17'),       # Jalan diagonal panjang ke atas-kiri (jalan utama)

    ('N4', 'N5'),        # Jalan penghubung kiri

    # --- Cluster Tengah (N5, N6, N7, N8, N9) ---
    ('N5', 'N7'),        # Jalan dari kiri ke tengah
    ('N5', 'N9'),        # Jalan diagonal ke tengah-atas
    ('N6', 'N7'),        # Jalan pendek tengah-bawah
    ('N6', 'N8'),        # Jalan dari tengah-bawah ke tengah
    ('N7', 'N8'),        # Jalan penghubung tengah
    ('N7', 'N9'),        # Jalan naik dari tengah

    # --- Cluster Atas-Tengah (N9, N10, N11) ---
    ('N8', 'N12'),       # Jalan dari tengah ke kanan-tengah
    ('N9', 'N10'),       # Jalan naik ke atas-tengah
    ('N10', 'N11'),      # Jalan pendek antar persimpangan atas
    ('N10', 'N17'),      # Jalan ke atas-kiri

    # --- Cluster Kanan (N12, N13, N14, N15, N16) ---
    ('N11', 'N12'),      # Jalan dari atas-tengah ke kanan
    ('N11', 'N18'),      # Jalan naik ke atas
    ('N12', 'N13'),      # Jalan naik di sisi kanan
    ('N12', 'N16'),      # Jalan turun ke kanan-bawah
    ('N13', 'N14'),      # Jalan ke persimpangan paling kanan
    ('N13', 'N19'),      # Jalan diagonal ke atas-tengah kanan
    ('N14', 'N15'),      # Jalan turun sisi kanan
    ('N14', 'N16'),      # Jalan ke kanan-bawah
    ('N14', 'N20'),      # Jalan naik ke atas-kanan

    ('N15', 'N16'),      # Jalan penghubung kanan-bawah

    # --- Cluster Atas (N17, N18, N19, N20, Goal) ---
    ('N17', 'N18'),      # Jalan atas kiri ke tengah
    ('N18', 'N19'),      # Jalan atas tengah
    ('N19', 'N20'),      # Jalan atas ke kanan
    ('N20', 'Goal'),     # Jalan terakhir menuju tujuan
]
# Total: 33 edge


# ============================================================
# SECTION 1b: GRAPH BUILDER FUNCTIONS
# ============================================================

def euclidean_distance(coord1, coord2):
    """Menghitung jarak Euclidean antara dua titik koordinat (x1,y1) dan (x2,y2)."""
    return round(math.sqrt((coord1[0] - coord2[0]) ** 2 +
                           (coord1[1] - coord2[1]) ** 2), 1)


def build_graph(node_coords, edges):
    """
    Membangun adjacency list (undirected weighted graph) dari koordinat dan daftar edge.
    Bobot setiap edge = jarak Euclidean antara kedua node.

    Args:
        node_coords (dict): {node_name: (x, y), ...}
        edges (list): [(node_a, node_b), ...]

    Returns:
        dict: {node: {neighbor: weight, ...}, ...}
    """
    graph = {node: {} for node in node_coords}
    for n1, n2 in edges:
        weight = euclidean_distance(node_coords[n1], node_coords[n2])
        graph[n1][n2] = weight  # Arah n1 → n2
        graph[n2][n1] = weight  # Arah n2 → n1 (undirected)
    return graph


def build_heuristic(node_coords, goal='Goal'):
    """
    Menghitung heuristik SLD (Straight-Line Distance) dari setiap node ke Goal.
    SLD selalu admissible karena jarak garis lurus ≤ jarak jalan sebenarnya.

    Args:
        node_coords (dict): {node_name: (x, y), ...}
        goal (str): Nama node tujuan

    Returns:
        dict: {node: sld_to_goal, ...}
    """
    goal_coord = node_coords[goal]
    return {node: euclidean_distance(node_coords[node], goal_coord)
            for node in node_coords}


def calculate_path_cost(graph, path):
    """
    Menghitung total biaya (jarak) dari sebuah path.

    Args:
        graph (dict): Adjacency list
        path (list): Urutan node [start, n1, n2, ..., goal]

    Returns:
        float: Total cost. Mengembalikan float('inf') jika edge tidak valid.
    """
    if not path or len(path) < 2:
        return 0.0 if path else float('inf')

    total = 0.0
    for i in range(len(path) - 1):
        edge_cost = graph[path[i]].get(path[i + 1], float('inf'))
        if edge_cost == float('inf'):
            return float('inf')
        total += edge_cost
    return round(total, 1)


# ============================================================
# SECTION 2: UNINFORMED SEARCH ALGORITHMS
# ============================================================

def bfs(graph, start, goal, heuristic=None):
    """
    Breadth-First Search (BFS).
    ─────────────────────────────────────────────────────────
    Strategi   : Eksplorasi level demi level (FIFO queue)
    Optimalitas: Optimal berdasarkan JUMLAH HOP (bukan bobot)
    Struktur   : collections.deque

    Args:
        graph (dict): Adjacency list
        start (str): Node awal
        goal (str): Node tujuan
        heuristic: Tidak digunakan (untuk konsistensi interface)

    Returns:
        tuple: (path, cost, nodes_explored)
            - path (list|None): Urutan node dari start ke goal
            - cost (float): Total jarak path
            - nodes_explored (int): Jumlah node yang dieksplorasi
    """
    # Queue FIFO berisi tuple (current_node, path_so_far)
    queue = collections.deque([(start, [start])])
    visited = {start}
    nodes_explored = 0

    while queue:
        current, path = queue.popleft()
        nodes_explored += 1

        # Goal test: apakah node saat ini adalah tujuan?
        if current == goal:
            cost = calculate_path_cost(graph, path)
            return path, cost, nodes_explored

        # Ekspansi tetangga (diurutkan alfabet untuk determinisme)
        for neighbor in sorted(graph[current].keys()):
            if neighbor not in visited and graph[current][neighbor] != float('inf'):
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    # Tidak ditemukan path ke goal
    return None, float('inf'), nodes_explored


def dfs(graph, start, goal, heuristic=None):
    """
    Depth-First Search (DFS).
    ─────────────────────────────────────────────────────────
    Strategi   : Eksplorasi sedalam mungkin sebelum backtrack (LIFO stack)
    Optimalitas: TIDAK optimal
    Struktur   : Python list (stack) dengan .pop()

    Args & Returns: Sama dengan BFS.
    """
    # Stack LIFO berisi tuple (current_node, path_so_far)
    stack = [(start, [start])]
    visited = set()
    nodes_explored = 0

    while stack:
        current, path = stack.pop()

        # Skip jika sudah pernah dikunjungi
        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1

        # Goal test
        if current == goal:
            cost = calculate_path_cost(graph, path)
            return path, cost, nodes_explored

        # Push tetangga ke stack (reverse sorted agar urutan alfabet ter-pop duluan)
        for neighbor in sorted(graph[current].keys(), reverse=True):
            if neighbor not in visited and graph[current][neighbor] != float('inf'):
                stack.append((neighbor, path + [neighbor]))

    return None, float('inf'), nodes_explored


def ucs(graph, start, goal, heuristic=None):
    """
    Uniform Cost Search (UCS) — setara dengan Dijkstra's Algorithm.
    ─────────────────────────────────────────────────────────
    Strategi   : Selalu ekspansi node dengan total cost g(n) TERKECIL
    Optimalitas: OPTIMAL pada graf berbobot positif
    Struktur   : heapq (min-priority queue)

    Priority queue entry: (g_cost, counter, node, path)
    Counter digunakan untuk tie-breaking agar heapq stabil.

    Args & Returns: Sama dengan BFS.
    """
    counter = 0
    pq = [(0, counter, start, [start])]  # (g_cost, tiebreak, node, path)
    visited = set()
    nodes_explored = 0

    while pq:
        cost, _, current, path = heapq.heappop(pq)

        # Skip jika sudah pernah dikunjungi (closed set)
        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1

        # Goal test
        if current == goal:
            return path, round(cost, 1), nodes_explored

        # Ekspansi tetangga
        for neighbor, weight in graph[current].items():
            if neighbor not in visited and weight != float('inf'):
                new_cost = cost + weight
                counter += 1
                heapq.heappush(pq, (new_cost, counter, neighbor,
                                    path + [neighbor]))

    return None, float('inf'), nodes_explored


# ============================================================
# SECTION 3: INFORMED (HEURISTIC) SEARCH ALGORITHMS
# ============================================================

def greedy_search(graph, start, goal, heuristic):
    """
    Greedy Search — Pure Greedy TANPA Backtracking.
    ─────────────────────────────────────────────────────────
    Strategi   : Pada setiap langkah, LANGSUNG pilih tetangga dengan h(n)
                 terkecil. Tidak menggunakan priority queue.
    Optimalitas: TIDAK optimal
    Kelemahan  : Bisa GAGAL (return None) jika terjebak di dead-end,
                 karena tidak ada mekanisme backtracking.

    Perbedaan dengan GBFS:
    - Greedy: Pilih 1 tetangga terbaik → langsung maju → stuck = gagal
    - GBFS:   Gunakan priority queue → bisa backtrack ke node lain

    Args:
        graph (dict): Adjacency list
        start (str): Node awal
        goal (str): Node tujuan
        heuristic (dict): {node: SLD_to_goal, ...}

    Returns:
        tuple: (path, cost, nodes_explored)
    """
    current = start
    path = [start]
    visited = {start}
    nodes_explored = 1

    while current != goal:
        # Kumpulkan tetangga yang belum dikunjungi
        unvisited_neighbors = [
            (heuristic[n], n)
            for n in graph[current]
            if n not in visited and graph[current][n] != float('inf')
        ]

        if not unvisited_neighbors:
            # DEAD-END: Tidak ada tetangga yang bisa dikunjungi
            # Pure Greedy tidak punya backtracking → GAGAL
            return None, float('inf'), nodes_explored

        # Pilih tetangga dengan h(n) terkecil (paling dekat ke Goal secara SLD)
        _, best_neighbor = min(unvisited_neighbors)

        visited.add(best_neighbor)
        path.append(best_neighbor)
        current = best_neighbor
        nodes_explored += 1

    cost = calculate_path_cost(graph, path)
    return path, cost, nodes_explored


def gbfs(graph, start, goal, heuristic):
    """
    Greedy Best-First Search (GBFS).
    ─────────────────────────────────────────────────────────
    Strategi   : Ekspansi node dengan h(n) terkecil menggunakan priority queue.
                 Memiliki backtracking melalui priority queue.
    Optimalitas: TIDAK optimal (hanya mempertimbangkan h(n), bukan g(n))
    Struktur   : heapq dengan priority = h(n)

    Perbedaan dengan A*:
    - GBFS: priority = h(n) saja
    - A*:   priority = g(n) + h(n)

    Args & Returns: Sama dengan greedy_search.
    """
    counter = 0
    pq = [(heuristic[start], counter, start, [start])]  # (h, tiebreak, node, path)
    visited = set()
    nodes_explored = 0

    while pq:
        h_val, _, current, path = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1

        # Goal test
        if current == goal:
            cost = calculate_path_cost(graph, path)
            return path, cost, nodes_explored

        # Ekspansi tetangga
        for neighbor, weight in graph[current].items():
            if neighbor not in visited and weight != float('inf'):
                counter += 1
                heapq.heappush(pq, (heuristic[neighbor], counter, neighbor,
                                    path + [neighbor]))

    return None, float('inf'), nodes_explored


def a_star(graph, start, goal, heuristic):
    """
    A* (A-Star) Search Algorithm.
    ─────────────────────────────────────────────────────────
    Strategi   : Ekspansi node dengan f(n) = g(n) + h(n) terkecil
                 - g(n): Actual cost dari start ke node n
                 - h(n): Estimated cost dari n ke goal (SLD)
    Optimalitas: OPTIMAL jika heuristik admissible (h(n) ≤ h*(n))
                 SLD selalu admissible → A* PASTI optimal
    Struktur   : heapq dengan priority = f(n) = g(n) + h(n)

    Optimasi: Menyimpan g_costs terbaik untuk setiap node agar tidak
    memasukkan entry yang lebih buruk ke priority queue.

    Args & Returns: Sama dengan greedy_search.
    """
    counter = 0
    g_start = 0
    f_start = g_start + heuristic[start]
    # Entry: (f_value, counter, g_value, node, path)
    pq = [(f_start, counter, g_start, start, [start])]
    visited = set()
    g_costs = {start: 0}  # Best known g(n) untuk setiap node
    nodes_explored = 0

    while pq:
        f_val, _, g_val, current, path = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        nodes_explored += 1

        # Goal test
        if current == goal:
            return path, round(g_val, 1), nodes_explored

        # Ekspansi tetangga
        for neighbor, weight in graph[current].items():
            if neighbor not in visited and weight != float('inf'):
                new_g = g_val + weight

                # Hanya masukkan ke PQ jika ini cost terbaik yang pernah ditemukan
                if neighbor not in g_costs or new_g < g_costs[neighbor]:
                    g_costs[neighbor] = new_g
                    f_new = new_g + heuristic[neighbor]
                    counter += 1
                    heapq.heappush(pq, (f_new, counter, new_g, neighbor,
                                        path + [neighbor]))

    return None, float('inf'), nodes_explored


# ============================================================
# SECTION 4: ANT COLONY OPTIMIZATION (ACO)
# ============================================================

def aco(graph, start, goal, heuristic=None,
        num_ants=20, num_iterations=50,
        alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100.0):
    """
    Ant Colony Optimization (ACO) — Ant System variant.
    ─────────────────────────────────────────────────────────
    Algoritma swarm intelligence terinspirasi perilaku semut dalam
    menemukan jalur terpendek menggunakan pheromone trail.

    Mekanisme:
    1. Setiap semut membangun path dari Start ke Goal secara probabilistik
    2. Probabilitas transisi: P(i→j) = [τ_ij^α × η_ij^β] / Σ[τ_ik^α × η_ik^β]
       - τ_ij: Pheromone pada edge (i,j) — pengalaman kolektif
       - η_ij: Visibility = 1/distance(i,j) — informasi lokal
    3. Setelah semua semut selesai, pheromone di-update:
       - Evaporasi: τ_ij = (1 - ρ) × τ_ij
       - Deposit:   τ_ij += Q / L_k (untuk edge di path semut k)

    Parameters:
        graph (dict): Adjacency list
        start (str): Node awal
        goal (str): Node tujuan
        heuristic: Tidak digunakan (ACO punya visibility sendiri)
        num_ants (int): Jumlah semut per iterasi (default: 20)
        num_iterations (int): Jumlah iterasi (default: 50)
        alpha (float): Bobot pengaruh pheromone (default: 1.0)
        beta (float): Bobot pengaruh visibility (default: 2.0)
        evaporation_rate (float): Laju evaporasi pheromone ρ (default: 0.5)
        Q (float): Konstanta deposit pheromone (default: 100.0)

    Returns:
        tuple: (best_path, best_cost, total_nodes_explored)
    """
    random.seed(42)  # Seed untuk reproducibility hasil

    # --- Inisialisasi Pheromone ---
    # Setiap edge mendapat pheromone awal τ₀ = 1.0
    pheromone = {}
    for node in graph:
        for neighbor in graph[node]:
            edge_key = tuple(sorted([node, neighbor]))
            if edge_key not in pheromone:
                pheromone[edge_key] = 1.0

    def get_pheromone(n1, n2):
        """Ambil level pheromone pada edge (n1, n2)."""
        return pheromone[tuple(sorted([n1, n2]))]

    def set_pheromone(n1, n2, value):
        """Set level pheromone pada edge (n1, n2)."""
        pheromone[tuple(sorted([n1, n2]))] = value

    # --- Tracking hasil terbaik ---
    best_path = None
    best_cost = float('inf')
    all_explored_nodes = set()  # Semua unique node yang pernah dikunjungi

    # ============================================
    # LOOP ITERASI UTAMA ACO
    # ============================================
    for iteration in range(num_iterations):
        iteration_paths = []  # Path yang berhasil ditemukan iterasi ini
        iteration_costs = []  # Cost masing-masing path

        # --- Setiap semut membangun path ---
        for ant in range(num_ants):
            current = start
            ant_path = [start]
            ant_visited = {start}
            stuck = False

            while current != goal:
                # Kumpulkan tetangga yang valid (belum dikunjungi, edge terbuka)
                candidates = []
                attractiveness_values = []

                for neighbor, weight in graph[current].items():
                    if (neighbor not in ant_visited and
                            weight != float('inf') and weight > 0):
                        tau = get_pheromone(current, neighbor)  # Pheromone level
                        eta = 1.0 / weight                      # Visibility (1/jarak)

                        # Hitung attractiveness: τ^α × η^β
                        attractiveness = (tau ** alpha) * (eta ** beta)
                        candidates.append(neighbor)
                        attractiveness_values.append(attractiveness)

                if not candidates:
                    stuck = True  # Semut terjebak di dead-end
                    break

                # --- Roulette Wheel Selection ---
                # Normalisasi probabilitas
                total_attract = sum(attractiveness_values)
                probabilities = [a / total_attract for a in attractiveness_values]

                # Pilih berdasarkan probabilitas kumulatif
                r = random.random()
                cumulative = 0.0
                chosen = candidates[-1]  # Default: kandidat terakhir
                for i, prob in enumerate(probabilities):
                    cumulative += prob
                    if r <= cumulative:
                        chosen = candidates[i]
                        break

                # Semut bergerak ke node terpilih
                ant_visited.add(chosen)
                ant_path.append(chosen)
                current = chosen

            # Catat semua node yang pernah dikunjungi
            all_explored_nodes.update(ant_visited)

            # Jika semut berhasil sampai Goal, simpan path
            if not stuck:
                cost = calculate_path_cost(graph, ant_path)
                iteration_paths.append(ant_path)
                iteration_costs.append(cost)

                # Update best path global
                if cost < best_cost:
                    best_cost = cost
                    best_path = ant_path[:]

        # --- Evaporasi Pheromone (semua edge) ---
        # τ_ij = (1 - ρ) × τ_ij
        for edge_key in pheromone:
            pheromone[edge_key] *= (1 - evaporation_rate)

        # --- Deposit Pheromone (hanya edge di path yang berhasil) ---
        # Δτ_ij = Q / L_k  (untuk setiap edge di path semut k)
        for ant_path, ant_cost in zip(iteration_paths, iteration_costs):
            if ant_cost > 0 and ant_cost != float('inf'):
                deposit = Q / ant_cost
                for i in range(len(ant_path) - 1):
                    edge_key = tuple(sorted([ant_path[i], ant_path[i + 1]]))
                    pheromone[edge_key] += deposit

    # --- Return hasil terbaik ---
    nodes_explored = len(all_explored_nodes)

    if best_path:
        return best_path, round(best_cost, 1), nodes_explored
    else:
        return None, float('inf'), nodes_explored


# ============================================================
# SECTION 5: SIMULATION ENGINE
# ============================================================

def run_algorithm(name, func, graph, start, goal, heuristic):
    """
    Wrapper untuk menjalankan satu algoritma dengan high-precision timer.

    Menggunakan time.perf_counter_ns() untuk presisi nanosecond.

    Args:
        name (str): Nama algoritma
        func (callable): Fungsi algoritma
        graph (dict): Adjacency list
        start (str): Node awal
        goal (str): Node tujuan
        heuristic (dict): Dictionary heuristik SLD

    Returns:
        dict: {
            'name':     str,         # Nama algoritma
            'status':   str,         # 'Success' atau 'Fail'
            'path':     list|None,   # Path yang ditemukan
            'cost':     float,       # Total jarak
            'explored': int,         # Jumlah node dieksplorasi
            'time_ms':  float        # Waktu eksekusi dalam milidetik
        }
    """
    # Timer mulai (nanosecond precision)
    t_start = time.perf_counter_ns()

    path, cost, explored = func(graph, start, goal, heuristic)

    # Timer selesai
    t_end = time.perf_counter_ns()

    elapsed_ms = (t_end - t_start) / 1_000_000  # nanosecond → millisecond

    return {
        'name':     name,
        'status':   'Success' if path is not None else 'Fail',
        'path':     path,
        'cost':     cost,
        'explored': explored,
        'time_ms':  elapsed_ms,
    }


def format_path(path, max_len=35):
    """
    Format path list menjadi string ringkas untuk tampilan tabel.
    Jika terlalu panjang, tampilkan awal...akhir.
    """
    if path is None:
        return "- (tidak ditemukan)"

    path_str = " -> ".join(path)
    if len(path_str) > max_len:
        # Ringkas: tampilkan 2 node awal + ... + 2 node akhir
        nodes = path
        if len(nodes) > 4:
            path_str = (" -> ".join(nodes[:2]) + " -> ... -> " +
                        " -> ".join(nodes[-2:]))
    return path_str


def print_header():
    """Cetak header program."""
    print()
    print("=" * 90)
    print("   SIMULASI PERBANDINGAN ALGORITMA PATHFINDING")
    print("   Evaluasi Adaptabilitas pada Pencarian Rute Pesan-Antar Makanan")
    print("   Terhadap Penutupan Portal Jalan Dinamis di Kawasan Tembalang")
    print("=" * 90)


def print_graph_info(graph, heuristic, start, goal):
    """Cetak informasi graf."""
    num_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
    print(f"\n   Graf         : {len(graph)} node, {num_edges} edge (undirected weighted)")
    print(f"   Start        : {start}")
    print(f"   Goal         : {goal}")
    print(f"   SLD(S -> G)  : {heuristic[start]:.1f} unit")


def print_results_table(results, phase_name):
    """
    Cetak tabel perbandingan hasil algoritma dengan format box-drawing.

    Args:
        results (list): List of dicts dari run_algorithm()
        phase_name (str): Nama fase untuk header tabel
    """
    # Lebar kolom
    w_name = 14
    w_stat = 9
    w_path = 38
    w_cost = 11
    w_expl = 10
    w_time = 13
    w_total = w_name + w_stat + w_path + w_cost + w_expl + w_time + 5  # +5 for separators

    print()
    # Top border
    print("+" + "-" * w_name + "+" + "-" * w_stat + "+" + "-" * w_path +
          "+" + "-" * w_cost + "+" + "-" * w_expl + "+" + "-" * w_time + "+")

    # Phase header
    header_text = f"  {phase_name}"
    print("|" + header_text.ljust(w_total) + "|")

    # Column header separator
    print("+" + "-" * w_name + "+" + "-" * w_stat + "+" + "-" * w_path +
          "+" + "-" * w_cost + "+" + "-" * w_expl + "+" + "-" * w_time + "+")

    # Column headers
    print("|" + " Algorithm".ljust(w_name) +
          "|" + " Status".ljust(w_stat) +
          "|" + " Path".ljust(w_path) +
          "|" + " Cost".ljust(w_cost) +
          "|" + " Explored".ljust(w_expl) +
          "|" + " Time (ms)".ljust(w_time) + "|")

    # Header-data separator
    print("+" + "-" * w_name + "+" + "-" * w_stat + "+" + "-" * w_path +
          "+" + "-" * w_cost + "+" + "-" * w_expl + "+" + "-" * w_time + "+")

    # Data rows
    for r in results:
        name = f" {r['name']}".ljust(w_name)
        status = f" {r['status']}".ljust(w_stat)
        path_str = f" {format_path(r['path'], w_path - 3)}".ljust(w_path)

        if r['cost'] == float('inf'):
            cost = " inf".ljust(w_cost)
        else:
            cost = f" {r['cost']:.1f}".ljust(w_cost)

        explored = f" {r['explored']}".ljust(w_expl)
        time_ms = f" {r['time_ms']:.4f}".ljust(w_time)

        print("|" + name + "|" + status + "|" + path_str +
              "|" + cost + "|" + explored + "|" + time_ms + "|")

    # Bottom border
    print("+" + "-" * w_name + "+" + "-" * w_stat + "+" + "-" * w_path +
          "+" + "-" * w_cost + "+" + "-" * w_expl + "+" + "-" * w_time + "+")


def print_path_details(results):
    """Cetak detail path lengkap untuk setiap algoritma."""
    print("\n   Detail Path Lengkap:")
    print("   " + "-" * 70)
    for r in results:
        if r['path']:
            full_path = " -> ".join(r['path'])
            print(f"   {r['name']:12s}: {full_path}")
            print(f"   {'':12s}  (Cost: {r['cost']:.1f}, Jumlah node: {len(r['path'])})")
        else:
            print(f"   {r['name']:12s}: GAGAL menemukan path")
        print()


def print_comparison(results_p1, results_p2):
    """
    Cetak tabel perbandingan adaptabilitas antara Phase 1 dan Phase 2.

    Menampilkan perubahan cost, waktu, dan status adaptabilitas.
    """
    print("\n" + "=" * 90)
    print("   ANALISIS ADAPTABILITAS: Phase 1 vs Phase 2")
    print("=" * 90)

    # Header
    header = (f"   {'Algorithm':<12} | {'Cost P1':>9} | {'Cost P2':>9} | "
              f"{'Delta':>9} | {'Time P1':>10} | {'Time P2':>10} | {'Adaptif?':>8}")
    print(header)
    print("   " + "-" * 84)

    for r1, r2 in zip(results_p1, results_p2):
        c1 = r1['cost']
        c2 = r2['cost']

        c1_str = f"{c1:.1f}" if c1 != float('inf') else "inf"
        c2_str = f"{c2:.1f}" if c2 != float('inf') else "inf"

        # Hitung selisih cost
        if c1 != float('inf') and c2 != float('inf'):
            delta = c2 - c1
            delta_str = f"{delta:+.1f}"
            adaptif = "Ya"
        elif c2 == float('inf'):
            delta_str = "N/A"
            adaptif = "Tidak"
        else:
            delta_str = "N/A"
            adaptif = "N/A"

        row = (f"   {r1['name']:<12} | {c1_str:>9} | {c2_str:>9} | "
               f"{delta_str:>9} | {r1['time_ms']:>9.4f}ms | {r2['time_ms']:>9.4f}ms | "
               f"{adaptif:>8}")
        print(row)

    print("   " + "-" * 84)

    # Ringkasan
    p2_success = sum(1 for r in results_p2 if r['status'] == 'Success')
    p2_fail = sum(1 for r in results_p2 if r['status'] == 'Fail')

    print(f"\n   Ringkasan Adaptabilitas:")
    print(f"   - Algoritma berhasil reroute : {p2_success}/{len(results_p2)}")
    print(f"   - Algoritma gagal reroute    : {p2_fail}/{len(results_p2)}")

    # Cari algoritma dengan cost terbaik di Phase 2
    successful_p2 = [r for r in results_p2 if r['status'] == 'Success']
    if successful_p2:
        best = min(successful_p2, key=lambda r: r['cost'])
        fastest = min(successful_p2, key=lambda r: r['time_ms'])
        print(f"   - Cost optimal (Phase 2)     : {best['name']} ({best['cost']:.1f})")
        print(f"   - Tercepat (Phase 2)         : {fastest['name']} ({fastest['time_ms']:.4f} ms)")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():
    """
    Fungsi utama simulasi.

    Flow:
    1. Build graf dan heuristik
    2. Phase 1: Jalankan 7 algoritma pada graf normal
    3. Dynamic Event: Tutup edge kritis di path optimal
    4. Phase 2: Jalankan 7 algoritma pada graf termodifikasi
    5. Analisis perbandingan
    """
    print_header()

    # === Build Graph & Heuristic ===
    graph = build_graph(NODE_COORDS, EDGES)
    heuristic = build_heuristic(NODE_COORDS, 'Goal')

    start_node = 'Start'
    goal_node = 'Goal'

    print_graph_info(graph, heuristic, start_node, goal_node)

    # Tampilkan heuristik SLD semua node (untuk referensi)
    print("\n   Heuristik SLD ke Goal:")
    print("   " + "-" * 50)
    for node in sorted(heuristic.keys()):
        bar = "#" * int(heuristic[node] / 30)
        print(f"   {node:>6s}: {heuristic[node]:>7.1f}  {bar}")

    # === Daftar Algoritma ===
    algorithms = [
        ("BFS",       bfs),
        ("DFS",       dfs),
        ("UCS",       ucs),
        ("Greedy",    greedy_search),
        ("GBFS",      gbfs),
        ("A*",        a_star),
        ("ACO",       aco),
    ]

    # ========================================================
    # PHASE 1: STATIC ROUTING (Graf Normal)
    # ========================================================
    print("\n\n" + "=" * 90)
    print("   PHASE 1: STATIC ROUTING (Graf Normal)")
    print("=" * 90)

    results_phase1 = []
    for name, func in algorithms:
        result = run_algorithm(name, func, graph, start_node, goal_node, heuristic)
        results_phase1.append(result)

    print_results_table(results_phase1, "PHASE 1: STATIC ROUTING (Graf Normal)")
    print_path_details(results_phase1)

    # ========================================================
    # DYNAMIC EVENT: Penutupan Jalan
    # ========================================================
    # Ambil path optimal dari A* untuk menentukan edge kritis
    a_star_result = next(r for r in results_phase1 if r['name'] == 'A*')

    if a_star_result['path'] and len(a_star_result['path']) >= 3:
        optimal_path = a_star_result['path']

        # Pilih edge di TENGAH path optimal untuk ditutup
        # Ini memastikan dampak maksimal pada rerouting
        mid_idx = len(optimal_path) // 2
        closed_n1 = optimal_path[mid_idx]
        closed_n2 = optimal_path[mid_idx + 1]
        original_weight = graph[closed_n1][closed_n2]

        print("\n" + "=" * 90)
        print("   DYNAMIC EVENT: PENUTUPAN PORTAL JALAN")
        print("=" * 90)
        print(f"   Skenario : Portal jalan ditutup secara tiba-tiba")
        print(f"   Ruas     : {closed_n1} <-> {closed_n2}")
        print(f"   Bobot    : {original_weight:.1f} -> inf (tidak bisa dilalui)")
        print(f"   Alasan   : Edge ini berada di tengah path optimal A*")
        print(f"   Path A*  : {' -> '.join(optimal_path)}")

        # Deep copy graf dan tutup jalan (set bobot = infinity)
        graph_modified = copy.deepcopy(graph)
        graph_modified[closed_n1][closed_n2] = float('inf')
        graph_modified[closed_n2][closed_n1] = float('inf')

        # ====================================================
        # PHASE 2: REROUTING (Setelah Penutupan Jalan)
        # ====================================================
        print("\n\n" + "=" * 90)
        print("   PHASE 2: REROUTING (Setelah Penutupan Jalan)")
        print("=" * 90)

        results_phase2 = []
        for name, func in algorithms:
            result = run_algorithm(name, func, graph_modified,
                                   start_node, goal_node, heuristic)
            results_phase2.append(result)

        print_results_table(results_phase2,
                            "PHASE 2: REROUTING (Setelah Penutupan Jalan)")
        print_path_details(results_phase2)

        # ====================================================
        # ANALISIS PERBANDINGAN
        # ====================================================
        print_comparison(results_phase1, results_phase2)

    else:
        print("\n   [!] A* gagal menemukan path di Phase 1.")
        print("       Tidak dapat menentukan edge untuk ditutup.")

    # === Selesai ===
    print("\n" + "=" * 90)
    print("   SIMULASI SELESAI")
    print("=" * 90)
    print()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
