"""
╔═══════════════════════════════════════════════════════════════════╗
║                     A* PATHFINDING ALGORITHM                      ║
║                      Heurística: Manhattan                         ║
╚═══════════════════════════════════════════════════════════════════╝

EXPLICACIÓN DE COLORES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

● VERDE: Punto de INICIO
  → Donde comienza la búsqueda del algoritmo A*

● ROJO: Punto OBJETIVO / META
  → A dónde quieres llegar, lo que busca encontrar A*

■ GRIS/ROJO: Paredes u OBSTÁCULOS
  → No se pueden cruzar, bloquean el camino

■ AZUL: Nodos en la LISTA ABIERTA
  → Nodos candidatos a explorar próximamente
  → El algoritmo los revisará en el siguiente paso
  → Representa la "frontera" de la búsqueda

■ PÚRPURA: Nodos en la LISTA CERRADA
  → Nodos ya explorados completamente
  → El algoritmo ya pasó por aquí
  → No se volverán a visitar

■ AMARILLO: CAMINO FINAL ÓPTIMO
  → La ruta más corta encontrada desde Inicio hasta Objetivo
  → Es el resultado de la búsqueda de A*

FLUJO DEL ALGORITMO A*:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Inicializar nodo inicio con g=0 y h=Manhattan(inicio, fin)
2. Agregar inicio a lista abierta
3. MIENTRAS lista_abierta no esté vacía:
   a) Extraer nodo con menor f(n) = g(n) + h(n) de ABIERTA
   b) Mover a CERRADA (colorear de púrpura)
   c) SI es objetivo: RECONSTRUIR CAMINO (colorear de amarillo)
   d) PARA cada vecino no explorado:
      - Calcular g = g_padre + costo_movimiento
      - Calcular h = Manhattan(vecino, fin)
      - Calcular f = g + h
      - SI es mejor que antes: agregar a ABIERTA (colorear de azul)
4. IMPRIMIR lista cerrada en orden de expansión
"""

import pygame
import math
from queue import PriorityQueue
import sys

# ==================== CONFIGURACIÓN ====================

pygame.init()

# Pantalla
ANCHO = 1000
ALTO = 800
VENTANA = pygame.display.set_mode((ANCHO, ALTO), pygame.RESIZABLE)
pygame.display.set_caption("🔍 A* PATHFINDER - MANHATTAN HEURISTIC")

# Fuentes
FUENTE_TITULO = pygame.font.Font(None, 32)
FUENTE_INFO = pygame.font.Font(None, 18)
FUENTE_PEQUEÑA = pygame.font.Font(None, 14)

# Paleta de colores
COLORES = {
    'fondo': (25, 25, 40),
    'grid': (50, 50, 70),
    'pared': (40, 40, 60),
    'pared_borde': (150, 60, 60),
    'inicio': (100, 255, 100),     # Verde (START)
    'fin': (255, 100, 100),        # Rojo (GOAL)
    'abierta': (100, 150, 255),    # Azul (OPEN LIST)
    'cerrada': (180, 100, 255),    # Púrpura (CLOSED LIST)
    'camino': (255, 255, 100),     # Amarillo (FINAL PATH)
    'panel': (15, 15, 30),
    'borde_panel': (100, 150, 255),
    'texto': (200, 200, 255),
    'texto_exito': (100, 255, 100),
    'texto_error': (255, 100, 100)
}

# ==================== CLASE NODO ====================

class Nodo:
    """
    Representa una celda en la cuadrícula.
    Mantiene información de costos A* y estado de búsqueda.
    """
    def __init__(self, fila, col, ancho_celda):
        self.fila = fila
        self.col = col
        self.x = 0
        self.y = 0
        self.ancho = ancho_celda
        
        # Tipo de celda
        self.es_pared_flag = False
        self.es_inicio_flag = False
        self.es_fin_flag = False
        
        # Costos A*
        self.g = float('inf')      # Costo desde inicio
        self.h = 0                 # Heurística Manhattan
        self.f = float('inf')      # g + h
        self.padre = None
        
        # Estado en búsqueda
        self.en_abierta = False    # En lista abierta
        self.en_cerrada = False    # En lista cerrada
        self.en_camino = False     # Parte del camino final
    
    def establecer_pared(self):
        """Marca la celda como obstáculo"""
        self.es_pared_flag = True
    
    def establecer_inicio(self):
        """Marca como nodo inicial"""
        self.es_inicio_flag = True
    
    def establecer_fin(self):
        """Marca como nodo objetivo"""
        self.es_fin_flag = True
    
    def resetear(self):
        """Resetea estado para nueva búsqueda"""
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.padre = None
        self.en_abierta = False
        self.en_cerrada = False
        self.en_camino = False
    
    def calcular_manhattan(self, nodo_fin):
        """
        Calcula heurística Manhattan: distancia Manhattan al objetivo.
        h(n) = |x_n - x_fin| + |y_n - y_fin|
        """
        self.h = abs(self.fila - nodo_fin.fila) + abs(self.col - nodo_fin.col)
    
    def obtener_color(self):
        """Retorna color para renderización según estado"""
        if self.es_pared_flag:
            return COLORES['pared']
        elif self.es_inicio_flag:
            return COLORES['inicio']
        elif self.es_fin_flag:
            return COLORES['fin']
        elif getattr(self, 'en_camino', False):  # Flag especial para nodos en camino final
            return COLORES['camino']
        elif self.en_cerrada:
            return COLORES['cerrada']
        elif self.en_abierta:
            return COLORES['abierta']
        else:
            return COLORES['fondo']
    
    def renderizar(self, ventana):
        """Dibuja la celda en la ventana"""
        color = self.obtener_color()
        margin = 1
        
        # Rectángulo principal
        pygame.draw.rect(ventana, color,
                        (self.x + margin, self.y + margin,
                         self.ancho - margin*2, self.ancho - margin*2))
        
        # Borde para paredes
        if self.es_pared_flag:
            pygame.draw.rect(ventana, COLORES['pared_borde'],
                            (self.x + margin, self.y + margin,
                             self.ancho - margin*2, self.ancho - margin*2), 2)
    
    def __lt__(self, otro):
        """Comparador para PriorityQueue (menor f primero)"""
        return self.f < otro.f


# ==================== CLASE GRID ====================

class Grid:
    """Gestiona la cuadrícula y su renderización"""
    def __init__(self, filas, cols, ancho_disponible, alto_disponible):
        self.filas = filas
        self.cols = cols
        self.ancho_celda = min(ancho_disponible // filas, alto_disponible // cols)
        
        # Crear nodos
        self.nodos = []
        for i in range(filas):
            fila_nodos = []
            for j in range(cols):
                nodo = Nodo(i, j, self.ancho_celda)
                fila_nodos.append(nodo)
            self.nodos.append(fila_nodos)
        
        # Calcular offset para centrar
        ancho_total = self.filas * self.ancho_celda
        alto_total = self.cols * self.ancho_celda
        self.offset_x = (ancho_disponible - ancho_total) // 2
        self.offset_y = 100  # Bajo el header
        
        # Actualizar posiciones
        self._actualizar_posiciones()
    
    def _actualizar_posiciones(self):
        """Actualiza coordenadas x, y de cada nodo"""
        for i in range(self.filas):
            for j in range(self.cols):
                nodo = self.nodos[i][j]
                nodo.x = self.offset_x + j * self.ancho_celda
                nodo.y = self.offset_y + i * self.ancho_celda
    
    def obtener_nodo(self, fila, col):
        """Acceso seguro a nodo"""
        if 0 <= fila < self.filas and 0 <= col < self.cols:
            return self.nodos[fila][col]
        return None
    
    def convertir_click_a_celda(self, pos):
        """Convierte coordenadas de ratón a fila, col"""
        x, y = pos
        x -= self.offset_x
        y -= self.offset_y
        
        if x < 0 or y < 0:
            return None, None
        
        col = x // self.ancho_celda
        fila = y // self.ancho_celda
        
        if fila >= self.filas or col >= self.cols:
            return None, None
        
        return fila, col
    
    def obtener_vecinos(self, nodo):
        """
        Retorna lista de vecinos adyacentes (8-vecindad):
        Arriba, Abajo, Izquierda, Derecha + 4 diagonales
        """
        vecinos = []
        # Movimientos cardinales (costo 1) + diagonales (costo √2 ≈ 1.414)
        movimientos = [
            (-1, 0, 1.0),   # Arriba
            (1, 0, 1.0),    # Abajo
            (0, -1, 1.0),   # Izquierda
            (0, 1, 1.0),    # Derecha
            (-1, -1, 1.414),  # Diagonal arriba-izquierda
            (-1, 1, 1.414),   # Diagonal arriba-derecha
            (1, -1, 1.414),   # Diagonal abajo-izquierda
            (1, 1, 1.414)     # Diagonal abajo-derecha
        ]
        
        for df, dc, costo in movimientos:
            nf, nc = nodo.fila + df, nodo.col + dc
            vecino = self.obtener_nodo(nf, nc)
            if vecino and not vecino.es_pared_flag:
                vecinos.append((vecino, costo))
        
        return vecinos
    
    def renderizar(self, ventana):
        """Dibuja la cuadrícula y todos los nodos"""
        for i in range(self.filas):
            for j in range(self.cols):
                self.nodos[i][j].renderizar(ventana)
        
        # Dibujar líneas del grid
        color_grid = COLORES['grid']
        for i in range(self.filas + 1):
            pygame.draw.line(ventana, color_grid,
                           (self.offset_x, self.offset_y + i * self.ancho_celda),
                           (self.offset_x + self.filas * self.ancho_celda,
                            self.offset_y + i * self.ancho_celda), 1)
        
        for j in range(self.cols + 1):
            pygame.draw.line(ventana, color_grid,
                           (self.offset_x + j * self.ancho_celda, self.offset_y),
                           (self.offset_x + j * self.ancho_celda,
                            self.offset_y + self.cols * self.ancho_celda), 1)
    
    def resetear(self):
        """Resetea todos los nodos excepto paredes, inicio y fin"""
        for i in range(self.filas):
            for j in range(self.cols):
                nodo = self.nodos[i][j]
                if not nodo.es_pared_flag and not nodo.es_inicio_flag and not nodo.es_fin_flag:
                    nodo.resetear()


# ==================== ALGORITMO A* ====================

def a_estrella(grid, inicio, fin, callback_status=None):
    """
    Implementa A* con heurística Manhattan.
    
    Parámetros:
    - grid: Grid con nodos
    - inicio: Nodo inicial
    - fin: Nodo objetivo
    - callback_status: Función para actualizar UI (opcional)
    
    Retorna:
    - lista_cerrada: Nodos expandidos en orden
    - camino: Secuencia de nodos del camino final (vacío si no existe)
    """
    
    lista_abierta = PriorityQueue()
    lista_cerrada = []
    conjunto_abierto = set()
    
    # PASO 1: Inicializar nodo inicio
    inicio.g = 0
    inicio.calcular_manhattan(fin)  # h = Manhattan distance
    inicio.f = inicio.g + inicio.h
    
    # PASO 2: Agregar inicio a lista abierta
    lista_abierta.put((inicio.f, id(inicio), inicio))
    conjunto_abierto.add(inicio)
    inicio.en_abierta = True
    
    iteracion = 0
    
    # PASO 3: MIENTRAS lista abierta no esté vacía
    while not lista_abierta.empty():
        iteracion += 1
        
        # Extraer nodo con menor f de lista abierta
        _, _, nodo_actual = lista_abierta.get()
        
        # Mover a lista cerrada
        if nodo_actual in conjunto_abierto:
            conjunto_abierto.remove(nodo_actual)
            nodo_actual.en_abierta = False
        
        nodo_actual.en_cerrada = True
        lista_cerrada.append(nodo_actual)
        
        # Callback para actualizar UI
        if callback_status:
            callback_status(
                f"Expandiendo: ({nodo_actual.fila},{nodo_actual.col}) | "
                f"g={nodo_actual.g:.0f} h={nodo_actual.h:.0f} f={nodo_actual.f:.0f} | "
                f"Abierta: {len(conjunto_abierto)} Cerrada: {len(lista_cerrada)}"
            )
            # Renderizar
            pygame.display.update()
            pygame.time.delay(50)
        
        # SI es objetivo: RECONSTRUIR CAMINO y terminar
        if nodo_actual == fin:
            camino = []
            nodo = fin
            while nodo is not None:
                camino.insert(0, nodo)
                nodo = nodo.padre
            
            # Marcar nodos del camino para visualización
            for nodo_camino in camino:
                nodo_camino.en_camino = True
            
            return lista_cerrada, camino
        
        # PASO 4: Explorar vecinos
        for vecino, costo_movimiento in grid.obtener_vecinos(nodo_actual):
            # Si ya está en cerrada, skip
            if vecino.en_cerrada:
                continue
            
            # Calcular nuevo g (costo desde inicio)
            # Movimiento cardinal = 1, diagonal = √2 ≈ 1.414
            tentativo_g = nodo_actual.g + costo_movimiento
            
            # Si ya estaba en abierta pero encontramos mejor camino
            if vecino in conjunto_abierto:
                if tentativo_g < vecino.g:
                    vecino.g = tentativo_g
                    vecino.padre = nodo_actual
                    vecino.calcular_manhattan(fin)
                    vecino.f = vecino.g + vecino.h
            else:
                # Nuevo nodo
                vecino.g = tentativo_g
                vecino.padre = nodo_actual
                vecino.calcular_manhattan(fin)
                vecino.f = vecino.g + vecino.h
                
                lista_abierta.put((vecino.f, id(vecino), vecino))
                conjunto_abierto.add(vecino)
                vecino.en_abierta = True
    
    # Si llegamos aquí: NO HAY CAMINO
    return lista_cerrada, []


# ==================== IMPRIMIR RESULTADOS ====================

def imprimir_resultados(lista_cerrada, camino):
    """Imprime en consola la lista cerrada y camino encontrado"""
    print("\n" + "="*70)
    if camino:
        # Calcular costo total del camino
        costo_total = 0.0
        for i in range(1, len(camino)):
            nodo_actual = camino[i]
            nodo_anterior = camino[i-1]
            # Calcular costo: 1 para cardinal, 1.414 para diagonal
            delta_fila = abs(nodo_actual.fila - nodo_anterior.fila)
            delta_col = abs(nodo_actual.col - nodo_anterior.col)
            if delta_fila == 1 and delta_col == 1:  # Diagonal
                costo_total += 1.414
            else:  # Cardinal
                costo_total += 1.0
        
        print(f"✓ ¡CAMINO ENCONTRADO!")
        print("="*70)
        print(f"\nNodos expandidos (lista cerrada): {len(lista_cerrada)}")
        print(f"Longitud del camino: {len(camino)}")
        print(f"Costo total del camino: {costo_total:.3f}")
    else:
        print(f"✗ NO SE ENCONTRÓ CAMINO")
        print("="*70)
        print(f"Nodos explorados: {len(lista_cerrada)}")
    
    print("\nLISTA CERRADA (Nodos expandidos en orden):")
    print("-"*70)
    print(f"{'#':>3} {'POS':>9} {'g(n)':>8} {'h(n)':>5} {'f(n)':>8} {'PADRE':>9}")
    print("-"*70)
    
    for i, nodo in enumerate(lista_cerrada, 1):
        padre_str = f"({nodo.padre.fila},{nodo.padre.col})" if nodo.padre else "INICIO"
        print(f"{i:3d} ({nodo.fila},{nodo.col}) {nodo.g:8.3f} {nodo.h:5.0f} {nodo.f:8.3f} {padre_str:>9}")
    
    print("-"*70)
    
    if camino:
        print(f"\nCAMINO FINAL ({len(camino)} nodos):")
        print("-"*70)
        for i, nodo in enumerate(camino):
            print(f"  {i}: ({nodo.fila},{nodo.col})", end="")
            if i < len(camino) - 1:
                # Indicar si es movimiento diagonal
                nodo_siguiente = camino[i+1]
                delta_fila = abs(nodo_siguiente.fila - nodo.fila)
                delta_col = abs(nodo_siguiente.col - nodo.col)
                if delta_fila == 1 and delta_col == 1:
                    print(" ↖↗↙↘ ", end="")
                else:
                    print(" → ", end="")
            if (i + 1) % 8 == 0:
                print()
        print("\n" + "="*70 + "\n")


# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """Loop principal de Pygame"""
    
    # Crear grid
    FILAS, COLS = 11, 11
    grid = Grid(FILAS, COLS, ANCHO - 340, ALTO - 120)
    
    # Estado
    inicio = None
    fin = None
    info_actual = "Haz click IZQ: Inicio (verde) → Fin (rojo) → Paredes | DERECHA para borrar | ENTER para buscar"
    camino_encontrado = False
    lista_cerrada = []
    camino = []
    mouse_presionado = False  # Rastrear si el ratón está presionado
    
    # Loop
    reloj = pygame.time.Clock()
    corriendo = True
    
    while corriendo:
        reloj.tick(60)
        
        # ========== MANEJO DE RATÓN CONTINUO ==========
        # Dibujar paredes mientras el botón está presionado
        if pygame.mouse.get_pressed()[0]:  # Botón izquierdo presionado
            pos = pygame.mouse.get_pos()
            fila, col = grid.convertir_click_a_celda(pos)
            
            if fila is not None:
                nodo = grid.obtener_nodo(fila, col)
                if nodo and not nodo.es_inicio_flag and not nodo.es_fin_flag:
                    nodo.establecer_pared()
        
        if pygame.mouse.get_pressed()[2]:  # Botón derecho presionado (borrar)
            pos = pygame.mouse.get_pos()
            fila, col = grid.convertir_click_a_celda(pos)
            
            if fila is not None:
                nodo = grid.obtener_nodo(fila, col)
                if nodo:
                    nodo.es_pared_flag = False
                    nodo.es_inicio_flag = False
                    nodo.es_fin_flag = False
                    if nodo == inicio:
                        inicio = None
                    elif nodo == fin:
                        fin = None
        
        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False
            
            # Click izquierdo: colocar inicio, fin
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = pygame.mouse.get_pos()
                fila, col = grid.convertir_click_a_celda(pos)
                
                if fila is not None:
                    nodo = grid.obtener_nodo(fila, col)
                    
                    if nodo:
                        if not inicio and not nodo.es_pared_flag:
                            inicio = nodo
                            nodo.establecer_inicio()
                            info_actual = "✓ Inicio colocado. Ahora coloca FIN (click izquierdo)"
                        elif not fin and nodo != inicio and not nodo.es_pared_flag:
                            fin = nodo
                            nodo.establecer_fin()
                            info_actual = "✓ Fin colocado. Arrastra para dibujar paredes | ENTER para buscar"
            
            # Teclas
            if event.type == pygame.KEYDOWN:
                # ENTER: ejecutar A*
                if event.key == pygame.K_RETURN and inicio and fin:
                    grid.resetear()
                    info_actual = "🔍 Ejecutando A*..."
                    pygame.display.update()
                    
                    def callback(msg):
                        nonlocal info_actual
                        info_actual = msg
                    
                    lista_cerrada, camino = a_estrella(grid, inicio, fin, callback)
                    
                    if camino:
                        camino_encontrado = True
                        info_actual = f"✓ CAMINO ENCONTRADO: {len(camino)} nodos | Expansiones: {len(lista_cerrada)}"
                        imprimir_resultados(lista_cerrada, camino)
                    else:
                        info_actual = f"✗ NO HAY CAMINO | Nodos explorados: {len(lista_cerrada)}"
                        imprimir_resultados(lista_cerrada, camino)
                
                # R: reset
                elif event.key == pygame.K_r:
                    inicio = None
                    fin = None
                    camino_encontrado = False
                    lista_cerrada = []
                    camino = []
                    grid = Grid(FILAS, COLS, ANCHO - 340, ALTO - 120)
                    info_actual = "Grid reseteado. Coloca inicio (IZQ click)"
        
        # Renderizar
        VENTANA.fill(COLORES['fondo'])
        
        # Header
        titulo = FUENTE_TITULO.render("A* PATHFINDER - Manhattan Heuristic", True, COLORES['texto'])
        VENTANA.blit(titulo, (20, 10))
        
        # Info
        info_texto = FUENTE_INFO.render(info_actual, True, COLORES['texto'])
        VENTANA.blit(info_texto, (20, 50))
        
        # Grid
        grid.renderizar(VENTANA)
        
        # Panel lateral
        panel_x = ANCHO - 320
        pygame.draw.rect(VENTANA, COLORES['panel'], (panel_x, 0, 320, ALTO))
        pygame.draw.line(VENTANA, COLORES['borde_panel'], (panel_x, 0), (panel_x, ALTO), 3)
        
        # Título del panel
        titulo_panel = FUENTE_PEQUEÑA.render("=== LEYENDA A* ===", True, COLORES['texto_exito'])
        VENTANA.blit(titulo_panel, (panel_x + 20, 15))
        
        # Leyenda detallada con explicaciones
        legend_y = 50
        leyenda_items = [
            ("● VERDE = Punto Inicio", COLORES['inicio']),
            ("  (donde comienza la búsqueda)", COLORES['texto']),
            
            ("● ROJO = Punto Objetivo/Meta", COLORES['fin']),
            ("  (a dónde quieres llegar)", COLORES['texto']),
            
            ("■ GRIS = Paredes/Obstáculos", COLORES['pared_borde']),
            ("  (no se pueden cruzar)", COLORES['texto']),
            
            ("■ AZUL = Nodos en ABIERTA", COLORES['abierta']),
            ("  (por explorar próximamente)", COLORES['texto']),
            
            ("■ PÚRPURA = Nodos en CERRADA", COLORES['cerrada']),
            ("  (ya fueron explorados)", COLORES['texto']),
            
            ("■ AMARILLO = Camino Final", COLORES['camino']),
            ("  (la ruta más óptima encontrada)", COLORES['texto']),
        ]
        
        for texto, color in leyenda_items:
            surf = FUENTE_PEQUEÑA.render(texto, True, color)
            VENTANA.blit(surf, (panel_x + 20, legend_y))
            legend_y += 22
        
        # Separador
        legend_y += 5
        pygame.draw.line(VENTANA, COLORES['borde_panel'], (panel_x + 20, legend_y), (panel_x + 300, legend_y), 1)
        
        # Instrucciones
        legend_y += 15
        titulo_instr = FUENTE_PEQUEÑA.render("=== CONTROLES ===", True, COLORES['texto_exito'])
        VENTANA.blit(titulo_instr, (panel_x + 20, legend_y))
        legend_y += 28
        
        instrucciones = [
            "1. Click → Inicio",
            "2. Click → Objetivo",
            "3. Arrastra → Paredes",
            "4. Der.Arr. → Borrar",
            "5. ENTER → Buscar",
            "6. R → Limpiar todo",
        ]
        
        for instr in instrucciones:
            surf = FUENTE_PEQUEÑA.render(instr, True, COLORES['texto'])
            VENTANA.blit(surf, (panel_x + 20, legend_y))
            legend_y += 25
        
        # Estadísticas
        if lista_cerrada:
            legend_y += 10
            stats_y = legend_y
            
            pygame.draw.line(VENTANA, COLORES['borde_panel'], (panel_x + 20, stats_y), (panel_x + 300, stats_y), 1)
            stats_y += 15
            
            color_stats = COLORES['texto_exito'] if camino else COLORES['texto_error']
            
            stats = [
                f"Expansiones: {len(lista_cerrada)}",
                f"Camino: {len(camino)} nodos",
            ]
            
            for stat in stats:
                surf = FUENTE_PEQUEÑA.render(stat, True, color_stats)
                VENTANA.blit(surf, (panel_x + 20, stats_y))
                stats_y += 25
        
        pygame.display.update()
    
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
