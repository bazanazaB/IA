# 🔍 A* PATHFINDER - MANHATTAN HEURISTIC

## Descripción

Implementación interactiva del algoritmo **A* (A-Star)** con heurística **Manhattan** usando Pygame. Permite visualizar el proceso de búsqueda de caminos óptimos en una cuadrícula.

## 📋 Características Principales

### Algoritmo A*
- ✅ **Heurística Manhattan**: `h(n) = |x_n - x_fin| + |y_n - y_fin|`
- ✅ **Óptimo**: Encuentra el camino más corto
- ✅ **Lista Abierta**: Nodos candidatos (azul)
- ✅ **Lista Cerrada**: Nodos expandidos (púrpura)
- ✅ **Costos A***: g(n), h(n), f(n) = g(n) + h(n)

### Visualización Pygame
- 🎨 **Colores intuitivos**:
  - Verde: Nodo inicio
  - Rojo: Nodo objetivo
  - Azul: Lista abierta (candidatos)
  - Púrpura: Lista cerrada (expandidos)
  - Amarillo: Camino final

- 🖱️ **Interfaz interactiva**:
  - Click izquierdo: Colocar inicio → fin → paredes
  - Click derecho: Borrar
  - ENTER: Ejecutar búsqueda
  - R: Reset

### Salida en Consola
- Tabla con **lista cerrada completa** en orden de expansión
- Estadísticas: g(n), h(n), f(n), nodo padre
- **Camino final** con coordenadas

## 🚀 Cómo Usar

### 1. Requisitos
```bash
python -m pip install pygame
```

### 2. Ejecutar
```bash
python Algoritmo.py
```

### 3. Interacción

1. **Coloca el INICIO** (click izquierdo)
   - Aparecerá en verde
2. **Coloca el FIN** (click izquierdo nuevamente)
   - Aparecerá en rojo
3. **Coloca PAREDES** (click izquierdo adicionales)
   - Gris oscuro
4. **Inicia la búsqueda** (presiona ENTER)
   - El algoritmo se ejecutará
   - Visualizarás los nodos expandidos
   - La consola mostrará resultados

### 4. Controles
| Tecla | Acción |
|-------|--------|
| IZQ (Click) | Inicio → Fin → Paredes |
| DERECHA (Click) | Borrar celda |
| ENTER | Ejecutar A* |
| R | Reset |

## 📊 Flujo del Algoritmo A*

```
┌─────────────────────────────────────────────────────────┐
│ 1. INICIALIZAR                                          │
│    - g(inicio) = 0                                      │
│    - h(inicio) = Manhattan(inicio, fin)                 │
│    - f(inicio) = g + h                                  │
│    - Agregar inicio a LISTA ABIERTA                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 2. MIENTRAS lista_abierta NO vacía:                     │
│    a) Extraer nodo con menor f(n)                       │
│    b) Mover a LISTA CERRADA                             │
│    c) ¿Es objetivo? → RECONSTRUIR CAMINO               │
│    d) PARA cada vecino:                                 │
│       - Calcular g = g_padre + 1                        │
│       - Calcular h = Manhattan(vecino, fin)             │
│       - Calcular f = g + h                              │
│       - SI es mejor: actualizar                         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 3. RESULTADO                                            │
│    - Imprimir LISTA CERRADA en consola                  │
│    - Mostrar CAMINO FINAL                               │
│    - Visualizar en grid                                 │
└─────────────────────────────────────────────────────────┘
```

## 📈 Ejemplo de Salida

```
======================================================================
✓ ¡CAMINO ENCONTRADO!
======================================================================

Nodos expandidos (lista cerrada): 42
Longitud del camino: 15
Costo total: 14

LISTA CERRADA (Nodos expandidos en orden):
----------------------------------------------------------------------
  # POS         g(n)  h(n)   f(n)     PADRE
----------------------------------------------------------------------
  1 (7,7)         0    10    10    INICIO
  2 (6,7)         1     9    10    (7,7)
  3 (8,7)         1     9    10    (7,7)
  4 (7,6)         1    10    11    (7,7)
  5 (7,8)         1    10    11    (7,7)
...
 42 (1,1)        14     0    14    (2,1)
----------------------------------------------------------------------

CAMINO FINAL (15 nodos):
(7,7) → (6,7) → (5,7) → (4,7) → (3,7) → (2,7) → (1,7) → (1,6) 
→ (1,5) → (1,4) → (1,3) → (1,2) → (1,1)
======================================================================
```

## 🔧 Estructura del Código

### Clase `Nodo`
```python
class Nodo:
    g: float        # Costo desde inicio
    h: float        # Heurística Manhattan
    f: float        # g + h
    padre: Nodo     # Padre en el árbol de búsqueda
    en_abierta: bool
    en_cerrada: bool
```

### Clase `Grid`
```python
class Grid:
    nodos: List[List[Nodo]]  # Matriz de nodos
    ancho_celda: int         # Tamaño en píxeles
    offset_x, offset_y: int  # Desplazamiento en pantalla
```

### Función `a_estrella()`
```python
def a_estrella(grid, inicio, fin, callback_status=None):
    # Retorna:
    # - lista_cerrada: nodos expandidos
    # - camino: secuencia del camino final
```

## 🎯 Requisitos Implementados

✅ Algoritmo A* con heurística Manhattan  
✅ Lista abierta y cerrada con visualización  
✅ Costos g(n), h(n), f(n) correctamente calculados  
✅ Cuadrícula con obstáculos  
✅ Interfaz Pygame completa  
✅ Click para colocar inicio, fin, paredes  
✅ Impresión de lista cerrada en consola  
✅ Reconstrucción del camino final  
✅ Código limpio con clases  
✅ Totalmente funcional y listo para usar  

## 📝 Notas

- El algoritmo garantiza **optimalidad** (camino más corto)
- La heurística Manhattan **admite 4-vecindad** (no diagonales)
- Los nodos se expanden en orden de menor f(n)
- La tabla en consola muestra el **orden de expansión exacto**

## 🐛 Troubleshooting

Si pygame no funciona:
```bash
py -3.11 -m pip install pygame --force-reinstall
```

Si la ventana no abre:
- Verifica resolución de pantalla
- Intenta cambiar ANCHO/ALTO en líneas 19-20

## ©️ Licencia

Código educativo. Úsalo libremente. 🎓
