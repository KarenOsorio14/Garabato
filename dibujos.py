"""Funciones compartidas: se usan IGUAL en el entrenamiento y en la app.

Un dibujo es una lista de trazos; cada trazo es [xs, ys] (coordenadas de los puntos).
"""
import numpy as np
from PIL import Image, ImageDraw

LIENZO = 256      # se dibuja primero en grande y luego se reduce
MARGEN = 24       # espacio libre alrededor del dibujo
GROSOR = 14       # grosor del trazo en el lienzo grande
LADO = 28         # tamaño final de la imagen (28x28)
MAX_PUNTOS = 128  # largo máximo de la secuencia para la LSTM


def normalizar_trazos(trazos):
    """Lleva el dibujo a la esquina (mínimo 0) y escala el lado mayor a 255,
    igual que el formato 'simplified' de Quick Draw."""
    trazos = [t for t in trazos if len(t[0]) > 0]
    xs = np.concatenate([np.asarray(t[0], dtype=float) for t in trazos])
    ys = np.concatenate([np.asarray(t[1], dtype=float) for t in trazos])
    minx, miny = xs.min(), ys.min()
    escala = 255.0 / max(xs.max() - minx, ys.max() - miny, 1.0)
    return [[(np.asarray(t[0], float) - minx) * escala,
             (np.asarray(t[1], float) - miny) * escala] for t in trazos]


def dibujar(trazos, lado=LADO):
    """Convierte los trazos en una imagen lado x lado (fondo negro, trazo blanco),
    con el dibujo centrado."""
    t = normalizar_trazos(trazos)
    ancho = max(x.max() for x, _ in t)
    alto = max(y.max() for _, y in t)
    util = LIENZO - 2 * MARGEN
    s = util / 255.0
    dx = MARGEN + (util - ancho * s) / 2
    dy = MARGEN + (util - alto * s) / 2

    img = Image.new("L", (LIENZO, LIENZO), 0)
    d = ImageDraw.Draw(img)
    r = GROSOR / 2
    for x, y in t:
        puntos = list(zip(x * s + dx, y * s + dy))
        if len(puntos) == 1:
            px, py = puntos[0]
            d.ellipse([px - r, py - r, px + r, py + r], fill=255)
        else:
            d.line(puntos, fill=255, width=GROSOR, joint="curve")
            for px, py in (puntos[0], puntos[-1]):   # puntas redondeadas
                d.ellipse([px - r, py - r, px + r, py + r], fill=255)
    return np.asarray(img.resize((lado, lado), Image.LANCZOS), dtype=np.uint8)


def a_secuencia(trazos, max_puntos=MAX_PUNTOS):
    """Convierte los trazos en una secuencia para la LSTM.
    Cada paso: [dx, dy, fin_de_trazo, 1]. El último 1 marca que el paso existe
    (los pasos de relleno quedan en 0 y la LSTM los ignora con Masking)."""
    t = normalizar_trazos(trazos)
    puntos = []
    for x, y in t:
        n = len(x)
        for i in range(n):
            puntos.append([x[i] / 255.0, y[i] / 255.0, 1.0 if i == n - 1 else 0.0])
    p = np.array(puntos, dtype=np.float32)
    seq = p.copy()
    seq[1:, :2] = p[1:, :2] - p[:-1, :2]          # desplazamientos entre puntos
    largo = len(seq)
    seq = seq[:max_puntos]
    salida = np.zeros((max_puntos, 4), dtype=np.float32)
    salida[:len(seq), :3] = seq
    salida[:len(seq), 3] = 1.0
    return salida, largo
