"""Garabato — app de Streamlit para aprender inglés dibujando.

La persona ve una palabra en inglés, la dibuja y la IA intenta reconocerla.
El dibujo se convierte en imagen con la MISMA función usada para entrenar (dibujos.py).
"""
import json
import random
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras

from dibujos import dibujar

CARPETA = Path(__file__).parent / "modelo"

st.set_page_config(page_title="Garabato", page_icon="✏️", layout="centered")


@st.cache_resource
def cargar():
    modelo = keras.models.load_model(CARPETA / "modelo_dibujos.keras")
    palabras = json.load(open(CARPETA / "palabras.json", encoding="utf-8"))
    config = json.load(open(CARPETA / "config.json"))
    return modelo, palabras, config


modelo, PALABRAS, CONFIG = cargar()
CLASES = list(PALABRAS)
UMBRAL = float(CONFIG.get("umbral", 0.5))


def trazos_del_lienzo(json_data):
    """Saca los trazos del lienzo como [[xs], [ys]] (mismo formato que Quick Draw)."""
    trazos = []
    for obj in (json_data or {}).get("objects", []):
        if obj.get("type", "").lower() != "path":
            continue
        xs, ys = [], []
        for cmd in obj.get("path", []):
            if len(cmd) >= 3:            # M x y | L x y | Q cx cy x y -> el punto final son los 2 últimos
                xs.append(float(cmd[-2]))
                ys.append(float(cmd[-1]))
        if xs:
            trazos.append([xs, ys])
    return trazos


def predecir(trazos):
    img = dibujar(trazos)
    prob = modelo.predict(img[None, ..., None], verbose=0)[0]
    return img, prob


# ---------- estado del juego ----------
if "palabra" not in st.session_state:
    st.session_state.palabra = random.choice(CLASES)
    st.session_state.puntos = 0
    st.session_state.intentos = 0
    st.session_state.lienzo = 0
    st.session_state.resuelta = False


def nueva_palabra():
    opciones = [c for c in CLASES if c != st.session_state.palabra]
    st.session_state.palabra = random.choice(opciones)
    st.session_state.lienzo += 1          # cambia la key -> lienzo limpio
    st.session_state.resuelta = False


# ---------- interfaz ----------
st.title("✏️ Garabato")
st.caption("Aprende inglés dibujando. No tienes que dibujar bien: ¡un garabato basta!")

modo = st.radio("Modo", ["🎯 Juego", "🖌️ Libre"], horizontal=True, label_visibility="collapsed")

if modo == "🎯 Juego":
    p = st.session_state.palabra
    st.markdown(f"## Draw: **{p}**")
    st.caption(f"En español: {PALABRAS[p]}")
else:
    st.markdown("## Dibuja lo que quieras y la IA intenta adivinar")
    with st.expander("Palabras que conoce la IA"):
        st.write(", ".join(f"**{c}** ({PALABRAS[c]})" for c in CLASES))

col1, col2 = st.columns([3, 2])
with col1:
    lienzo = st_canvas(
        stroke_width=6, stroke_color="#111111", background_color="#FFFFFF",
        height=320, width=320, drawing_mode="freedraw",
        key=f"lienzo_{st.session_state.lienzo}",
    )

with col2:
    marcador = st.empty()
    adivinar = st.button("🤔 ¿Qué dibujé?", type="primary", use_container_width=True)
    if modo == "🎯 Juego":
        st.button("➡️ Otra palabra", on_click=nueva_palabra, use_container_width=True)
    else:
        st.button("🧽 Borrar", on_click=lambda: st.session_state.update(lienzo=st.session_state.lienzo + 1),
                  use_container_width=True)

if adivinar:
    trazos = trazos_del_lienzo(lienzo.json_data)
    if not trazos:
        st.info("Primero dibuja algo en el recuadro blanco.")
    else:
        img, prob = predecir(trazos)
        orden = np.argsort(prob)[::-1]
        top, conf = CLASES[orden[0]], float(prob[orden[0]])

        if modo == "🎯 Juego" and not st.session_state.resuelta:
            st.session_state.intentos += 1

        if conf < UMBRAL:
            st.warning(f"🤷 No estoy segura de qué es… ¿lo intentas otra vez? (mi mejor idea: **{top}** con {conf:.0%})")
        elif modo == "🎯 Juego":
            objetivo = st.session_state.palabra
            if top == objetivo:
                if not st.session_state.resuelta:
                    st.session_state.puntos += 1
                    st.session_state.resuelta = True
                st.success(f"🎉 ¡Sí! Es un **{top}** ({PALABRAS[top]}).")
                st.balloons()
            else:
                st.error(f"Mmm, a mí me parece un **{top}** ({PALABRAS[top]}). ¿Lo intentas otra vez?")
        else:
            st.success(f"Creo que es un **{top}** ({PALABRAS[top]}) — {conf:.0%}")

        st.markdown("**Mis 3 mejores ideas:**")
        for i in orden[:3]:
            st.progress(float(prob[i]), text=f"{CLASES[i]} ({PALABRAS[CLASES[i]]}): {prob[i]:.0%}")

        with st.expander("👀 Lo que ve la IA (28×28 píxeles)"):
            st.image(img, width=140, clamp=True)
            st.caption("Tu dibujo se centra, se ajusta de tamaño y se reduce a 28×28, igual que los dibujos con los que se entrenó.")

if modo == "🎯 Juego":
    marcador.metric("Puntos", f"{st.session_state.puntos} / {st.session_state.intentos}")

with st.expander("ℹ️ Sobre este proyecto"):
    st.markdown(f"""
- **Modelo:** {CONFIG.get('modelo', 'CNN')}, entrenado con dibujos de *Quick, Draw!* (Google, CC BY 4.0).
- **Conoce {len(CLASES)} palabras.** Si dibujas otra cosa, igual intentará adivinar una de estas.
- Si la confianza es menor a {UMBRAL:.0%}, responde *"no estoy segura"* en vez de adivinar.
- **Limitaciones:** se entrenó con dibujos de personas de muchos países, la mayoría adultos y con tiempo límite; puede fallar con dibujos muy detallados o de niños pequeños.
""")
