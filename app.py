"""Garabato — aprende inglés dibujando.

Recorrido: Inicio -> Elige un tema -> Dibuja -> Resultado -> Final de la ronda.
La IA (una CNN) reconoce el dibujo. El dibujo se convierte en imagen con la MISMA
función usada para entrenar (dibujos.py).
"""
import json
import random
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras

from dibujos import dibujar

RAIZ = Path(__file__).parent
MODELO = RAIZ / "modelo"
IMG = RAIZ / "imagenes"

st.set_page_config(page_title="Garabato", page_icon=str(IMG / "gato_hola.png"), layout="centered")

CRAYONES = {"tinta": "#2B2B2B", "rojo": "#D64545", "naranja": "#E8892B", "amarillo": "#F2C230", "verde": "#3B8C5A",
            "azul": "#2F5FA8", "morado": "#7B5EA7", "rosa": "#E07A9E", "cafe": "#8B5A3C"}
IMAGEN_TEMA = {
    "Frutas y comida": "tema_frutas_y_comida.png", "Animales": "tema_animales.png",
    "Casa y objetos": "tema_casa_y_objetos.png", "Naturaleza": "tema_naturaleza.png",
    "Transporte": "tema_transporte.png",
}

# ---------------------------------------------------------------- estilo
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Patrick+Hand&display=swap');
:root { --beige:#F6EFE2; --crema:#FFFBF4; --azul:#2F5FA8; --cielo:#CFE0F5; --tinta:#2B2B2B; --durazno:#F2B8A2; --gris:#6B6A66; }

.stApp { background: var(--beige); }
header[data-testid="stHeader"], footer, #MainMenu { visibility: hidden; height: 0; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 720px; }

h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--tinta) !important; }
p, li, label, button, div[data-testid="stMarkdownContainer"], div[data-testid="stCaptionContainer"],
div[data-testid="stExpander"] summary p { font-family: 'Patrick Hand', cursive !important; }
span[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
div[data-testid="stMarkdownContainer"] p { font-size: 1.25rem; color: var(--tinta); }

.centro { text-align: center; }
.titulo { font-family: 'Fraunces', serif !important; font-weight: 600; font-size: 4rem !important; color: var(--tinta); margin: 0; line-height: 1; }
.sub { font-family: 'Patrick Hand', cursive; font-size: 1.5rem !important; color: var(--gris); margin: .4rem 0 0; }
.seccion { font-family: 'Fraunces', serif !important; font-weight: 500; font-size: 2rem !important; color: var(--tinta); margin: 0 0 .2rem; }
.miga { font-family: 'Patrick Hand', cursive; font-size: 1.15rem !important; color: var(--gris); margin: 0; }

/* botones tipo cuaderno */
.stButton > button {
  background: var(--crema); color: var(--tinta); border: 2.5px solid var(--tinta); border-radius: 16px;
  box-shadow: 3px 3px 0 var(--tinta); font-size: 1.35rem; padding: .3rem 1.1rem; transition: transform .08s, box-shadow .08s;
}
.stButton > button:hover { transform: translate(1px,1px); box-shadow: 2px 2px 0 var(--tinta); color: var(--tinta); border-color: var(--tinta); }
.stButton > button:focus:not(:active) { color: var(--tinta); border-color: var(--tinta); }
.stButton > button[kind="primary"], .stButton > button[kind="primary"]:hover, .stButton > button[kind="primary"]:focus:not(:active) {
  background: var(--azul); color: var(--crema); }
.stButton button p { font-size: 1.35rem !important; color: inherit !important; }
.stButton button[kind="primary"] p { color: var(--crema) !important; }

/* tarjeta de la palabra */
.tarjeta { background: var(--crema); border: 2.5px solid var(--tinta); border-radius: 22px; box-shadow: 4px 4px 0 var(--tinta);
           padding: .6rem 1rem .8rem; text-align: center; }
.tarjeta .pide { font-family: 'Patrick Hand', cursive; font-size: 1.25rem; color: var(--gris); margin: 0; }
.tarjeta .palabra { font-family: 'Fraunces', serif !important; font-weight: 600; font-size: 3.2rem !important; color: var(--azul); margin: 0; line-height: 1.1; }
.tarjeta .trad { font-family: 'Patrick Hand', cursive; font-size: 1.25rem; color: var(--gris); margin: 0; }

/* tarjetas de tema */
div[class*="st-key-tema_"] { background: var(--crema); border: 2.5px solid var(--tinta); border-radius: 22px;
                             box-shadow: 4px 4px 0 var(--tinta); padding: .6rem .8rem .9rem; }
div[class*="st-key-tema_"] button { width: 100% !important; box-shadow: none; border-width: 2px; }
.nombre-tema { font-family: 'Fraunces', serif !important; font-weight: 500; font-size: 1.35rem !important; text-align: center; margin: 0; color: var(--tinta); }
.cuantas { font-family: 'Patrick Hand', cursive; font-size: 1.1rem !important; text-align: center; margin: 0 0 .4rem; color: var(--gris); }

/* crayones */
div[class*="st-key-crayon_"] button { width: 38px; height: 38px; min-height: 0; padding: 0; border-radius: 50%;
                                                  box-shadow: 2px 2px 0 var(--tinta); }
.st-key-crayon_tinta button { background: #2B2B2B !important; }
.st-key-crayon_azul button { background: #2F5FA8 !important; }
.st-key-crayon_verde button { background: #3B8C5A !important; }
.st-key-crayon_naranja button { background: #E8892B !important; }
.st-key-crayon_rosa button { background: #E07A9E !important; }
.st-key-crayon_rojo button { background: #D64545 !important; }
.st-key-crayon_amarillo button { background: #F2C230 !important; }
.st-key-crayon_morado button { background: #7B5EA7 !important; }
.st-key-crayon_cafe button { background: #8B5A3C !important; }
.st-key-crayones { flex-wrap: nowrap !important; gap: 10px !important; }
.st-key-crayones > div { width: auto !important; min-width: 0 !important; flex: 0 0 auto !important; }
@media (max-width: 380px) { div[class*="st-key-crayon_"] button { width:26px !important; height:26px !important; } .st-key-crayones { gap: 6px !important; } }

/* imágenes centradas y sin botón de pantalla completa */
div[data-testid="stFullScreenFrame"] { display: flex; justify-content: center; width: 100%; }
div[data-testid="stFullScreenFrame"] > div { margin: 0 auto; }
div[data-testid="stElementToolbar"] { display: none; }
div[class*="st-key-tema_"] div[data-testid="stElementContainer"]:has(img) { width: 100% !important; }

/* lienzo */
iframe { border-radius: 18px; }
div[class*="st-key-lienzo_"] { border: 2.5px solid var(--tinta); border-radius: 20px; box-shadow: 4px 4px 0 var(--tinta);
                               background: var(--crema); padding: 6px; width: fit-content; margin: 0 auto; }

/* globos de respuesta */
.globo { border: 2.5px solid var(--tinta); border-radius: 20px; padding: .7rem 1.1rem; text-align: center;
         font-family: 'Patrick Hand', cursive; font-size: 1.6rem; color: var(--tinta); margin: .4rem 0 1rem; }
.globo.bien { background: var(--cielo); } .globo.casi { background: var(--durazno); } .globo.duda { background: var(--crema); }

/* lista final */
.fila { display: flex; justify-content: space-between; align-items: center; background: var(--crema);
        border: 2px solid var(--tinta); border-radius: 14px; padding: .35rem .9rem; margin: .35rem 0;
        font-family: 'Patrick Hand', cursive; font-size: 1.25rem; color: var(--tinta); }
.fila .marca { font-size: 1.05rem; padding: .05rem .6rem; border-radius: 10px; }
.fila .marca.bien { background: var(--cielo); } .fila .marca.pend { background: var(--durazno); }
/* Refinamiento visual global: interfaz editorial, sobria y consistente */
:root { --beige:#F7F1E7; --crema:#FFFCF7; --azul:#3568B5; --cielo:#E8EFF9; --tinta:#292824; --durazno:#F4E4D9; --gris:#77736D; }
.stApp { background: var(--beige); color: var(--tinta); }
.block-container { max-width: 900px; padding-top: 2.25rem; padding-bottom: 3rem; }
h1, h2, h3 { letter-spacing: -.025em; }
p, li, label, div[data-testid="stMarkdownContainer"] { line-height: 1.5; }
.stButton > button { background: var(--crema); color: var(--tinta); border: 1px solid #D9D0C3; border-radius: 12px;
  box-shadow: 0 1px 2px rgba(45,38,28,.05); font-size: 1.05rem; padding: .55rem 1rem; min-height: 42px;
  transition: background .15s ease, border-color .15s ease, transform .15s ease; }
.stButton > button:hover { background:#F4EDE2; border-color:#BEB19F; color:var(--tinta); transform:translateY(-1px); box-shadow:0 3px 8px rgba(45,38,28,.07); }
.stButton > button:focus:not(:active) { border-color:var(--azul); color:var(--tinta); box-shadow:0 0 0 2px rgba(53,104,181,.16); }
.stButton > button[kind="primary"], .stButton > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:focus:not(:active) { background:var(--azul); color:#fff; border:1px solid #2D5A9D; box-shadow:0 2px 5px rgba(42,82,143,.16); }
.stButton button p, .stButton button[kind="primary"] p { font-family:'Nunito',sans-serif !important; font-size:1rem !important; font-weight:700; color:inherit !important; }
.tarjeta { background:var(--crema); border:1px solid #E1D6C7; border-radius:18px; box-shadow:0 4px 14px rgba(75,57,35,.055);
  padding:1rem 1.2rem 1.1rem; }
.tarjeta .palabra { font-size:2.8rem !important; }
div[class*="st-key-tema_"] { background:var(--crema); border:1px solid #E4DACC; border-radius:18px;
  box-shadow:0 3px 12px rgba(75,57,35,.05); padding:1rem; transition:transform .15s ease, box-shadow .15s ease; }
div[class*="st-key-tema_"]:hover { transform:translateY(-2px); box-shadow:0 7px 18px rgba(75,57,35,.08); }
div[class*="st-key-tema_"] button { border-radius:10px; }
div[class*="st-key-crayon_"] button { width:30px; height:30px; border:2px solid rgba(41,40,36,.14); box-shadow:none; }
div[class*="st-key-lienzo_"] { border:1px solid #DED3C4; border-radius:16px; box-shadow:0 4px 14px rgba(75,57,35,.06); padding:7px; }
iframe { border-radius:12px; }
.globo { border:1px solid #DED3C4; border-radius:16px; padding:1rem 1.2rem; font-size:1.35rem; }
.globo.bien { background:#EEF4EA; } .globo.casi { background:#FBF0E8; } .globo.duda { background:var(--crema); }
.fila { background:var(--crema); border:1px solid #E3D9CC; border-radius:12px; padding:.65rem .9rem; font-size:1.08rem; }
.fila .marca { border-radius:8px; }
@media (max-width:640px) {
  .block-container { padding:1.25rem 1rem 2rem !important; }
  .seccion { font-size:1.7rem !important; }
  .tarjeta .palabra { font-size:2.35rem !important; }
  .fila { gap:.5rem; font-size:.98rem; flex-wrap:wrap; }
  .fila .marca { font-size:.85rem; }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- modelo
@st.cache_resource
def cargar():
    modelo = keras.models.load_model(MODELO / "modelo_dibujos.keras")
    palabras = json.load(open(MODELO / "palabras.json", encoding="utf-8"))
    config = json.load(open(MODELO / "config.json"))
    ruta = MODELO / "temas.json"
    temas = json.load(open(ruta, encoding="utf-8")) if ruta.exists() else {"Todas las palabras": list(palabras)}
    return modelo, palabras, temas, config


modelo, PALABRAS, TEMAS, CONFIG = cargar()
CLASES = list(PALABRAS)
UMBRAL = float(CONFIG.get("umbral", 0.5))


def trazos_del_lienzo(json_data):
    """Trazos del lienzo como [[xs], [ys]] (mismo formato que Quick Draw)."""
    trazos = []
    for obj in (json_data or {}).get("objects", []):
        if obj.get("type", "").lower() != "path":
            continue
        xs = [float(c[-2]) for c in obj.get("path", []) if len(c) >= 3]
        ys = [float(c[-1]) for c in obj.get("path", []) if len(c) >= 3]
        if xs:
            trazos.append([xs, ys])
    return trazos


def predecir(trazos):
    img = dibujar(trazos)
    prob = modelo.predict(img[None, ..., None], verbose=0)[0]
    return img, prob


# ---------------------------------------------------------------- estado
s = st.session_state
s.setdefault("pantalla", "inicio")
s.setdefault("crayon", "tinta")
s.setdefault("lienzo", 0)


def ir(pantalla):
    s.pantalla = pantalla


def empezar_tema(tema):
    s.tema = tema
    s.ronda = random.sample(TEMAS[tema], len(TEMAS[tema]))
    s.indice = 0
    s.aciertos = {}
    s.resultado = None
    s.lienzo += 1
    s.pantalla = "dibujar"


def siguiente():
    s.indice += 1
    s.lienzo += 1
    s.resultado = None
    s.pantalla = "final" if s.indice >= len(s.ronda) else "dibujar"


def otra_vez():
    s.lienzo += 1
    s.resultado = None
    s.pantalla = "dibujar"


def borrar():
    s.lienzo += 1


@st.cache_data
def imagen_nitida(nombre, ancho):
    # st.image reduce la imagen al ancho exacto y en celular se ve borrosa;
    # aquí se guarda a 3 veces el ancho para pantallas de alta densidad.
    import base64, io
    from PIL import Image
    im = Image.open(IMG / nombre)
    lado = min(im.width, ancho * 3)
    im = im.resize((lado, round(im.height * lado / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def imagen(nombre, ancho):
    if not (IMG / nombre).exists():
        nombre = "gato_hola.png"
    st.markdown(f'<div style="text-align:center; margin-bottom:0.9rem"><img src="data:image/png;base64,{imagen_nitida(nombre, ancho)}" '
                f'style="width:{ancho}px; max-width:100%; height:auto" alt=""></div>', unsafe_allow_html=True)


def centrado(nombre, ancho):
    imagen(nombre, ancho)  # el CSS centra todas las imágenes


# ---------------------------------------------------------------- 1. inicio
import base64


@st.cache_data
def imagen_base64(nombre):
    return base64.b64encode((IMG / nombre).read_bytes()).decode()





# Fondo: formas orgánicas en las cuatro esquinas (versión horizontal y vertical para que no se deformen).
FONDO_ESCRITORIO = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 900' preserveAspectRatio='xMidYMid slice'>
<path d='M0 0 H230 C 214 70 160 92 120 140 C 76 194 64 236 0 252 Z' fill='#F3DFC4'/>
<path d='M1440 0 H1240 C 1252 54 1300 92 1352 106 C 1398 118 1420 150 1440 164 Z' fill='#F3DFC4'/>
<path d='M0 900 V600 C 70 560 150 590 196 650 C 242 712 300 770 420 812 C 470 830 480 870 470 900 Z' fill='#F3DFC4'/>
<path d='M1440 900 V650 C 1380 610 1290 626 1230 690 C 1180 744 1160 830 1170 900 Z' fill='#F3DFC4'/></svg>"""
FONDO_MOVIL = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 390 844' preserveAspectRatio='xMidYMid slice'>
<path d='M0 0 H96 C 88 28 64 40 44 56 C 22 74 16 92 0 98 Z' fill='#F3DFC4'/>
<path d='M390 0 H320 C 326 28 350 48 390 56 Z' fill='#F3DFC4'/>
<path d='M0 844 V730 C 40 714 80 730 104 764 C 124 794 150 822 168 844 Z' fill='#F3DFC4'/>
<path d='M390 844 V748 C 356 734 316 752 296 790 C 286 810 286 830 290 844 Z' fill='#F3DFC4'/></svg>"""
CLIP = """<svg viewBox="0 0 40 80" aria-hidden="true"><path d="M14 66 V18 C14 10 26 10 26 18 V60 C26 70 8 70 8 60 V24"
 fill="none" stroke="#2B2B2B" stroke-width="3.4" stroke-linecap="round"/></svg>"""
BRILLO = """<svg viewBox="0 0 60 50" aria-hidden="true"><path d="M22 6 L38 18 M16 30 L38 30 M22 46 L38 40"
 stroke="#F7C948" stroke-width="6.5" stroke-linecap="round"/></svg>"""
SUB_AMARILLO = """<svg viewBox="0 0 300 14" preserveAspectRatio="none" aria-hidden="true">
<path d="M4 9 C 80 4, 200 3, 296 5" fill="none" stroke="#F7C948" stroke-width="5" stroke-linecap="round"/></svg>"""
SUB_AZUL = """<svg viewBox="0 0 420 30" preserveAspectRatio="none" aria-hidden="true">
<path d="M8 24 C 120 12, 260 6, 412 8" fill="none" stroke="#3B67B8" stroke-width="7" stroke-linecap="round"/></svg>"""


def pantalla_inicio():
    f_esc = base64.b64encode(FONDO_ESCRITORIO.encode()).decode()
    f_mov = base64.b64encode(FONDO_MOVIL.encode()).decode()
    gato = imagen_base64("gato_portada.png")
    # Escala de espaciado (base 8): 8 · 16 · 24 · 32 · 48 · 64
    st.markdown(f"""<style>
      @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@500;600;700;800&display=swap');
      :root {{ --e1: 8px; --e2: 16px; --e3: 24px; --e4: 32px; --e5: 48px; --e6: 64px; }}
      .stApp {{ background: #FAF1E4 url("data:image/svg+xml;base64,{f_esc}") no-repeat center / cover fixed; }}
      .block-container {{ max-width: 1180px !important; padding: clamp(32px, 9vh, 104px) var(--e5) var(--e4) !important; }}
      div[data-testid="stVerticalBlock"] {{ gap: 0 !important; }}

      /* columna izquierda: tarjeta, frase, botón y enlace en un solo eje centrado */
      .marco-hoja {{ position: relative; padding: var(--e4) var(--e2) 0 var(--e5); }}
      .hoja {{ position: relative; background-color: #FFFCF7; border: 1px solid #DED2C1; border-radius: 24px;
              background-image: linear-gradient(#F1E9DE 1px, transparent 1px), linear-gradient(90deg, #F1E9DE 1px, transparent 1px);
              background-size: 26px 26px; box-shadow: 0 7px 18px rgba(80, 61, 38, .07);
              padding: var(--e5) var(--e5) var(--e4); transform: rotate(-3deg); text-align: center; }}
      .hoja .clip {{ position: absolute; left: var(--e5); top: -30px; width: 32px; }}
      .hoja .clip svg, .brillo svg {{ width: 100%; height: auto; display: block; }}
      .hoja .ojo {{ font-family: 'Nunito', sans-serif !important; font-weight: 600; font-size: 1.75rem !important; line-height: 1.15 !important;
                   color: #2B2B2B !important; margin: 0 !important; }}
      .hoja .sub-am {{ display: block; width: 66%; height: 11px; margin: 6px auto 0; }}
      .hoja .sub-am svg, .hoja .sub-az svg {{ width: 100%; height: 100%; display: block; }}
      .hoja .heroe {{ font-family: 'Fraunces', serif !important; font-weight: 800; font-size: clamp(4rem, 6.2vw, 6.2rem) !important; white-space: nowrap;
                     line-height: 1 !important; color: #222 !important; margin: var(--e2) 0 0 !important; letter-spacing: -1px; }}
      .hoja .sub-az {{ display: block; width: 92%; height: 24px; margin: var(--e2) auto 0; }}
      .brillo {{ position: absolute; left: 0; top: 44%; width: 56px; }}

      .bajada {{ font-family: 'Nunito', sans-serif !important; font-weight: 600; font-size: 1.95rem !important; color: #333 !important;
                 text-align: center; line-height: 1.45 !important; margin: var(--e5) 0 var(--e4) !important; }}
      .bajada mark {{ background: #F7C948; border-radius: 999px; padding: 0 12px; color: #2B2B2B; font-weight: 700; }}
      .st-key-empezar, .st-key-ver_info {{ width: 100% !important; display: flex !important; justify-content: center; }}
      .st-key-empezar .stButton, .st-key-ver_info .stButton {{ display: flex; justify-content: center; }}
      .st-key-empezar button {{ width: 360px !important; min-height: 60px; background: #3568B5 !important; border: 1px solid #2D5A9D !important;
                                border-radius: 13px !important; box-shadow: 0 2px 5px rgba(42, 82, 143, .16) !important;
                                transform: none !important; }}
      .st-key-empezar button:hover {{ background: #2F5EAA !important; box-shadow: 0 4px 9px rgba(42, 82, 143, .18) !important; }}
      .st-key-empezar button p {{ font-family: 'Nunito', sans-serif !important; font-weight: 700; font-size: 1.35rem !important; color: #FFF !important; }}
      .st-key-empezar button p::after {{ content: ""; }}
      .st-key-ver_info {{ margin-top: var(--e3); }}
      .st-key-ver_info button {{ background: transparent !important; border: none !important; box-shadow: none !important;
                                 transform: none !important; min-height: 0; padding: var(--e1) 0 !important; }}
      .st-key-ver_info button p {{ font-family: 'Nunito', sans-serif !important; font-weight: 600; font-size: 1.25rem !important;
                                   color: #333 !important; text-decoration: underline; text-underline-offset: 6px; }}
      .st-key-ver_info button p::before, .st-key-ver_info button p::after {{ content: ""; display: inline-block; width: 84px; height: 2px;
                                   background: #D9C9B2; vertical-align: middle; margin: 0 var(--e4); }}

      /* columna derecha: el gato con su sombra en el piso */
      .gato-portada {{ position: relative; width: 100%; max-width: 440px; margin: 0 auto; }}
      .gato-portada img {{ position: relative; width: 100%; display: block; z-index: 1; }}
      .gato-portada .piso {{ position: absolute; left: 8%; right: 2%; bottom: -8px; height: 48px; border-radius: 50%; background: #F1DEC2; }}
      .gato-movil {{ display: none; }}

      .info {{ max-width: 640px; margin: var(--e4) auto 0; background: #FFFBF4; border: 2px solid #D8C3A4; border-radius: 22px;
              box-shadow: 0 12px 24px rgba(120, 90, 50, .12); padding: var(--e3) var(--e4); }}
      .info p {{ font-family: 'Nunito', sans-serif !important; font-size: 1.05rem !important; line-height: 1.5 !important; margin: 0 0 var(--e1) !important; }}

      @media (max-width: 1100px) {{
        .bajada {{ font-size: 1.6rem !important; }}
        .st-key-empezar button {{ width: 340px !important; min-height: 62px; }}
        .st-key-empezar button p {{ font-size: 1.55rem !important; }}
      }}
      /* celular: tarjeta, gato, frase, botón y enlace en una sola columna */
      @media (max-width: 640px) {{
        .stApp {{ background-image: url("data:image/svg+xml;base64,{f_mov}"); }}
        .block-container {{ padding: var(--e4) var(--e3) var(--e3) !important; }}
        .st-key-columna_gato {{ display: none !important; }}
        .gato-movil {{ display: block; width: 62%; max-width: 250px; margin: var(--e4) auto 0; }}
        .marco-hoja {{ padding: var(--e3) 0 0; }}
        .hoja {{ padding: var(--e4) var(--e3) var(--e3); border-radius: 22px; }}
        .hoja .clip {{ width: 24px; top: -22px; left: var(--e3); }}
        .hoja .ojo {{ font-size: 1.15rem !important; }}
        .hoja .heroe {{ font-size: clamp(3rem, 15vw, 3.9rem) !important; margin-top: var(--e1) !important; }}
        .hoja .sub-az {{ height: 16px; margin-top: var(--e1); }}
        .brillo {{ width: 30px; left: -6px; top: 40%; }}
        .marco-hoja {{ padding-left: var(--e3) !important; }}
        .bajada {{ font-size: 1.3rem !important; margin: var(--e3) 0 var(--e3) !important; }}
        .st-key-empezar .stButton {{ width: 100%; }}
        .st-key-empezar button {{ width: 100% !important; max-width: 320px; min-height: 54px; border-radius: 12px !important; }}
        .st-key-empezar button p {{ font-size: 1.2rem !important; }}
        .st-key-ver_info {{ margin-top: var(--e2); }}
        .st-key-ver_info button p {{ font-size: 1.1rem !important; }}
        .st-key-ver_info button p::before, .st-key-ver_info button p::after {{ width: 32px; margin: 0 var(--e2); }}
        .info {{ padding: var(--e2) var(--e3); }}
      }}
    </style>""", unsafe_allow_html=True)

    figura = f"""<div class="piso"></div><img src="data:image/png;base64,{gato}" alt="Gato con boina azul abrazando un lápiz">"""
    izq, der = st.columns([1.3, 1], vertical_alignment="center", gap="large")
    with izq:
        st.markdown(f"""
        <div class="marco-hoja"><span class="brillo">{BRILLO}</span>
          <div class="hoja"><span class="clip">{CLIP}</span>
            <p class="ojo">Aprende inglés dibujando</p><span class="sub-am">{SUB_AMARILLO}</span>
            <p class="heroe">Garabato</p><span class="sub-az">{SUB_AZUL}</span>
          </div>
        </div>
        <div class="gato-portada gato-movil">{figura}</div>""", unsafe_allow_html=True)
        st.markdown('<p class="bajada">No tienes que dibujar bien:<br>un <mark>garabato</mark> basta.</p>', unsafe_allow_html=True)
        st.button("Empezar", key="empezar", type="primary", on_click=ir, args=("temas",))
        if st.button("¿Qué es Garabato?", key="ver_info"):
            s.mostrar_info = not s.get("mostrar_info", False)
    with der:
        with st.container(key="columna_gato"):
            st.markdown(f'<div class="gato-portada">{figura}</div>', unsafe_allow_html=True)

    if s.get("mostrar_info"):
        st.markdown(f"""<div class="info">
          <p><b>Por qué dibujar:</b> hay evidencia de que dibujar una palabra ayuda a recordarla mejor que escribirla
          (<i>efecto del dibujo</i>, Wammes, Meade y Fernandes, 2016).</p>
          <p><b>La IA:</b> una red neuronal convolucional entrenada con dibujos de <i>Quick, Draw!</i> (Google, CC BY 4.0).
          Conoce {len(CLASES)} palabras en {len(TEMAS)} temas.</p>
          <p><b>Honestidad:</b> si su confianza es menor a {UMBRAL:.0%}, prefiere decir que no está segura. Aprendió de
          garabatos rápidos, así que con dibujos muy detallados puede fallar.</p></div>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- 2. temas
def pantalla_temas():
    st.markdown('<p class="seccion centro">Elige un tema</p>', unsafe_allow_html=True)
    st.markdown('<p class="miga centro">¿Qué quieres dibujar hoy?</p>', unsafe_allow_html=True)
    st.markdown('<p class="miga centro">Elige un tema y te mostraré palabras en inglés, una por una. '
                'Tú las dibujas y la IA, sin saber qué palabra te tocó, intenta adivinar qué dibujaste.</p>', unsafe_allow_html=True)
    st.write("")
    nombres = list(TEMAS)
    for fila in [nombres[:3], nombres[3:]]:
        cols = st.columns(3) if len(fila) == 3 else st.columns([0.5, 1, 1, 0.5])[1:3]
        for col, tema in zip(cols, fila):
            with col:
                with st.container(key=f"tema_{nombres.index(tema)}"):
                    imagen(IMAGEN_TEMA.get(tema, "gato_hola.png"), 170)
                    st.markdown(f'<p class="nombre-tema">{tema}</p><p class="cuantas">{len(TEMAS[tema])} palabras</p>',
                                unsafe_allow_html=True)
                    st.button("Elegir", key=f"elegir_{tema}", on_click=empezar_tema, args=(tema,), use_container_width=True)
        st.write("")
    _, c, _ = st.columns([1, 1, 1])
    c.button("Volver", use_container_width=True, on_click=ir, args=("inicio",))


# ---------------------------------------------------------------- 3. dibujar
PISTAS = {
    "apple": "Fruta roja o verde que se come con cáscara.",
    "pear": "Fruta verde, más ancha abajo que arriba.",
    "banana": "Fruta amarilla y alargada que se pela.",
    "strawberry": "Fruta roja y pequeña con semillitas por fuera.",
    "watermelon": "Fruta grande, verde por fuera y roja por dentro.",
    "grapes": "Frutas pequeñas que vienen en racimo.",
    "ice cream": "Postre frío que se come en cono.",
    "pizza": "Comida redonda que se corta en triángulos.",
    "cat": "Animal que maúlla y tiene bigotes.",
    "dog": "Animal que ladra y mueve la cola.",
    "rabbit": "Animal de orejas largas que salta.",
    "bird": "Animal con plumas y pico que vuela.",
    "butterfly": "Insecto con alas de colores.",
    "fish": "Animal que vive en el agua y tiene aletas.",
    "snake": "Animal largo y sin patas que se arrastra.",
    "cow": "Animal de la finca que da leche.",
    "frog": "Animal verde que salta y croa.",
    "elephant": "Animal muy grande con trompa.",
    "house": "Lugar donde vive una familia.",
    "umbrella": "La abres cuando llueve.",
    "guitar": "Instrumento musical de seis cuerdas.",
    "chair": "Mueble para sentarse.",
    "cup": "Recipiente con oreja para tomar café o chocolate.",
    "clock": "Marca las horas.",
    "book": "Tiene páginas y se lee.",
    "key": "Sirve para abrir una puerta.",
    "tree": "Planta grande con tronco y hojas.",
    "flower": "Parte de la planta, de colores y con olor.",
    "sun": "Brilla en el cielo de día.",
    "moon": "Brilla en el cielo de noche.",
    "cloud": "Está en el cielo y puede traer lluvia.",
    "rainbow": "Aparece en el cielo después de llover, con muchos colores.",
    "mountain": "Terreno muy alto, mucho más que una loma.",
    "car": "Vehículo de cuatro ruedas para pocas personas.",
    "bicycle": "Vehículo de dos ruedas con pedales.",
    "bus": "Vehículo grande que lleva a muchos pasajeros.",
    "airplane": "Vehículo con alas que vuela.",
    "sailboat": "Barco que se mueve con el viento.",
    "train": "Vehículo largo que va sobre rieles.",
    "truck": "Vehículo grande que carga mercancía.",
}


def botones_palabra(palabra):
    # Escuchar usa la voz del navegador (speechSynthesis). La pista aparece sin recargar la pantalla.
    import json as _json
    import streamlit.components.v1 as components
    components.html(f"""
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@700&family=Patrick+Hand&display=swap" rel="stylesheet">
    <style>
      body {{ margin: 0; background: transparent; text-align: center; }}
      .fila {{ display: flex; justify-content: center; gap: 10px; }}
      button {{ font-family: 'Nunito', sans-serif; font-weight: 700; font-size: 15px; color: #2F5FA8;
               background: #FFFBF4; border: 1.5px solid #2F5FA8; border-radius: 999px; padding: 8px 18px;
               display: inline-flex; align-items: center; gap: 8px; cursor: pointer; }}
      button:active {{ background: #EAF1FB; }}
      #texto {{ font-family: 'Patrick Hand', cursive; font-size: 18px; color: #6B5D4C; margin: 8px 12px 0; line-height: 1.25; }}
    </style>
    <div class="fila">
      <button id="b" type="button" aria-label="Escuchar la palabra en inglés">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2F5FA8" stroke-width="2.2"
             stroke-linecap="round" stroke-linejoin="round"><path d="M4 9v6h4l5 4V5L8 9H4z"/>
             <path d="M16.5 8.5a5 5 0 0 1 0 7"/><path d="M19.5 5.5a9 9 0 0 1 0 13"/></svg>
        Escuchar
      </button>
      <button id="p" type="button">Pista</button>
    </div>
    <p id="texto">¿No sabes qué es? Pide una pista.</p>
    <script>
      const palabra = {_json.dumps(palabra)};
      document.getElementById("b").onclick = () => {{
        const v = window.speechSynthesis;
        if (!v) return;
        v.cancel();
        const u = new SpeechSynthesisUtterance(palabra);
        u.lang = "en-US"; u.rate = 0.85;
        const voz = v.getVoices().find(x => x.lang && x.lang.startsWith("en"));
        if (voz) u.voice = voz;
        v.speak(u);
      }};
      const pista = {_json.dumps(PISTAS.get(palabra, ""))};
      document.getElementById("p").onclick = () => {{
        document.getElementById("texto").textContent = "Pista: " + pista;
      }};
    </script>""", height=100)


def pantalla_dibujar():
    palabra = s.ronda[s.indice]
    a, b = st.columns([3, 1], vertical_alignment="center")
    a.markdown(f'<p class="miga">{s.tema} · palabra {s.indice + 1} de {len(s.ronda)}</p>', unsafe_allow_html=True)
    b.button("Salir", use_container_width=True, on_click=ir, args=("temas",))

    st.markdown(f"""<div class="tarjeta"><p class="pide">Draw:</p><p class="palabra">{palabra}</p>
                </div>""", unsafe_allow_html=True)
    botones_palabra(palabra)

    with st.container(key=f"lienzo_{s.lienzo}"):
        lienzo = st_canvas(stroke_width=6, stroke_color=CRAYONES[s.crayon], background_color="#FFFBF4",
                           height=360, width=360, drawing_mode="freedraw", key=f"canvas_{s.lienzo}")

    with st.container(horizontal=True, horizontal_alignment="center", gap="small", key="crayones"):
        for nombre in CRAYONES:
            st.button(" ", key=f"crayon_{nombre}", on_click=lambda n=nombre: s.update(crayon=n))
    st.markdown(f"<style>.st-key-crayon_{s.crayon} button {{ outline: 3px solid #2F5FA8; outline-offset: 3px; }}</style>",
                unsafe_allow_html=True)
    st.markdown('<p class="miga centro">Dibuja la palabra en el recuadro, rápido y sencillo, como en 20 segundos. Cuando termines, toca <b>Listo</b>.</p>', unsafe_allow_html=True)

    _, c1, c2, _ = st.columns([1, 2, 2, 1])
    listo = c1.button("Listo", type="primary", use_container_width=True)
    c2.button("Borrar", use_container_width=True, on_click=borrar)

    if listo:
        trazos = trazos_del_lienzo(lienzo.json_data)
        if not trazos:
            st.markdown('<div class="globo duda">Primero dibuja algo en el recuadro.</div>', unsafe_allow_html=True)
            return
        img, prob = predecir(trazos)
        orden = np.argsort(prob)[::-1]
        top, conf = CLASES[orden[0]], float(prob[orden[0]])
        tipo = "duda" if conf < UMBRAL else ("bien" if top == palabra else "casi")
        if tipo == "bien":
            s.aciertos[palabra] = True
        s.resultado = {"tipo": tipo, "top": top, "img": img,
                       "top3": [(CLASES[i], float(prob[i])) for i in orden[:3]]}
        s.pantalla = "resultado"
        st.rerun()


# ---------------------------------------------------------------- 4. resultado
def pantalla_resultado():
    r, palabra = s.resultado, s.ronda[s.indice]
    st.markdown(f'<p class="miga">{s.tema} · palabra {s.indice + 1} de {len(s.ronda)}</p>', unsafe_allow_html=True)
    pose = {"bien": "gato_feliz.png", "casi": "gato_animo.png", "duda": "gato_confundido.png"}[r["tipo"]]
    centrado(pose, 250)
    t = r["top"]
    texto = {"bien": f"¡Sí! Dibujaste <b>{t}</b> ({PALABRAS[t]}).",
             "casi": f"¡Casi! A mí me parece: <b>{t}</b> ({PALABRAS[t]}).<br>La palabra era <b>{palabra}</b> ({PALABRAS[palabra]}).",
             "duda": f"Mmm… no estoy segura de qué es.<br>La palabra era <b>{palabra}</b> ({PALABRAS[palabra]})."}[r["tipo"]]
    st.markdown(f'<div class="globo {r["tipo"]}">{texto}</div>', unsafe_allow_html=True)
    st.markdown('<p class="miga centro">La IA no sabe qué palabra te tocó: solo mira tu dibujo y adivina.</p>',
                unsafe_allow_html=True)

    ultima = s.indice + 1 >= len(s.ronda)
    texto_sig = "Ver resultados" if ultima else "Siguiente palabra"
    if r["tipo"] == "bien":
        _, c1, _ = st.columns([1, 1, 1])
        c1.button(texto_sig, type="primary", use_container_width=True, on_click=siguiente)
    else:
        st.markdown('<p class="miga centro">Puedes dibujarla de nuevo o pasar a la siguiente palabra.</p>',
                    unsafe_allow_html=True)
        _, c1, c2, _ = st.columns([1, 2, 2, 1])
        c1.button("Intentar otra vez", type="primary", use_container_width=True, on_click=otra_vez)
        c2.button(texto_sig, use_container_width=True, on_click=siguiente)

    st.write("")
    with st.expander("Ver lo que vio la IA"):
        a, b = st.columns([1, 2], vertical_alignment="center")
        with a:
            imagen("gato_lupa.png", 120)
        with b:
            st.image(r["img"], width=130, clamp=True)
            st.caption("Tu dibujo, centrado y reducido a 28×28 píxeles, igual que los dibujos con los que aprendió la IA.")
        st.markdown("**Sus 3 ideas** (qué tan segura está de cada una):")
        for p, pr in r["top3"]:
            st.progress(pr, text=f"{p} ({PALABRAS[p]}) · {pr:.0%}")


# ---------------------------------------------------------------- 5. final
def pantalla_final():
    n_bien = sum(s.aciertos.get(p, False) for p in s.ronda)
    centrado("gato_feliz.png" if n_bien >= len(s.ronda) / 2 else "gato_animo.png", 240)
    st.markdown(f'<p class="seccion centro">Dibujaste {n_bien} de {len(s.ronda)}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="miga centro">{s.tema}</p>', unsafe_allow_html=True)
    st.write("")
    filas = "".join(
        f'<div class="fila"><span><b>{p}</b> · {PALABRAS[p]}</span>'
        f'<span class="marca {"bien" if s.aciertos.get(p) else "pend"}">{"la IA lo reconoció" if s.aciertos.get(p) else "para practicar"}</span></div>'
        for p in s.ronda)
    st.markdown(filas, unsafe_allow_html=True)
    st.write("")
    _, c1, c2, _ = st.columns([1, 2, 2, 1])
    c1.button("Otra ronda", type="primary", use_container_width=True, on_click=empezar_tema, args=(s.tema,))
    c2.button("Cambiar de tema", use_container_width=True, on_click=ir, args=("temas",))


{"inicio": pantalla_inicio, "temas": pantalla_temas, "dibujar": pantalla_dibujar,
 "resultado": pantalla_resultado, "final": pantalla_final}[s.pantalla]()
