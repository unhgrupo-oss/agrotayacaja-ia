import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title='AgroTayacaja IA', page_icon='🌱', layout='wide', initial_sidebar_state='expanded')

CLASSES_PATH = Path('agrotayacaja_clases.json')
HEAD_PATH = Path('agrotayacaja_head_weights.json')

st.markdown('''
<style>
.block-container{padding-top:1.4rem;max-width:1200px}
.hero{padding:1.4rem 1.5rem;border-radius:22px;background:linear-gradient(135deg,#14532d 0%,#16a34a 65%,#65a30d 100%);color:white;box-shadow:0 10px 30px rgba(20,83,45,.18);margin-bottom:1.2rem}
.hero h1{margin:0;font-size:2.6rem}.hero p{margin:.45rem 0 0;opacity:.95}
.pill{display:inline-block;padding:.32rem .7rem;margin:.2rem .25rem .2rem 0;border-radius:999px;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.28);font-size:.86rem}
.card{padding:1rem 1.1rem;border:1px solid rgba(120,120,120,.16);border-radius:18px;background:rgba(255,255,255,.72);box-shadow:0 6px 18px rgba(0,0,0,.05)}
.result{padding:1.15rem 1.25rem;border-radius:18px;border:1px solid rgba(22,163,74,.20);background:linear-gradient(180deg,rgba(240,253,244,.92),rgba(255,255,255,.92))}
div[data-testid='stMetric']{background:rgba(255,255,255,.72);border:1px solid rgba(120,120,120,.14);padding:.75rem 1rem;border-radius:16px}
.stButton>button{border-radius:14px;font-weight:700;min-height:3rem}.footer{margin-top:1.6rem;padding-top:1rem;border-top:1px solid rgba(120,120,120,.16);font-size:.86rem;opacity:.72;text-align:center}
</style>
''', unsafe_allow_html=True)

@st.cache_data
def cargar_meta():
    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def cargar_modelo():
    import base64
    with open(HEAD_PATH, 'r', encoding='utf-8') as f:
        w = json.load(f)
    base = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
    base.trainable = False
    inp = tf.keras.Input(shape=(224,224,3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(3, activation='softmax', name='clasificador')(x)
    model = tf.keras.Model(inp, out)
    kernel = np.frombuffer(base64.b64decode(w['kernel_b64']), dtype=np.float32).reshape(w['kernel_shape'])
    bias = np.frombuffer(base64.b64decode(w['bias_b64']), dtype=np.float32)
    model.get_layer('clasificador').set_weights([kernel, bias])
    return model

meta = cargar_meta()
labels_es = meta['labels_es']
img_size = tuple(meta['img_size'])
model = cargar_modelo()

def predecir(imagen):
    img = imagen.convert('RGB').resize(img_size)
    arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
    probs = model.predict(arr, verbose=0)[0]
    i = int(np.argmax(probs))
    return labels_es[i], float(probs[i]), probs

def nivel(c):
    if c >= .85: return 'Alta', 'La predicción es relativamente consistente para este prototipo.'
    if c >= .65: return 'Media', 'Conviene contrastar con otra fotografía tomada desde otro ángulo.'
    return 'Baja', 'Tome otra fotografía con mejor luz y consulte a un especialista.'

def recomendacion(clase):
    if clase == 'Sana': return 'No se detectó una de las dos enfermedades consideradas por este prototipo. Mantenga el monitoreo y pruebe otra fotografía si observa síntomas.'
    if clase == 'Tizón temprano': return 'El modelo encontró patrones compatibles con tizón temprano. Tome imágenes adicionales y contraste el resultado con una evaluación agronómica.'
    return 'El modelo encontró patrones compatibles con tizón tardío. Registre nuevas fotografías y solicite una evaluación agronómica antes de tomar decisiones de manejo.'

with st.sidebar:
    st.markdown('## 🌱 AgroTayacaja IA')
    st.caption('Prototipo académico · UNH')
    st.divider(); st.markdown('### Alcance actual'); st.write('**Cultivo:** Papa')
    for c in labels_es: st.write('• '+c)
    st.divider(); st.markdown('### Próximas fases'); st.write('🌽 Maíz\n\n🫘 Haba\n\n🫛 Arveja\n\n📚 RAG agronómico\n\n🤖 Agente de apoyo')
    st.divider(); st.caption('El resultado es preliminar y no sustituye un diagnóstico agronómico.')

st.markdown('''<div class="hero"><h1>🌱 AgroTayacaja IA</h1><p>Clasificación preliminar de enfermedades en hojas de papa mediante visión artificial.</p><div style="margin-top:.9rem"><span class="pill">🥔 Papa</span><span class="pill">🧠 MobileNetV2</span><span class="pill">📷 Visión por computadora</span><span class="pill">⚡ Prototipo funcional</span></div></div>''', unsafe_allow_html=True)

st.markdown('### 1. Cargar una imagen')
modo = st.radio('Modo', ['Subir archivo','Usar cámara'], horizontal=True, label_visibility='collapsed')
archivo = st.file_uploader('Sube una fotografía de una hoja de papa', type=['jpg','jpeg','png']) if modo=='Subir archivo' else st.camera_input('Toma una fotografía de la hoja')

if archivo is None:
    st.info('Sube o toma una fotografía para comenzar el análisis.')
    st.markdown('<div class="card"><b>Consejos:</b><br>• Buena iluminación.<br>• Que la hoja ocupe gran parte de la imagen.<br>• Evite fondos muy cargados.<br>• Pruebe imágenes distintas a las del entrenamiento.</div>', unsafe_allow_html=True)
else:
    imagen = Image.open(archivo).convert('RGB')
    ci, ca = st.columns([1.15,1], gap='large')
    with ci: st.image(imagen, caption='Imagen seleccionada', width='stretch')
    with ca:
        st.markdown('### 2. Analizar')
        st.write('El modelo comparará la fotografía con patrones aprendidos para **Sana, Tizón temprano y Tizón tardío**.')
        if st.button('🔍 Analizar hoja', type='primary', width='stretch'):
            with st.spinner('Analizando la imagen...'):
                clase, conf, probs = predecir(imagen)
            st.session_state['r'] = (clase, conf, probs.tolist())
    if 'r' in st.session_state:
        clase, conf, probs = st.session_state['r']; probs = np.array(probs)
        niv, txt = nivel(conf)
        st.divider(); st.markdown('### 3. Resultado del análisis')
        st.markdown(f'<div class="result"><div style="font-size:.9rem;opacity:.7">Resultado preliminar</div><div style="font-size:2rem;font-weight:800">{clase}</div><div>Confianza del modelo: <b>{conf*100:.2f}%</b></div></div>', unsafe_allow_html=True)
        c1,c2,c3=st.columns(3); c1.metric('Clase detectada',clase); c2.metric('Confianza',f'{conf*100:.2f}%'); c3.metric('Nivel de confianza',niv)
        st.caption(txt); st.markdown('#### Probabilidades por clase')
        for i in np.argsort(probs)[::-1]:
            p=float(probs[i]); st.write(f'**{labels_es[int(i)]} — {p*100:.2f}%**'); st.progress(max(0.0,min(1.0,p)))
        st.markdown('#### Interpretación'); st.write(recomendacion(clase))
        if conf < .65: st.warning('La confianza es baja. Repita la prueba con una fotografía más clara antes de interpretar el resultado.')
        with st.expander('ℹ️ ¿Qué significa este resultado?'):
            st.write('El modelo fue entrenado con imágenes de PlantVillage y reconoce únicamente las tres clases incluidas en este prototipo. Una probabilidad alta no equivale a una confirmación agronómica.')

st.markdown('<div class="footer">AgroTayacaja IA · Prototipo académico de Inteligencia Artificial<br>Primera versión: clasificación de hojas de papa · Tayacaja, Huancavelica</div>', unsafe_allow_html=True)
