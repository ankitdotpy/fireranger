import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import io
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

st.write('''
	# Fire Ranger
	#### Deep Learning model to predict wildfires from satellite images
''')

@st.cache
def load_model():
	model = tf.kears.models.load_model('./models/final_model.h5')
	return model
	
model = load_model()

img = Image.open('./images/test.jpg')
img = np.array(img).reshape((1,350,350,3))

uploaded_img = st.file_uploader(
	label = 'Upload a 350x350 satellite image',
	type = ['png','jpg','jpeg'])

if uploaded_img is not None:
	file = uploaded_img.read()
	img = Image.open(io.BytesIO(file))
	img = np.array(img).reshape((1,350,350,3))

pred = model.predict(img)

st.image(image = img)
st.text(f'Wildfire prediction probability is {pred[0,1]*100}%')

