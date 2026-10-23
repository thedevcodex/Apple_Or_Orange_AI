import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img

model = tf.keras.models.load_model('apple_or_orange.keras')

st.set_page_config(
    page_title="AppleOrange AI",
    page_icon = "https://img.icons8.com/?size=100&id=5J0YM2MKuxZO&format=png&color=000000",
    layout='wide',
    initial_sidebar_state='expanded'
)
st.markdown("""
<style>
.stApp{
        background: linear-gradient(135deg, #0B1D16,#163D2B,
        #245A3C);
      
}
div.stButton > button {
    background-color: red;
    color: white;
    border: none;
    border-radius: 10px;
}
</style>

""",unsafe_allow_html=True)


st.title("AppleOrange AI")
st.caption("CNN-Powered Image Classification | Built by devcodex")  

tab1,tab2,tab3 = st.tabs(['Home',"Prediction","About"])

with tab1:
    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image("img.jpg", use_container_width=True)
    st.write("")
    st.markdown(""" 

Welcome to **🍎AppleOrange AI🍊**, an image classification application built using Deep Learning. It uses a custom Convolutional Neural Network (CNN) developed from scratch with TensorFlow and Keras to classify images into two categories: Apple and Orange.

Navigate to the Prediction tab, upload an image of an apple or orange, and let the model predict the fruit category.

Note: Predictions may not always be accurate.

        """)


with tab2:
    image = st.file_uploader("Upload Apple or Orange Image",
                             ['jpg','jpeg','png','webp'])
    if image:
        col1,col2,col3 = st.columns([1,2,1])
        with col2:
            st.image(image,width=800)
            st.write(' ')
            button_pred = st.button("Predict")
            if button_pred:
                img = load_img(image,target_size=(300,300))
                img_arr = np.array(img)/255.0
                img_arr = np.expand_dims(img_arr,axis=0)
                pred = model.predict(img_arr)
                if pred[0][0]>0.5:
                    st.warning("Orange")
                else:
                    st.warning("Apple")


with tab3:
    st.markdown("""

### Project Overview
AppleOrange AI is an image classification project developed using **Convolutional Neural Networks (CNNs)** with TensorFlow and Keras. The model is trained to classify images into two categories: Apple and Orange.

### 🎯 Project Objectives
- Understand how CNNs learn features from images.
- Apply deep learning techniques to image classification.
- Build an interactive web application using Streamlit.
- Explore the practical implementation of a CNN model from scratch.

### ⚙️ Technologies Used
- **Python** — Core programming language
- **TensorFlow & Keras** — CNN model development and training
- **NumPy** — Numerical operations
- **Pillow (PIL)** — Image processing
- **Streamlit** — Interactive web application

### 🔄 How It Works
1. Upload an image through the Prediction tab.
2. The image is resized and preprocessed to match the model's input requirements.
3. The trained CNN analyzes the image and generates a prediction.
4. The application displays the predicted fruit category: Apple or Orange.

### ⚠️ Limitations
The model may produce incorrect predictions depending on image quality, lighting, dataset diversity, and training performance. Best results may be obtained with clear images of apples or oranges.



- **GitHub:** [github.com/thedevcodex](https://github.com/thedevcodex)
- **LinkedIn:** [linkedin.com/in/sudhasudalaimani](https://www.linkedin.com/in/sudhasudalaimani/)

An AI and Deep Learning learning project focused on understanding and applying CNN-based image classification.""")