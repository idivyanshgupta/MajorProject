import streamlit as st 
import pandas as pd
import os
from PIL import Image
from collections import deque
import numpy as np
from recording import record
from css_loader import my_css
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
st.set_option('deprecation.showfileUploaderEncoding', False)
my_css("style.css")

classes_list = ["Walking With Dog", "Diving", "Skate Boarding", "Horse Race","Playing Piano"]
sidebar_title = "<div><span class='highlight red'>Human Activity Recognition using CNN </span><br></br></div>"
st.sidebar.markdown(sidebar_title, unsafe_allow_html=True)

sidebar_subtitle = "<div><span class='highlight blue'>Major Project</span><br></br></div>"
st.sidebar.markdown(sidebar_subtitle, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .reportview-container 
    {
        background-color: aliceblue
    }
   
    </style>
    """,
    unsafe_allow_html=True
)


mainoption = st.sidebar.radio("Please select an option?",
    ("Record Video","Upload Video", "Recognize Activity")
)


def file_display(option='.'):
    imagef = Image.open(option)
    st.image(imagef, caption='Selected Image',width=400)


def activity_predictor(image_path, model, w=32, h=32):
    img = image.load_img(image_path,target_size=(w,h))
    x= image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    return model.predict(x)


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    images = {}
    exts=[".jpg",".png",".tif",".bmp"]
    for file in filenames:
        f,e = os.path.splitext(file)
        if e in exts:
            images[file]=os.path.join(folder_path,file)
    
    try:
        selected_filename = st.selectbox('Select a file', list(images.keys()))
        return images[selected_filename]
    except:
        st.error("No File Found !")
   



if __name__ == '__main__':
    # Select a file
   
        
    
    if mainoption=="Record Video":
        
        st.title("Record Video")
        rec = st.button("Click to Record a 10s Video")
        if rec:
            record()

    if mainoption=="Upload Video":
        st.title("Upload Video")
        file = st.file_uploader("Please upload a 10s Video",type=["mp4","mkv","avi"])
        if file:
            data = file.read()
            vid = open(os.path.join('..','uploads',file.name),'wb')
            vid.write(data)
            vid.close()
            st.success("Video Uploaded Successfully ! ")
            st.video(data)

    if mainoption=="Recognize Activity":
        st.title("Recognize Activity")
        
        if st.checkbox('Select Directory'):
            content = os.listdir('..')
            content=list(filter(lambda item:os.path.isdir(os.path.join("..", item)),content))
            content = list(filter(lambda item:not item.startswith("."),content))


            folder_path= st.selectbox('Select a folder', content)
            if os.path.exists(os.path.join("..",folder_path)) :
                filename = file_selector(folder_path=os.path.join("..",folder_path))
                if filename:
                    st.write('You selected `%s`' % filename)
                    file_display(option=filename)
                    if os.path.exists("../model/activity_predict.h5"):
                            model=load_model("../model/activity_predict.h5")
                          
                            pred = activity_predictor(image_path=filename,model=model)
                            st.write(f"The activity being performed is **{classes_list[np.argmax(pred)]}**")
                            
                    else:
                        st.error("No Model Found")

                else:
                            st.error("No Images Found")
            else:
                st.write('Invalid Path')


