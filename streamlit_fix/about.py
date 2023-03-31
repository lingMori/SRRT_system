import streamlit as st
from PIL import Image



class About:
    class Model:
        pageTitle = "About us"

    def view(self, model):
        st.title(model.pageTitle)
        # st.sidebar.write('联系我们\n\r''Wenbin Wang  kklplus@gmail\n\r ' 'zhanghaoyang ')
        st.sidebar.markdown('# 联系我们\n\r''## Wenbin Wang (master) \n\r  kklpluswwb@gmail.com\n\r ' '## HaoYang Zhang (coworker) \n\r 754597673@qq.com \n\r ')

        st.subheader('Generative model of Channel Characteristics, as an emerging approach, plays an increasingly influential role in channel modeling. In this article, we present a super-resolution model for Channel characteristics generating. Based on  convolutional neural networks(CNN), we have also in corporated residual connections in this architecture.')




        st.markdown('Our team consists of Wang Wenbin and Zhang Haoyang from the School of Computer Science, Beijing Jiaotong University, and Cheng Yunhao from the School of Telecommunications. This software is based on the results of the first-stage channel modeling research. It packs the channel modeling process into a software form and summarizes and compares the running results.')

        st.markdown('# A Multi-Task Learning Model for Super Resolution of Wireless Channel Characteristics')

        col1, col2 = st.columns(2)
        with col1:
            image = Image.open('image/net1.jpg')
            st.image(image, caption='net1', use_column_width=True)
        with col2:
            image = Image.open('image/net2.jpg')
            st.image(image, caption='net2', use_column_width=True)



        # st.markdown('![banner](../image/net1.jpg)')
        # st.markdown('![badge](../image/net2.jpg)')
        # st.markdown('[![standard-readme compliant](https://img.shields.io/badge/Multi_Task%20-Super '
        #             'Resolution-brightgreen.svg?style=flat-square)]('
        #             'https://github.com/lingMori/A-Multi-Task-Learning-Model-for-Super-Resolution-of-Wireless-Channel'
        #             '-Characteristics)', unsafe_allow_html=True)
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')
        st.markdown('')

        video_file = open('bjtu.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)
