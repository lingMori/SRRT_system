import streamlit as st
import numpy as np
import os
import sys
from PIL import Image
sys.path.append("..")
from code_backup_08_12 import mlt_main
import pandas as pd

class DataTraining:
    class Model:
        pageTitle = "Data Training"
        mat_name = []
        train = []
        test = []
        scale = 2
        test_ratio = 0.2
        seed = 42
        batch_size = 1
        step_size = 30
        aug_switch = True
        lr = 1e-5
        gamma = 0.7
        weight_decay = 0
        maskvalue = 1e-5
        eopchs_P1 = 100
        epochs_P2 = 100




    def view(self, model):
        st.title(model.pageTitle)
        model.mat_name = []

        uploaded_file = st.sidebar.file_uploader("上传mat文件", type="mat")

        # st.write(uploaded_file)
        if uploaded_file != None:
            path_savefile=os.path.join(uploaded_file.name)
            path_savefile = '..\\code_backup_08_12\\data\\' + path_savefile
            # st.write(path_savefile)

            with open(path_savefile, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.sidebar.success("Saved file :{} in tempDir".format(uploaded_file.name))


        # 将当前目录下的所有文件名称读取进来

        path_data='..\\code_backup_08_12\\data\\'#文件路径
        a = os.listdir(path_data)

        for j in a:
            # 判断是否为mat文件，如果是则存储到列表中
            if os.path.splitext(j)[1] == '.mat':
                model.mat_name.append(j)
        # st.write(model.mat_name)

        with st.container():
            col1, col2, col3, col4 = st.columns(4)

            length = len(model.mat_name)
            with col1:
                model.train = st.multiselect(
                    '选择训练集:',
                    model.mat_name,
                )
            with col2:
                st.write('You selected:', model.train)
            with col3:
                model.test = st.multiselect(
                    '选择测试集:',
                    model.mat_name,
                )
            with col4:
                st.write('You selected:', model.test)

        options = np.arange(1, 301, 1)
        a = len(options)

        # st.write(options)
        col12, col22 = st.columns(2)
        with col12:
            model.epochs_P1 = st.select_slider("请选择epochs_P1：",
                                               options=options,
                                               value=10,
                                               )

            options22 = np.arange(1, 401, 1)
        with col22:
            model.epochs_P2 = st.select_slider("请选择epochs_P2：",
                                               options=options22,
                                               value=100,
                                               )
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                model.scale = st.selectbox(

                    " 选择超分尺度",
                    [2, 4, 8],  # 也可以用元组
                    index=1
                )
            with col2:
                options = np.arange(0, 1.1, 0.1)
                model.test_ratio = st.select_slider("请选择test_ratio：",
                                                    options=options,
                                                    value=0.2,
                                                    )
            with col3:
                options = np.arange(0, 101, 1)
                model.seed = st.select_slider("请选择seed：",
                                              options=options,
                                              value=42,
                                              )
            with col4:
                options = np.arange(1, 11, 1)
                model.batch_size = st.select_slider("请选择batch_size：",
                                                    options=options,
                                                    value=1,
                                                    )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            options = np.arange(1, 51, 1)
            model.step_size = st.select_slider("请选择step_size：",
                                               options=options,
                                               value=30,
                                               )
        with col2:

            model.aug_swith = st.selectbox(
                '请选择aug_swith:',
                [True, False],
            )



        with col3:
            model.lr = st.selectbox(
                '请选择lr:',
                [1e-6, 1e-5, 1e-4],
            )
        with col4:
            options = np.arange(0, 1.1, 0.1)
            model.gamma = st.select_slider("请选择gamma：",
                                           options=options,
                                           value=0,
                                           )

        col1, col2 = st.columns(2)
        with col1:
            options = np.arange(0, 1.1, 0.1)
            model.weight_decay = st.select_slider("请选择weight_decay：",
                                                  options=options,
                                                  value=0,
                                                  )
        with col2:
            model.maskvalue = st.selectbox(
                '请选择maskvalue:',
                [1e-6, 1e-5, 1e-4],
            )

        st.subheader("您选择的参数如下：")


        col1, col2, col3, col4 = st.columns(4)  # 位置

        with col1:
            st.metric('scale', "{} ".format(model.scale))
            st.metric('test_ratio', "{:0.1f} ".format(model.test_ratio))
            st.metric('seed', "{} ".format(model.seed))
        with col2:

            st.metric('batch_size', "{} ".format(model.batch_size))
            st.metric('step_size', "{} ".format(model.step_size))
            st.metric('aug_swith', model.aug_swith)


        with col3:
            st.metric('lr', model.lr)
            st.metric('gamma', "{:0.1f} ".format(model.gamma))
            st.metric('weight_decay', "{} ".format(model.weight_decay))



        with col4:
            st.metric('maskvalue',model.maskvalue)
            st.metric('epochs_P1', model.epochs_P1)
            st.metric('epochs_P2',model.epochs_P2)

        st.markdown('确认无误后，点击下方按钮开始训练：')

        if st.button('Start'):
            st.write('开始训练')
            mlt_main.train_mlt(model)
            st.markdown('训练完成！！！')

