import streamlit as st
import numpy as np
import pandas as pd
import os
import plotly.express as px



class Dashboard:
    class Model:
        pageTitle = "Dashboard"

    def view(self, model):
        st.title(model.pageTitle)

        row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))  # 设置位置
        with row0_1:
            st.title('Comparison Of Channel Simulation Training Process')
        row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
        with row3_1:
            st.subheader(
                'Use this software to help you more intuitively see the changes in channel characteristics during training with different data sets.')
        ### Data Import ##
        file_name = []

        def name():
            # 将当前目录下的所有文件名称读取进来
            a = os.listdir('../code_backup_08_12/result')
            for j in a:
                # 判断是否为CSV文件，如果是则存储到列表中
                if os.path.splitext(j)[1] == '.csv':
                    file_name.append(j)

        name()
        length = len(file_name)


        st.text('')
        options2 = st.sidebar.multiselect(
            'Please select the data you want to compare :',
            file_name,
        )


        st.sidebar.write('You selected:', options2)
        # st.write(222)
        # st.write(options2)
        st.subheader("Please select the element you want to draw")
        genre = st.selectbox(  # 选择内容
            " ",
            file_name,  # 也可以用元组
            index=0
        )

        b = format(genre)
        b = '..\\code_backup_08_12\\result\\' + b#文件路径

        col1, col2, col3, col4 = st.columns(4)  # 位置
        # K_mae', 'Phi_mae', 'Theta_mae','P_mae','T_mae','TPR_mae','FPR_mae'
        df = pd.read_csv(b)  # street4_scale=8.csv
        with col1:
            st.metric('K_mae', "{:0.3f} dB".format(df.iloc[-1]['K_mae']))
            st.metric('Phi_mae', "{:0.3f} dB".format(df.iloc[-1]['Phi_mae']))

        with col2:
            st.metric('Theta_mae', "{:0.3f} dB".format(df.iloc[-1]['Theta_mae']))
            st.metric('P_mae', "{:0.3f} dB".format(df.iloc[-1]['P_mae']))

        with col3:
            st.metric('T_mae', "{:0.3f} dB".format(df.iloc[-1]['T_mae']))
            st.metric('TPR_mae', "{:0.3f} ".format(df.iloc[-1]['TPR_mae']))

        with col4:
            st.metric('FPR_mae', "{:0.3f} dB".format(df.iloc[-1]['FPR_mae']))



        df_all = [0] * length


        len22 = len(options2)

        for i in range(0, len22):
                path = '..\\code_backup_08_12\\result\\' + options2[i]

                df_all[i] = pd.read_csv(path)  # 读数据



        # st.dataframe(df)#将数据框显示为交互式表格
        st.sidebar.subheader('2.选择运行的epoch范围')
        options = np.array(df['epoch']).tolist()
        a = len(options)
        # st.write(a)
        epoch1=1
        for i in range(0, a - 1):
            if options[i] > options[i + 1]:
                epoch1 = i + 1
                break
            # st.write(epoch1)
        for i in range(epoch1, a):
                options[i] = options[i] + epoch1

        (start_time, end_time) = st.sidebar.select_slider("请选择时间序列长度：",
                                                          options=options,
                                                          value=(1, 40),
                                                          )
        st.sidebar.write("epoch开始:", start_time)
        st.sidebar.write("epoch结束:", end_time)
        df.index = options
        df = df[start_time:end_time]


        # st.dataframe(df)
        row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
        with row6_1:
            st.subheader("Please select the element you want to draw")
        genre = st.selectbox(

            " ",
            ['K_mae', 'Phi_mae', 'Theta_mae', 'P_mae', 'T_mae', 'TPR_mae', 'FPR_mae', 'K_std', 'Phi_std', 'Theta_std',
             'P_std', 'T_std'],  # 也可以用元组
            index=1
        )
        # K_mae	Phi_mae	Theta_mae	P_mae	T_mae	TPR_mae	FPR_mae	K_std	Phi_std	Theta_std	P_std	T_std

        # st.write('You selected {}.'.format(genre))


#############################################################
        selected = format(genre)
        # st.write(selected,'is:')
        data = df.drop(['epoch'], axis=1)
        selected_data = data[selected]
        arry = []
        for i in range(0, len22):
            path = '..\\code_backup_08_12\\result\\' + options2[i]

            df = pd.read_csv(path)
            newnum=format(df.iloc[-1][selected])
            newnum2=float(newnum)
            arry.append(newnum2)

######################################################
        data = {'Name': options2, 'shuzhi': arry}
        df_bar = pd.DataFrame(data)

        st.subheader("柱状图")

        fig = px.bar(
            df_bar,
            x="Name",
            y="shuzhi",
            # title="Books Read by Year",
            color_discrete_sequence=["#9EE6CF"],
        )

        col1,col2=st.columns(2)
        with col1:
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

        with col2:
                st.subheader('Data comparison display chart')
                st.markdown('\n In order to more intuitively show the difference in the results of different training parameter configurations. You can select the running results you want to compare in the left sidebar, and select the data you want to compare in the selection bar above.'
                                       )

        selected_data2 = selected_data
        if length > 0:
            for i in range(0, len22):
                if options2[i] != 0:
                    ii = str(i)
                    df_all[i].index = options
                    df_all[i] = df_all[i][start_time:end_time]
                    df_s = df_all[i][[selected]]
                    df_new = df_s.rename(columns={selected: selected + '-' + ii})
                    selected_data2 = pd.concat([selected_data2, df_new], axis=1)


        row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
        with row3_1:
            st.markdown("")
            see_data = st.expander('You can click here to see the raw data first 👉')
            with see_data:
                st.dataframe(data=selected_data2.reset_index(drop=True))
        st.text('')
        # st.write(selected_data)

        row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
        with row6_1:
            st.subheader('Data comparison display chart')
        row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))
        with row7_1:
            st.markdown(
                'You can select the data content you want to compare and analyze in the left sidebar. (All the .csv files in the sidebar are automatically generated by the program running, and are in the same folder as the current program.) You can select multiple sets of data for comparison at the same time, or adjust the epoch to observe the results achieved in different rounds of training training effect')
        with row7_2:

            st.line_chart(selected_data2)



###############################################################################
        '''
             动态折线图演示示例
             '''
        # col1,col2=st.columns(2)
        # a=[1.0,3.1],[2.0,2.1]
        # # st.write(a)
        # info = st.empty()
        # with col1:
        #     newline1=st.line_chart(a[0])
        # with col2:
        #     newline2=st.line_chart(a[1])
        # for i in range(0,10):
        #     info.success("第" + str(i + 1) + "次已完成")
        #     num=1.0-3*i+(i*i)/10
        #
        #     newline1.add_rows([num])
        #     newline2.add_rows([num+3-i*i])
        #     time.sleep(0.1)
        # a = [1.0, 3.1], [2.0, 2.1]
        # newline = st.line_chart(a)
        '''
        动态折线图演示示例
        '''
#########################################################################################################


        # # 背景图片的网址
        # img_url = 'https://bpic.588ku.com/back_pic/05/95/17/745d4b86c48a09c.jpg'  # 背景图片地址
        #
        # # 修改背景样式
        # st.markdown('''<style>.css-fg4pbf{background-image:url(''' + img_url + ''');
        # background-size:10% 2000%;background-attachment:fixed;}</style>
        # ''', unsafe_allow_html=True)
        #
        # # 侧边栏样式
        # st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div >
        # section.css-1lcbmhc.e1fqkh3o3 > div.css-1adrfps.e1fqkh3o2
        # {background:rgba(255,255,255,0.5)}</style>''', unsafe_allow_html=True)
        #
        # # 底边样式
        # st.markdown('''<style>#root > div:nth-child(1) > div > div > div > div >
        # section.main.css-1v3fvcr.egzxvld3 > div > div > div
        # {background-size:100% 100% ;background:rgba(207,207,207,0.9);
        # color:red; border-radius:5px;} </style>''', unsafe_allow_html=True)