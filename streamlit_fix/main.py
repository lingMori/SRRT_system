
import streamlit as st
from streamlit_option_menu import option_menu
import os
from dashboard import Dashboard
from about import About
from data_training import DataTraining


st.set_page_config(
    page_title="信道建模APP",
    page_icon="11.ico",
    layout="wide"
)


class Model:
    menuTitle = "信道建模APP"
    option1 = "Dashboard"
    option2 = "Data_training"
    option3 = "About"

    menuIcon = "menu-up"
    icon1 = "speedometer"
    icon2 = "activity"
    icon3 = "motherboard"
    icon4 = "graph-up-arrow"
    icon5 = "clipboard-data"
    icon6 = "gear"
    icon7 = "chat"


def view(model):
    with st.sidebar:
        menuItem = option_menu(model.menuTitle,
                               [model.option1, model.option2, model.option3],
                               icons=[model.icon4, model.icon3, model.icon6],
                               menu_icon=model.menuIcon,
                               default_index=0,

                               styles={
                                   "container": {"padding": "5!important", "background-color": "#050505"},
                                   "icon": {"color": "black", "font-size": "25px"},
                                   "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px",
                                                "--hover-color": "#eee"},
                                   "nav-link-selected": {"background-color": "#037ffc"},
                               })
    if menuItem == model.option1:
        Dashboard().view(Dashboard.Model())

    if menuItem == model.option2:
        DataTraining().view(DataTraining.Model())

    if menuItem == model.option3:
        About().view(About.Model())


view(Model())
