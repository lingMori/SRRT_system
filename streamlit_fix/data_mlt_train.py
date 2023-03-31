import streamlit as st
import numpy as np
import random


def train_mlt(epoch):
    trloss = np.zeros(6)
    teloss = np.zeros(7)
    trloss[0] = random.uniform(100, 200)
    for j in range(0, 5):
        trloss[j] = random.uniform(100, 200)

    for j in range(0, 6):
        trloss[j] = random.uniform(1, 10)
    st.write(f"Epoch : {epoch + 1} (trainloss) phi :{trloss[0]:.4f}   theta :{trloss[1]:.4f}   K :{trloss[2]:.4f}\
       P :{trloss[3]:.4f}   T :{trloss[4]:.4f}   L :{trloss[5]:.4f}")
    st.write(f"Epoch : {epoch + 1} (testloss)  phi :{teloss[0]:.4f}   theta :{teloss[1]:.4f}   K :{teloss[2]:.4f}\
       P :{teloss[3]:.4f}   T :{teloss[4]:.4f}   TPR :{teloss[5]:.4f}   FPR :{teloss[6]:.4f}\n")
