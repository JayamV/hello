import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

def main():

    st.markdown(""" 
    <style>
    body {
        background: linear-gradient(135deg, #edf2f4, #8ecae6);
        font-family: 'Arial', sans-serif;
    }
    .main-title {
        font-size: 2.5rem;
        color: #023e8a;
        text-align: center;
        font-weight: bold;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #0077b6;
        text-align: center;
        margin-bottom: 20px;
    }
    footer {
        color: #6c757d;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(""" 
    ### Why early detection of Breast Cancer is important?
    #### 💪Higher Survival Rates: 
    Early detection increases the chances of successful treatment and long-term survival.
    #### 🌿Less Aggressive Treatment:
    Timely diagnosis often allows for simpler and less invasive treatment options.
    #### 🌐Lower Risk of Metastasis: 
    Identifying cancer early reduces the likelihood of it spreading to other parts of the body.
    #### 💰Reduced Healthcare Costs: 
    Early treatment is typically more cost-effective than managing advanced-stage cancer.

    """)
 

    st.markdown(""" 
    ### About AdaBoost Classifier
    AdaBoost (Adaptive Boosting) is a powerful ensemble learning algorithm that combines multiple weak learners to create a strong classifier, enhancing the accuracy of predictions.
    """)
 



if __name__ == "__main__":
    main()

    
