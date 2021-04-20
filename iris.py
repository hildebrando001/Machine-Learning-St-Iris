import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


st.write("""
# Iris Flower Prediction
This app predict Iris flower type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width  = st.sidebar.slider('Sepal width',  2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width  = st.sidebar.slider('Petal width',  0.1, 2.5, 0.2)
    data = {'Sepal length':sepal_length,
            'Sepal width' :sepal_width,
            'Petal length':petal_length,
            'Petal width' :petal_width}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()
st.subheader('User input parameters')
st.write(df)

iris = datasets.load_iris() # load the iris dataset
X = iris.data
Y = iris.target

clf = RandomForestClassifier() # classifier
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# defining columns layout
col1, col2 = st.beta_columns(2) # ([1, 2])

# col1 content
col1.subheader('Class labels')
col1.write(iris.target_names)

col1.subheader('Prediction')
col1.write(iris.target_names[prediction])

col1.subheader('Prediction probability')
col1.write(prediction_proba)

# col2 content
col2.subheader(f"Iris {iris.target_names[prediction]}")


# st.write(dir(st.sidebar))
# col2.write(dir(col1))
