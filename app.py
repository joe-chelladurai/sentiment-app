import streamlit as st
from transformers import pipeline
import pandas as pd

st.title('Analyse Sentiments with Transformers')
st.write('Joe Chelladurai')
st.write('It may a few seconds to run the model.*')

a = st.radio("Dataset:", ['Example', 'Upload'], 0)

d = {'text': ["I felt that they were excluding me", 
              "This was definitely the best place where I felt welcome", 
              "There were some hard days, but mostly good days"]}

example_df = pd.DataFrame(data = d)

if a == 'Example':
    st.dataframe(example_df)
else:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        st.dataframe(uploaded_df)
    else:
        st.write("Please upload a dataset")
  
submit = st.button('Analyse Sentiments')

if submit:
    classifier = pipeline("sentiment-analysis")
    if a == 'Example':
        df = example_df
    else:
        df = uploaded_df

    df = (
    df
    .assign(sentiment = lambda x: x.iloc[:, 0].apply(lambda s: classifier(s)))
    .assign(
         label = lambda x: x['sentiment'].apply(lambda s: (s[0]['label'])),
         score = lambda x: x['sentiment'].apply(lambda s: (s[0]['score']))
    )
)
    st.dataframe(df)
   