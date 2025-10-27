import streamlit as st 

#versatile display function
st.write('Hello, Streamlit!')

#widgets (input from people)
name = st.text_input('Enter your name: ')
age = st.number_input('Enter your age: ', min_value=1, max_value=120)