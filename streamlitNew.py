import streamlit as st 

#versatile display function
st.write('Hello, Streamlit!')

#widgets (input from people)
name = st.text_input('Enter your name: ')
age = st.number_input('Enter your age: ', min_value=1, max_value=120)

st.sidebar.title('Navigation')
option = st.sidebar.radio('Choose page', ['Home', 'About'])

#layout
col1, col2 = st.columns(2)
with col1:
    st.write('Left side')
with col2:
    st.write('Right side')

#feedback and messages
st.success('It worked')
st.error('Something went wrong')
st.warning('Be careful')