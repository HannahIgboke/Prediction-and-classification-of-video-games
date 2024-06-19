#Step 1: import all necessary libraries
import pandas as pd
import numpy as np
import streamlit as st
import joblib

#Step 2: Load model, scaler and columns
model = joblib.load('LR_model.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl') #to get the columns used in the training data

#Step 3: Function to take the input from users
def predict_sales(platform, genre, publisher, critic_score, critic_count, user_score, user_count, developer,  rating, release_era):
    input_data = pd.DataFrame({
        'platform_'+ platform: [1],
        'genre_' + genre: [1],
        'publisher_' + publisher: [1],
        'critic_score': [critic_score],
        'critic_count': [critic_count],
        'user_score': [user_score],
        'user_count': [user_count],
        'developer_' + developer: [1],
        'rating_' + rating: [1],
        'release_era_' + release_era: [1]
    })

    #Step 4: Ensure all columns are present
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    input_data = input_data[columns]

    #Step 5: Scale numerical columns
    num_cols = ['critic_score', 'critic_count', 'user_score', 'user_count']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    #Step 6: Make predictions
    prediction = model.predict(input_data)

    return prediction

#Step 7: Streamlit UI
st.set_page_config(page_title='Video game sales predictor', layout='wide')


st.markdown("<h1 style='text-align: center; color:blue;'> 🎮Video game sales predictor</h1>", unsafe_allow_html=True)

st.markdown("""
            <div style='text-align: center;'>
            Welcome! Input the details below to predict the estimated sales of your game. 
            </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    platform = st.selectbox('Platform', ['PS2', 'X360', 'PS3', 'Wii', 'DS', 'PSP', 'PS', 'PC', 'XB', 'Others'])
    genre = st.selectbox('Genre', ['Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Puzzle', 'Racing', 'Role-Playing', 'Shooter', 'Simulation', 'Sports', 'Strategy'])
    publisher = st.selectbox('Publisher', ['Electronic Arts', 'Activision', 'Namco Bandai Games', 'Ubisoft', 'Konami Digital Entertainment', 'THQ', 'Nintendo', 'Sony Computer Entertainment', 'Sega', 'Others'])
    developer=st.selectbox('Developer', ['Ubisoft', 'EA Sports', 'EA Canada', 'Konami', 'Capcom', 'EA Tiburon', 'Electronic Arts', 'Ubisoft Montreal', 'Others'])
    rating = st.selectbox('Rating', ['E', 'M', 'T', 'E10+', 'K-A', 'AO', 'EC', 'RP'])
   
with col2:
    critic_score = st.number_input('Critic score', min_value=0.00, max_value=100.00)
    critic_count = st.number_input('Critic count')
    user_score = st.number_input('User score', min_value =0.00, max_value=10.00)
    user_count = st.number_input('User count')
    release_era = st.selectbox('Release_era', ['pre-2000s', '2010-2010', 'post-2010'])

if st.button('Predict'):
    prediction = predict_sales(platform, genre, publisher, critic_score, critic_count, user_score, user_count, developer,  rating, release_era)
    if prediction == 'High':
        st.success('Based on your inputs, the game is estimated to bring in high sales (more than 1 million game copies sales).' )
    else:
        st.info('Low sales. The game is estimated to bring in little sales. Consider revising your choice for the game features.')
        st.markdown('Visit [this notebook](https://github.com/HannahIgboke/Prediction-and-classification-of-video-games/blob/main/Notebooks/Exploratory%20Data%20Analysis.ipynb) to gain understanding into the most important features to include!')