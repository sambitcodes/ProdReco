import streamlit as st
import pandas as pd
import pickle
import re
import numpy as np
from PIL import Image
import time

st.set_page_config(page_title = "ProdReco",page_icon="☢️",layout="wide",initial_sidebar_state="collapsed")

#Getting Data
data = pd.read_csv(r'data/flipkart-data.csv')


#Getting Similarity Models
similarity_w2v = pickle.load(open(r'models/similarity_w2v.pkl','rb'))


#Fetch Product Images
def poster_string(ind):
    text = data['image'][ind]
    text = str(text)
    text = text.replace('"','')
    pattern = re.compile('[\([{})\]]')
    text = pattern.sub(r'',text)
    pos = text.split(",")
    return pos[0]


#Fetch Recommended Products, Product image Links, Product Index and Similarity list
def recommend(product,similarity):
    product_index = data[data['product_name'] == product].index[0]
    distances = similarity[product_index]
    product_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_prods = []
    posters_links = []
    indexes = []
    sim_list= []
    for x in product_list:
        recommended_prods.append(data.iloc[x[0]].product_name)
        posters_links.append(poster_string(x[0]))
        indexes.append(x[0])
        prod_sim = np.round((x[1] * 100), 2)
        sim_list.append(prod_sim)

    return recommended_prods,posters_links,indexes,sim_list


#Logo and Descr
with st.container():
    logo_col, desc_col = st.columns([0.2, 0.8])
    with logo_col:
        with st.container():
            open_img = Image.open('img/img1.png')
            st.image(open_img, use_container_width=True)

    with desc_col:
        with st.container(border=True):
            st.subheader('A minimalistic *:red[Product Recommendation]* system')
            st.markdown('''Inspired by *:blue-background[Flipkart]* ''')


#Product Input container
with st.container(border=True):
    product_names_list = set(data['product_name'].values)
    selected_product = st.selectbox('Choose a product from list',product_names_list, index = None)
st.button('Recommend', key='recommend_button')
if st.session_state.get('recommend_button'):

    #progress bar
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
    st.balloons()

    product_index = data[data['product_name'] == selected_product].index[0]
    product_poster = poster_string(product_index)


    #selected product details
    with st.expander("Selected Product Details"):

        image_col,product_col = st.columns([0.2, 0.8])
        with image_col:
            with st.container(height=300):
                st.image(product_poster)

        with product_col:
            with st.container():
                
                query, answer = st.columns([0.2, 0.8])
                with query:
                    with st.container(border=True):
                        st.markdown(':red[Selected Product]')
                        st.markdown(':red[Product Id]')
                        st.markdown(':red[Retail Price]')
                        st.markdown(':red[Discounted Price]')
                        st.markdown(':red[Description]')
                with answer:
                    with st.container(border=True):
                        st.markdown(':blue[{}]'.format(selected_product))
                        st.markdown(':blue[{}]'.format(data['pid'][product_index]))
                        st.markdown(':green[Rs. {}]'.format(data['retail_price'][product_index]))
                        st.markdown(':green[Rs. {}]'.format(data['discounted_price'][product_index]))
                        st.markdown(':blue[{}]'.format(data['description'][product_index]))

    #recommended products details
    products,posters,index,similarity = recommend(selected_product,similarity_w2v)
    name = np.array(products)
    similarity = np.array(similarity)
    retail_price = []
    discounted_price = []
    product_id = []
    for indx in range(len(index)):
        discounted_price.append(data['discounted_price'][index[indx]])
        retail_price.append(data['retail_price'][index[indx]])
        product_id.append(data['pid'][index[indx]])
        
    prod1, prod2, prod3, prod4, prod5 = st.columns(5)

    with prod1:
        with st.container(border=True):
            st.markdown('ID:  :blue[{}]'.format(data['pid'][index[0]]))

        st.text(products[0])

        with st.container(height=300):
            st.image(posters[0])

        st.write(similarity[0])

        st.markdown('Retail Price :   :red[{}]'.format(data['retail_price'][index[0]]))
        st.markdown('Discounted Price :   :green[{}]'.format(data['discounted_price'][index[0]]))


        st.markdown('Description :  :blue[{}]'.format(data['description'][index[0]]))
        st.markdown("Rating :  :blue[{}]".format(data['product_rating'][index[0]]))
        

    with prod2:
        with st.container(border=True):
            st.markdown('ID:  :blue[{}]'.format(data['pid'][index[1]]))

        st.text(products[1])

        with st.container(height=300):
            st.image(posters[1])

        st.write(similarity[1])

        st.markdown('Retail Price :   :red[{}]'.format(data['retail_price'][index[1]]))
        st.markdown('Discounted Price :   :green[{}]'.format(data['discounted_price'][index[1]]))


        st.markdown('Description :  :blue[{}]'.format(data['description'][index[1]]))
        st.markdown("Rating :  :blue[{}]".format(data['product_rating'][index[1]]))
        

    with prod3:
        with st.container(border=True):
            st.markdown('ID:  :blue[{}]'.format(data['pid'][index[2]]))

        st.text(products[2])

        with st.container(height=300):
            st.image(posters[2])

        st.write(similarity[2])

        st.markdown('Retail Price :   :red[{}]'.format(data['retail_price'][index[2]]))
        st.markdown('Discounted Price :   :green[{}]'.format(data['discounted_price'][index[2]]))


        st.markdown('Description :  :blue[{}]'.format(data['description'][index[2]]))
        st.markdown("Rating :  :blue[{}]".format(data['product_rating'][index[2]]))
        

    with prod4:
        with st.container(border=True):
            st.markdown('ID:  :blue[{}]'.format(data['pid'][index[3]]))

        st.text(products[3])

        with st.container(height=300):
            st.image(posters[3])

        st.write(similarity[3])

        st.markdown('Retail Price :   :red[{}]'.format(data['retail_price'][index[3]]))
        st.markdown('Discounted Price :   :green[{}]'.format(data['discounted_price'][index[3]]))


        st.markdown('Description :  :blue[{}]'.format(data['description'][index[3]]))
        st.markdown("Rating :  :blue[{}]".format(data['product_rating'][index[3]]))
        
    with prod5:
        with st.container(border=True):
            st.markdown('ID:  :blue[{}]'.format(data['pid'][index[4]]))

        st.text(products[4])

        with st.container(height=300):
            st.image(posters[4])

        st.write(similarity[4])

        st.markdown('Retail Price :   :red[{}]'.format(data['retail_price'][index[4]]))
        st.markdown('Discounted Price :   :green[{}]'.format(data['discounted_price'][index[4]]))


        st.markdown('Description :  :blue[{}]'.format(data['description'][index[4]]))
        st.markdown("Rating :  :blue[{}]".format(data['product_rating'][index[4]]))
        

                
