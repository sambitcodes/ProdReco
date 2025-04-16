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
        

                




#         prod_index = data[data['product_name'] == selected_product].index[0]
#         prod_poster = poster_string(prod_index)

#         #selected product details
#         with st.expander("PRODUCT DETAILS"):
#             st.error('SELECTED PRODUCT :::: {}'.format(selected_product))
#             st.image(prod_poster)
#             st.success('RETAIL PRICE =  Rs.  {} '.format(data['retail_price'][prod_index]))
#             st.success('DISCOUNTED PRICE =  Rs.  {} '.format(data['discounted_price'][prod_index]))
#             st.error('DESCRIPTION  :::   {}'.format(data['description'][prod_index]))

#         #recommended products details
#         with st.expander("RECOMMENDED PRODUCTS"):
#             st.success('PRODUCTS SUCCESFULLY RECOMMENDED')
#             products,posters,indx,sim = recommend(selected_product,similarity_tfidf)
#             nam = np.array(products)
#             sim = np.array(sim)
#             ret_price = []
#             disc_price = []
#             pid = []
#             for i in range(len(indx)):
#                 disc_price.append(data['discounted_price'][indx[i]])
#                 ret_price.append(data['retail_price'][indx[i]])
#                 pid.append(data['pid'][indx[i]])


#             col1, col2, col3, col4, col5 = st.columns(5)
#             with col1:
#                 st.text(products[0])
#                 st.image(posters[0])
#                 st.write(data['discounted_price'][indx[0]])
#                 st.write(sim[0])
#                 st.text(data['description'][indx[0]])
#             with col2:
#                 st.text(products[1])
#                 st.image(posters[1])
#                 st.write(data['discounted_price'][indx[1]])
#                 st.write(sim[1])
#                 st.text(data['description'][indx[1]])
#             with col3:
#                 st.text(products[2])
#                 st.image(posters[2])
#                 st.write(data['discounted_price'][indx[2]])
#                 st.write(sim[2])
#                 st.text(data['description'][indx[2]])
#             with col4:
#                 st.text(products[3])
#                 st.image(posters[3])
#                 st.write(data['discounted_price'][indx[3]])
#                 st.write(sim[3])
#                 st.text(data['description'][indx[3]])
#             with col5:
#                 st.text(products[4])
#                 st.image(posters[4])
#                 st.write(data['discounted_price'][indx[4]])
#                 st.write(sim[4])
#                 st.text(data['description'][indx[4]])

#         dat = np.reshape([[pid], [nam], [sim]], (3, 5))
#         dat2 = np.reshape([[pid], [ret_price], [disc_price]], (3, 5))
#         df = pd.DataFrame(dat.T, columns=['Product Id', 'Recommended Product', 'Similarity(%age)'],
#                           index=[0, 1, 2, 3, 4])
#         df2 = pd.DataFrame(dat2.T, columns=['Product Id', 'Retail Price', 'Discounted Price (Rs.)'],
#                            index=[0, 1, 2, 3, 4])

#         #similarity details
#         with st.expander("SEE SIMILARITY STATUS"):
#             st.header('SIMILARITY TABLE', anchor=None)
#             st.dataframe(df)
#             chart_data = pd.DataFrame(sim, columns=["Similarity %"])
#             st.bar_chart(chart_data)

#         #prices details
#         with st.expander("SEE PRICES TABLE"):
#             st.header('PRICE TABLE', anchor=None)
#             st.dataframe(df2)
#             chart_data2 = pd.DataFrame(ret_price, columns=["Retail"])
#             st.bar_chart(chart_data2)
#             chart_data3 = pd.DataFrame(disc_price, columns=["Discounted"])
#             st.bar_chart(chart_data3)



# #BagOfWords Vectorizer
# elif add_selectbox == 'BagOfWords':

#     product_names_list = set(data['product_name'].values)
#     selected_product = st.selectbox('Choose A product', product_names_list, index = None)

#     if st.button('Recommend'):
#         my_bar = st.progress(0)

#         for percent_complete in range(100):
#             time.sleep(0.01)
#             my_bar.progress(percent_complete + 1)
#         st.balloons()

#         prod_index = data[data['product_name'] == selected_product].index[0]
#         prod_poster = poster_string(prod_index)

#         with st.expander("PRODUCT DETAILS"):
#             st.error('SELECTED PRODUCT :::: {}'.format(selected_product))
#             st.image(prod_poster)
#             st.success('RETAIL PRICE =  Rs.  {} '.format(data['retail_price'][prod_index]))
#             st.success('DISCOUNTED PRICE =  Rs.  {} '.format(data['discounted_price'][prod_index]))
#             st.error('DESCRIPTION  :::   {}'.format(data['description'][prod_index]))

#         with st.expander("RECOMMENDED PRODUCTS"):
#             st.success('PRODUCTS SUCCESFULLY RECOMMENDED')
#             products, posters, indx, sim = recommend(selected_product,similarity_bow)
#             nam = np.array(products)
#             sim = np.array(sim)
#             ret_price = []
#             disc_price = []
#             pid = []
#             for i in range(len(indx)):
#                 disc_price.append(data['discounted_price'][indx[i]])
#                 ret_price.append(data['retail_price'][indx[i]])
#                 pid.append(data['pid'][indx[i]])

#             col1, col2, col3, col4, col5 = st.columns(5)
#             with col1:
#                 st.text(products[0])
#                 st.image(posters[0])
#                 st.write(data['discounted_price'][indx[0]])
#                 st.write(sim[0])
#                 st.text(data['description'][indx[0]])
#             with col2:
#                 st.text(products[1])
#                 st.image(posters[1])
#                 st.write(data['discounted_price'][indx[1]])
#                 st.write(sim[1])
#                 st.text(data['description'][indx[1]])
#             with col3:
#                 st.text(products[2])
#                 st.image(posters[2])
#                 st.write(data['discounted_price'][indx[2]])
#                 st.write(sim[2])
#                 st.text(data['description'][indx[2]])
#             with col4:
#                 st.text(products[3])
#                 st.image(posters[3])
#                 st.write(data['discounted_price'][indx[3]])
#                 st.write(sim[3])
#                 st.text(data['description'][indx[3]])
#             with col5:
#                 st.text(products[4])
#                 st.image(posters[4])
#                 st.write(data['discounted_price'][indx[4]])
#                 st.write(sim[4])
#                 st.text(data['description'][indx[4]])

#         dat = np.reshape([[pid], [nam], [sim]], (3, 5))
#         dat2 = np.reshape([[pid], [ret_price], [disc_price]], (3, 5))
#         df = pd.DataFrame(dat.T, columns=['Product Id', 'Recommended Product', 'Similarity(%age)'],index=[0, 1, 2, 3, 4])
#         df2 = pd.DataFrame(dat2.T, columns=['Product Id', 'Retail Price', 'Discounted Price (Rs.)'],index=[0, 1, 2, 3, 4])


#         with st.expander("SEE SIMILARITY STATUS"):
#             st.header('SIMILARITY TABLE', anchor=None)
#             st.dataframe(df)
#             chart_data = pd.DataFrame(sim, columns=["Similarity %"])
#             st.bar_chart(chart_data)

#         with st.expander("SEE PRICES TABLE"):
#             st.header('PRICE TABLE', anchor=None)
#             st.dataframe(df2)
#             chart_data2 = pd.DataFrame(ret_price, columns=["Retail"])
#             st.bar_chart(chart_data2)
#             chart_data3 = pd.DataFrame(disc_price, columns=["Discounted"])
#             st.bar_chart(chart_data3)


# elif add_selectbox == 'TF-IDF v/s BagOfWords':
#     # choose product from menu
#     product_names_list = set(data['product_name'].values)
#     selected_product = st.selectbox('Choose A product', product_names_list, index = None)

#     # recommend product
#     if st.button('Recommend'):
#         my_bar = st.progress(0)

#         # progress bar
#         for percent_complete in range(100):
#             time.sleep(0.01)
#             my_bar.progress(percent_complete + 1)
#         st.balloons()

#         prod_index = data[data['product_name'] == selected_product].index[0]
#         prod_poster = poster_string(prod_index)

#         # selected product details
#         with st.expander("PRODUCT DETAILS"):
#             st.error('SELECTED PRODUCT :::: {}'.format(selected_product))
#             st.image(prod_poster)
#             st.success('RETAIL PRICE =  Rs.  {} '.format(data['retail_price'][prod_index]))
#             st.success('DISCOUNTED PRICE =  Rs.  {} '.format(data['discounted_price'][prod_index]))
#             st.error('DESCRIPTION  :::   {}'.format(data['description'][prod_index]))

#         # recommended products details
#         with st.expander("RECOMMENDED PRODUCTS"):

#             products, posters, indx, sim = recommend(selected_product, similarity_tfidf)
#             products2, posters2, indx2, sim2 = recommend(selected_product, similarity_bow)

#             nam = np.array(products)
#             sim = np.array(sim)
#             ret_price = []
#             disc_price = []
#             pid = []

#             nam2 = np.array(products2)
#             sim2 = np.array(sim2)
#             ret_price2 = []
#             disc_price2 = []
#             pid2 = []

#             for i in range(len(indx)):
#                 disc_price.append(data['discounted_price'][indx[i]])
#                 ret_price.append(data['retail_price'][indx[i]])
#                 pid.append(data['pid'][indx[i]])

#             for i in range(len(indx2)):
#                 disc_price2.append(data['discounted_price'][indx2[i]])
#                 ret_price2.append(data['retail_price'][indx2[i]])
#                 pid2.append(data['pid'][indx2[i]])

#             st.success('TFIDF - RECOMMENDATION')
#             col1, col2, col3, col4, col5 = st.columns(5)
#             with col1:
#                 st.text(products[0])
#                 st.image(posters[0])
#                 st.write(sim[0])
#             with col2:
#                 st.text(products[1])
#                 st.image(posters[1])
#                 st.write(sim[1])
#             with col3:
#                 st.text(products[2])
#                 st.image(posters[2])
#                 st.write(sim[2])
#             with col4:
#                 st.text(products[3])
#                 st.image(posters[3])
#                 st.write(sim[3])
#             with col5:
#                 st.text(products[4])
#                 st.image(posters[4])
#                 st.write(sim[4])
#             st.success('BagOfWords - RECOMMENDATION')
#             col1, col2, col3, col4, col5 = st.columns(5)
#             with col1:
#                 st.text(products2[0])
#                 st.image(posters2[0])
#                 st.write(sim2[0])
#             with col2:
#                 st.text(products2[1])
#                 st.image(posters2[1])
#                 st.write(sim2[1])
#             with col3:
#                 st.text(products2[2])
#                 st.image(posters2[2])
#                 st.write(sim2[2])
#             with col4:
#                 st.text(products2[3])
#                 st.image(posters2[3])
#                 st.write(sim2[3])
#             with col5:
#                 st.text(products2[4])
#                 st.image(posters2[4])
#                 st.write(sim2[4])

#         dat = np.reshape([[pid],[pid2], [nam], [nam2]], (4, 5))
#         dat2 = np.reshape([[sim], [sim2]], (2, 5))
#         df = pd.DataFrame(dat.T, columns=['TFIDF Product Id','BOW Product Id', 'Tfidf Recommendation','Bow Recommendation'],index=[0, 1, 2, 3, 4])
#         df2 = pd.DataFrame(dat2.T, columns=['TFIDF Similarity(%)', 'BoW Similarity(%)'],index=[0, 1, 2, 3, 4])

#         # similarity details
#         with st.expander("SEE COMPARISON "):
#             st.header('RECOMMENDATION', anchor=None)
#             st.dataframe(df)
#             st.header('SIMILARITIES', anchor=None)
#             st.dataframe(df2)
#             chart_data = pd.DataFrame(np.reshape([[sim], [sim2]], (5, 2)), columns=['TFIDF Similarity(%)', 'BoW Similarity(%)'])
#             st.line_chart(chart_data)
