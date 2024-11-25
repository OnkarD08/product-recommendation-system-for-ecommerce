import streamlit as st
import pickle
import pandas as pd
# import numpy as np
import time
# import requests
# from PIL import Image
import urllib.request
# import glob
import cv2 as cv

st.set_page_config(layout="wide")
# set wide layout by default

# importing the similarity matrix and dataframe generated in preprocessing from jupyter notebook
similarity = pickle.load(open('similarity2.pkl', 'rb'))

products_dict = pickle.load(open('products_cb_dict2.pkl', 'rb'))
products_df = pd.DataFrame(products_dict)

popular_dict = pickle.load(open('popular_dict2.pkl', 'rb'))
popular = pd.DataFrame(popular_dict)

corrMatrix_dict = pickle.load(open('corrMatrix2.pkl', 'rb'))
corrMatrix = pd.DataFrame(corrMatrix_dict)

# Streamlit title and sub header element
st.title('Product Recommendation System')
st.subheader('Project By: BC260 Onkar Dongare, BC254 Rushikesh Tekale, BC256 Hitesh Umare, BC258 Rutwik Wanve')

# streamlit select box element which shows all the products list from the dataset
option = st.selectbox('Search for Products', products_df['name'].values)


# EXPERIMENTAL___________________________________________
# def fetch_image(product_id):
#     image_list = []
#     for filename in glob.glob('C:/Users/Onkar D/PycharmProjects/product-recommender-system/'
#                               'images_renamed/{}.jpg'.format(product_id)):
#         im = Image.open(filename)
#         image_list.append(im)
#     return image_list

# def fetch_image(product_id):
#     response = requests.get(
#         'https://res.cloudinary.com/onkard08/image/upload/v1648056501/recsys/{}.jpg'.format(product_id))
#     data = response.json
#     print(data)
#     return  # "https://res.cloudinary.com/onkard08/image/upload/v1648055234/recsys/" + data []
# EXPERIMENTAL_______________________________________________

# ****** CONTENT BASED RECOMMENDATIONS :******

# function for getting image of recommended product.
def fetch_image(product_id):
    urllib.request.urlretrieve('https://res.cloudinary.com/onkard08/image/upload/v1648056501/recsys/{}.jpg'
                               .format(product_id), "img.jpg")
    # img = Image.open("img.jpg", mode='r')
    img = cv.imread("img.jpg")
    img = img[:, :, ::-1]  # colour correction to RGB
    return img


# Function for recommendations
def recommend_contentbased(product):
    index = products_df[products_df['name'] == product].index[0]
    distances = similarity[index]
    products_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
    recommended_products = []
    image_list = []
    for i in products_list:
        product_id = products_df.iloc[i[0]].id
        recommended_products.append(products_df['name'].iloc[i[0]])
        image_list.append(fetch_image(product_id))
    return recommended_products, image_list


# ****** COLLABORATIVE FILTERING :******
def collab(name, rating):
    similar_ratings = corrMatrix[name] * (rating - 2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    return similar_ratings


def recommend_collab():
    similar_products = pd.DataFrame()
    result = pd.DataFrame()
    id_df = pd.DataFrame()
    recs = []
    prod_id = []
    prod_images = []
    for i in range(0, 15):
        similar_products = similar_products.append(collab(option, 4.5))
    result = similar_products.transpose()
    # result.columns = ['score']
    result.index.name = 'product'
    result = result.reset_index()
    recs = result['product'].head(15).to_list()[1:11]
    for i in range(0, 10):
        id_df = products_df[products_df['name'].str.contains(recs[i], na=False, regex=False)]
        id_df = id_df['id'].to_string(index=False)
        prod_id.append(id_df)
    for i in range(0, 10):
        if len(prod_id[i]) > 5:
            prod_id[i] = prod_id[i][:5]
        prod_images.append(fetch_image(prod_id[i]))

    return recs, prod_images


# ****** POPULARITY BASED RECOMMENDATIONS :******
def recommend_popular():
    popular_products = []
    popular_images = []
    temp = []
    sponsored_name = []
    sponsored_id = []
    sponsored_image = []
    for i in range(0, 10):
        product_id = popular['id'].iloc[i]
        popular_products.append(popular['name'].iloc[i])
        popular_images.append(fetch_image(product_id))
    temp = products_df.sample()
    sponsored_id = temp['id'].to_list()
    sponsored_name = temp['name'].to_list()
    sponsored_image.append(fetch_image(sponsored_id[0]))
    return popular_products, popular_images, sponsored_image, sponsored_name


# Recommend button operation______________________________
with st.container():
    if st.button('Recommend'):
        with st.spinner('Wait for it...'):
            time.sleep(7)
        st.success('Done!')

        st.header('Sponsored:')
        recommendations, images, spon_img, spon_name = recommend_popular()
        col1, col2 = st.columns(2)
        with col1:
            st.caption(spon_name[0])
            st.image(spon_img[0])
# ______________________________________________________________________
        st.header('Trending and Most Rated Products right now:')
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
        with col1:
            st.caption(recommendations[0])
            st.image(images[0])
            # col1.image(images[0], use_column_width=False)

        with col2:
            st.caption(recommendations[1])
            st.image(images[1])

        with col3:
            st.caption(recommendations[2])
            st.image(images[2])

        with col4:
            st.caption(recommendations[3])
            st.image(images[3])

        with col5:
            st.caption(recommendations[4])
            st.image(images[4])

        with col6:
            st.caption(recommendations[5])
            st.image(images[5])

        with col7:
            st.caption(recommendations[6])
            st.image(images[6])

        with col8:
            st.caption(recommendations[7])
            st.image(images[7])

        with col9:
            st.caption(recommendations[8])
            st.image(images[8])

        with col10:
            st.caption(recommendations[9])
            st.image(images[9])

        # _______________________________________________________________________________________

        st.header('Similar items to your search:')
        recommendations, images = recommend_contentbased(option)
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)

        with col1:
            st.caption(recommendations[0])
            st.image(images[0])
            # col1.image(images[0], use_column_width=False)

        with col2:
            st.caption(recommendations[1])
            st.image(images[1])

        with col3:
            st.caption(recommendations[2])
            st.image(images[2])

        with col4:
            st.caption(recommendations[3])
            st.image(images[3])

        with col5:
            st.caption(recommendations[4])
            st.image(images[4])

        with col6:
            st.caption(recommendations[5])
            st.image(images[5])

        with col7:
            st.caption(recommendations[6])
            st.image(images[6])

        with col8:
            st.caption(recommendations[7])
            st.image(images[7])

        with col9:
            st.caption(recommendations[8])
            st.image(images[8])

        with col10:
            st.caption(recommendations[9])
            st.image(images[9])

        # _____________________________________________________________________________________
        st.header('Other users were also interested in:')
        recommendations, images = recommend_collab()
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)

        with col1:
            st.caption(recommendations[0])
            st.image(images[0])
            # col1.image(images[0], use_column_width=False)

        with col2:
            st.caption(recommendations[1])
            st.image(images[1])

        with col3:
            st.caption(recommendations[2])
            st.image(images[2])

        with col4:
            st.caption(recommendations[3])
            st.image(images[3])

        with col5:
            st.caption(recommendations[4])
            st.image(images[4])

        with col6:
            st.caption(recommendations[5])
            st.image(images[5])

        with col7:
            st.caption(recommendations[6])
            st.image(images[6])

        with col8:
            st.caption(recommendations[7])
            st.image(images[7])

        with col9:
            st.caption(recommendations[8])
            st.image(images[8])

        with col10:
            st.caption(recommendations[9])
            st.image(images[9])

    # for j in recommendations:
    # st.write(j[0], j[1])
