# from IPython.core.display import HTML
import base64

import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title='Employee Decision Predictor',
    page_icon='icon.png'
)


html_temp = """
	<div style ="background-color:#1D9C26; padding:13px">
	<h1 style ="color:black; text-align:center; ">Streamlit Employee Classifier </h1>
	</div>
	"""
st.markdown(html_temp, unsafe_allow_html = True)


# İmages of Stay or Leave
image = Image.open("stay_or_go.jpg")
st.image(image, use_column_width=True)


# Display Dataset
st.header("_HR Dataset_")
df = pd.read_csv('HR_Dataset.csv')
st.write(df.sample(5))


def user_input_features():
    st.sidebar.title("```Please Enter Specs to Estimate Whether Your Employee Will Stay```")

    satisfaction_level = st.sidebar.slider("Satisfaction Level", 0.09, 1.0, 0.5)
    last_evaluation = st.sidebar.slider("Evaluation Score", 0.360, 1.0, 0.5)
    number_project = st.sidebar.selectbox('Number of Projects', [2, 3, 4, 5, 6, 7])
    average_montly_hours = st.sidebar.number_input('Avg. Monthly Working Hours', min_value=96, max_value=310, value=200, step=1)
    time_spend_company = st.sidebar.selectbox('Years in the Company', [2, 3, 4, 5, 6, 7, 8, 10])
    agree1 = st.sidebar.checkbox('Employee had a work accident')
    if agree1:
        work_accident = 1
    else:
        work_accident = 0
    agree2 = st.sidebar.checkbox('The employee received a promotion in the last 5 years')
    if agree2:
        promotion_last_5years = 1
    else:
        promotion_last_5years = 0
    choice = st.sidebar.selectbox('Department',
        ("Information Technology (IT)",
        "Research and Development (R & D)",
        "Accounting",
        "Human Resources",
        "Management",
        "Marketing",
        "Product Management",
        "Sales",
        "Support",
        "Technical"),
    )
    if choice == "Information Technology (IT)":
        departments = "IT"
    elif choice == "Research and Development (R & D)":
        departments = "RandD"
    elif choice == "Accounting":
        departments = "accounting"
    elif choice == "Human Resources":
        departments = "hr"
    elif choice == "Management":
        departments = "management"
    elif choice == "Marketing":
        departments = "marketing"
    elif choice == "Product Management":
        departments = "product_mng"
    elif choice == "Sales":
        departments = "sales"
    elif choice == "Support":
        departments = "support"
    elif choice == "Technical":
        departments = "technical" 
    choice2 = st.sidebar.radio('Salary Level', ["Low", "Medium", "High"])
    if choice2 == "Low":
        salary = "low"
    elif choice2 == "Medium":
        salary = "medium"
    elif choice2 == "High":
        salary = "high"
    new_df = {"satisfaction_level":satisfaction_level,
              "last_evaluation":last_evaluation,
              "number_project":number_project,
              "average_montly_hours":average_montly_hours,
              "time_spend_company":time_spend_company,
              "work_accident":work_accident,
              "promotion_last_5years":promotion_last_5years,
              "departments":departments,
              "salary":salary}
    features = pd.DataFrame(new_df, index=[0])
    return features
input_df = user_input_features()





resa = input_df.rename(columns={"satisfaction_level":"Satisfaction Level",
              "last_evaluation":"Evaluation Score",
              "number_project":"# of Projects",
              "average_montly_hours":"Monthly Hours",
              "time_spend_company":"Years in Company",
              "work_accident":"Work Accident",
              "promotion_last_5years":"Received Promotion",
              "departments":"Department",
              "salary":"Salary Level"})
resa['Monthly Hours'] = resa['Monthly Hours'].astype('int')
resa['Work Accident'] = resa['Work Accident'].map({0:'No', 1:'Yes'})
resa['Received Promotion'] = resa['Received Promotion'].map({0:'No', 1:'Yes'})
resa['Department'] = resa['Department'].map({'IT':'Information Technology',
                        'RandD': 'R & D',
                        'accounting':'Accounting',
                        'hr':'Human Resources',
                        'management':'Management',
                        'marketing':'Marketing',
                        'product_mng':'Product Management',
                        'sales':'Sales',
                        'support':'Support',
                        'technical':'Technical'})
resa['Salary Level'] = resa['Salary Level'].map({'low':'Low', 'medium':'Medium', 'high':'High'})

# Selected Display (Geliştirilecek)

st.markdown("""<h3 style='text-align:left; color:#1D9C26;'>Your Selected</h3>
""", unsafe_allow_html=True)
st.write(resa)

xgb_model = pickle.load(open('KNN_model', 'rb'))
knn_model = pickle.load(open('KNN_model', 'rb'))
rf_model = pickle.load(open('RF_model', 'rb'))
ann_model = pickle.load(open('RF_model', 'rb'))

pred_xgb = xgb_model.predict(input_df)
pred_xgb = ['Left' if pred_xgb == 1 else 'Stayed']
pred_knn = knn_model.predict(input_df)
pred_knn = ['Left' if pred_knn == 1 else 'Stayed']
pred_rf = rf_model.predict(input_df)
pred_rf = ['Left' if pred_rf == 1 else 'Stayed']
pred_ann = ann_model.predict(input_df)
pred_ann = ['Left' if ann_model == 1 else 'Stayed']

st.title('')


#  Model Seçimi
st.markdown("""<h3 style='text-align:left; color:#1D9C26;'>Choose Your Model</h3>
""", unsafe_allow_html=True)

model_selected = st.selectbox('Pick one model and get your prediction',
        ['K-Nearest Neighbors', 'Random Forest', 'XGBoosting', 'Artificial Neural Network'])


if model_selected == 'K-Nearest Neighbors':
    x1 = pred_xgb
elif model_selected == 'XGBoosting':
    x1 = pred_knn
elif model_selected == 'Random Forest':
    x1 = pred_rf
else: x1 = pred_ann


x2 = ['It looks like s/he is leaving. ' if x1[0] == 'Left' else 'Relax, s/he is still yours.']

if st.button('Predict'):
    if model_selected == 'XGBoosting':
        st.metric('XGBoosting Prediction', value=x2[0])
    elif model_selected == 'K-Nearest Neighbors':
        st.metric('KNN Prediction', value=x2[0])
    elif model_selected == 'Random Forest':
        st.metric('Random Forest Prediction', value=x2[0])
    else: st.metric('Artificial Neural Network', value=x2[0])

    if x1[0] == 'Left':
        st.image('left-no.jpg', width=500)
    else:
        st.image('smile-ok.jpg', width=500)

# To hide Streamlit style
hide_st_style = """
        <style>
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        header {visibility:hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)

# to add a background image
# @st.cache(allow_output_mutation=True)
# def get_base64(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_bg(png_file):
#     bin_str = get_base64(png_file)
#     page_bg_img = """
#         <style>
#         .stApp {
#         background-image: url("data:image/png;base64,%s");
#         background-size: cover;
#         }
#         </style>
#     """ % bin_str
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     return
    

# set_bg('background_image.png')

