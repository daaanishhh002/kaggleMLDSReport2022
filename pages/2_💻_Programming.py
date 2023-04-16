import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Kaggle Machine Learning & Data Science Report 2022", layout="wide", page_icon='👨‍💻')

plt.style.use('ggplot')
sns.set_theme(palette = 'deep')

url = "https://raw.githubusercontent.com/daaanishhh4218/kaggleMLDSReport2022/main/kaggle_survey_2022_responses.csv" 
df = pd.read_csv(url)

df = df.rename(columns = {'Duration (in seconds)': 'Q1'})

schema = df.iloc[0, :]
schema = pd.DataFrame(schema)
schema = schema.rename(columns = {0: 'Question'})

df = df.drop(axis = 0, index = 0)

def to_transform(start, stop, fname, sname, col_name, data = df):
    """
    Collapses multiple columns into 2 columns
    with one being the category while the
    other is the frequency of the category.
    """
    fname = data.loc[:, start: stop]
    sname = pd.Series([])
    
    def values(x):
        return fname[x].value_counts()

    for col in fname.columns:
        x = values(col)
        sname = pd.concat([sname, x], axis = 0)
        
    sname = sname.reset_index(name = 'Count')
    sname = sname.rename(columns = {'index': col_name})
    sname = sname.sort_values('Count', ascending = False, ignore_index = True)
    
    return sname

def to_show(sname, title):
    """
    Plots horizontal barplot.
    """
    fig, ax = plt.subplots()
    sns.barplot(data = sname, x = 'Count', y = sname.columns[0], palette = 'deep', ax = ax)
    
    title = title.title()
    plt.title(title, weight = 'bold')
    plt.xlabel('Count', labelpad = 10)
    plt.ylabel(sname.columns[0])
    
    return fig

st.header('Some Infographics Related To Programming')
st.markdown('---')


fig, ax = plt.subplots()
df['Q11'] = df['Q11'].str.replace('never', 'never\n')
sort = list(df['Q11'].value_counts().index)
sns.countplot(data = df, y = 'Q11', order = sort)
plt.title('Coding For How Long?', weight = 'bold')
plt.xlabel(None, labelpad = 15)
plt.ylabel(None)
plt.tight_layout()
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.write('Caption for second chart')
    with col2:
        st.pyplot(fig)

st.write('\n')

fig, ax = plt.subplots()
lang = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Language')
sns.barplot(data = lang, y = 'Language', x = 'Count', palette = 'deep')
plt.title('Top Languages Used On A Regular Basis', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None, labelpad = 10)   
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)
    with col2:
        st.write('Caption for second chart')

st.write('\n')

fig, ax = plt.subplots()
visual = to_transform('Q15_1', 'Q15_15', 'v_df', 'visual', 'Visualisation Library')
visual['Visualisation Library'] = visual['Visualisation Library'].drop(4)
fig= to_show(sname = visual, title = 'Top Visualisation Libraries Used On A regualar Basis')
plt.xlabel(None)
plt.ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col2:
        st.pyplot(fig)
    with col1:
        st.write('Caption for second chart')

st.write('\n')

fig, ax = plt.subplots()
df['Q16'] = df['Q16'].str.replace('machine', 'machine\n')
sns.countplot(data = df, y = 'Q16', palette = 'deep')
plt.title('Using Machine Learning Methods For How Long?', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)
    with col2:
        st.write('Caption for second chart')
        
def container(col1, col2):
    
    with st.container():
        col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)
    with col2:
        st.write('Caption for second chart')
        
fig, ax = plt.subplots()
ml_frame = to_transform('Q17_1', 'Q17_15', 'ml_df', 'ml_frame', 'ML Framework')
ml_frame['ML Framework'] = ml_frame['ML Framework'].drop(6)
fig = to_show(ml_frame, 'regularly used machine learning frameworks')
plt.xlabel(None)
plt.ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col2:
        st.pyplot(fig)
    with col1:
        st.write('Caption for second chart')

st.write('\n')

fig, ax = plt.subplots()
ml_algo = to_transform('Q18_1', 'Q18_14', 'ml_df', 'ml_algo', 'ML Algorithm')
ml_algo['ML Algorithm'] = ml_algo['ML Algorithm'].str.replace('(', '\n(')
ml_algo['ML Algorithm'] = ml_algo['ML Algorithm'].drop([9, 13])
fig = to_show(ml_algo, 'top regularly used machine learning algorithms')
plt.xlabel(None)
plt.ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)
    with col2:
        st.write('Caption for second chart')

st.write('\n')

fig, ax = plt.subplots(1, 1, figsize = (25, 15))
cv = to_transform('Q19_1', 'Q19_8', 'cv_df', 'cv', 'CV Algorithm')
cv['CV Algorithm'] = cv['CV Algorithm'].str.replace('(', '\n(')
cv['CV Algorithm'] = cv['CV Algorithm'].drop([4, 7])
fig = to_show(cv, 'computer vision algorithms used on a regular basis')
plt.xlabel(None)
plt.ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col2:
        st.pyplot(fig)
    with col1:
        st.write('Caption for second chart')

st.write('\n')

fig, ax = plt.subplots()
nlp = to_transform('Q20_1', 'Q20_6', 'nlp_df', 'nlp', 'NLP Algorithm')
nlp['NLP Algorithm'] = nlp['NLP Algorithm'].str.replace('(', '\n(')
nlp['NLP Algorithm'] = nlp['NLP Algorithm'].drop(3)
fig = to_show(nlp, 'regularly used natural language processing algorithms')
plt.xlabel(None)
plt.ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)
    with col2:
        st.write('Caption for second chart')

st.write('\n')

fig, ax = plt.subplots()
cloud = to_transform('Q31_1', 'Q31_12', 'cloud_df', 'cloud', 'Cloud Service')
cloud['Cloud Service'] = cloud['Cloud Service'].drop([3, 6])
fig = to_show(cloud, 'top used cloud service')
plt.xlabel(None)
plt.ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col2:
        st.pyplot(fig)
    with col1:
        st.write('Caption for second chart')

st.write('\n')

fig, ax = plt.subplots()
db = to_transform('Q35_1', 'Q35_16', 'db_df', 'db', 'Database')
db['Database'] = db['Database'].drop([5, 14])
fig = to_show(db, 'top used databases')
plt.xlabel(None)
plt.ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)
    with col2:
        st.write('Caption for second chart')

st.write('\n')

fig, ax = plt.subplots()
bi = to_transform('Q36_1', 'Q36_15', 'bi_df', 'bi', 'BI Tool')
bi['BI Tool'] = bi['BI Tool'].drop([0, 6])
fig = to_show(bi, 'top used business intelligence tools')
plt.xlabel(None)
plt.ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col2:
        st.pyplot(fig)
    with col1:
        st.write('Caption for second chart')
