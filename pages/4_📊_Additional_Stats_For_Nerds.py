import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

plt.style.use('ggplot')
sns.set_theme(font = 'Georgia', palette = 'deep')
st.set_page_config(page_title="Kaggle Machine Learning & Data Science Report 2022", layout="wide", page_icon='👨‍💻')

path = "C:\\Users\\dzuz1\\Desktop\\Python\\datasets\\kaggle_survey_2022_responses.csv"
df = pd.read_csv(path)

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

st.header('Leftout Visuals')
st.markdown('---')

fig1, ax = plt.subplots()
sns.countplot(data = df, x = 'Q9')
plt.title('Any Research Published?', weight = 'bold')
plt.xlabel(None)
plt.ylabel(None, labelpad = 15)
plt.tight_layout()
fig2, ax = plt.subplots()
research = to_transform('Q10_1', 'Q10_3', 'research_df', 'research', 'Research Level')
research['Research Level'] = research['Research Level'].str.replace('machine', '\nmachine')
sns.barplot(data = research, y = 'Research Level', x = 'Count')
plt.title('Scope of Research', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None)
plt.tight_layout()
col1, col2 = st.columns(2)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.write(fig2)

fig1, ax = plt.subplots()
sort = list(df['Q24'].value_counts().index)
sns.countplot(data = df, y = 'Q24', 
              order = sort, palette = 'deep')
plt.title('In Which Industries You Can Expect To Work', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None)
fig2, ax = plt.subplots()
sort = list(df['Q25'].value_counts().index)
sns.countplot(data = df, y = 'Q25', order = sort)
plt.title('Size of Companies', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None)
plt.xlim(1250, 2150)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)


monitor = to_transform('Q40_1', 'Q40_15', 'monitor_df', 'monitor', 'ML Monitoring Tool')
fig1 = to_show(monitor, 'top used tools to monitor ml models')
plt.xlabel(None)
plt.ylabel(None)
ai = to_transform('Q41_1', 'Q41_9', 'ai_df', 'ai', 'AI Ethical Product')
ai['AI Ethical Product'] = ai['AI Ethical Product'].str.replace('(', '\n(')
fig2 = to_show(ai, 'AI ethical products for machine learning')
fig2.set_figheight(7)
fig2.set_figwidth(5)
plt.xlabel(None)
plt.ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)

fig1, axs = plt.subplots()
ide = to_transform('Q13_1', 'Q13_14', 'ide_df', 'ide', 'IDE')
sns.barplot(data = ide, y = 'IDE', x = 'Count', palette = 'deep')
plt.title('IDEs Used On A Regular Basis', weight = 'bold')
plt.xlabel(None)
plt.ylabel(None)
fig2, axs = plt.subplots()
nb = to_transform('Q14_1', 'Q14_16', 'nb_df', 'nb', 'Hosted Notebook')
sns.barplot(data = nb, y = 'Hosted Notebook', x = 'Count', palette = 'deep')
plt.title('Hosted Notebooks Being Used', weight = 'bold')
plt.xlabel(None)
plt.ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)

hard = to_transform('Q42_1', 'Q42_9', 'hard_df', 'hard', 'Hardware Used For Machine/Deep Learning')
fig1 = to_show(hard, 'Hardware Used For Machine/Deep Learning')
plt.xlabel(None)
plt.ylabel(None)
fig2, ax = plt.subplots()
sns.countplot(data = df, x = 'Q43')
ax.set_title('How Frequent Is The Use Of Hardware In Machine/Deep Learning', weight = 'bold')
ax.set_xlabel(None)
ax.set_ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)
