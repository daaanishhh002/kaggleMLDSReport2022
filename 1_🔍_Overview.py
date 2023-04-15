import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

plt.style.use('ggplot')
sns.set_theme(font = 'Georgia', palette = 'deep')
st.set_page_config(page_title="Kaggle Machine Learning & Data Science Report 2022", layout="wide")

path = "https://github.com/daaanishhh4218/kaggleMLDSReport2022/blob/main/stars(classification).csv"
df = pd.read_csv(path)
st.table(df)

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
    fig = plt.subplots()
    sns.barplot(data = sname, x = 'Count', y = sname.columns[0], palette = 'deep', ax = ax)
    
    title = title.title()
    plt.title(title, weight = 'bold')
    plt.xlabel('Count', labelpad = 10)
    plt.ylabel(sname.columns[0])
    
    return fig


st.header('Demographical Information')
st.markdown('---')

fig, ax = plt.subplots()
sort = sorted(list(df['Q2']\
                   .value_counts()\
                   .index))
sns.countplot(data = df, x = 'Q2', 
              hue = 'Q3', order = sort, ax = ax)
plt.title('Distribution of Age and Gender', 
          weight = 'bold')
ax.set_xlabel('Age Group', labelpad = 10)
ax.set_ylabel(None, labelpad = 10)
ax.legend(title = 'Gender')
plt.tight_layout()
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.write('Caption for second chart')
    with col2:
        st.pyplot(fig)


fig, ax = plt.subplots()
sns.countplot(data = df, x = 'Q5', ax = ax)
plt.title('Is The Participant a Student?', weight = 'bold')
plt.xlabel(None)
plt.ylabel(None, labelpad = 10)
plt.ylim(11600, 12100)
plt.tight_layout()
with st.container():
    col1, col2 = st.columns(2)
    with col2:
        st.write('Caption for second chart')
    with col1:
        st.pyplot(fig)


fig, ax = plt.subplots()
temp = df['Q4'].value_counts().drop('Other').head()
temp = temp.reset_index()
temp['index'] = temp['index'].str.replace('of', '\nof')
sns.barplot(data = temp, x = 'Q4', y = 'index')
plt.title('Countries With The Most Amount of Participants', 
          weight = 'bold', pad = 10)
plt.xlabel(None, labelpad = 10)
plt.ylabel(None, labelpad = 10)
plt.tight_layout()
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.write('Caption for second chart')
    with col2:
        st.pyplot(fig)


df['Q8'] = df['Q8'].str.replace('education', 'education\n')
df['Q8'] = df['Q8'].str.replace('university', 'university\n')
df['Q8'] = df['Q8'].str.replace('earning', 'earning\n')
fig, ax = plt.subplots(1, 1)
sns.countplot(data = df, y = 'Q8', ax = ax)
plt.title('Education Status of Participants', weight = 'bold')
plt.xlabel(None, labelpad = 15)
plt.ylabel(None, labelpad = 15)
with st.container():
    col1, col2 = st.columns(2)
    with col2:
        st.success('This is a success message!', icon="✅")
    with col1:
        st.pyplot(fig)


fig, axs = plt.subplots(3, 2, figsize = (30, 20))
fig.suptitle('Education Requirement For Some ML/DS Jobs\n', size = 35, weight = 'bold')
job = df.groupby('Q23')
#'Data Scientist'
job_df = job.get_group('Data Scientist')['Q8'].value_counts()
job_df = job_df.reset_index()
sns.barplot(data= job_df, x = 'index', y = 'Q8', palette = 'deep', ax = axs[0][0])
ax = axs[0][0]
ax.set_xticklabels(job_df['index'].unique(), fontsize=14)
ax.set_yticklabels(ax.get_yticks(), size = 14)
ax.set_ylabel(None)
ax.set_xlabel(None, labelpad = 10)
ax.set_title('Data Scientist', loc = 'right', size = 25)
#'Software Engineer'
job_df = job.get_group('Software Engineer')['Q8'].value_counts()
job_df = job_df.reset_index()
sns.barplot(data= job_df, x = 'index', y = 'Q8', palette = 'deep', ax = axs[1][1])
ax = axs[1][1]
ax.set_xticklabels(job_df['index'].unique(), fontsize=14)
ax.set_yticklabels(ax.get_yticks(), size = 14)
ax.set_ylabel(None)
ax.set_xlabel(None, labelpad = 10)
ax.set_title('Software Engineer', loc = 'right', size = 20)
#'Data Analyst'
job_df = job.get_group('Data Analyst (Business, Marketing, Financial, Quantitative, etc)')['Q8'].value_counts()
job_df = job_df.reset_index()
sns.barplot(data= job_df, x = 'index', y = 'Q8', palette = 'deep', ax = axs[1][0])
ax = axs[1][0]
ax.set_xticklabels(job_df['index'].unique(), fontsize=14)
ax.set_yticklabels(ax.get_yticks(), size = 14)
ax.set_ylabel(None)
ax.set_xlabel(None, labelpad = 10)
ax.set_title('Data Analyst', loc = 'right', size = 25)
#ML Engineer
job_df = job.get_group('Machine Learning/ MLops Engineer')['Q8'].value_counts()
job_df = job_df.reset_index()
sns.barplot(data= job_df, x = 'index', y = 'Q8', palette = 'deep', ax = axs[0][1])
ax = axs[0][1]
ax.set_xticklabels(job_df['index'].unique(), fontsize=14)
ax.set_yticklabels(ax.get_yticks(), size = 14)
ax.set_ylabel(None)
ax.set_xlabel(None, labelpad = 10)
ax.set_title('Machine Learning Engineer', loc = 'right', size = 25)
#'Research Scientist'
job_df = job.get_group('Research Scientist')['Q8'].value_counts()
job_df = job_df.reset_index()
sns.barplot(data= job_df, x = 'index', y = 'Q8', palette = 'deep', ax = axs[2][0])
ax = axs[2][0]
ax.set_xticklabels(job_df['index'].unique(), fontsize=14)
ax.set_yticklabels(ax.get_yticks(), size = 14)
ax.set_ylabel(None)
ax.set_xlabel(None, labelpad = 10)
ax.set_title('Research Scientist', loc = 'right', size = 25)
#'Manager'
job_df = job.get_group('Manager (Program, Project, Operations, Executive-level, etc)')['Q8'].value_counts()
job_df = job_df.reset_index()
sns.barplot(data= job_df, x = 'index', y = 'Q8', palette = 'deep', ax = axs[2][1])
ax = axs[2][1]
ax.set_xticklabels(job_df['index'].unique(), fontsize=14)
ax.set_yticklabels(ax.get_yticks(), size = 14)
ax.set_ylabel(None)
ax.set_xlabel(None, labelpad = 10)
ax.set_title('Manager', loc = 'right', size = 25)
plt.tight_layout()
with st.container():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write('Caption for second chart')
    with col2:
        st.pyplot(fig)


fig, axes = plt.subplots(2, 2, figsize = (30, 16))
fig.suptitle('Education Level Among Some Countries\n', size = 30, weight = 'bold')
country = df.groupby('Q4')
#'India'
c_df = country.get_group('India')
sns.countplot(data = c_df, y = 'Q8', ax = axes[0][0], order = list(c_df['Q8'].value_counts().index))
ax = axes[0][0]
ax.set_yticklabels(c_df['Q8'].value_counts().index, fontsize=20)
ax.set_xticklabels(ax.get_xticks(), size = 15)
ax.set_title('India', loc = 'right', size = 25)
ax.set_ylabel(None)
ax.set_xlabel(None)
#'United States of America'
c_df = country.get_group('United States of America')
sns.countplot(data = c_df, y = 'Q8', ax = axes[0][1], order = list(c_df['Q8'].value_counts().index))
ax = axes[0][1]
ax.set_yticklabels(c_df['Q8'].value_counts().index, fontsize=20)
ax.set_xticklabels(ax.get_xticks(), size = 15)
ax.set_title('USA', loc = 'right', size = 25)
ax.set_ylabel(None)
ax.set_xlabel(None)
#'UK'
c_df = country.get_group('United Kingdom of Great Britain and Northern Ireland')
sns.countplot(data = c_df, y = 'Q8', ax = axes[1][0], order = list(c_df['Q8'].value_counts().index))
ax = axes[1][0]
ax.set_yticklabels(c_df['Q8'].value_counts().index, fontsize=20)
ax.set_xticklabels(ax.get_xticks(), size = 15)
ax.set_title('UK', loc = 'right', size = 25)
ax.set_ylabel(None)
ax.set_xlabel(None)
#'Canada'
c_df = country.get_group('Canada')
sns.countplot(data = c_df, y = 'Q8', ax = axes[1][1], order = list(c_df['Q8'].value_counts().index))
ax = axes[1][1]
ax.set_yticklabels(c_df['Q8'].value_counts().index, fontsize=20)
ax.set_xticklabels(ax.get_xticks(), size = 15)
ax.set_title('Canada', loc = 'right', size = 25)
ax.set_ylabel(None)
ax.set_xlabel(None)
plt.tight_layout()
with st.container():
    col1, col2 = st.columns([3, 1])
    with col2:
        st.write('Caption for second chart')
    with col1:
        st.pyplot(fig)


fig, axs = plt.subplots(3, 2, figsize = (25, 17))
country = df.groupby('Q4')
fig.suptitle('Annual Income Per Country\n', weight = 'bold', size = 30)
fig.supxlabel(None, weight = 'bold', size = 20)
# 'India'
x = country.get_group('India')['Q29'].\
                        value_counts().head(10)\
                        .plot(kind = 'barh', ax = axs[0][0], 
                              color = sns.palettes.color_palette('deep'))
ticks = country.get_group('India')['Q29'].value_counts().head(10).index
x.set_title('India', size = 20, loc = 'right')
x.set_ylabel(None)
x.set_yticklabels(ticks, fontsize=20)
x.set_xticklabels(x.get_xticks(), size = 14)
#x.set_xlabel('Count', labelpad = 10)
x.invert_yaxis()
# 'United States of America'
x = country.get_group('United States of America')['Q29'].\
                        value_counts().head(10)\
                        .plot(kind = 'barh', ax = axs[0][1], 
                              color = sns.palettes.color_palette('deep'))
x.set_title('USA', size = 20, loc = 'right')
x.set_ylabel(None)
ticks = country.get_group('United States of America')['Q29'].value_counts().head(10).index
x.set_yticklabels(ticks, fontsize=20)
x.set_xticklabels(x.get_xticks(), size = 14)
#x.set_xlabel('Count', labelpad = 10)
x.invert_yaxis()
# 'Canada'
x = country.get_group('Canada')['Q29'].\
                        value_counts().head(10)\
                        .plot(kind = 'barh', ax = axs[1][0], 
                              color = sns.palettes.color_palette('deep'))
x.set_title('Canada', size = 20, loc = 'right')
x.set_ylabel(None)
ticks = country.get_group('Canada')['Q29'].value_counts().head(10).index
x.set_yticklabels(ticks, fontsize=20)
x.set_xticklabels(x.get_xticks(), size = 14)
#x.set_xlabel('Count', labelpad = 10)
x.invert_yaxis()
# 'United Kingdom of Great Britain and Northern Ireland'
x = country.get_group('United Kingdom of Great Britain and Northern Ireland')['Q29'].\
                        value_counts().head(10)\
                        .plot(kind = 'barh', ax = axs[1][1], 
                              color = sns.palettes.color_palette('deep'))
x.set_title('UK', size = 20, loc = 'right')
x.set_ylabel(None)
ticks = country.get_group('United Kingdom of Great Britain and Northern Ireland')['Q29'].value_counts().head(10).index
x.set_yticklabels(ticks, fontsize=20)
x.set_xticklabels(x.get_xticks(), size = 14)
#x.set_xlabel('Count', labelpad = 10)
x.invert_yaxis()
# 'Australia'
x = country.get_group('Australia')['Q29'].\
                        value_counts().head(10)\
                        .plot(kind = 'barh', ax = axs[2][0], 
                              color = sns.palettes.color_palette('deep'))
x.set_title('Australia', size = 20, loc = 'right')
x.set_ylabel(None)
ticks = country.get_group('Australia')['Q29'].value_counts().head(10).index
x.set_yticklabels(ticks, fontsize=20)
x.set_xticklabels(x.get_xticks(), size = 14)
#x.set_xlabel('Count', labelpad = 10)
x.invert_yaxis()
# 'United Arab Emirates'
x = country.get_group('United Arab Emirates')['Q29'].\
                        value_counts().head(10)\
                        .plot(kind = 'barh', ax = axs[2][1], 
                              color = sns.palettes.color_palette('deep')) 
x.set_title('UAE', size = 20, loc = 'right')
x.set_ylabel(None)
ticks = country.get_group('United Arab Emirates')['Q29'].value_counts().head(10).index
x.set_yticklabels(ticks, fontsize=20)
x.set_xticklabels(x.get_xticks(), size = 14)
#x.set_xlabel('Count', labelpad = 10)
x.invert_yaxis()
plt.tight_layout()
with st.container():
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write('Caption for second chart')
    with col2:
        st.pyplot(fig)

media = to_transform('Q44_1', 'Q44_12', 'media_df', 'media', 'Media Source')
fig, ax = plt.subplots()
to_show(media, 'favourite media souces for machine learning and data science')
ax.set_title('Favourite Media Sources For Machine Learning And Data Science', weight = 'bold')
ax.set_xlabel(None)
ax.set_ylabel(None)
plt.tight_layout()
with st.container():
    col1, col2 = st.columns(2)
    with col2:
        st.write('Caption for second chart')
    with col1:
        st.pyplot(fig)
