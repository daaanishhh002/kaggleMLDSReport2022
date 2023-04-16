import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

plt.style.use('ggplot')
sns.set_theme(font = 'Georgia', palette = 'deep')
st.set_page_config(page_title="Kaggle Machine Learning & Data Science Report 2022", layout="wide", page_icon='👨‍💻')

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

st.header(':black[Job and Industry Related Visuals]')
st.markdown('---')

fig, ax = plt.subplots()
sort = list(df['Q23'].value_counts().index)
sns.countplot(data = df, y = 'Q23', 
                    order = sort, palette = 'deep')
plt.title('Jobs In The ML/DS Space', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)
    with col2:
        st.write('Caption for second chart')

st.write('\n')

fig, ax = plt.subplots()
role = to_transform('Q28_1', 'Q28_8', 'role_df', 'role', 'Job Role')
role['Job Role'] = role['Job Role'].str.replace('uses', '\nuses')
role['Job Role'] = role['Job Role'].str.replace('influence', 'influence\n')
role['Job Role'] = role['Job Role'].str.replace('applying', 'applying\n')
role['Job Role'] = role['Job Role'].str.replace('to improve', 'to\nimprove')
role['Job Role'] = role['Job Role'].str.replace('operationally', '\noperationally')
role['Job Role'] = role['Job Role'].str.replace(' art', '\nart')
role['Job Role'] = role['Job Role'].str.replace('part', '\npart')
plt.figure(figsize = (8, 6.5))
fig = to_show(role, 'important role at work')
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
sort = list(df['Q26'].value_counts().index)
sns.countplot(data = df, x = 'Q26', order = sort)
plt.title('Typical Size Of Data Science Department In Companies', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig)
    with col2:
        st.write('Caption for second chart')

st.write('\n')

fig, axs = plt.subplots(3, 2, figsize = (23, 15))
fig.suptitle('Industries Where Some ML/DS Profesionals Work In\n', weight = 'bold', size = 20)
job = df.groupby('Q23')
job_df = job.get_group('Data Scientist')['Q24'].value_counts()
g = job_df.head().plot.pie(ax = axs[0][0], startangle = 45)
g.set_title('Data Scientist', loc = 'center', size = 15)
#g.set_label(g.get_label(), fontsize = 15)
g.set_ylabel(None)
job_df = job.get_group('Software Engineer')['Q24'].value_counts()
g = job_df.head().plot.pie(ax = axs[2][0], startangle = 45)
g.set_title('Software Engineer', loc = 'center', size = 15) 
g.set_ylabel(None)
job_df = job.get_group('Data Analyst (Business, Marketing, Financial, Quantitative, etc)')['Q24'].value_counts()
g = job_df.head().plot.pie(ax = axs[1][0], startangle = 0)
g.set_title('Data Analyst', loc = 'center', size = 15)
g.set_ylabel(None)
job_df = job.get_group('Machine Learning/ MLops Engineer')['Q24'].value_counts()
g = job_df.head().plot.pie(ax = axs[0][1], startangle = 45)
g.set_title('ML Engineer', loc = 'center', size = 15)
g.set_ylabel(None)
job_df = job.get_group('Research Scientist')['Q24'].value_counts()
g = job_df.head().plot.pie(ax = axs[1][1], startangle = 45)
g.set_title('Research Scientist', loc = 'center', size = 15)
g.set_ylabel(None)
job_df = job.get_group('Manager (Program, Project, Operations, Executive-level, etc)')['Q24'].value_counts()
g = job_df.head().plot.pie(ax = axs[2][1], startangle = 45)
g.set_title('Manager', loc = 'center', size = 15)
g.set_ylabel(None)
plt.tight_layout()
with st.container():
    col1, col2 = st.columns([1, 2])
    with col2:
        st.pyplot(fig)
    with col1:
        st.write('Caption for second chart')

st.write('\n')

fig, axes = plt.subplots(3, 2, figsize = (25, 15))
fig.suptitle('Languages Used By Some ML/DS Professionals\n', size = 30, weight = 'bold')
jobrole = df.groupby('Q23')
#'Data Scientist'
ds_df = jobrole.get_group('Data Scientist')
ds_df = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Languages', data = ds_df)
sns.barplot(data = ds_df, x = ds_df['Languages'][:5], y = 'Count', ax = axes[0][0], palette = 'deep')
ax = axes[0][0]
ax.set_title('Data Scientist', loc = 'right', size = 20)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_xticklabels(ds_df['Languages'][:5], fontsize = 14)
ax.set_yticklabels(ax.get_yticks(), fontsize = 15)
#'Machine Learning/ MLops Engineer'
ds_df = jobrole.get_group('Machine Learning/ MLops Engineer')
ds_df = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Languages', data = ds_df)
sns.barplot(data = ds_df, x = ds_df['Languages'][:5], y = 'Count', ax = axes[0][1], palette = 'deep')
ax = axes[0][1]
ax.set_title('Machine Learning Engineer', loc = 'right', size = 20)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_xticklabels(ds_df['Languages'][:5], fontsize = 14)
ax.set_yticklabels(ax.get_yticks(), fontsize = 15)
#'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'
ds_df = jobrole.get_group('Data Analyst (Business, Marketing, Financial, Quantitative, etc)')
ds_df = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Languages', data = ds_df)
sns.barplot(data = ds_df, x = ds_df['Languages'][:5], y = 'Count', ax = axes[1][0], palette = 'deep')
ax = axes[1][0]
ax.set_title('Data Analyst', loc = 'right', size = 20)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_xticklabels(ds_df['Languages'][:5], fontsize = 14)
ax.set_yticklabels(ax.get_yticks(), fontsize = 15)
#'Research Scientist'
ds_df = jobrole.get_group('Research Scientist')
ds_df = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Languages', data = ds_df)
sns.barplot(data = ds_df, x = ds_df['Languages'][:5], y = 'Count', ax = axes[1][1], palette = 'deep')
ax = axes[1][1]
ax.set_title('Research Scientist', loc = 'right', size = 20)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_xticklabels(ds_df['Languages'][:5], fontsize = 14)
ax.set_yticklabels(ax.get_yticks(), fontsize = 15)
# 'Software Engineer'
ds_df = jobrole.get_group('Software Engineer')
ds_df = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Languages', data = ds_df)
sns.barplot(data = ds_df, y = 'Count', x = ds_df['Languages'][:5], ax = axes[2][0], palette = 'deep')
ax = axes[2][0]
ax.set_title('Software Engineer', loc = 'right', size = 20)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_xticklabels(ds_df['Languages'][:5], fontsize = 14)
ax.set_yticklabels(ax.get_yticks(), fontsize = 15)
# 'Manager'
ds_df = jobrole.get_group('Manager (Program, Project, Operations, Executive-level, etc)')
ds_df = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Languages', data = ds_df)
sns.barplot(data = ds_df, y = 'Count', x = ds_df['Languages'][:5], ax = axes[2][1], palette = 'deep')
ax = axes[2][1]
ax.set_title('Manager', loc = 'right', size = 20)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_xticklabels(ds_df['Languages'][:5], fontsize = 14)
ax.set_yticklabels(ax.get_yticks(), fontsize = 15)
plt.tight_layout()
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.write('Caption for second chart')

st.write('\n')

fig, axes = plt.subplots(3, 2, figsize = (25, 15))
jobrole = df.groupby('Q23')
fig.suptitle('Most Used Machine Learning Algorithms And By Whom\n', weight = 'bold', size = 35)
#'Data Scientist'
ds_df = jobrole.get_group('Data Scientist')
ds_df = to_transform('Q18_1', 'Q18_14', 'lang_df', 'lang', 'ML Algorithm', data = ds_df)
g = sns.barplot(data = ds_df, y = ds_df['ML Algorithm'][:5], x = 'Count', ax = axes[0][0], palette = 'deep')
g.set_title('Data Scientist', loc = 'right', size = 20)
g.set_ylabel(None)
g.set_xlabel(None)
g.set_yticklabels(ds_df['ML Algorithm'][:5], fontsize = 16)
g.set_xticklabels(g.get_xticks(), size = 14)
#'Machine Learning/ MLops Engineer'
ds_df = jobrole.get_group('Machine Learning/ MLops Engineer')
ds_df = to_transform('Q18_1', 'Q18_14', 'lang_df', 'lang', 'ML Algorithm', data = ds_df)
g = sns.barplot(data = ds_df, y = ds_df['ML Algorithm'][:5], x = 'Count', ax = axes[0][1], palette = 'deep')
g.set_title('Machine Learning Engineer', loc = 'right', size = 20)
g.set_ylabel(None)
g.set_xlabel(None)
g.set_yticklabels(ds_df['ML Algorithm'][:5], fontsize = 16)
g.set_xticklabels(g.get_xticks(), size = 14)
#'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'
ds_df = jobrole.get_group('Data Analyst (Business, Marketing, Financial, Quantitative, etc)')
ds_df = to_transform('Q18_1', 'Q18_14', 'lang_df', 'lang', 'ML Algorithm', data = ds_df)
g = sns.barplot(data = ds_df, y = ds_df['ML Algorithm'][:5], x = 'Count', ax = axes[1][0], palette = 'deep')
g.set_title('Data Analyst', loc = 'right', size = 20)
g.set_ylabel(None)
g.set_xlabel(None)
g.set_yticklabels(ds_df['ML Algorithm'][:5], fontsize = 16)
g.set_xticklabels(g.get_xticks(), size = 14)
#'Research Scientist'
ds_df = jobrole.get_group('Research Scientist')
ds_df = to_transform('Q18_1', 'Q18_14', 'lang_df', 'lang', 'ML Algorithm', data = ds_df)
g = sns.barplot(data = ds_df, y = ds_df['ML Algorithm'][:5], x = 'Count', ax = axes[2][0], palette = 'deep')
g.set_title('Research Scientist', loc = 'right', size = 20)
g.set_ylabel(None)
g.set_xlabel(None)
g.set_yticklabels(ds_df['ML Algorithm'][:5], fontsize = 16)
g.set_xticklabels(g.get_xticks(), size = 14)
# 'Software Engineer'
ds_df = jobrole.get_group('Software Engineer')
ds_df = to_transform('Q18_1', 'Q18_14', 'lang_df', 'lang', 'ML Algorithm', data = ds_df)
g = sns.barplot(data = ds_df, y = ds_df['ML Algorithm'][:5], x = 'Count', ax = axes[1][1], palette = 'deep')
g.set_title('Software Engineer', loc = 'right', size = 20)
g.set_ylabel(None)
g.set_xlabel(None)
g.set_yticklabels(ds_df['ML Algorithm'][:5], fontsize = 16)
g.set_xticklabels(g.get_xticks(), size = 14)
# 'Manager'
ds_df = jobrole.get_group('Manager (Program, Project, Operations, Executive-level, etc)')
ds_df = to_transform('Q18_1', 'Q18_14', 'lang_df', 'lang', 'ML Algorithm', data = ds_df)
g = sns.barplot(data = ds_df, y = ds_df['ML Algorithm'][:5], x = 'Count', ax = axes[2][1], palette = 'deep')
g.set_title('Manager', loc = 'right', size = 20)
g.set_ylabel(None)
g.set_xlabel(None)
g.set_yticklabels(ds_df['ML Algorithm'][:5], fontsize = 16)
g.set_xticklabels(g.get_xticks(), size = 14)
plt.tight_layout()
with st.container():
    col1, col2 = st.columns([1, 2])
    with col2:
        st.pyplot(fig)
    with col1:
        st.write('Caption for second chart')

st.write('\n')

fig, axs = plt.subplots(2, 2, figsize = (23, 10))
fig.suptitle('Different Job Titles By Countries\n', weight = 'bold', size = 25)
country = df.groupby('Q4')
job = country.get_group('India')['Q23'].value_counts()
job = job.reset_index()
sns.barplot(data= job, y = job['index'][:10], x = job['Q23'][:10], palette = 'deep', ax = axs[0][0])
ax = axs[0][0]
ax.set_ylabel(None)
ax.set_xlabel(None, labelpad = 10)
ax.set_title('India', loc = 'right', size = 15)
ax.set_yticklabels(job['index'][:10], size = 14)
ax.set_xticklabels(ax.get_xticks(), size = 15)
job = country.get_group('United States of America')['Q23'].value_counts()
job = job.reset_index()
sns.barplot(data= job, y = job['index'][:10], x = job['Q23'][:10], palette = 'deep', ax = axs[0][1])
ax = axs[0][1]
ax.set_ylabel(None)
ax.set_xlabel(None, labelpad = 10)
ax.set_title('USA', loc = 'right', size = 15)
ax.set_yticklabels(job['index'][:10], size = 14)
ax.set_xticklabels(ax.get_xticks(), size = 15)
job = country.get_group('Canada')['Q23'].value_counts()
job = job.reset_index()
sns.barplot(data= job, y = job['index'][:10], x = job['Q23'][:10], palette = 'deep', ax = axs[1][0])
ax = axs[1][0]
ax.set_ylabel(None)
ax.set_xlabel(None, labelpad = 10)
ax.set_title('Canada', loc = 'right', size = 15)
ax.set_yticklabels(job['index'][:10], size = 14)
ax.set_xticklabels(ax.get_xticks(), size = 15)
job = country.get_group('United Kingdom of Great Britain and Northern Ireland')['Q23'].value_counts()
job = job.reset_index()
sns.barplot(data= job, y = job['index'][:10], x = job['Q23'][:10], palette = 'deep', ax = axs[1][1])
ax = axs[1][1]
ax.set_ylabel(None)
ax.set_xlabel(None, labelpad = 10)
ax.set_title('UK', loc = 'right', size = 15)
ax.set_yticklabels(job['index'][:10], size = 14)
ax.set_xticklabels(ax.get_xticks(), size = 15)
plt.tight_layout()
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.write('Caption for second chart')

st.write('\n')

fig, axs = plt.subplots(5, 3, figsize = (30, 30))
country_job_salary = df.groupby(['Q4', 'Q23'])
fig.suptitle('Annual Income Of Certain Jobs By Country\n', weight = 'bold', size = 30)
g = country_job_salary.get_group(('India', 'Data Scientist'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[0][0], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('India|Data Scientist', loc = 'right', size = 25)
tick = country_job_salary.get_group(('India', 'Data Scientist'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# usa, data scientist
g = country_job_salary.get_group(('United States of America', 'Data Scientist'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[0][1], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('USA|Data Scientist', loc = 'right', size = 25)
tick = country_job_salary.get_group(('United States of America', 'Data Scientist'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# canada, data scientist
g = country_job_salary.get_group(('Canada', 'Data Scientist'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[0][2], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('Canada|Data Scientist', loc = 'right', size = 25)
tick = country_job_salary.get_group(('Canada', 'Data Scientist'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# india, ml
g = country_job_salary.get_group(('India', 'Machine Learning/ MLops Engineer'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[1][0], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('India|ML Engineer', loc = 'right', size = 25)
tick = country_job_salary.get_group(('India', 'Machine Learning/ MLops Engineer'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# usa, ml
g = country_job_salary.get_group(('United States of America', 'Machine Learning/ MLops Engineer'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[1][1], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('USA|ML Engineer', loc = 'right', size = 25)
tick = country_job_salary.get_group(('United States of America', 'Machine Learning/ MLops Engineer'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# canada, ml
g = country_job_salary.get_group(('Canada', 'Machine Learning/ MLops Engineer'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[1][2], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('Canada|ML Engineer', loc = 'right', size = 25)
tick = country_job_salary.get_group(('Canada', 'Machine Learning/ MLops Engineer'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# india, research
g = country_job_salary.get_group(('India', 'Research Scientist'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[2][0], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('India|Research Scientist', loc = 'right', size = 25)
tick = country_job_salary.get_group(('India', 'Research Scientist'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# usa, research
g = country_job_salary.get_group(('United States of America', 'Research Scientist'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[2][1], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('USA|Research Scientist', loc = 'right', size = 25)
tick = country_job_salary.get_group(('United States of America', 'Research Scientist'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# canada, research
g = country_job_salary.get_group(('Canada', 'Research Scientist'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[2][2], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('Canada|Research Scientist', loc = 'right', size = 25)
tick = country_job_salary.get_group(('Canada', 'Research Scientist'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# india, se
g = country_job_salary.get_group(('India', 'Software Engineer'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[3][0], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('India|Sofware Engineer', loc = 'right', size = 25)
tick = country_job_salary.get_group(('India', 'Software Engineer'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# usa, se
g = country_job_salary.get_group(('United States of America', 'Software Engineer'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[3][1], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('USA|Sofware Engineer', loc = 'right', size = 25)
tick = country_job_salary.get_group(('United States of America', 'Software Engineer'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# canada, se
g = country_job_salary.get_group(('Canada', 'Software Engineer'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[3][2], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('Canada|Sofware Engineer', loc = 'right', size = 25)
tick = country_job_salary.get_group(('Canada', 'Software Engineer'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# india, da
g = country_job_salary.get_group(('India', 'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[4][0], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('India|Data Analyst', loc = 'right', size = 25)
tick = country_job_salary.get_group(('India', 'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# usa, da
g = country_job_salary.get_group(('United States of America', 'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[4][1], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('USA|Data Analyst', loc = 'right', size = 25)
tick = country_job_salary.get_group(('United States of America', 'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
# canada, da
g = country_job_salary.get_group(('Canada', 'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[4][2], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('Canada|Data Analyst', loc = 'right', size = 25)
tick = country_job_salary.get_group(('Canada', 'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'))['Q29'].value_counts().head(7).index
g.set_yticklabels(tick, size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
plt.tight_layout()
with st.container():
    col1, col2 = st.columns([1, 3])
    with col2:
        st.pyplot(fig)
    with col1:
        st.write('Caption for second chart')

st.write('\n')

fig, axs = plt.subplots(3, 2, figsize = (25, 15))
fig.suptitle('What One Can Expect To Do As Per The Job Title\n', weight = 'bold', size = 30)
#df['Q28_1'] = df['Q28_1'].str.replace('uses', '\nuses')
df['Q28_2'] = df['Q28_2'].str.replace('uses', 'uses\n')
#df['Q28_3'] = df['Q28_3'].str.replace('applying', 'applying\n')
df['Q28_4'] = df['Q28_4'].str.replace('that', '\nthat')
#df['Q28_5'] = df['Q28_5'].str.replace('operationally', '\noperationally')
#df['Q28_1'] = df['Q28_1'].str.replace(' art', '\nart')
#df['Q28_1'] = df['Q28_1'].str.replace('part', '\npart')
jobrole = df.groupby('Q23')
ds_df = jobrole.get_group('Data Scientist')
ds_df = to_transform('Q28_1', 'Q28_8', 'role_df', 'ds_df', 'Job Role', data = ds_df)
g = sns.barplot(data = ds_df, x = 'Count', y = 'Job Role', ax = axs[0][0])
g.set_title('Data Scientist', loc = 'right', size = 15)
g.set_ylabel(None)
g.set_xlabel(None)
g.set_yticklabels(ds_df['Job Role'], size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
ds_df = jobrole.get_group('Machine Learning/ MLops Engineer')
ds_df = to_transform('Q28_1', 'Q28_8', 'role_df', 'ds_df', 'Job Role', data = ds_df)
g = sns.barplot(data = ds_df, x = 'Count', y = 'Job Role', ax = axs[0][1])
g.set_title('ML Engineer', loc = 'right', size = 15)
g.set_ylabel(None)
g.set_xlabel(None)
g.set_yticklabels(ds_df['Job Role'], size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
ds_df = jobrole.get_group('Data Analyst (Business, Marketing, Financial, Quantitative, etc)')
ds_df = to_transform('Q28_1', 'Q28_8', 'role_df', 'ds_df', 'Job Role', data = ds_df)
g = sns.barplot(data = ds_df, x = 'Count', y = 'Job Role', ax = axs[1][0])
g.set_title('Data Analyst', loc = 'right', size = 15)
g.set_ylabel(None)
g.set_xlabel(None)
g.set_yticklabels(ds_df['Job Role'], size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
ds_df = jobrole.get_group('Software Engineer')
ds_df = to_transform('Q28_1', 'Q28_8', 'role_df', 'ds_df', 'Job Role', data = ds_df)
g = sns.barplot(data = ds_df, x = 'Count', y = 'Job Role', ax = axs[1][1])
g.set_title('Software Engineer', loc = 'right', size = 15)
g.set_ylabel(None)
g.set_xlabel(None)
g.set_yticklabels(ds_df['Job Role'], size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
ds_df = jobrole.get_group('Research Scientist')
ds_df = to_transform('Q28_1', 'Q28_8', 'role_df', 'ds_df', 'Job Role', data = ds_df)
g = sns.barplot(data = ds_df, x = 'Count', y = 'Job Role', ax = axs[2][0])
g.set_title('Research Scientist', loc = 'right', size = 15)
g.set_ylabel(None)
g.set_xlabel(None)
g.set_yticklabels(ds_df['Job Role'], size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
ds_df = jobrole.get_group('Manager (Program, Project, Operations, Executive-level, etc)')
ds_df = to_transform('Q28_1', 'Q28_8', 'role_df', 'ds_df', 'Job Role', data = ds_df)
g = sns.barplot(data = ds_df, x = 'Count', y = 'Job Role', ax = axs[2][1])
g.set_title('Manager', loc = 'right', size = 15)
g.set_ylabel(None)
g.set_xlabel(None)
g.set_yticklabels(ds_df['Job Role'], size = 15)
g.set_xticklabels(g.get_xticks(), size = 15)
plt.tight_layout()
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.pyplot(fig)
    with col2:
        st.write('Caption for second chart')
