import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

plt.style.use('ggplot')
sns.set_theme(font = 'Georgia', palette = 'deep')
st.title('hfehgfuewg')
url = "C:\\Users\\dzuz1\\Desktop\\Python\\datasets\\kaggle_survey_2022_responses.csv"
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

st.set_page_config(page_title="Kaggle Machine Learning & Data Science Report 2022", layout="centered")


st.title(':blue[2022 Kaggle Machine Learning & Data Science Survey]')
st.markdown('##### Danish Ahmed')
st.markdown('---')
text = \
"""
Machine Learning and Data Science related jobs are coming up to be one of most 
sought after jobs currently. I plan to pursue a career in the field of
machine learning, specifically in the healthcare sector. The way our lives have
been affected positively by machine learning, I believe that we can also help
improve the lives of many people by deploying machine learning and data science
in the healthcare sector.\n
To understand the industry, its requirements and its trends, I gone through the 
Kaggle Machine Learning & Data Science Survey of 2022, 
and I have visualised the survey results for better understandability. I have 
specifically focused on comparing India with other countries, especially the United
States of America, Canada and the United Kingdom. I have emphasised more on 
visualising factors realted to employability, yearly income and necessary skills.
I have created this project to practise my data visualisation skills, and also
as a way to better understand the data science market, that way this project
also acts as a guide for me. \n
You may find that I haven't presented all possible
observations, because I have made this report to help me get a good job. 
This could very well be passed off as a personal report, because I have 
emphasised more on my interests and my goals, rather than a complete 
all - around report.
"""
st.header('Introduction')
st.markdown(text)
ps = \
"""
Please note that all statements and findings are based on the 
"2022 Kaggle Machine Learning & Data Science Survey". It may or may not reflect
the actual trends occuring in the data science and machine learning world.
"""
st.caption(ps)
st.header('Summary')
summary = \
"""
I start off by\n 
The most significant age bucket is between the ages of 18 and 29, 
and most of them are men. 
We observe that almost 50% of all participants are students. We can therefore
say that Kaggle is used by students as well as professionals.It is found that
the a few results of this survey are heavily skewed due to Indian participants 
due to most particpants being Indian. ML/DS is a highly skilled domain, which is
why most people are atleast graduates. It would be very difficult for a person
without a degree to secure a fine ML/DS job. Most people have a master's degree
in the ML/DS space. In the western countries, most people have a master's degree
in the ML/DS space. In western countries, the annual pay of an ML/DS is above
100,000 USD.\n
To get a good job in the ML/DS space, one must be proficient with Python and SQL.
Learning R and Bash would help a lot. Libraries like Matplotlib, Seaborn and Plotly
should be mastered. ML frameworks like SciKit-learn, TensorFlow, Keras, Pytorch
and Xgboost should be very familar.ML algorithms like Regression, 
Random Forests, CNNs and GBMs are very heavily used. Algorithms and models 
related to Computer Vision and Natural Language Processing are also mentioned 
in the report.\n

"""
st.write(summary)
col1, col2, col3 = st.columns(3)

with col1:
    st.write(summary)

with col2:
    st.write('keuhgfuewahg')


st.header('Some Programming Related Infographics')
#st.subheader('How Much Experince Do They Have In programming?')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
df['Q11'] = df['Q11'].str.replace('never', 'never\n')
sort = list(df['Q11'].value_counts().index)
sns.countplot(data = df, y = 'Q11', order = sort)

plt.title('Coding For How Long?', weight = 'bold')
plt.xlabel(None, labelpad = 15)
plt.ylabel(None)
plt.tight_layout()
st.pyplot(fig)
#plt.show()

#st.subheader('Languages That Are Used On A Regular Basis')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
lang = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Language')

sns.barplot(data = lang, y = 'Language', x = 'Count', palette = 'deep')

plt.title('Top Languages Used On A Regular Basis', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None, labelpad = 10)
st.pyplot(fig)
#plt.show()


#st.subheader('rando3')
st.caption('enter graph and text here')

fig, axes = plt.subplots(3, 2, figsize = (25, 15))
jobrole = df.groupby('Q23')

#'Data Scientist'
ds_df = jobrole.get_group('Data Scientist')
ds_df = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Languages', data = ds_df)
sns.barplot(data = ds_df, x = ds_df['Languages'][:5], y = 'Count', ax = axes[0][0], palette = 'deep')
ax = axes[0][0]
ax.set_title('Data Scientist', loc = 'right', size = 20)
ax.set_ylabel(None)
ax.set_xlabel(None)

#'Machine Learning/ MLops Engineer'
ds_df = jobrole.get_group('Machine Learning/ MLops Engineer')
ds_df = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Languages', data = ds_df)
sns.barplot(data = ds_df, x = ds_df['Languages'][:5], y = 'Count', ax = axes[0][1], palette = 'deep')
ax = axes[0][1]
ax.set_title('Machine Learning Engineer', loc = 'right', size = 20)
ax.set_ylabel(None)
ax.set_xlabel(None)

#'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'
ds_df = jobrole.get_group('Data Analyst (Business, Marketing, Financial, Quantitative, etc)')
ds_df = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Languages', data = ds_df)
sns.barplot(data = ds_df, x = ds_df['Languages'][:5], y = 'Count', ax = axes[1][0], palette = 'deep')
ax = axes[1][0]
ax.set_title('Data Analyst', loc = 'right', size = 20)
ax.set_ylabel(None)
ax.set_xlabel(None)

#'Research Scientist'
ds_df = jobrole.get_group('Research Scientist')
ds_df = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Languages', data = ds_df)
sns.barplot(data = ds_df, x = ds_df['Languages'][:5], y = 'Count', ax = axes[1][1], palette = 'deep')
ax = axes[1][1]
ax.set_title('Research Scientist', loc = 'right', size = 20)
ax.set_ylabel(None)
ax.set_xlabel(None)

# 'Software Engineer'
ds_df = jobrole.get_group('Software Engineer')
ds_df = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Languages', data = ds_df)
sns.barplot(data = ds_df, y = 'Count', x = ds_df['Languages'][:5], ax = axes[2][0], palette = 'deep')
ax = axes[2][0]
ax.set_title('Software Engineer', loc = 'right', size = 20)
ax.set_ylabel(None)
ax.set_xlabel(None)

# 'Data Administrator'
ds_df = jobrole.get_group('Manager (Program, Project, Operations, Executive-level, etc)')
ds_df = to_transform('Q12_1', 'Q12_15', 'lang_df', 'lang', 'Languages', data = ds_df)
sns.barplot(data = ds_df, y = 'Count', x = ds_df['Languages'][:5], ax = axes[2][1], palette = 'deep')
ax = axes[2][1]
ax.set_title('Manager', loc = 'right', size = 20)
ax.set_ylabel(None)
ax.set_xlabel(None)


fig.suptitle('Languages Used By Some ML/DS Professionals', size = 30, weight = 'bold')

plt.tight_layout()
plt.show()
st.pyplot(fig)


#st.subheader('Visualisation Libraries Used On A Regular Basis')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
visual = to_transform('Q15_1', 'Q15_15', 'v_df', 'visual', 'Visualisation Library')

fig= to_show(sname = visual, title = 'Top Visualisation Libraries Used On A regualar Basis')
plt.xlabel(None)
plt.ylabel(None)
st.pyplot(fig)

#st.subheader('For How Long Have They Been Using Machine Learning Methods?')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
df['Q16'] = df['Q16'].str.replace('machine', 'machine\n')
sns.countplot(data = df, y = 'Q16', palette = 'deep')

plt.title('Using Machine Learning Methods For How Long?', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None)
#plt.tight_layout()
st.pyplot(fig)
#plt.show()

#st.subheader('Machine Learning Frameworks Used On A Regular Basis')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
ml_frame = to_transform('Q17_1', 'Q17_15', 'ml_df', 'ml_frame', 'ML Framework')

fig = to_show(ml_frame, 'regularly used machine learning frameworks')
plt.xlabel(None)
plt.ylabel(None)
st.pyplot(fig)


#st.subheader('Regularly Used Machine Learning Algorithms')
st.caption('enter graph and text here')

ml_algo = to_transform('Q18_1', 'Q18_14', 'ml_df', 'ml_algo', 'ML Algorithm')
ml_algo['ML Algorithm'] = ml_algo['ML Algorithm'].str.replace('(', '\n(')
#ml_algo['ML Algorithm'] = ml_algo['ML Algorithm'].drop('Other')
fig, ax = plt.subplots()

fig = to_show(ml_algo, 'top regularly used machine learning algorithms')
plt.xlabel(None)
plt.ylabel(None)
st.pyplot(fig)

#st.subheader('Regularly Used Computer Vision Algorithms')
st.caption('enter graph and text here')
fig, ax = plt.subplots(1, 1, figsize = (25, 15))
cv = to_transform('Q19_1', 'Q19_8', 'cv_df', 'cv', 'CV Algorithm')

cv['CV Algorithm'] = cv['CV Algorithm'].str.replace('(', '\n(')

fig = to_show(cv, 'computer vision algorithms used on a regular basis')
#plt.figure(figsize=(25, 15))
plt.xlabel(None)
plt.ylabel(None)
st.pyplot(fig)


# st.subheader('rando stuff')
st.caption('enter graph and text here')

fig, axes = plt.subplots(3, 2, figsize = (25, 15))
jobrole = df.groupby('Q23')

fig.suptitle('Most Used Machine Learning Algorithms w.r.t. Job Title\n', weight = 'bold', size = 35)

#'Data Scientist'
ds_df = jobrole.get_group('Data Scientist')
ds_df = to_transform('Q18_1', 'Q18_14', 'lang_df', 'lang', 'ML Algorithm', data = ds_df)
g = sns.barplot(data = ds_df, y = ds_df['ML Algorithm'][:5], x = 'Count', ax = axes[0][0], palette = 'deep')
g.set_title('Data Scientist', loc = 'right', size = 20)
g.set_ylabel(None)
g.set_xlabel(None)



#'Machine Learning/ MLops Engineer'
ds_df = jobrole.get_group('Machine Learning/ MLops Engineer')
ds_df = to_transform('Q18_1', 'Q18_14', 'lang_df', 'lang', 'ML Algorithm', data = ds_df)
g = sns.barplot(data = ds_df, y = ds_df['ML Algorithm'][:5], x = 'Count', ax = axes[0][1], palette = 'deep')
g.set_title('Machine Learning Engineer', loc = 'right', size = 20)
g.set_ylabel(None)
g.set_xlabel(None)


#'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'
ds_df = jobrole.get_group('Data Analyst (Business, Marketing, Financial, Quantitative, etc)')
ds_df = to_transform('Q18_1', 'Q18_14', 'lang_df', 'lang', 'ML Algorithm', data = ds_df)
g = sns.barplot(data = ds_df, y = ds_df['ML Algorithm'][:5], x = 'Count', ax = axes[1][0], palette = 'deep')
g.set_title('Data Analyst', loc = 'right', size = 20)
g.set_ylabel(None)
g.set_xlabel(None)


#'Research Scientist'
ds_df = jobrole.get_group('Research Scientist')
ds_df = to_transform('Q18_1', 'Q18_14', 'lang_df', 'lang', 'ML Algorithm', data = ds_df)
g = sns.barplot(data = ds_df, y = ds_df['ML Algorithm'][:5], x = 'Count', ax = axes[2][0], palette = 'deep')
g.set_title('Research Scientist', loc = 'right', size = 20)
g.set_ylabel(None)
g.set_xlabel(None)


# 'Software Engineer'
ds_df = jobrole.get_group('Software Engineer')
ds_df = to_transform('Q18_1', 'Q18_14', 'lang_df', 'lang', 'ML Algorithm', data = ds_df)
g = sns.barplot(data = ds_df, y = ds_df['ML Algorithm'][:5], x = 'Count', ax = axes[1][1], palette = 'deep')
g.set_title('Software Engineer', loc = 'right', size = 20)
g.set_ylabel(None)
g.set_xlabel(None)

# 'Data Administrator'
ds_df = jobrole.get_group('Manager (Program, Project, Operations, Executive-level, etc)')
ds_df = to_transform('Q18_1', 'Q18_14', 'lang_df', 'lang', 'ML Algorithm', data = ds_df)
g = sns.barplot(data = ds_df, y = ds_df['ML Algorithm'][:5], x = 'Count', ax = axes[2][1], palette = 'deep')
g.set_title('Manager', loc = 'right', size = 20)
g.set_ylabel(None)
g.set_xlabel(None)

plt.tight_layout()
st.pyplot(fig)
plt.tight_layout()
plt.show()

# st.subheader('rando7')
st.caption('enter graph and text here')
fig, axs = plt.subplots(5, 3, figsize = (30, 30))
country_job_salary = df.groupby(['Q4', 'Q23'])
fig.suptitle('Annual Income Of Certain Jobs By Country\n', weight = 'bold', size = 30)
#fig.supxlabel('Count', weight = 'bold', size = 20)

# india, data scientist
g = country_job_salary.get_group(('India', 'Data Scientist'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[0][0], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('India|Data Scientist', loc = 'right', size = 25)

# usa, data scientist
g = country_job_salary.get_group(('United States\n of America', 'Data Scientist'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[0][1], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('USA|Data Scientist', loc = 'right', size = 25)

# canada, data scientist
g = country_job_salary.get_group(('Canada', 'Data Scientist'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[0][2], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('Canada|Data Scientist', loc = 'right', size = 25)

# india, ml
g = country_job_salary.get_group(('India', 'Machine Learning/ MLops Engineer'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[1][0], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('India|ML Engineer', loc = 'right', size = 25)

# usa, ml
g = country_job_salary.get_group(('United States\n of America', 'Machine Learning/ MLops Engineer'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[1][1], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('USA|ML Engineer', loc = 'right', size = 25)

# canada, ml
g = country_job_salary.get_group(('Canada', 'Machine Learning/ MLops Engineer'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[1][2], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('Canada|ML Engineer', loc = 'right', size = 25)

# india, research
g = country_job_salary.get_group(('India', 'Research Scientist'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[2][0], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('India|Research Scientist', loc = 'right', size = 25)

# usa, research
g = country_job_salary.get_group(('United States\n of America', 'Research Scientist'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[2][1], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('USA|Research Scientist', loc = 'right', size = 25)

# canada, research
g = country_job_salary.get_group(('Canada', 'Research Scientist'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[2][2], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('Canada|Research Scientist', loc = 'right', size = 25)

# india, se
g = country_job_salary.get_group(('India', 'Software Engineer'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[3][0], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('India|Sofware Engineer', loc = 'right', size = 25)

# usa, se
g = country_job_salary.get_group(('United States\n of America', 'Software Engineer'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[3][1], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('USA|Sofware Engineer', loc = 'right', size = 25)


# canada, se
g = country_job_salary.get_group(('Canada', 'Software Engineer'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[3][2], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('Canada|Sofware Engineer', loc = 'right', size = 25)

# india, da
g = country_job_salary.get_group(('India', 'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[4][0], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('India|Data Analyst', loc = 'right', size = 25)

# usa, da
g = country_job_salary.get_group(('United States\n of America', 'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[4][1], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('USA|Data Analyst', loc = 'right', size = 25)

# canada, da
g = country_job_salary.get_group(('Canada', 'Data Analyst (Business, Marketing, Financial, Quantitative, etc)'))['Q29'].value_counts().head(7).plot(kind = 'barh', ax = axs[4][2], color = sns.palettes.color_palette('deep'))
g.invert_yaxis()
g.set_title('Canada|Data Analyst', loc = 'right', size = 25)

#plt.savefig('annual_income_job_country.png', dpi = 100, bbox_inches = 'tight')
plt.tight_layout()
st.pyplot(fig)

#st.subheader('Rando9')
st.caption('enter graph and text here')
fig, axs = plt.subplots(3, 2, figsize = (28, 20))

#ds_df = job.get_group('Data Scientist')
#ds_df = to_transform('Q28_1', 'Q28_8', 'role_df', 'ds_df', 'Job Role', data = ds_df)
#g = sns.barplot(data = ds_df, x = 'Count', y = 'Job Role', ax = axs[0][0])
#g.set_title('Data Scientist', loc = 'right', size = 25)
#g.set_ylabel(None)
#g.set_xlabel(None)
#
#
#ds_df = job.get_group('Machine Learning/ MLops Engineer')
#ds_df = to_transform('Q28_1', 'Q28_8', 'role_df', 'ds_df', 'Job Role', data = ds_df)
#g = sns.barplot(data = ds_df, x = 'Count', y = 'Job Role', ax = axs[0][1])
#g.set_title('Machine Learning Engineer', loc = 'right', size = 25)
#g.set_ylabel(None)
#g.set_xlabel(None)
#
#
#ds_df = job.get_group('Data Analyst (Business, Marketing, Financial, Quantitative, etc)')
#ds_df = to_transform('Q28_1', 'Q28_8', 'role_df', 'ds_df', 'Job Role', data = ds_df)
#g = sns.barplot(data = ds_df, x = 'Count', y = 'Job Role', ax = axs[1][0])
#g.set_title('Data Analyst', loc = 'right', size = 25)
#g.set_ylabel(None)
#g.set_xlabel(None)
#
#
#ds_df = job.get_group('Software Engineer')
#ds_df = to_transform('Q28_1', 'Q28_8', 'role_df', 'ds_df', 'Job Role', data = ds_df)
#g = sns.barplot(data = ds_df, x = 'Count', y = 'Job Role', ax = axs[1][1])
#g.set_title('Software Engineer', loc = 'right', size = 25)
#g.set_ylabel(None)
#g.set_xlabel(None)
#
#
#ds_df = job.get_group('Research Scientist')
#ds_df = to_transform('Q28_1', 'Q28_8', 'role_df', 'ds_df', 'Job Role', data = ds_df)
#g = sns.barplot(data = ds_df, x = 'Count', y = 'Job Role', ax = axs[2][0])
#g.set_title('Research Scientist', loc = 'right', size = 25)
#g.set_ylabel(None)
#g.set_xlabel(None)
#
#
#ds_df = job.get_group('Manager (Program, Project, Operations, Executive-level, etc)')
#ds_df = to_transform('Q28_1', 'Q28_8', 'role_df', 'ds_df', 'Job Role', data = ds_df)
#g = sns.barplot(data = ds_df, x = 'Count', y = 'Job Role', ax = axs[2][1])
#g.set_title('Manager', loc = 'right', size = 25)
#g.set_ylabel(None)
#g.set_xlabel(None)
#
#fig.suptitle('Job Title v/s What They Actually Do', weight = 'bold', size = 35)
#plt.tight_layout()
#plt.show()
#
#st.pyplot(fig)
#
#st.subheader('Regularly Used Natural Language Processing Algorithms')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
nlp = to_transform('Q20_1', 'Q20_6', 'nlp_df', 'nlp', 'NLP Algorithm')
nlp['NLP Algorithm'] = nlp['NLP Algorithm'].str.replace('(', '\n(')

fig = to_show(nlp, 'regularly used natural language processing algorithms')
plt.xlabel(None)
plt.ylabel(None)

st.pyplot(fig)

st.header('Job and Industry Related Visuals')
#st.subheader('Job Titles Held By The Participants')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
#df['Q23'] = df['Q23'].str.replace('(', '\n(')
sort = list(df['Q23'].value_counts().index)

sns.countplot(data = df, y = 'Q23', 
              order = sort, palette = 'deep')

plt.title('Frequency of Job Titles Among Participants', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None)
st.pyplot(fig)
plt.show()

#st.subheader('rando stuff2')
st.caption('enter graph and txt here')
fig.suptitle('Industries Where Some ML/DS Profesionals Work In', weight = 'bold', size = 20)
fig, axs = plt.subplots(3, 2, figsize = (23, 15))
job = df.groupby('Q23')

job_df = job.get_group('Data Scientist')['Q24'].value_counts()
g = job_df.head().plot.pie(ax = axs[0][0], startangle = 45)
g.set_title('Data Scientist', loc = 'center', size = 15)
g.set_ylabel(None)

job_df = job.get_group('Software Engineer')['Q24'].value_counts()
g = job_df.head().plot.pie(ax = axs[0][1], startangle = 45)
g.set_title('Software Engineer', loc = 'center', size = 15) 
g.set_ylabel(None)

job_df = job.get_group('Data Analyst (Business, Marketing, Financial, Quantitative, etc)')['Q24'].value_counts()
g = job_df.head().plot.pie(ax = axs[1][0], startangle = 0)
g.set_title('Data Analyst', loc = 'center', size = 15)
g.set_ylabel(None)

job_df = job.get_group('Machine Learning/ MLops Engineer')['Q24'].value_counts()
g = job_df.head().plot.pie(ax = axs[1][1], startangle = 45)
g.set_title('ML Engineer', loc = 'center', size = 15)
g.set_ylabel(None)

job_df = job.get_group('Research Scientist')['Q24'].value_counts()
g = job_df.head().plot.pie(ax = axs[2][0], startangle = 45)
g.set_title('Research Scientist', loc = 'center', size = 15)
g.set_ylabel(None)

job_df = job.get_group('Data Engineer')['Q24'].value_counts()
g = job_df.head().plot.pie(ax = axs[2][1], startangle = 45)
g.set_title('Data Engineer', loc = 'center', size = 15)
g.set_ylabel(None)

#plt.tight_layout()
plt.show()
st.pyplot(fig)

#st.subheader('In Which Industry Do Most ML/DS People Work?')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
sort = list(df['Q24'].value_counts().index)

sns.countplot(data = df, y = 'Q24', 
              order = sort, palette = 'deep')

plt.title('In What Industry Do The Participants Work?', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None)
st.pyplot(fig)
plt.show()

#st.subheader('Size of The Comapanies To Which The Particiapnsta work in')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
sort = list(df['Q25'].value_counts().index)

sns.countplot(data = df, y = 'Q25', order = sort)

plt.title('Size of The Company Where Participants Work At', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None)

plt.xlim(1250, 2150)
st.pyplot(fig)
plt.show()

#st.subheader('Size Of the DS Department')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
sort = list(df['Q26'].value_counts().index)

sns.countplot(data = df, x = 'Q26', order = sort)

plt.title('Size of Data Science Departments Where\nML/DS Professionals Work At', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None)
st.pyplot(fig)

plt.show()


#st.subheader('What are they actually doing')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
role = to_transform('Q28_1', 'Q28_8', 'role_df', 'role', 'Job Role')

role['Job Role'] = role['Job Role'].str.replace('uses', 'uses\n')
role['Job Role'] = role['Job Role'].str.replace('that', 'that\n')

plt.figure(figsize = (8, 6.5))
fig = to_show(role, 'important role at work')
plt.xlabel(None)
plt.ylabel(None)

st.pyplot(fig)

#st.subheader('Which Cloud Service Is Used Mostly')
st.caption('enter graph and text here')
fig, ax = plt.subplots()

cloud = to_transform('Q31_1', 'Q31_12', 'cloud_df', 'cloud', 'Cloud Service')

fig = to_show(cloud, 'top used cloud service')
plt.xlabel(None)
plt.ylabel(None)

st.pyplot(fig)

#st.subheader('Which Database Is Used Mostly')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
db = to_transform('Q35_1', 'Q35_16', 'db_df', 'db', 'Database')

fig = to_show(db, 'top used databases')
plt.xlabel(None)
plt.ylabel(None)
st.pyplot(fig)

#st.subheader('Which BI Tool Is Used Mostly')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
bi = to_transform('Q36_1', 'Q36_15', 'bi_df', 'bi', 'BI Tool')

fig = to_show(bi, 'top used business intelligence tools')
plt.xlabel(None)
plt.ylabel(None)
st.pyplot(fig)

st.header('Research Related Infrmation')
#st.subheader('Do Most People research in this field')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
sns.countplot(data = df, x = 'Q9')

plt.title('Has The Participant Published Any Research?', weight = 'bold')
plt.xlabel(None)
plt.ylabel(None, labelpad = 15)

st.pyplot(fig)

# st.subheader('What is the scope of their research')
st.caption('enter graph and text here')
fig, ax = plt.subplots()
research = to_transform('Q10_1', 'Q10_3', 'research_df', 'research', 'Research Level')
research['Research Level'] = research['Research Level'].str.replace('machine', '\nmachine')

sns.barplot(data = research, y = 'Research Level', x = 'Count')

plt.title('Scope of Research', weight = 'bold')
plt.xlabel(None, labelpad = 10)
plt.ylabel(None)

st.pyplot(fig)

st.header('Additional Stats For Nerds')
