import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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

import streamlit as st

df['Q4'] = df['Q4'].str.replace('United States of America', 'United States\n of America')

st.set_page_config(page_title="Kaggle Machine Learning & Data Science Report 2022", layout="centered", page_icon='👨‍💻')


st.title(':blue[2022 Kaggle Machine Learning & Data Science Survey Analysis]')
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
st.header('Executive Summary')
summary = \
"""
Most people who have given this survey are aged between 18 and 30. More than half of the people are men. Nearly 50% of the 
entries are from people are from India. Due to which a huge shift in the survey results has been observed. This survey has been
attended by students as well as professionals with an equal split. The Machine Learning and Data Science space is a very skilled
profession. More than 70% are graduates, and close to 50% have a master's degree. If we compare the education level with the
western world, we observe that they mostly have a master's degree, and have far more doctoral degrees. Another interesting insight 
is the annual income. A drastic difference in annual income when comparing India to other western countries. This could very well be 
explained by the higher education requirements and higher market demand of machine learning and data science jobs in the western world.
People from India are mostly freelancing or not earning much in this field. Therefore it is safe to assume that going abroad for a
master's degree and also working there would be a much safer bet.\n
Moving on, we see that most people enrolled in this survey are relatively new coders, by that I mean they have been coding for
around 1-3 years mostly. Python and R are the most used languages. For data visualisation, most people use Matplotlib, Seaborn and Plotly.
So mastering those would be bare a minimum. Majority of the people are new at implementing Machine Learning methods. And for the 
Machine Learning enthusiast, Scikit-learn, TensorFlow and Keras is necessary to learn and practise. Regression, Random Forests and CNNs
are the most used algorithms right now. So starting with these algorithms and mastering them would help a lot. Computer Vision models like
VGG, Inception, etc are used widely in the industry. Chat GPT-3 is a very powerful AI Chatbot, it is possible due to transformer 
language model. Another NLP algorithm which is quite ubiquitous is word embeddings/vectors type modes like GLoVe and fastText.
Besides these, some very important tools like Cloud Services such as AWS and GCP, and Databases such as MySQL and PostgreSQL, and BI 
Tools like Tableau and Power BI are important. So familiarity with these would be very well received.\n
Data Scientist and Data Analyst are the most famous popular jobs in the Machine Learning and Data Science Space. 
Followed by Software Engineers, Teaching professionals, Managerial positions and the end Research Scientists, and Machine Learning Engineers.
This might be due to the fact that Machine Learning requires high level mathematical information, which isn't favourable by many people.
So if one wants to become a good Machine Learning Engineer, they must also be good at mathematics and other tools necessary.
Most people, except Machine Learning Engineers, irrespective of their job titles are analyzing and trying to make sense of data for a business point of view.
Comparatively very few people are building and experimenting with ML models. The people 
who practise that consist of Machine Learning Engineers, Data Scientists and Research Scientists. Typically the Data Science Department consists of more that 20 people.
Majority of people in this space work in Technology and Accounting/Finance related industries. Python, SQL and R are the most popular languages among high level Machine 
Learning Engineers and Data Scientists. Instead of R, aspiring Machine Learning Engineers can learn Bash and C++, and for Research Scientists, they can replace it by
MATLAB. Besides Regression, Random Forests and CNNs, GBMs are widely used across the industry.\n
If we look at the jobs by country, we discover 
that the top 3 jobs are Data Scientist, Data Analyst and Managerial roles. There are very few Machine Learning and Research Scientist jobs. 
So if one dreams of becoming that, they must be the best in the game. At the end, we again observe that India isn't the job country for the ML/DS space. 
The US and Canada have jobs that pays 6 figures for the same job. Between US and Canada, The US has a greater pay, it could be because of the high living costs 
associated in the US. So going to The States would be very much lucrative and wil also help in job growth.\n

"""
st.write(summary)

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 6))
courses = to_transform('Q6_1', 'Q6_12', fname = 'org1_df', sname = 'org1', col_name = 'Courses')
courses['Courses'] = courses['Courses'].str.replace('(', '\n(')
sns.barplot(data = courses, y = 'Courses', x = 'Count', ax = ax1, palette='deep')
ax1.set_ylabel(None); ax1.set_xlabel(None)
ax1.set_yticklabels(courses['Courses'], size = 14)
ax1.set_xticklabels(ax1.get_xticks(), size = 12)
courses = to_transform('Q7_1', 'Q7_7', fname = 'org2_df', sname = 'org2', col_name = 'Courses')
courses['Courses'] = courses['Courses'].str.replace('(', '\n(')
sns.barplot(data = courses, y = 'Courses', x = 'Count', ax = ax2, palette='deep');
ax = ax2; ax.invert_xaxis(); ax2.yaxis.set_ticks_position('right'); 
ax2.set_ylabel(None); ax2.set_xlabel(None)
ax2.set_yticklabels(courses['Courses'], size = 14)
ax2.set_xticklabels(ax2.get_xticks(), size = 12)
fig.suptitle('What Introduced Them To Data Science\nVS\nMost Helpful Course', weight = 'bold', size = 15)
plt.tight_layout()
st.pyplot(fig)

caption = \
"""
This graph clearly depicts why people are running away from educational institutes. The method of learning isn't efficient at all.
People learn more easily when they use the tools that education provides them to solve their problems. Project based learning is what
is very much needed in institutions. A person should be thought how to solve a problem and not to memorise the solution. This makes the 
student a problem solver and enables them to think out of the box. These people are very much valued by companies.\n
I have made this project to practise my data visualiation skills and also get to know about trends in the ML/DS space. I now feel very comfortable with all the skills
used in making this project. There came many problems and confusions while making this project, but overcoming them by myself has given me a
lot of confidence. 
"""
st.write(caption)
