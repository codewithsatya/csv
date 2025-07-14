# WEEK 5 ASSIGNMENT
# Title: Data Analysis and Visualization with Python
# Duration: 1 Week
#  Objective:
#  To introduce students to the fundamentals of data analysis using pandas and data visualization using matplotlib and seaborn. Students will learn how to clean, explore, and visualize structured datasets, laying the groundwork for machine learning.

# Learning Outcomes
# By the end of Week 5, students will be able to:
#1)Use pandas to read, filter, and manipulate CSV data
# import pandas as pd
# data={
#     'name':['priya','rahul','anjali','ravi','meena'],
#     'roll_number':[24,25,26,27,28],
#     'grade': ['A', 'B', 'A', 'C', 'B'],
#     'marks': [88, 72, 90, 60, 75]
# }
# df = pd.DataFrame(data)
# df.to_csv('students.csv', index=False)
# print("students.csv created successfully.")

#2)Perform basic statistical analysis and summaries
# import pandas as pd
# df=pd.read_csv('students.csv')
# print(df.head())
# 📊 Basic Summary Statistics
# print(df.describe())
# 🔢 Get mean, median, mode
# print("mean marks:",df['marks'].mean())
# print("median marks:",df['marks'].mean())
# print("most common grade:",df['grade'].mode()[0])
# 📋 Value counts (frequency)
# print(df['grade'].value_counts())
# # 🧮 Grouped statistics
# print(df.groupby('grade')['marks'].mean())
# print(df.groupby('grade')['name'].count())
# Identify missing or inconsistent data and handle them appropriately
# 🧹Check for missing values
# print(df.isnull().sum())

#3)Create meaningful plots (bar charts, histograms, line plots, scatter plots)
# import pandas as pd 
# import matplotlib.pyplot as plt
# import seaborn as sns
# df=pd.read_csv('students.csv')
# 📊 Bar Chart — Students per Grade
# sns.countplot(x='grade', data=df)
# plt.title("Number of Students in Each Grade")
# plt.xlabel("Grade")
# plt.ylabel("Count")
# plt.show()
# 📉Histogram — Distribution of Marks
# df['marks'] = pd.to_numeric(df['marks'], errors='coerce')
# sns.histplot(df['marks'], bins=5, kde=True)
# plt.title("Distribution of Marks")
# plt.xlabel("Marks")
# plt.ylabel("Frequency")
# plt.show() 
# 📈 Line Plot — Marks by Roll Number
# plt.plot(df['roll_number'], df['marks'], marker='o')
# plt.title("Marks by Roll Number")
# plt.xlabel("Roll Number")
# plt.ylabel("Marks")
# plt.grid(True)
# plt.show()
# 📌 Scatter Plot — Marks vs Roll Number
# sns.scatterplot(x='roll_number',y='marks',hue='grade',data=df)
# plt.title("marks vs Roll Number(Colored by grade)")
# plt.xlabel("Roll number")
# plt.ylabel("Marks")
# plt.show()
# 🖼️ Optional: Use Seaborn Style
# sns.set(style="whitegrid")
#4)Combine insights and visuals into structured reports or notebooks
# 📥 Import Libraries and Load Data
# sns.set(style="whitegrid")
# df=pd.read_csv("students.csv")
# df.head()
# 🧹Data Cleaning & Summary
# print("Missing values:\n",df.isnull().sum())
# df['marks']=pd.to_numeric(df['marks'],errors='coerce')
# print("summary:\n",df.describe())
# 📈 Visualizations with Interpretation
# 🔹 Bar Chart – Students per Grade
# sns.countplot(x='grade',data=df)
# plt.title("number of student per grade")
# plt.show()
# 🔹 Histogram – Marks Distribution
# sns.histplot(df['marks'],bins=5,kde=True)
# plt.title("distribution of marks")
# plt.show()
# 🔹 Line Plot – Marks by Roll Number
# plt.plot(df['roll_number'],df['marks'],marker='o')
# plt.title("marks by the roll_number")
# plt.xlabel("roll_number")
# plt.ylabel("marks")
# plt.show()
# 🔹 Scatter Plot – Marks vs Roll Number by Grade
# sns.scatterplot(x='roll_number',y='marks',hue='grade',data=df)
# plt.title("scatter plot:marks vs roll_number (coloured by grade)")
# plt.show()
# 📊 Statistical Summary
# print(df.groupby('grade')['marks'].mean())

# Topics Covered
#1)Introduction to pandas: Series, DataFrames
# pandas is a powerful Python library used for data analysis and manipulation. It provides two main data structures:

#📌Series
# A Series is a one-dimensional labeled array that can hold any data type — integers, strings, floats, etc.
# import pandas as pd
# s=pd.Series([10,20,30,40])
# print(s)
# Each value has an index (on the left) and the data (on the right).
# You can also specify custom indexes:
# s=pd.Series([10,20,30],index=['math','science','english'])
# print(s)
# 📌 DataFrame
# A DataFrame is a two-dimensional table (like an Excel spreadsheet). It is made of rows and columns, where:
#a)Each column is a Series
#b)The entire table is a DataFrame
data={
    'name':['alice','bob','charlie'],
    'marks':[85,92,78]
}
# df=pd.DataFrame(data)
# print(df)
# You can access columns like:
# print(df['marks'])
#2)Reading CSV files using pd.read_csv()
# 📥 Reading CSV Files with pandas
# 🔹 What is a CSV?
# CSV stands for Comma-Separated Values.
# It is a plain text file where each line is a row and columns are separated by commas.
# import pandas as pd
# df=pd.read_csv('students.csv')
# # df = pd.read_csv('C:/Users/Admin/Desktop/students.csv')
# df.to_csv('output.csv', index=False)

# print(df.head())

#3)Exploring data: .head(), .info(), .describe()
# 🟦 head() – View First Few Rows
# df.head()
# df.head(10)
# 🟦 info() – View Structure and Nulls
# df.info()
# 🟦describe() – View Summary Statistics
# df.describe
# 📘 Load the CSV
# import pandas as pd
# df=pd.read_csv('students.csv')
# 📘Explore the Data
# print(df.head())
# print(df.info())
# print(df.describe())
#4)Selecting rows/columns using loc[] and iloc[]
# 🟨 loc[] – Label-based Selection
# Use loc[] when you want to select by row labels or column names.
# import pandas as pd 
# df=pd.read_csv('students.csv')
# 🔹 Select a specific row by index label:
# df.loc[0]
# 🔹 Select multiple rows:
# df.loc[[0,2]]
# 🔹 Select specific columns:
# df.loc[:,['name','marks']]
# 🔹 Select rows with a condition:
# df.loc[df['grade']=='A']
# 🟦 iloc[] – Position-based Selection
# Use .iloc[] when you want to select by row/column numbers (integers).
# 🔹 Select a specific row:
# df.iloc[0]
# 🔹 Select a range of rows:
# df.iloc[1:4]
# 🔹 Select rows and columns by number:
# df.iloc[0:3,1:3]

#5)Filtering, sorting, and grouping data
# 🧹Filtering Data
# Filtering means selecting rows that meet a certain condition.
# # df[df['marks']>80]
# df[df['grade']=='A']
# df[(df['grade']=='A')&(df['marks']>85)]
# 🔃Sorting Data
# # Sorting means arranging data in ascending or descending order.
# 🔸 Sort by marks (ascending):
# df.sort_values(by='marks')
# 🔸 Sort by marks (descending):
# df.sort_values(by='marks', ascending=False)
# 🔸 Sort by multiple columns:
# df.sort_values(by=['grade','marks'],ascending=[True,False])
#🧮Grouping Data (Aggregation)
# Grouping is useful for computing summaries like average, max, count, etc., per group.
# 🔸 Group by grade and get average marks:
# df.groupby('grade')['marks'].mean()
# 🔸 Count of students per grade:
# df.groupby('grade').size()
# 🔸 Summary stats for each grade:
# df.groupby('grade')['marks'].describe()
#6)Handling missing values (isnull(), dropna(), fillna())
# 🟨 isnull() – Detect Missing Values
# Use isnull() to check which cells have missing values (NaN).

# df.isnull()
# 🔹 This returns a DataFrame of True/False values.

# To see which rows have any missing values:


# df[df.isnull().any(axis=1)]
# To count missing values in each column:

# df.isnull().sum()
# 🗑️ 2. dropna() – Remove Missing Values
# Removes rows or columns that contain missing values.

# 🔸 Remove rows with any NaN
# df.dropna()
# 🔸 Remove columns with NaN:
# df.dropna(axis=1)
# 🔸 Remove rows where all columns are NaN:

# df.dropna(how='all')
# 🧯 3. fillna() – Fill Missing Values
# Use this to replace missing values with something meaningful.

# 🔹 Fill with a fixed value:

# df.fillna(0)
# 🔹 Fill with mean of column:
# df['marks'].fillna(df['marks'].mean(), inplace=True)
# 🔹 Fill with method:
# Forward fill (use previous row’s value):
# df.fillna(method='ffill')
# Backward fill (use next row’s value):

# df.fillna(method='bfill')
# import pandas as pd
# data={
#     'name':['priya','rahul','anjali',None],
#     'marks':[88,None,90,75],
#     'grade':['A','B',None,'C']
# }
# df=pd.DataFrame(data)
# print(df.isnull().sum())
# df_cleaned=df.fillna("unknown")
# print(df_cleaned)

#7)Basic aggregations: mean(), sum(), count(), groupby()
# ✅ mean() – Average
# df['marks'].mean()
# 🔹 Gets the average marks of all students.
# ✅ sum() – Total
# df['marks'].sum()
# 🔹 Adds up all values in the marks column.

# ✅ count() – Number of Non-NaN Values
# df['name'].count()
# 🔹 Counts how many students have a non-empty name.

# ✅  groupby() – Aggregation by Category
# Used to perform aggregation grouped by a specific column, e.g., grade.

# 🔸 Average marks per grade:

# df.groupby('grade')['marks'].mean()
# 🔸 Total marks per grade:
# df.groupby('grade')['marks'].sum()
# 🔸 Student count per grade:
# df.groupby('grade').size()
# 🔸 Multiple aggregations:
# df.groupby('grade')['marks'].agg(['mean', 'sum', 'count'])

# import pandas as pd
# data={
#     'name':['priya','rahul','anjali','ravi','meena'],
#     'grade':['A','B','A','C','B'],
#     'marks':[88,72,90,60,75]
# }
# df=pd.DataFrame(data)

#8)Introduction to matplotlib and seaborn


#a)Bar charts, histograms, pie charts, scatter plots
# 📊 Introduction to Matplotlib
# Matplotlib is a powerful Python library used for creating static, interactive, and animated visualizations.
# The most commonly used module is matplotlib.pyplot.


# import matplotlib.pyplot as plt
# 📈 Introduction to Seaborn
# Seaborn is built on top of Matplotlib and provides a high-level interface for attractive and informative statistical graphics.


# import seaborn as sns
# 🔹 Common Plot Types
# 1. Bar Chart
# Used to show comparisons among discrete categories.


# categories = ['A', 'B', 'C']
# values = [10, 15, 7]

# plt.bar(categories, values)
# plt.title('Bar Chart')
# plt.xlabel('Category')
# plt.ylabel('Value')
# plt.show()
# 2. Histogram
# Shows the distribution of a numerical variable.


# import numpy as np

# data = np.random.randn(1000)  

# plt.hist(data, bins=30, color='skyblue')
# plt.title('Histogram')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()
# 3. Pie Chart
# Used to show proportions.

# labels = ['A', 'B', 'C']
# sizes = [40, 35, 25]

# plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
# plt.title('Pie Chart')
# plt.axis('equal')  
# plt.show()
# 4. Scatter Plot
# Used to show relationships between two numeric variables.

# x = [1, 2, 3, 4, 5]
# y = [5, 4, 2, 1, 0]

# plt.scatter(x, y, color='red')
# plt.title('Scatter Plot')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()

#b)Plot formatting: titles, labels, legends
# 🎨 Plot Formatting
# ✅ Titles, Labels, Legends
# import matplotlib.pyplot as plt
# import pandas as pd

# data = {
#     'name': ['priya', 'rahul', 'anjali', 'ravi', 'meena'],
#     'roll_number': [24, 25, 26, 27, 28],
#     'grade': ['A', 'B', 'A', 'C', 'B'],
#     'marks': [88, 72, 90, 60, 75]
# }

# df = pd.DataFrame(data)


# X = df['roll_number']
# Y = df['marks']
# plt.plot(X, Y, label='Line 1')

# plt.title('Sample Plot')         
# plt.xlabel('X Axis')             
# plt.ylabel('Y Axis')            
# plt.legend()                     
# plt.grid(True)                   
# plt.show()
# 🐧 Example with Seaborn
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd

# tips = sns.load_dataset('tips')

# sns.barplot(x='day', y='total_bill', data=tips)
# plt.title('Average Total Bill per Day')
# plt.show()

# 1️⃣ Title

# plt.title("Your Title Here")
# This adds a title at the top of the plot.

# 2️⃣ X-axis and Y-axis Labels

# plt.xlabel("X-axis Label")
# plt.ylabel("Y-axis Label")
# These label the horizontal (X) and vertical (Y) axes.

# 3️⃣ Legend

# plt.legend()
# This shows a box that explains what each plotted line, point, or bar means (if you've set labels).

# You must pass a label="..." in your plot line/bar/scatter for this to work.

# import pandas as pd
# import matplotlib.pyplot as plt
# data={
#     'name':['priya','rahul','anjali','ravi','meena'],
#     'roll_number':[24,25,26,27,28],
#     'grade':['A','B','A','C','B'],
#     'marks':[88, 72, 90, 60, 75]
# }
# df=pd.DataFrame(data)
# plt.plot(df['roll_number'],df['marks'],marker='o',color='blue',label='marks')
# plt.title('students marks bby roll number')
# plt.xlabel('roll number')
# plt.ylabel('marks')
# plt.legend()
# plt.grid(True)
# plt.show()


#9)Exporting results: to_csv(), saving plots as images
# After analyzing or modifying your data in pandas, you may want to save it to a CSV file.
# ✅ Exporting DataFrames to CSV – to_csv()
# After analyzing or modifying your data in pandas, you may want to save it to a CSV file.
# import pandas as pd
# data={
#     'name':['priya','rahul','anjali','ravi','meena'],
#     'roll_number':[24,25,26,27,28],
#     'grade':['A','B','A','C','B'],
#     'marks':[88,72,90,60,75]
# }
# df.to_csv('students.csv',index=False)

# ✅ 2. Saving Plots as Images – plt.savefig()
# You can save any matplotlib figure as an image file (PNG, JPG, PDF, etc.).

# import matplotlib.pyplot as plt
# plt.plot(df['roll_number'],df['marks'],marker='o',label='marks')
# plt.title('students marks')
# plt.xlabel('roll number')
# plt.ylabel('marks')
# plt.legend()
# plt.savefig('student_marks.png',dpi=300,bbox_inches='tight')
# plt.show()

# Assignment Tasks
# Task 1: Data Exploration with Pandas
# Use a dataset such as students_performance.csv (or use the one provided):
# Sample columns:
# javascript
# CopyEdit
# StudentID, Name, Gender, Math, English, Science

#1)Load the data into a DataFrame
import pandas as pd
data={
    'name':['priya','rahul','anjali','ravi','meena','aryan','neeta','simran','kunal','ramesh'],
    'student_id':[101,102,103,104,105,106,107,108,109,110],
    'gender': ['F','M','F','M','F','M','F','M','F','M'],
    'math': [85,76,90,60,88,70,95,55,82,78],
    'english':[78,74,85,58,92,66,90,60,86,72],
    'science':[92,88,95,65,91,80,98,59,87,75]
}
df = pd.DataFrame(data)
# df.to_csv('students_performance.csv', index=False)
# print("students_performance.csv created successfully.")

# import pandas as pd
df=pd.read_csv('students_performance.csv')
print(df.head())

# # 2)Display the first 10 rows
# print(df.head(10))
#3)Show dataset summary using .info() and .describe()
# 🧩 info() – Dataset Structure
# print("dataset info:")
# print(df.info())

# 📊 describe() – Statistical Summary
# print("statistical summary:")
# print(df.describe())
#4)Check for and handle missing values
# print("missing value in each column:")
# print(df.isnull().sum())

#5)Sort students by total score (Math + English + Science)
# import pandas as pd

# df = pd.read_csv("students_performance.csv")

# print("Column names:", df.columns)


# df['TotalScore'] = df['math'] + df['english'] + df['science']

# df_sorted = df.sort_values(by='TotalScore', ascending=False)


# print("Students sorted by total score (highest first):")
# print(df_sorted[['student_id', 'name', 'math', 'english', 'science', 'TotalScore']].head(10))

#6)Create a new column AverageScore
# import pandas as pd
# df = pd.read_csv("students_performance.csv")
# print("Actual column names:", df.columns.tolist())
# df['TotalScore'] = df['math'] + df['english'] + df['science']
# df['AverageScore'] = df['TotalScore'] / 3
# df_sorted = df.sort_values(by='TotalScore', ascending=False)
# print("\nTop 10 students sorted by total score:")
# print(df_sorted[['student_id', 'name', 'math', 'english', 'science', 'TotalScore', 'AverageScore']].head(10))

# Task 2: Grouping and Aggregations
#1)Group by Gender and calculate average scores for each subject
# import pandas as pd
# df=pd.read_csv("students_performance.csv")
# print("column names:",df.columns.tolist())
# Average_Scores_by_gender = df.groupby('gender')[['math', 'english', 'science']].mean()

# print("\n🔹 Average scores for each subject grouped by Gender:")
# print(Average_Scores_by_gender)

#2)Find how many students scored above 85 in Math
# import pandas as pd
# df=pd.read_csv("students_performance.csv")
# high_math_scorers=df[df['math']>85]
# count=high_math_scorers.shape[0]

# print(f"\n Number of students who scored above 85 in Math: {count}")
#3)List top 5 students by average score
# import pandas as pd
# df=pd.read_csv("students_performance.csv")
# df['TotalScore']=df['math']+df['english']+df['science']
# df['AverageScore']=df['TotalScore']/3
# top_5_students=df.sort_values(by='AverageScore',ascending=False).head(5)
# print("\n Top 5 students by Average Score:")
# print(top_5_students[['student_id', 'name', 'math', 'english', 'science', 'AverageScore']])

# Task 3: Visualization
# Using matplotlib or seaborn, create the following:
#1)Bar chart of average scores per subject
# import pandas as pd
# import matplotlib.pyplot as plt
# df=pd.read_csv("students_performance.csv")
# average_scores=df[['math','english','science']].mean()
# plt.figure(figsize=(6,4))
# average_scores.plot(kind='bar',color=['skyblue','lightgreen','salmon'])
# plt.title("Average Scores per Subject")
# plt.ylabel("Average Score")
# plt.xlabel("subjects")
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()
#2)Pie chart of gender distribution
# import pandas as pd
# import matplotlib.pyplot as plt
# df=pd.read_csv("students_performance.csv")
# gender_count=df['gender'].value_counts()
# plt.figure(figsize=(6,6))
# plt.pie(
#     gender_count,
#     labels=gender_count.index,
#     autopct='%1.1f%%',
#     startangle=90,
#     colors=['#66b3ff', '#ff9999']
# )

# plt.title("Gender Distribution of Students")
# plt.axis('equal')  
# plt.show()


#3)Histogram of Math scores

# ✅ Option 1: Using Seaborn (recommended)
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# df=pd.read_csv("students_performance.csv")
# plt.figure(figsize=(6,4))
# sns.histplot(df['math'],bins=10,kde=True,color='skyblue')
# plt.title("Distribution of math scores")
# plt.xlabel("math score")
# plt.ylabel("number of students")
# plt.tight_layout()
# plt.show()

# ✅ Option 2: Using Matplotlib only
# import pandas as pd
# import matplotlib.pyplot as plt
# df=pd.read_csv("students_performance.csv")
# plt.figure(figsize=(6,4))
# plt.hist(df['math'],bins=10,color='lightcoral',edgecolor='black')
# plt.title("histogram of math scores")
# plt.xlabel("math score")
# plt.ylabel("number of students")
# plt.tight_layout()
# plt.show()

#4)Scatter plot of Math vs Science scores
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# df=pd.read_csv("students_performance.csv")
# plt.figure(figsize=(6,4))
# sns.scatterplot(x='math',y='science',hue='gender',data=df,palette='Set2')

# plt.title("Math vs Science Scores")
# plt.xlabel("Math Score")
# plt.ylabel("Science Score")
# plt.legend(title='gender')
# plt.tight_layout()
# plt.show()

#5)Line chart showing average scores by student ID or rank
# ✅ Line Chart by Student Rank (sorted by average score)
# import pandas as pd
# import matplotlib.pyplot as plt
# df=pd.read_csv("students_performance.csv")
# df['totalscore']=df['math']+df['english']+df['science']
# df['averagescore']=df['totalscore']/3
# df_sorted=df.sort_values(by='averagescore',ascending=False).reset_index(drop=True)
# plt.figure(figsize=(8,4))
# plt.plot(df_sorted.index+1,df_sorted['averagescore'],marker='o',linestyle='-',color='green')
# plt.title("average score by student rank")
# plt.xlabel("rank(1=Top student)")
# plt.ylabel("average score")
# plt.grid("True")
# plt.tight_layout()
# plt.show()

# Mini Project: Student Performance Dashboard (Jupyter Notebook)
# Goal: Create an interactive or narrative dashboard using a Jupyter Notebook that summarizes student performance visually and statistically.
# Requirements:
#1)Load and clean a CSV dataset
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sn
# df=pd.read_csv("students_performance.csv")
# df.head()
# ✅ Clean Data
# print(df.isnull().sum())
# df=df.dropna()
# print(df.columns.tolist())
#2)Calculate and display descriptive stats
# df['totalscore']=df['math']+df['english']+df['science']
# df['averagescore']=df['totalscore']/3
#3)Show top and bottom 5 performers
# 📌 Show Top 5 Performers (Highest AverageScore):
# top_5=df.sort_values(by='averagescore',ascending=False).head(5)
# print("\nTop 5 performers:")
# print(top_5[['student_id', 'name', 'math', 'english', 'science', 'averagescore']])
# 📌 Show bottom 5 Performers (lowest AverageScore):
# bottom_5 = df.sort_values(by='averagescore', ascending=True).head(5)
# print("\nBottom 5 performers:")
# print(bottom_5[['student_id', 'name', 'math', 'english', 'science', 'averagescore']])
#4)Visualize at least 4 types of plots
# ✅ Prerequisites (Run first if not already done)
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

df = pd.read_csv("students_performance.csv")



df['totalscore'] = df['math'] + df['english'] + df['science']
df['averagescore'] = df['totalscore'] / 3


# sns.set(style="whitegrid")


# 📊 Bar Chart – Average Scores per Subject

# avg_scores = df[['math', 'english', 'science']].mean()

# plt.figure(figsize=(6, 4))
# avg_scores.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])

# plt.title("Average Scores per Subject")
# plt.ylabel("Average Score")
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()


# 🥧 Pie Chart – Gender Distribution

# gender_counts = df['gender'].value_counts()

# plt.figure(figsize=(5, 5))
# plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
# plt.title("Gender Distribution")
# plt.axis('equal')
# plt.show()


# 📈 Line Chart – Average Score by Rank

# ranked_df = df.sort_values(by='averagescore', ascending=False).reset_index(drop=True)

# plt.figure(figsize=(8, 4))
# plt.plot(ranked_df.index + 1, ranked_df['averagescore'], marker='o', linestyle='-', color='green')

# plt.title("Average Score by Student Rank")
# plt.xlabel("Rank (1 = Top)")
# plt.ylabel("Average Score")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# 🔘 Scatter Plot – Math vs Science
# plt.figure(figsize=(6, 4))
# sns.scatterplot(data=df, x='math', y='science', hue='gender', palette='Set2')

# plt.title("Math vs Science Scores")
# plt.xlabel("Math Score")
# plt.ylabel("Science Score")
# plt.legend(title="gender")
# plt.tight_layout()
# plt.show()


#5)Answer specific questions (e.g.):
#a)Who is the top scorer in Science?
# top_science=df[df['science']==df['science'].max()]
# print("top scorer in science")
# print(top_science[['student_id', 'name', 'science']])
#b)What is the gender-based performance trend?
# gender_performance=df.groupby('gender')[['math','english','science','averagescore']].mean()
# print("\n Gender-based Performance Trend (Average Scores):")
# print(gender_performance)
#c)What subject has the highest average score?
# subject_means=df[['math','english','science']].mean()
# highest_avg_subject=subject_means.idxmax()
# print("\n Subject with the Highest Average Score:", highest_avg_subject)
# print(subject_means)
# 
# 
# Bonus (Optional):
#1)Use seaborn's heatmap() to visualize correlations
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = pd.read_csv("students_performance.csv")



# df.columns = [col.strip().capitalize() for col in df.columns]


# df['Totalscore'] = df['Math'] + df['English'] + df['Science']
# df['Averagescore'] = df['Totalscore'] / 3

# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
# plt.title("Correlation Heatmap")
# plt.show()

#2)Create a downloadable CSV of filtered results
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = pd.read_csv("students_performance.csv")


# df.columns = [col.strip().capitalize() for col in df.columns]

# df['Totalscore'] = df['Math'] + df['English'] + df['Science']
# df['Averagescore'] = df['Totalscore'] / 3


# top_students = df[df['Averagescore'] > 85]

# top_students.to_csv("top_students.csv", index=False)

# print("Filtered results saved to top_students.csv")


#3)Add markdown cells to explain your steps and findings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("students_performance.csv")
df.head()
df.columns = [col.strip().capitalize() for col in df.columns]
df.head()
df['Totalscore'] = df['Math'] + df['English'] + df['Science']
df['Averagescore'] = df['Totalscore'] / 3
df.head()
df['Totalscore'] = df['Math'] + df['English'] + df['Science']
df['Averagescore'] = df['Totalscore'] / 3
df.head()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
top_students = df[df['Averagescore'] > 85]
top_students
top_students.to_csv("top_students.csv", index=False)
print("Filtered results saved to top_students.csv")
