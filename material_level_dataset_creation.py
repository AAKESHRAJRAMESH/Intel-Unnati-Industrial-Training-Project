# -----------------------------------------------------------------------Code to Create Dataset ------------------------------------------------------

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

num_samples = 1000

data = {
    'Age': np.round(np.random.normal(12, 3, num_samples)).astype(int),
    'IQ': np.random.normal(100, 15, num_samples),
    'Time per Day (hrs)': np.random.exponential(1.5, num_samples),
    'Assessment Score': np.random.randint(40, 100, num_samples),
    'Level of Student': random.choices(['Beginner', 'Intermediate', 'Advanced'], k=num_samples),
    'Level of Course': random.choices(['Beginner', 'Intermediate', 'Advanced'], k=num_samples),
    'Course Name': random.choices(['Math', 'English', 'Science', 'History'], k=num_samples),
    'Consistency': random.choices(['Regular', 'Irregular'], k=num_samples),
}

df = pd.DataFrame(data)

df['Age'] = df['Age'].clip(3, 18)

df['Consistency_Num'] = df['Consistency'].map({'Regular': 1, 'Irregular': 0})
df['Student_Level_Num'] = df['Level of Student'].map({'Beginner': 1, 'Intermediate': 2, 'Advanced': 3})
df['Course_Level_Num'] = df['Level of Course'].map({'Beginner': 1, 'Intermediate': 2, 'Advanced': 3})

def generate_present_material(student_level):
    if student_level == 1:
        return random.choices(['Beginner', 'Intermediate'], weights=[0.8, 0.2])[0]
    elif student_level == 2:
        return random.choices(['Beginner', 'Intermediate', 'Advanced'], weights=[0.2, 0.6, 0.2])[0]
    else:
        return random.choices(['Intermediate', 'Advanced'], weights=[0.2, 0.8])[0]

df['Present Material Level'] = df['Student_Level_Num'].apply(generate_present_material)

# Revised determine_material_level function
def determine_material_level(row):
    score = row['Assessment Score']
    student_level = row['Student_Level_Num']
    course_level = row['Course_Level_Num']
    iq = row['IQ']
    consistency = row['Consistency_Num']
    time = row['Time per Day (hrs)']
    course = row['Course Name']
    present_material = row['Present Material Level']
    relative_performance = row['Relative Performance']

    base_level = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}[present_material]

    # Stronger influence of Relative Performance
    adjustment = relative_performance / 10 + (iq - 100) / 20 + (consistency * 1) + (time - 1.5)

    if course == 'Math':
        adjustment += 1.5
    if course == 'History':
        adjustment -= 1.5

    if abs(adjustment) >= 1:
        base_level += int(round(adjustment))
    else:
        if adjustment > 0.3:
            base_level += 1
        elif adjustment < -0.3:
            base_level -= 1

    base_level = max(1, min(3, base_level))

    return {1: 'Beginner', 2: 'Intermediate', 3: 'Advanced'}[base_level]

# Feature Engineering: Relative Performance
df['Relative Performance'] = (df['Assessment Score'] - (df['Student_Level_Num'] + df['Course_Level_Num']) * 15)

df['Material Level'] = df.apply(determine_material_level, axis=1)

# Numerical representations for correlation matrix
df['Present_Material_Level_Num'] = df['Present Material Level'].map({'Beginner': 1, 'Intermediate': 2, 'Advanced': 3})
df['Material_Level_Num'] = df['Material Level'].map({'Beginner': 1, 'Intermediate': 2, 'Advanced': 3})

# Visualizations

# 1. Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=16, kde=True)
plt.title('Age Distribution')
plt.show()

# 2. Assessment Score Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Assessment Score'], bins=20, kde=True)
plt.title('Assessment Score Distribution')
plt.show()

# 3. Material Level Count
plt.figure(figsize=(8, 6))
sns.countplot(x='Material Level', data=df)
plt.title('Material Level Counts')
plt.show()

# 4. Present vs Predicted Material Level
plt.figure(figsize=(10, 8))
sns.countplot(x='Present Material Level', hue='Material Level', data=df)
plt.title('Present vs Predicted Material Level')
plt.show()

# 5. IQ vs Assessment Score
plt.figure(figsize=(8, 6))
sns.scatterplot(x='IQ', y='Assessment Score', data=df)
plt.title('IQ vs Assessment Score')
plt.show()

# 6. Time per Day (hrs) Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Time per Day (hrs)'], kde=True)
plt.title('Time per Day (hrs) Distribution')
plt.show()

# 7 Correlation Heatmap

correlation_matrix = df[['Age', 'IQ', 'Time per Day (hrs)', 'Assessment Score', 'Consistency_Num', 'Student_Level_Num', 'Course_Level_Num', 'Present_Material_Level_Num', 'Material_Level_Num', 'Relative Performance']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

df.to_csv('Material_Level.csv', index=False)
