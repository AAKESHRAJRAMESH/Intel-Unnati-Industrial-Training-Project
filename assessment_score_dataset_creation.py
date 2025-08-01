# ___________________________________________ Code to generate dataset ____________________________________________________________________

import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from faker import Faker
import numpy as np

fake = Faker()

# --- Configuration ---
genders = ['Male', 'Female']
earning_classes = ['Low', 'Middle', 'High']
parent_occupations = ['Engineer', 'Doctor', 'Teacher', 'Farmer', 'Business Owner', 'Government Employee', 'Artist', 'Unemployed', 'Other']
levels_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
levels_list = list(levels_map.keys())
courses = ['Math', 'Science', 'History', 'Computer Science', 'Physics', 'Chemistry', 'Biology', 'English', 'Art', 'Geography']
material_types = ['pdf', 'pptx', 'txt', 'docx', 'video', 'interactive_module']
consistencies_map = {'Regular': 1, 'Irregular': 0}
consistencies_list = list(consistencies_map.keys())
health_levels_map = {'Very Poor': 1, 'Poor': 2, 'Average': 3, 'Good': 4, 'Excellent': 5}
health_levels_list = list(health_levels_map.keys())

num_records = 1000
num_rare_cases = 20

# --- Helper Functions ---

def get_level_from_value(value, level_map):
    """Finds the level name corresponding to a numeric value."""
    for name, val in level_map.items():
        if val == value:
            return name
    return None

# --- Main Generation Function ---

def generate_student():
    age = random.randint(3, 18)
    gender = random.choice(genders)
    parent_occupation = random.choice(parent_occupations)
    earning_class = random.choice(earning_classes)
    country = fake.country()


    if age <= 7:
        student_level_val = 1
    elif age <= 13:
        student_level_val = random.choice([1, 2])
    else:
        student_level_val = random.choice([2, 3])
    student_level = get_level_from_value(student_level_val, levels_map)

    # --- Decouple Course and Material Level ---
    course_level_val = student_level_val + random.choice([-1, 0, 0, 0, 1])
    course_level_val = max(1, min(course_level_val, 3))
    course_level = get_level_from_value(course_level_val, levels_map)

    material_level_val = course_level_val + random.choice([-1, 0, 0, 1])
    material_level_val = max(1, min(material_level_val, 3))
    material_level = get_level_from_value(material_level_val, levels_map)

    course_name = random.choice(courses)
    material_type = random.choice(material_types)

    # --- Refine other features ---
    base_study_time = {1: (0.5, 2.5), 2: (1.0, 4.0), 3: (1.5, 6.0)}
    study_time = round(random.uniform(*base_study_time[course_level_val]) + random.gauss(0, 0.5), 1)
    study_time = max(0.1, study_time)

    # IQ with slightly wider range and less strict level dependence
    iq = random.randint(70, 135) + random.choice([-5, 0, 5])

    consistency = random.choice(consistencies_list)
    health_desc = random.choice(health_levels_list)
    health = health_levels_map[health_desc]
    # --- More Nuanced Assessment Score Calculation ---
    base_score = 50 + (iq - 100) * 0.3 + (course_level_val - 1.5) * 5

    # Multiplicative effect of consistency on study time effectiveness
    consistency_factor = 1.0 if consistency == 'Regular' else 0.6
    effective_study_time = study_time * consistency_factor
    # Apply diminishing returns to study time
    study_benefit = 15 * np.log1p(effective_study_time)

    # Health impact (more significant at extremes)
    health_impact = 0
    if health <= 2:
        health_impact = -10 * (3 - health)
    elif health >= 4:
        health_impact = 5 * (health - 3)

    # Level Mismatch Penalty (if student level is much lower than course level)
    level_mismatch_penalty = -10 * max(0, course_level_val - student_level_val - 1)

    # Combine factors and add noise
    calculated_score = base_score + study_benefit + health_impact + level_mismatch_penalty
    noise = random.gauss(0, 8)
    assessment_score = round(calculated_score + noise)

    # Clamp score to 0-100 range
    assessment_score = max(0, min(assessment_score, 100))

    return {
        'Age': age,
        'Gender': gender,
        'Parent Occupation': parent_occupation,
        'Earning Class': earning_class,
        'Level of Student': student_level,
        'Level of Course': course_level,
        'Course Name': course_name,
        'Time per Day (hrs)': study_time,
        'Material Level': material_level,
        'IQ': iq,
        'Consistency': consistency,
        'Health': health,
        'Assessment Score': assessment_score,
        'Health Description': health_desc
    }

# --- Generate Rare/Edge Cases ---
def generate_rare_case():
    """ Generates more diverse and potentially challenging edge cases """
    student = generate_student()

    # Apply a specific modification to make it a rare case
    case_type = random.randint(1, 7)

    if case_type == 1: # Very High IQ, Poor Health/Consistency
        student.update({'IQ': random.randint(135, 150), 'Consistency': 'Irregular', 'Health': random.choice([1, 2])})
    elif case_type == 2: # Lower IQ, Excellent Health/Consistency
        student.update({'IQ': random.randint(70, 85), 'Consistency': 'Regular', 'Health': random.choice([4, 5])})
    elif case_type == 3: # Significant Level Mismatch (Student < Course)
        student.update({'Level of Student': 'Beginner', 'Level of Course': 'Advanced'})
        student['Material Level'] = random.choice(['Advanced', 'Intermediate']) # Material likely matches course
    elif case_type == 4: # High Performing Beginner
        student.update({'Level of Student': 'Beginner', 'Level of Course': 'Beginner', 'Assessment Score': random.randint(85, 98)})
        student['Consistency'] = 'Regular'
        student['Health'] = random.choice([4, 5])
    elif case_type == 5: # Very High Study Time, Low Score
        student.update({'Time per Day (hrs)': round(random.uniform(6.0, 8.0), 1), 'Assessment Score': random.randint(30, 55)})
        student['Consistency'] = 'Irregular' # Possible reason for low score despite time
        student['IQ'] = random.randint(80, 100)
    elif case_type == 6: # Very Low Study Time, High Score
        student.update({'Time per Day (hrs)': round(random.uniform(0.1, 0.8), 1), 'Assessment Score': random.randint(80, 95)})
        student['IQ'] = random.randint(120, 140) # Possible reason: High IQ
        student['Consistency'] = 'Regular'
    elif case_type == 7: # Advanced Student taking Beginner Course
        student.update({'Level of Student': 'Advanced', 'Level of Course': 'Beginner'})
        student['Material Level'] = 'Beginner'



    return student


# --- Generate Dataset ---
students = [generate_student() for _ in range(num_records - num_rare_cases)]
students += [generate_rare_case() for _ in range(num_rare_cases)]

df = pd.DataFrame(students)

# --- Map Categorical to Numerical for Correlation ---
# Use the maps defined earlier
df['Consistency_Num'] = df['Consistency'].map(consistencies_map)
df['Material_Level_Num'] = df['Material Level'].map(levels_map)
df['Student_Level_Num'] = df['Level of Student'].map(levels_map)
df['Course_Level_Num'] = df['Level of Course'].map(levels_map)

# Select only relevant numeric columns for the heatmap
numeric_cols_for_corr = [
    'Age', 'Student_Level_Num', 'Course_Level_Num', 'Time per Day (hrs)',
    'Material_Level_Num', 'IQ', 'Consistency_Num', 'Health', 'Assessment Score'
]
numeric_df = df[numeric_cols_for_corr]

# Rename columns for better readability in the heatmap
numeric_df.columns = [
    'Age', 'Level of Student', 'Level of Course', 'Time per Day (hrs)',
    'Material Level', 'IQ', 'Consistency', 'Health', 'Assessment Score'
    ]

# Save the full dataset (including categorical)
df.to_csv('Assessment_Score.csv', index=False)
print("Assessment_Score.csv")

# --- Visualizations ---

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title("Feature Correlation Heatmap (Improved Dataset)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# IQ vs Assessment Score Scatter Plot (Hue by Consistency)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='IQ', y='Assessment Score', hue='Consistency', alpha=0.7)
plt.title("IQ vs Assessment Score (Colored by Consistency)")
plt.xlabel("IQ")
plt.ylabel("Assessment Score")
plt.show()

# Level of Student vs Assessment Score Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Level of Student', y='Assessment Score', order=levels_list) # Ensure correct order
plt.title("Student Level vs Assessment Score")
plt.xlabel("Level of Student")
plt.ylabel("Assessment Score")
plt.show()

# Consistency vs Assessment Score Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Consistency', y='Assessment Score', order=consistencies_list)
plt.title("Consistency vs Assessment Score")
plt.xlabel("Consistency")
plt.ylabel("Assessment Score")
plt.show()

# Study Time vs Assessment Score Scatter Plot (Hue by Course Level)
plt.figure(figsize=(8, 6))
# Map Course Level numerical back to string for hue legend clarity if needed
df['Course_Level_Str'] = df['Course_Level_Num'].map({v: k for k, v in levels_map.items()})
sns.scatterplot(data=df, x='Time per Day (hrs)', y='Assessment Score', hue='Course_Level_Str', alpha=0.7, hue_order=levels_list)
plt.title("Study Time vs Assessment Score (Colored by Course Level)")
plt.xlabel("Time per Day (hrs)")
plt.ylabel("Assessment Score")
plt.show()

# Health vs Assessment Score Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Health Description', y='Assessment Score', order=health_levels_list)
plt.title("Health vs Assessment Score")
plt.xlabel("Health Description")
plt.ylabel("Assessment Score")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()


print("Visualization complete!")
