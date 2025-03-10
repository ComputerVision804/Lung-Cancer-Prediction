import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Step 1: Load the CSV file into a Pandas DataFrame
df = pd.read_csv(r"C:\Users\Qc\Desktop\detection\cancer patient data sets.csv")


# Step 2: Drop rows with NaN values
df.dropna(inplace=True)

# Step 4: Convert 'Level' to numerical values (1 for Low, 2 for Medium, 3 for High)
level_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df['Level'] = df['Level'].map(level_mapping)

# Step 5: Explore the correlation between features
correlation_matrix = df[['Obesity', 'Snoring', 'Gender', 'Level', 'Smoking', 'Air Pollution', 'Dust Allergy', 'Alcohol use', 'Genetic Risk', 'Fatigue', 'OccuPational Hazards', 'Wheezing', 'chronic Lung Disease', 'Swallowing Difficulty', 'Dry Cough']].corr()
# Create a heatmap
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()
#check risk level
plt.figure(figsize=(10, 6))
explode = (0, 0, 0.15)
plt.pie(df['Level'].value_counts(), 
        labels=df['Level'].unique(), 
        explode=explode, autopct='%1.1f%%', 
        shadow=True, startangle=90, 
        colors=['green', 'yellow', 'red'])
plt.title('Level Distribution')
plt.show()
#Each individual column(independent variables) influenced in target column(dependent variables) 
# function for bar plotting
def occ_cht(col, df=df):
    return df.groupby(col)['Level'].value_counts(normalize=True).unstack().plot(kind='bar', title = (f'Lungs Disease Based on {col}'), figsize=(12,5))
occ_cht('Age')
occ_cht('Gender')
occ_cht('Air Pollution')
occ_cht('Alcohol use')
occ_cht('Dust Allergy')
occ_cht('Smoking')
occ_cht('Obesity')
occ_cht('Genetic Risk')
occ_cht('chronic Lung Disease')
occ_cht('OccuPational Hazards')
occ_cht('Fatigue')
occ_cht('Wheezing')
occ_cht('Swallowing Difficulty')
occ_cht('Dry Cough')
occ_cht('Snoring')
plt.show()
# Step 6: Save the cleaned DataFrame to a new CSV file  
df.to_csv(r"C:\Users\Qc\Desktop\detection\cleaned_cancer_patient_data.csv", index=False)
# Step 7: Print the first 5 rows of the cleaned DataFrame
print(df.head())
# Step 8: Print the shape of the cleaned DataFrame  
print(df.shape)
# Step 9: Print the data types of the cleaned DataFrame
print(df.dtypes)
# Step 10: Print the summary statistics of the cleaned DataFrame
print(df.describe())
# Step 11: Print the number of missing values in the cleaned DataFrame
print(df.isnull().sum())
# Step 12: Print the unique values in the 'Level' column
print(df['Level'].unique())
# Step 13: Print the value counts of the 'Level' column
print(df['Level'].value_counts())


sns.set_theme(style="white")

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(50, 130, n = 7, as_cmap=True)
#cmap = sns.diverging_palette(250, 20, n = 7, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, annot=True, fmt= ".2f", cmap=cmap, vmin=-3, vmax=3, center=0, square=True, linewidths=.7, cbar_kws={"shrink": .5})

