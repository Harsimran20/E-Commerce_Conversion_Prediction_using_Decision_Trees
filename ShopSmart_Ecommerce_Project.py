#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import pandas for data manipulation and analysis using DataFrames and Series
import pandas as pd
# Import NumPy for numerical computing, arrays, and mathematical operations
import numpy as np
# Import Matplotlib's pyplot module for creating charts and visualizations
import matplotlib.pyplot as plt
# Import Seaborn for higher-level statistical data visualization and attractive plot styles
import seaborn as sns


# In[2]:


# Load the Shop Smart e-commerce dataset into a pandas DataFrame for analysis
df = pd.read_csv("shop_smart_ecommerce.csv")


# In[3]:


df.head()


# In[27]:


# Check for missing (NaN/null) values in each column
df.isnull().sum()


# In[29]:


# Calculate the median for all numeric columns and replace NaN values
# in those columns with their respective column medians
df.fillna(df.median(numeric_only=True))


# In[31]:


# Count the total number of duplicate rows in the DataFrame
df.duplicated().sum()


# In[35]:


# Remove duplicate rows from the DataFrame and update it in place
df.drop_duplicates(inplace=True)


# In[37]:


# Separate input features (independent variables)
X = df.drop(columns=["Revenue"])
# Separate target variable (dependent variable) to be predicted
y = df["Revenue"]


# In[39]:


X.head()


# In[41]:


y.head()


# In[43]:


df.describe()


# In[12]:


# Select only numerical columns
numeric_df = df.select_dtypes(include=['number'])

# Plot correlation heatmap
sns.heatmap(numeric_df.corr(), annot=True)

plt.show()


# In[13]:


# Analyze class distribution of the target variable 'Revenue'
# Revenue = True  -> Customer completed a purchase
# Revenue = False -> Customer did not complete a purchase
# Understanding class balance is important because an imbalanced dataset
# can affect model performance and influence the choice of evaluation metric (F1-score).
sns.countplot(x='Revenue', data=df)
# Render the visualization
plt.show()


# In[14]:


# Convert categorical variables into dummy/indicator variables
# This transforms each category into a separate binary (0/1) column.
df = pd.get_dummies(
    df,
    columns=[
        'Month',        # Month of the visit
        'VisitorType',  # New vs returning visitor
        'Weekend'       # Whether the visit occurred on a weekend
    ],
    drop_first=True    # Drop one category from each feature to avoid multicollinearity
)


# In[45]:


# Import preprocessing techniques required for feature transformation

from sklearn.preprocessing import (
    LabelEncoder,     # Encodes target labels or ordinal categorical variables

    OneHotEncoder,    # Converts nominal categorical features such as
                      # Month, VisitorType, and Weekend into machine-readable format

    StandardScaler    # Scales numerical features to a common range;
                      # generally more useful for distance-based models,
                      # though not mandatory for Decision Trees
)


# In[47]:


# Import utility for splitting data into training and test sets.
from sklearn.model_selection import (
    train_test_split
)


# In[49]:


# Separate features and target variable
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# Split data into 80% training and 20% testing while maintaining class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[51]:


# Import the Decision Tree classifier algorithm
from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree model instance
model = DecisionTreeClassifier(random_state=42)


# In[53]:


# Train the model on the training set to learn parameters
# that minimize prediction error on the target variable
model.fit(X_train, y_train)


# In[55]:


from sklearn.metrics import classification_report, f1_score
# Use the trained Decision Tree model to make predictions on the test dataset
y_pred = dt.predict(X_test)
# Generate a detailed classification report containing
# precision, recall, F1-score, and support for each class
print(classification_report(y_test, y_pred))
# Calculate and display the overall F1-score
# F1-score is the primary evaluation metric for this imbalanced classification problem
print("F1 Score:", f1_score(y_test, y_pred))


# In[61]:


# Initialize a pruned Decision Tree Classifier
# Pruning parameters are used to reduce overfitting and improve generalization

model = DecisionTreeClassifier(
    max_depth=5,          # Limit tree depth
    min_samples_split=20, # Minimum samples required to split a node
    min_samples_leaf=10,  # Minimum samples required in a leaf node
    random_state=42       # Reproducibility
)


# In[65]:


# Calculate cost-complexity pruning parameters
path = model.cost_complexity_pruning_path(X_train, y_train)
# Obtain candidate pruning alpha values
ccp_alphas = path.ccp_alphas


# In[67]:


# Store the F1-score obtained for each pruning level
scores = []

# Train and evaluate a Decision Tree for every ccp_alpha value
# to identify the optimal level of pruning
for alpha in ccp_alphas:

    # Create a Decision Tree Classifier with the current pruning parameter
    clf = DecisionTreeClassifier(
        random_state=42,  # Ensure reproducible results
        ccp_alpha=alpha   # Apply Cost Complexity Pruning
    )

    # Train the model on the training dataset
    clf.fit(X_train, y_train)

    # Make predictions on the test dataset
    pred = clf.predict(X_test)

    # Calculate and store the F1-score for the current pruned tree
    # F1-score is used because the dataset is imbalanced
    scores.append(
        f1_score(y_test, pred)
    )


# In[69]:


# Import GridSearchCV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid to search over
params = {
    'max_depth':[3,5,7,10],  # Maximum depth of the decision tree
    'min_samples_split':[2,5,10,20],  # Minimum samples required to split an internal node
    'min_samples_leaf':[1,5,10],  # Minimum samples required to be at a leaf node
    'ccp_alpha':[0.0,0.001,0.005,0.01]  # Complexity parameter for minimal cost-complexity pruning
}

# Create GridSearchCV object with decision tree classifier
grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),  # Base estimator with fixed random state for reproducibility
    param_grid=params,  # Parameter grid to search
    scoring='f1',  # Use F1-score as evaluation metric
    cv=5  # 5-fold cross-validation
)

# Fit the grid search to find optimal hyperparameters
grid.fit(X_train,y_train)


# In[71]:


# Create a DataFrame to store feature names and their corresponding importance scores
importance = pd.DataFrame({
    'Feature': X.columns,  # Feature names from the input dataset
    'Importance': grid.best_estimator_.feature_importances_  # Importance scores from the best model
})
# Sort features by importance in descending order (most important first)
importance.sort_values(
    by='Importance',
    ascending=False
)


# In[73]:


# Get the best model from the grid search results
best_model = grid.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Calculate and print the F1 score to evaluate model performance
print("F1 Score:",
      f1_score(y_test, y_pred))


# In[ ]:




