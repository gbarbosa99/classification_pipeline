import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.io import arff

data = arff.loadarff('bone-marrow.arff')
df = pd.DataFrame(data[0])
df.drop(columns=['Disease'], inplace=True)


#Convert all columns to numeric, coerce errors to null values
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    
#Make sure binary columns are encoded as 0 and 1
for c in df.columns[df.nunique()==2]:
    df[c] = (df[c]==1)*1.0

# 1. Calculate the number of unique values for each column
print('Count of unique values in each column:')
print((df.columns.nunique()))

# 2. Set target, survival_status,as y; features (dropping survival status and time) as X
y = df.survival_status
X = df.drop(columns = ['survival_status', 'survival_time'])

# 3. Define lists of numeric and categorical columns based on number of unique values
num_cols = X.columns[X.nunique()>7]
cat_cols = X.columns[X.nunique()<=7]

# 4. Print columns with missing values
print(X.columns[X.isnull().sum()>0])

# 5. Split data into train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 1)

# 6. Create categorical preprocessing pipeline
# Using mode to fill in missing values and OHE
cat_vals = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'))])

# 7. Create numerical preprocessing pipeline
# Using mean to fill in missing values and standard scaling of features
num_vals = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scale', StandardScaler())])

# 8. Create column transformer that will preprocess the numerical and categorical features separately
preprocess = ColumnTransformer(transformers=[('num_preprocess', num_vals, num_cols), ('cat_preproces', cat_vals, cat_cols)])

# 9. Create a pipeline with preprocess, PCA, and a logistic regresssion model
pipeline = Pipeline([('preprocess', preprocess), ('pca', PCA()), ('clf', LogisticRegression())])

# 10. Fit the pipeline on the training data
pipeline.fit(x_train, y_train)
#Predict the pipeline on the test data
pipeline.score(x_test, y_test)

# 11. Define search space of hyperparameters
search_space = [{'clf':[LogisticRegression()],
                    'clf__C': np.logspace(-4, 2, 10),
                'pca':[PCA()],
                    'pca__n_components':np.linspace(30,37,3).astype(int)}]

#12. Search over hyperparameters abolve to optimize pipeline and fit
gs = GridSearchCV(pipeline, search_space, cv = 5)
gs.fit(x_train, y_train)

# 13. Save the best estimator from the gridsearch and print attributes and final accuracy on test set
best_model = gs.best_estimator_

# 14. Print attributes of best_model
print(classifier_type = best_model.named_steps['clf'])
print(best_model.named_steps['clf'].get_params())
print(best_model.named_steps['pca'].n_components)

# 15. Print final accuracy score 
print(best_model.score(x_test, y_test))
