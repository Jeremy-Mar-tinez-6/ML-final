## My Project

I applied machine learning techniques to investigate their use in weather prediction models. Below is my report.

***

## Introduction 

Weather is something that scientists have been trying to predict for decades. The advent of computers has allowed meteorologists to increase the accuracy of their models providing people with more accurate information. However, these computers require significant time and energy to run the models. With the advent of artificial intelligence and its growing use in STEM, I thought it would be worthwhile to explore machine learning capabilities in weather prediction. 

Using the nikhil7280/weather-type-classification dataset from kaggle, we can explore the viability of using machine learning in weather prediction. I chose to test the machine learning's capabilities in a supervised setting by running a decision tree, decision tree regression, and a classifier to see what it could predict the weather type using 10 other variables from the dataset

## Data
Here is what the first few rows of the dataset of 13,200 looks like. Since I was trying to do weather prediction with this dataset, the target variable is the last column, "Weather Type" 
<img width="2168" height="494" alt="Screenshot 2025-12-01 145420" src="https://github.com/user-attachments/assets/1ad9a73b-9d12-4d18-95ec-028466643c02" />
*Figure 1: Dataset*


Since the columns were a mix of numerical and categorical data, I changed all the values to be numerical.
```python
# Create a label encoder
le = LabelEncoder()

# Apply it to the categorical columns
df['Cloud Cover'] = le.fit_transform(df['Cloud Cover'])
df['Season'] = le.fit_transform(df['Season'])
df['Location'] = le.fit_transform(df['Location'])

map_weather_types = {'Sunny':0, 'Cloudy':1, 'Rainy':2, 'Snowy':3}
mapper = lambda x: map_weather_types[x]
df['Weather Type'] = df['Weather Type'].map(mapper)
```
<img width="2159" height="423" alt="Screenshot 2025-12-01 145448" src="https://github.com/user-attachments/assets/086418fb-935c-48f1-b1da-2e1327a7c65d" />
*Figure 2: Dataset with numerical values*


After making the datset all numerical values, I split it and then scaled the data 
```python
#we can use the values attribute to extract numpy arrays from the DataFrame:
target = "Weather Type"
X_data=df.drop([target],axis=1).values
y_data=df[target].values
y_data = y_data.reshape(-1,1)

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size)
#CROSS VALIDATION:
#now, let's do a K-fold cross validation and repeat the training K times.
data_idx = np.arange(len(X_train)) # [0, 1, 2 ... len(X_data) - 1]

kf = KFold(n_splits=5)
k = 0 #keep track of the fold
best_score = np.inf #keep track of the best score

fold = 1
for idx_train, idx_val in kf.split(data_idx):
    print("fold: {}".format(fold))
    print(idx_train.shape, idx_val.shape)
    fold = fold + 1

#now, let's do a K-fold cross validation and repeat the training K times using a pipeline
data_idx = np.arange(len(X_train)) # [0, 1, 2 ... len(X_data) - 1]

kf = KFold(n_splits=5)
k = 0 #keep track of the fold
best_score = np.inf #keep track of the best score

for idx_train, idx_val in kf.split(data_idx):
    X_train_k = X_train[idx_train]
    y_train_k = y_train[idx_train]

    X_val = X_train[idx_val]
    y_val = y_train[idx_val]

    pipe = Pipeline([('scaler', StandardScaler()), ('lr', Ridge(alpha=1.0))])
    pipe.fit(X_train_k, y_train_k)
    y_pred_lr = pipe.predict(X_val)

    score = np.sqrt(np.mean((y_val-y_pred_lr)**2))
    print("fold ", k, ": RMSE for linear regression (with scaling):", score)
    k = k + 1

    if score < best_score:
        best_model = copy.deepcopy(pipe)
        best_score = score

#test the best model from cross validation:
y_pred_lr = best_model.predict(X_test)
score = np.sqrt(np.mean((y_test-y_pred_lr)**2))
```


## Modelling
I wanted to test different methods of machine learning predictions, so I chose to use a Decision Tree, Random Forest Regressor, and a Classifer Model. I chose to use these three models because I had 10 variables I was trying to use as input variables to get to my target variable, and these three models can handle multiple features well. 

Decision Tree
```python
# Loading the Decision Tree Classifier and Tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Instantiate the DecisionTreeClassifier model:
dt = DecisionTreeClassifier(random_state=42)

# Training it:
dt.fit(X_train, y_train)
```

Random Forest Regressor
```python
# Create model
regr_multifeat = RandomForestRegressor(max_depth=100, random_state=0, oob_score=True)

# Fit model
regr_multifeat.fit(X_train_k, y_train_k.ravel())

# Predict
y_pred = regr_multifeat.predict(X_test)

# Evaluate
print('R^2 score on testing data =', regr_multifeat.score(X_test, y_test))
```

Classifier Model
```python
# Features 
X_classifier = df.drop(columns=[target]).values

# Labels 
y_classifier = np.argmax(Y_classes, axis=1)

# Split train/test
X_classifier_train, X_classifier_test, y_classifier_train, y_classifier_test = train_test_split(
    X_classifier, y_classifier, test_size=0.2, random_state=42, stratify=y_classifier)

# Scale features for MLPClassifier
scaler = StandardScaler()
X_classifier_train_scale= scaler.fit_transform(X_classifier_train)
X_classifier_test_scale = scaler.transform(X_classifier_test)

# Train MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_classifier_train_scale, y_classifier_train)
```

## Results

Decision Tree Results:
Letting the decision tree model run without a max depth allowed the model to fit the dataset with an accuracy of 89%. 
Accuracy: 0.8928030303030303
<img width="1192" height="790" alt="image" src="https://github.com/user-attachments/assets/b0c19a8d-ff5e-4605-acb7-def91ac021c1" />
*Figure 3: This shows the overfitted decision tree classifier model results.*


Accuracy with Max Depth of 3: 0.859469696969697
<img width="1570" height="1175" alt="image" src="https://github.com/user-attachments/assets/a0fc4c78-a04f-4bb0-a767-9ff7b1e9f44c" />
*Figure 4: This shows the fitted decision tree classifier model with a max depth = 3*

Random Forest Regressor Results: 
With a R^2 score of 0.87, the model did pretty well. Even though there are red dots that vary from the fit line, the model does not show density of the dots well, which matters in a dataset of 13,200 values. The R^2 score being relatively high shows the model does have some outliers/errors, but is not guessing. 
<img width="680" height="1112" alt="Screenshot 2025-12-01 134814" src="https://github.com/user-attachments/assets/c4624761-17d3-447e-8803-c3e5f5ad9442" />

*Figure 5: Comparing dataset values vs model predicted values.*

I then wanted to see which of the features was influencing the model the most. Surprisingly the visibility distance was responsible for about 50% of the model's choice
<img width="1001" height="650" alt="image" src="https://github.com/user-attachments/assets/510dfe5d-18a8-47b6-998d-144455e091ed" />
*Figure 6: Bar chart showing feature importance*


Classifier Model Results:

<img width="680" height="580" alt="image" src="https://github.com/user-attachments/assets/4ad1cecd-e182-4f30-bd2a-0ff83fa52aa7" />

*Figure 7: Bar chart showing model probability of each Weather Type outcome*

Checking the accuracy of the model
Prediction accuracy: 90.41666666666667 %
<img width="686" height="547" alt="image" src="https://github.com/user-attachments/assets/330d50f5-dfcb-4e6c-9d8d-1f7d85696755" />

*Figure 8: Scatterplot showing prediction accuracy of the predicted Y values vs testing Y values*

Checking the ROC of the model 

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/763542b7-929f-4f40-9d16-a00af1edf9e5" />

*Figure 9: ROC scores of each of the weather types in the model*


## Discussion



## Conclusion

Here is a brief summary. From this work, the following conclusions can be made:
* first conclusion
* second conclusion

Here is how this work could be developed further in a future project.

## References
[1] DALL-E 3

[back](./)

