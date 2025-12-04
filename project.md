## My Project

I applied machine learning techniques to investigate their use in weather prediction models. Below is my report.

***

## Introduction 

The weather has been a topic that scientists have been trying to predict for decades. The advent of computers has enabled meteorologists to improve the accuracy of their models, providing people with more reliable information. However, these computers require significant time and energy to run the models. With the advent of artificial intelligence and its growing use in STEM, it would be worthwhile to explore machine learning capabilities in weather prediction. 

Using the dataset at https://www.kaggle.com/datasets/nikhil7280/weather-type-classification, I explored the viability of machine learning for weather prediction. I chose to test machine learning's capabilities in a supervised setting by running a Decision Tree, a Random Forest Regressor, and a Classifier Model to predict weather type using 10 other variables from the dataset.

## Data
Here is what the first few rows of the dataset of 13,200 looks like. Since I was trying to do weather prediction with this dataset, the target variable is the last column, "Weather Type." 
<img width="1168" height="494" alt="Screenshot 2025-12-01 145420" src="https://github.com/user-attachments/assets/1ad9a73b-9d12-4d18-95ec-028466643c02" />
*Figure 1: Dataset*


Since the columns contained both numerical and categorical data, I converted all values to numerical.
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
<img width="1159" height="423" alt="Screenshot 2025-12-01 145448" src="https://github.com/user-attachments/assets/086418fb-935c-48f1-b1da-2e1327a7c65d" />
*Figure 2: Dataset with numerical values*


After making the dataset all numerical values, I split it and then scaled the data 
```python
# Extract numpy arrays from the DataFrame:
target = "Weather Type"
X_data=df.drop([target],axis=1).values
y_data=df[target].values
y_data = y_data.reshape(-1,1)

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size)

#CROSS VALIDATION:
#K-fold cross-validation and repeat the training K times.
data_idx = np.arange(len(X_train)) # [0, 1, 2 ... len(X_data) - 1]

kf = KFold(n_splits=5)
k = 0 #keep track of the fold
best_score = np.inf #keep track of the best score

fold = 1
for idx_train, idx_val in kf.split(data_idx):
    print("fold: {}".format(fold))
    print(idx_train.shape, idx_val.shape)
    fold = fold + 1

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

# Test the best model from cross-validation:
y_pred_lr = best_model.predict(X_test)
score = np.sqrt(np.mean((y_test-y_pred_lr)**2))
```


## Modeling
I wanted to test different machine learning methods, so I chose a Decision Tree, a Random Forest Regressor, and a Classifier Model. I chose these three models because I had 10 input variables and wanted to use them to predict my target variable, and they can handle multiple features well. I also wanted to add in a regressor model, so it could tell me about how my model was working with the data, and not necessarily its accuracy, which is what I wanted to see from the Decision Tree and Classifier Models. 

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

### Decision Tree Results:
Running the decision tree model without a max depth allowed it to fit the dataset with an accuracy of 91%. 
Accuracy: 0.9140151515151516
<img width="1193" height="790" alt="image" src="https://github.com/user-attachments/assets/463990fe-d36b-45d3-9dea-ff3253c9830c" />

*Figure 3: This shows the overfitted decision tree classifier model results.*


Accuracy with Max Depth of 3: 0.8776515151515152
<img width="1570" height="1175" alt="image" src="https://github.com/user-attachments/assets/d6639807-8600-4baa-a7ea-f30f478f98c1" />

*Figure 4: This shows the fitted decision tree classifier model with a max depth = 3*

I then wanted to see which feature was influencing the model the most. Surprisingly, the visibility distance was responsible for about 50% of the model's choice
<img width="1001" height="650" alt="image" src="https://github.com/user-attachments/assets/510dfe5d-18a8-47b6-998d-144455e091ed" />
*Figure 5: Bar chart showing feature importance*

Checking the ROC of the model

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/854b0935-4741-4854-a0d3-28c7049541f6" />

*Figure 6: ROC scores of each of the weather types in the model*

### Random Forest Regressor Results: 
With an R^2 score of 0.889722788377116, the model did pretty well. Even though there are red dots that deviate from the fit line, the model does not capture the dots' density well, which matters in a dataset of 13,200 values. The R^2 score being relatively high shows the model does have some outliers/errors, but it is not guessing. 

<img width="536" height="525" alt="image" src="https://github.com/user-attachments/assets/bec91b26-12be-4122-9635-fe6ab483c71a" />

*Figure 7: Comparing dataset values vs model predicted values.*


### Classifier Model Results:

<img width="680" height="580" alt="image" src="https://github.com/user-attachments/assets/4ad1cecd-e182-4f30-bd2a-0ff83fa52aa7" />

*Figure 8: Bar chart showing model probability of each Weather Type outcome*

Checking the accuracy of the model
Prediction accuracy: 90.41666666666667 %
<img width="686" height="547" alt="image" src="https://github.com/user-attachments/assets/330d50f5-dfcb-4e6c-9d8d-1f7d85696755" />

*Figure 9: Scatterplot showing prediction accuracy of the predicted Y values vs testing Y values*

Checking the ROC of the model 

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/763542b7-929f-4f40-9d16-a00af1edf9e5" />

*Figure 10: ROC scores of each of the weather types in the model*


## Discussion
The Decision Tree (Figure 3) and Classifier Model (Figures 8 & 9) performed at around 90% accuracy, which is on the lower end of computer model accuracy for 5-day forecasts [1]. The ROC curves in both models (Figures 6 & 10) also indicate that the models perform very well. The Random Forest Regressor Model (Figure 7) captured about 89% of the target's variability. So all three models performed very well. However, the Decision Tree & Classifier Models are the more important metrics for my question about using machine learning to predict weather type, as they provide prediction accuracy numbers. 
I was very surprised to see that visibility was such an important feature, as shown in Figure 5. I thought temperature or atmospheric pressure would play a bigger role in the model sorting.
Overall, I was very impressed with the model's results. 

## Conclusion

Here is a summary. From this work, the following conclusions can be made:
* Machine learning models are a viable tool in being able to identify or predict weather types, operating at around a 90% accuracy, which is on the lower end of computer model accuracy for forecasts [1]
* The MLP Classifier worked well, having a near-perfect AUC score, and the Decision Tree had an AUC score of about 0.94, meaning they are well-performing models. 

Here is how this work could be developed further in a future project:
* Expanding or contracting the number of variables used. Making the model more complex or simpler could further affect its accuracy. More specific data, like wind direction, could decrease the accuracy of the model, but it would make it more realistic 
* If we could add more atmospheric dynamics variables to the model, that would make it more in line with what computer models run. [2]


## References
[1] https://www.nesdis.noaa.gov/about/k-12-education/weather-forecasting/how-reliable-are-weather-forecasts
[2] https://www.weather.gov/about/models 

[back](./)

