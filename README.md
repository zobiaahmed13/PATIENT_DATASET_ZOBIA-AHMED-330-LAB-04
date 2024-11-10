# PATIENT_DATASET_ZOBIA-AHMED-330-LAB-04
(*): DATASET:
Patient ID	Age	BMI	Blood Pressure	Glucose Level	Physical Activity Level	Family History	Diet Type	Category
101	45	29	130	85	3.5	Yes	Balanced	Healthy
102	50	32	140	105	2	Yes	High Sugar	Pre-Diabetic
103	60	28	145	160	1	Yes	High Sugar	Diabetic
104	35	22	125	90	5	No	Balanced	Healthy
105	55	30	135	120	2	Yes	Low Carb	Pre-Diabetic
106	42	27	128	88	4	No	Balanced	Healthy
107	38	33	142	130	1.5	Yes	High Sugar	Diabetic
108	47	29	135	110	2.5	No	Balanced	Pre-Diabetic
109	52	28	130	100	2.5	Yes	High Sugar	Pre-Diabetic
110	61	26	138	165	1	Yes	Low Carb	Diabetic
111	43	30	133	92	3	No	Balanced	Healthy
112	36	31	129	120	2.3	No	High Sugar	Pre-Diabetic
113	48	34	141	130	1.8	Yes	High Sugar	Diabetic
114	39	25	126	89	4	No	Balanced	Healthy
115	53	30	134	115	2	Yes	Low Carb	Pre-Diabetic
116	46	32	144	140	1.2	Yes	High Sugar	Diabetic
117	40	28	132	105	3.5	No	Balanced	Healthy
118	49	27	127	110	2	Yes	Low Carb	Pre-Diabetic
119	37	29	136	118	1.9	No	High Sugar	Pre-Diabetic
120	58	33	143	155	1.5	Yes	High Sugar	Diabetic
121	55	28	135	95	3	Yes	Balanced	Healthy
122	42	32	142	160	1.3	Yes	High Sugar	Diabetic
123	47	27	128	100	3.2	No	Low Carb	Healthy
124	51	29	139	135	2	Yes	High Sugar	Pre-Diabetic
125	45	26	131	120	3	No	Balanced	Healthy
126	50	31	138	125	1.7	Yes	High Sugar	Pre-Diabetic
127	34	24	127	90	5	No	Low Carb	Healthy
128	59	29	140	150	1.4	Yes	High Sugar	Diabetic
129	46	28	133	110	2.5	Yes	Low Carb	Pre-Diabetic
130	38	30	130	115	3	No	Balanced	Healthy
131	62	33	145	165	1	Yes	High Sugar	Diabetic
132	49	27	128	105	2.3	No	Balanced	Pre-Diabetic
133	41	29	132	95	4	No	Low Carb	Healthy
134	56	28	137	130	1.5	Yes	High Sugar	Diabetic
135	35	25	126	90	4.5	No	Balanced	Healthy
136	57	32	139	145	1.2	Yes	High Sugar	Pre-Diabetic
137	48	30	136	110	2.5	Yes	Low Carb	Pre-Diabetic
138	52	31	141	120	2	No	High Sugar	Diabetic
139	44	27	133	105	3.3	No	Balanced	Healthy
140	60	34	147	170	1	Yes	High Sugar	Diabetic
![image](https://github.com/user-attachments/assets/78ddf59f-2586-468e-bf1d-1f84fb4b405d)

(*): HOME TASK CODE:
#ZOBIA AHMED / 2022F-BSE-330 / LAB 04 / HOMETASK:
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv("patient_data_zobia.csv")

# Encoding categorical variables
df['Family History'] = df['Family History'].map({'No': 0, 'Yes': 1})
df['Diet Type'] = df['Diet Type'].map({'Balanced': 0, 'High Sugar': 1, 'Low Carb': 2})
df['Category'] = df['Category'].map({'Healthy': 0, 'Pre-Diabetic': 1, 'Diabetic': 2})

# Splitting the data into training and testing sets (first 30 for training, last 10 for testing)
train_data = df.iloc[:30]
test_data = df.iloc[30:]

# Separate features and target variable
X_train = train_data.drop(['Patient ID', 'Category'], axis=1)
y_train = train_data['Category']
X_test = test_data.drop(['Patient ID', 'Category'], axis=1)
y_test = test_data['Category']

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Add the predicted values to the test data
test_data_with_predictions = test_data.copy()
test_data_with_predictions['Predicted Category'] = y_pred

print("ZOBIA AHMED / 2022F-BSE-330 / LAB 04 / HOMETASK:\n")
# Display the full test data with actual and predicted values
print("(1): Test Data Of Last 10 Rows With Predictions:")
print(test_data_with_predictions)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n(2): Accuracy of the model:", accuracy)
print("(3): Confusion Matrix:\n", conf_matrix)

