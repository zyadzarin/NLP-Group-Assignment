from preprocessing import preprocess_text
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

dataset_path = "Data/news_headlines_data.csv"
df = pd.read_csv(dataset_path)
df = df.iloc[:, [0, 2]]

df['Label'].value_counts().sort_index().plot(kind='bar', title="Count of Label based on NewsHeadlines").figsize=(10, 5)
#plt.show()

df['Headlines'] = preprocess_text(df['Headlines'])
print("After Preprocessing:")
print(df['Headlines'][3:7])

x = df['Headlines']
y = df['Label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 17, shuffle = True)

train_headlines = x_train
test_headlines = x_test

# implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(1,2))
traindataset=countvector.fit_transform(train_headlines)

# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=450,criterion='entropy')
randomclassifier.fit(traindataset,y_train)

## Predict for the Test Dataset
test_transform= []
test_transform = test_headlines
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)

# CONFUSION MATRIX
matrix=confusion_matrix(y_test,predictions)
print(matrix)
score=accuracy_score(y_test,predictions)
print(score)
report=classification_report(y_test,predictions)
print(report)