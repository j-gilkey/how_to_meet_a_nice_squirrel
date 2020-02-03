from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import squirrel_data_access
import resampling


df = squirrel_data_access.create_df()
X_train, X_test,y_train,y_test = squirrel_data_access.get_train_test_split(df, 'does_not_flee')

gnb = GaussianNB()
gnb.fit(X_train, y_train)

training_preds = gnb.predict(X_train)
test_preds = gnb.predict(X_test)

cm = confusion_matrix(y_test,test_preds)

training_accuracy = accuracy_score(y_train, training_preds)
test_accuracy = accuracy_score(y_test, test_preds)

training_classified = classification_report(y_train, training_preds)
test_classified = classification_report(y_test, test_preds)


print(training_classified)
print(test_classified)
