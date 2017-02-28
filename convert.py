import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


data = pd.read_csv('default.csv')
data.describe()
print '\nTotal DEFAULT percentage:\n', data['DEFAULT'].value_counts(normalize=True)

train_validation, test = train_test_split(data, test_size=0.5, random_state=42, stratify=data.DEFAULT)

train_validation.to_csv('train_validation.csv', index=False)

train, validation = train_test_split(train_validation, test_size=0.2, random_state=42, stratify=train_validation.DEFAULT)

# # verify
print len(set(train['ID']) | set(validation['ID']) | set(test['ID']))
print '\nTrain DEFAULT percentage:\n', train['DEFAULT'].value_counts(normalize=True)
print '\nValidation DEFAULT percentage:\n', validation['DEFAULT'].value_counts(normalize=True)
print '\nTest DEFAULT percentage:\n', test['DEFAULT'].value_counts(normalize=True)

train.to_csv('train.csv', index=False)
validation.to_csv('validation.csv', index=False)
test.to_csv('test.csv', index=False)
