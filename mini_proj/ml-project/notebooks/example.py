import warnings

from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# 참고: https://ywkim92.github.io/machine_learning/feature_selection/#sequencial-feature-selection

data, target = load_diabetes(return_X_y=True, as_frame=True)
train_X, test_X, train_y, test_y = train_test_split(data, target, test_size=0.2)
print(train_X.shape)

model = LogisticRegression()
# forward  : [x1, x2, x3]
# backward : [x1, x2, x3]
sfs = SequentialFeatureSelector(estimator=model,
                                n_features_to_select=5, direction='backward')

result = sfs.fit(train_X, train_y)
selected_features = sfs.support_

print(data.columns[selected_features])