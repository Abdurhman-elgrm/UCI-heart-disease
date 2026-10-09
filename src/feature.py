from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


numeric_features = ['age', 'trestbps', 'chol', 'thalch','ca', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope',  'thal']


numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

def build_processor():
    return ColumnTransformer([
        ('num' , numeric_transformer , numeric_features),
        ('cat' , categorical_transformer ,categorical_features)

    ])