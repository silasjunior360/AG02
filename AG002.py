import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 1. Ler CSV 
df = pd.read_csv("palmerpenguins.csv", sep=';')

# 2. Padronizar nomes das colunas
df.columns = df.columns.str.strip().str.lower()

# 3. Remover espacos em strings
df['species'] = df['species'].str.strip()
df['island'] = df['island'].str.strip()
df['sex'] = df['sex'].str.strip()

# 4. Remover valores nulos
df = df.dropna()

# 5. Converter categoricos para numeros
df['island'] = df['island'].replace({
    'Biscoe': 0,
    'Dream': 1,
    'Torgersen': 2
})

df['sex'] = df['sex'].replace({
    'FEMALE': 0,
    'MALE': 1
})

df['species'] = df['species'].replace({
    'Adelie': 0,
    'Chinstrap': 1,
    'Gentoo': 2
})


df = df.astype({
    'island': 'int64',
    'sex': 'int64',
    'species': 'int64'
})

# 6. Reordenar colunas
df = df[
    [
        'island',
        'sex',
        'culmen_length_mm',
        'culmen_depth_mm',
        'flipper_length_mm',
        'body_mass_g',
        'species'
    ]
]

# 7. Separar dados
X = df.drop('species', axis=1)
y = df['species']

# Debug 
print("Valores unicos de species:", y.unique())
print("Tipos:\n", df.dtypes)

# 8. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

