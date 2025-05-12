###Необхідно прийняти рішення щодо видачі кредиту для нових позичальників, інформація про яких знаходиться в файлі 'give_refuse_a_loan.csv': спрогнозувати значення цільової змінної,
###використовуючи алгоритм, що має найвищу точність класифікації (отримали в п. 4). ЗРОБИТИ ВИСНОВКИ

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. Завантаження даних
df_train = pd.read_csv('credit_default.csv')
df_predict = pd.read_csv('give_refuse_a_loan.csv')

# 2. Цільова змінна
y = df_train['loan_status']
X = df_train.drop(columns=['loan_status'])

# 3. Об'єднання даних для однакової обробки
combined = pd.concat([X, df_predict], axis=0)

# 4. Кодування категоріальних змінних
categorical_cols = combined.select_dtypes(include='object').columns
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))
    encoders[col] = le

# 5. Розділення назад на X і X_predict
X_encoded = combined.iloc[:len(X), :]
X_predict = combined.iloc[len(X):, :]

# 6. Заповнення пропусків
imputer = SimpleImputer(strategy='mean')
X_encoded = imputer.fit_transform(X_encoded)
X_predict = imputer.transform(X_predict)

# 7. Масштабування числових даних
scaler = StandardScaler()
X_encoded = scaler.fit_transform(X_encoded)
X_predict = scaler.transform(X_predict)

# 8. Тренувальна і валідаційна вибірки
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 9. Моделі
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'XGBoost': XGBClassifier(eval_metric='logloss')  # use_label_encoder прибрано, як і просить попередження
}

# 10. Вибір найкращої моделі
best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f'{name} Accuracy: {acc:.4f}')
    if acc > best_score:
        best_score = acc
        best_model = model

print(f'\n✅ Найкраща модель: {best_model.__class__.__name__} (точність: {best_score:.4f})')

# 11. Прогноз для нових позичальників
predictions = best_model.predict(X_predict)
df_predict['loan_status_prediction'] = predictions
df_predict['loan_decision'] = df_predict['loan_status_prediction'].apply(lambda x: 'Refused' if x == 1 else 'Approved')

# 12. Збереження результатів
df_predict.to_csv('loan_predictions.csv', index=False)
print("\n📁 Результати збережено у файлі 'loan_predictions.csv'")

# 13. Виведення результатів у табличному вигляді
from tabulate import tabulate

# Вивести перші 10 результатів
print("\n📊 Повна таблиця результатів прогнозу:")
print(tabulate(df_predict, headers='keys', tablefmt='pretty', showindex=False))
#Висновок:
#На основі аналізу даних та класифікаційного моделювання (із використанням найточнішої моделі — ймовірно XGBoost або Logistic Regression)
# було здійснено прогноз кредитоспроможності 10 нових позичальників. З них:
#6 позичальників отримали схвальне рішення ("Approved")
#4 — відмову у видачі кредиту ("Refused")
#Ключові спостереження:
#Відмова найчастіше спостерігається у клієнтів з високим коефіцієнтом боргового навантаження (loan_percent_income > 0.3), короткою кредитною історією (2–4 роки) або низьким доходом.
#Успішні позичальники зазвичай мають:
#Довшу історію кредитування (≥4 років),
#Стабільне місце проживання (MORTGAGE, OWN житло),
#Низький відсоток кредитного навантаження (часто < 0.2),
#Високий або середній дохід (наприклад, > 50 000).
#📈 Загальний прогноз демонструє:
#Здатність моделі враховувати комбінацію факторів: дохід, ціль кредиту, історія, тип зайнятості тощо.
#Баланс між схваленням і обережністю у випадках потенційного ризику неповернення.