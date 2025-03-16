import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from tensorflow import keras
import pandas as pd

pd.set_option('display.max_columns', 50)

X_train = pd.read_csv('processed/X_train.csv')
X_val = pd.read_csv('processed/X_val.csv')
X_test = pd.read_csv('processed/X_test.csv')
y_train = pd.read_csv('processed/y_train.csv')
y_val = pd.read_csv('processed/y_val.csv')
test = pd.read_csv('inputs/test.csv')

model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
)

model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
)

y_test = model.predict(X_test)

submission = pd.DataFrame(columns=['PassengerId', 'Transported'])

submission['Transported'] = pd.DataFrame(y_test) >= 0.50
submission['PassengerId'] = test['PassengerId']

submission.to_csv('outputs/submission.csv', index=False)
