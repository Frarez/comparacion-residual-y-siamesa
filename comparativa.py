import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Add, Subtract
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

filepath = 'lechuzasdataset.csv'

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    data = data.dropna()

    X = data[['Radiacion', 'Temperatura', 'Temperatura panel']].values
    y = data['Potencia'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_residual_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    residual = Dense(1)(x) 
    outputs = Add()([residual, Dense(1)(inputs)]) 
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model


def build_siamese_model(input_dim):
    def create_base_network():
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        return model

    base_network = create_base_network()

    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    outputs = Subtract()([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=outputs)
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return model

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"{model_name} Performance:\nMSE: {mse}\nMAE: {mae}\nR^2: {r2}")

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return mse, mae, r2

def main():
    global filepath

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(filepath)

    residual_model = build_residual_model(X_train.shape[1])
    siamese_model = build_siamese_model(X_train.shape[1])

    print("Training Residual Model")
    residual_metrics = train_and_evaluate_model(residual_model, X_train, y_train, X_val, y_val, X_test, y_test, "Residual Model")

    print("Training Siamese Model")
    siamese_metrics = train_and_evaluate_model(siamese_model, [X_train, X_train], y_train, [X_val, X_val], y_val, [X_test, X_test], y_test, "Siamese Model")

    print("\nComparison of Models")
    print(f"Residual Model Metrics: {residual_metrics}")
    print(f"Siamese Model Metrics: {siamese_metrics}")

if __name__ == "__main__":
    main()
