from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,train_test_split,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, GlobalAveragePooling1D, Flatten, Dropout, BatchNormalization, Add, Activation, Multiply, Reshape
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.models import save_model,load_model

app = Flask(__name__)

# Mock ACO Feature Selection (replace this with your actual ACO implementation)
def apply_aco(data):
    # Simulate ACO selecting top 5 features (replace with actual ACO logic)
    feature_indices = random.sample(range(data.shape[1]), 5)
    selected_features = data.columns[feature_indices]
    return selected_features

# Preprocessing function
def preprocess():
    # Load dataset
    df = pd.read_csv("IBM Telco Dataset_7043.csv")
    
    # Drop customerID as it's not relevant
    df.drop('customerID', axis='columns', inplace=True)

    # Convert TotalCharges to numeric, handling errors
    df.TotalCharges = pd.to_numeric(df.TotalCharges, errors='coerce')
    df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]
    df= df[df.TotalCharges!=' ']
    df.TotalCharges = pd.to_numeric(df.TotalCharges)
    df = df.dropna(subset=['TotalCharges'])
    df.isnull().sum()  # Check for missing values
    df = df.dropna()  # Drop rows with missing values
    df.replace('No internet service','No',inplace=True)
    df.replace('No phone service','No',inplace=True)    
    # Replace categorical values for yes/no
    yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']

    for col in yes_no_columns:
        df[col].replace({'Yes': 1, 'No': 0}, inplace=True)
    df1=df
    df1.head(3)
    # Encode gender
    df1['gender'].replace({'Female':1,'Male':0},inplace=True)
    df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
    columns_to_encode = [
        'InternetService_Fiber optic',
        'InternetService_No',
        'Contract_One year',
        'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed check',
        'InternetService_DSL',
        'PaymentMethod_Bank transfer (automatic)',
        'Contract_Month-to-month'
    ]
    label_encoder = LabelEncoder()

    # Iterate through each column and encode the boolean values
    for col in columns_to_encode:
        df2[col] = label_encoder.fit_transform(df2[col])
    cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']

    scaler = MinMaxScaler()
    df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

    # Save preprocessed data locally
    df2.to_csv("Preprocessed_IBM_Telco_Dataset_7043.csv", index=False)
    print("Preprocessing complete, saved to Preprocessed_IBM_Telco_Dataset_7043.csv")
    return df

def ACO_alg():
    df2 = pd.read_csv("Preprocessed_IBM_Telco_Dataset_7043.csv")
    X = df2.drop('Churn', axis=1)
    y = df2['Churn']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ACO parameters
    NUM_ANTS = 30
    NUM_ITER = 50
    EVAPORATION_RATE = 0.5
    ALPHA = 1.0
    BETA = 1.0

    # Initialize pheromone levels
    def initialize_pheromones(num_features):
        return np.ones(num_features)

    # Calculate fitness
    def calculate_fitness(selected_features):
        if len(selected_features) == 0:
            return 0

        X_train_selected = X_train.iloc[:, selected_features]
        X_test_selected = X_test.iloc[:, selected_features]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_selected, y_train)
        y_pred = clf.predict(X_test_selected)

        return accuracy_score(y_test, y_pred)

    # Generate a solution
    def generate_solution(pheromones, alpha, beta):
        num_features = len(pheromones)
        probabilities = np.zeros(num_features)

        for i in range(num_features):
            probabilities[i] = (pheromones[i] ** alpha) * ((1.0 / (i + 1)) ** beta)

        probabilities /= probabilities.sum()
        solution = np.random.choice([0, 1], size=num_features, p=[1 - probabilities.mean(), probabilities.mean()])
        selected_features = [index for index, bit in enumerate(solution) if bit == 1]

        return solution, selected_features

    # Update pheromones
    def update_pheromones(pheromones, solutions, fitness_values, evaporation_rate):
        pheromones *= (1 - evaporation_rate)
        for solution, fitness in zip(solutions, fitness_values):
            for i in range(len(solution)):
                if solution[i] == 1:
                    pheromones[i] += fitness

    # ACO main loop
    def ACO(num_ants, num_iter, num_features, evaporation_rate, alpha, beta):
        pheromones = initialize_pheromones(num_features)
        best_solution = None
        best_fitness = 0

        for t in range(num_iter):
            solutions = []
            fitness_values = []

            for ant in range(num_ants):
                solution, selected_features = generate_solution(pheromones, alpha, beta)
                fitness = calculate_fitness(selected_features)

                solutions.append(solution)
                fitness_values.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution

            update_pheromones(pheromones, solutions, fitness_values, evaporation_rate)
            # print(f"Iteration {t+1}/{num_iter}, Best Accuracy: {best_fitness}")
            print(f"Iteration {t+1}/{num_iter}, Best Accuracy: {best_fitness}, Selected Features: {selected_features}")

        return best_solution, best_fitness

    if __name__ == "__main__":
        num_features = X_train.shape[1]
        best_solution, best_fitness = ACO(NUM_ANTS, NUM_ITER, num_features, EVAPORATION_RATE, ALPHA, BETA)

        selected_features = [index for index, bit in enumerate(best_solution) if bit == 1]
        print("Best solution is: ", best_solution)
        print("Selected features are: ", X_train.columns[selected_features])
        print("Best fitness (accuracy): ", best_fitness)
    # df2=df2[['tenure', 'InternetService_Fiber optic','PaymentMethod_Credit card (automatic)','Churn']]
    columns_selected = X.columns[selected_features]
    sel_columns_dataset = df2[columns_selected]
    df2 = pd.concat([sel_columns_dataset, df2[['Churn']]], axis=1)
    df2.to_csv("ACO_selected_features_dataset1.csv", index=False)
    print("Done ACO feature selection")

def data_balancing(X, y):
    dataBalancing = "SMOTEEN"  # Change this to switch balancing methods
    if dataBalancing == "SMOTE-Tomek":
        smote_tomek = SMOTETomek()
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    elif dataBalancing == "SMOTEEN":
        smoteen = SMOTEENN()
        X_resampled, y_resampled = smoteen.fit_resample(X, y)
    else:  # Default to SMOTE
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)

    print("Resampled class distribution: \n", pd.Series(y_resampled).value_counts())
    return X_resampled, y_resampled

def build_churnnet(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(128, 5, padding='same')(inputs)
    x = residual_block(x, 128)
    x = squeeze_excite_block(x)
    x = channel_attention(x)
    x = spatial_attention_block(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

# Define the CNN building blocks (Attention and Residual blocks)
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True)

    avg_pool = GlobalAveragePooling1D()(input_feature)
    avg_pool = Reshape((1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling1D()(input_feature)
    max_pool = Reshape((1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return Multiply()([input_feature, cbam_feature])

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling1D()(input_tensor)
    se = Reshape((1, filters))(se)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    return Multiply()([input_tensor, se])

def spatial_attention_block(input_tensor):
    attention = Conv1D(1, kernel_size=7, padding='same', activation='sigmoid')(input_tensor)
    return Multiply()([input_tensor, attention])

def ACO_model():
    df2 = pd.read_csv("ACO_selected_features_dataset.csv")
    X = df2.drop('Churn', axis='columns').values
    y = df2['Churn'].values

    X, y = data_balancing(X, y)

    # Reshape for Conv1D
    X = X[..., np.newaxis]
    
    # Define the StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Store results for each fold
    fold_no = 1
    acc_per_fold = []
    loss_per_fold = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    mcc_scores = []
    auc_roc_scores = []

    for train_index, val_index in skf.split(X.reshape(X.shape[0], -1), y):
        print(f'Training fold {fold_no}...')

        # Split the data
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # Build and compile the model
        model = build_churnnet((X_train_fold.shape[1], 1))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train_fold, y_train_fold, epochs=30, batch_size=32, validation_data=(X_val_fold, y_val_fold), verbose=1)

        # Evaluate the model
        scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Make predictions
        y_pred = model.predict(X_val_fold)
        y_pred_binary = (y_pred > 0.5).astype(int)

        # Calculate additional metrics
        precision = precision_score(y_val_fold, y_pred_binary)
        recall = recall_score(y_val_fold, y_pred_binary)
        f1 = f1_score(y_val_fold, y_pred_binary)
        mcc = matthews_corrcoef(y_val_fold, y_pred_binary)
        auc_roc = roc_auc_score(y_val_fold, y_pred)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        mcc_scores.append(mcc)
        auc_roc_scores.append(auc_roc)

        fold_no += 1

    print(f'Average Accuracy: {np.mean(acc_per_fold)}')
    print(f'Average Precision: {np.mean(precision_scores)}')
    print(f'Average Recall: {np.mean(recall_scores)}')
    print(f'Average F1 Score: {np.mean(f1_scores)}')
    print(f'Average MCC: {np.mean(mcc_scores)}')
    print(f'Average AUC-ROC: {np.mean(auc_roc_scores)}')
    model.save('ACO_ChurnModel.h5')

# model = load_model('churn_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model('ACO_ChurnModel.h5')
    # Get the data from the request
    data = request.get_json()
    features = np.array([[data['tenure'], data['InternetService_Fiber optic'], data['PaymentMethod_Credit card (automatic)']]])
    # Make the prediction
    prediction = model.predict(features)
    # Convert the prediction to a Python float
    return jsonify({'prediction': int(prediction[0][0] > 0.5)})


@app.route('/ACO_FS', methods=['POST'])
def ACO_route():
    data = ACO_alg()
    return jsonify({"message": "Done feature selection"}), 200

@app.route('/ACO_model', methods=['POST'])
def ACOModel_route():
    data = ACO_model()
    return jsonify({"message": "model runned succesfully"}), 200

@app.route('/preprocess', methods=['POST'])
def preprocess_route():
    data = preprocess()
    return jsonify({"message": "Preprocessing complete and dataset saved!"}), 200

# @app.route('/', methods=['GET'])
# def model_save():
#     save_model()
#     return jsonify({"message": "model saved"}), 200

if __name__ == '__main__':
    app.run(debug=True)


