# Assignment 2 - Weather Wizard Team

## **Project Setup**

This guide provides steps to configure the project environment, process data, train the models (Gradient Boosting, Classification, and K-Means Clustering), and use these models for prediction.

---

## **1. Environment Setup**

### **Using Conda**

We use Conda for setting up the environment to ensure proper dependency management:

1. **Install Conda**:
   - Ensure Conda is installed. Download it from [Anaconda](https://www.anaconda.com/products/individual) or use Miniconda.

2. **Create a Conda environment**:
   - Create a new Conda environment with Python 3.8:
   ```bash
   conda create --name climate_prediction python=3.8
   ```

3. **Activate the environment**:
   - Activate the newly created environment:
   ```bash
   conda activate climate_prediction
   ```

4. **Install the required dependencies**:
   - Install the necessary libraries:
   ```bash
   conda install pandas numpy scikit-learn matplotlib seaborn
   conda install -c conda-forge imbalanced-learn
   ```

5. **Optional - Jupyter Notebook**:
   - If Jupyter is needed for data exploration:
   ```bash
   conda install jupyterlab
   ```

6. **Clone the project repository**:
   - If the project is on GitHub, clone it:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

---

## **2. Data Processing**

We preprocess the dataset to prepare it for model training.

### **Steps for Data Processing**:

1. **Load the Dataset**:
   - Load the dataset using Pandas:
   ```python
   import pandas as pd
   df = pd.read_csv('path/to/dataset.csv') #this is the raw file after we combine the source files together
   ```

2. **Handle Missing Values**:
   - Drop missing values:
   ```python
   df.dropna(inplace=True)
   ```

3. **Feature Selection & Scaling**:
   - For **Gradient Boosting**, we select features like `air_temperature`, `dew_point`, `wind_spd`, etc., and scale the data using `StandardScaler`.
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   df[['air_temperature', 'dew_point', 'wind_spd']] = scaler.fit_transform(df[['air_temperature', 'dew_point', 'wind_spd']])
   ```

4. **Splitting Data for Training and Testing**:
   - We split the data into training and testing sets:
   ```python
   from sklearn.model_selection import train_test_split
   X = df[['air_temperature', 'dew_point', 'wind_spd']]
   y = df['rel-humidity']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

---

## **3. Model Training**

We apply **Gradient Boosting** for regression tasks and **K-Means Clustering** for grouping similar data points.

### **Gradient Boosting**:

1. **Train the Model**:
   ```python
   from sklearn.ensemble import GradientBoostingRegressor
   gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
   gbr.fit(X_train, y_train)
   ```

2. **Evaluate the Model**:
   ```python
   from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
   y_pred = gbr.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   mae = mean_absolute_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
   print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
   ```

3. **Visualize Results**:
   ```python
   import matplotlib.pyplot as plt
   plt.scatter(y_test, y_pred)
   plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
   plt.xlabel('Actual Humidity')
   plt.ylabel('Predicted Humidity')
   plt.show()
   ```

---

### **K-Means Clustering**:

1. **Train the K-Means Model**:
   ```python
   from sklearn.cluster import KMeans
   kmeans = KMeans(n_clusters=3, random_state=42)
   kmeans.fit(X_train)
   labels = kmeans.labels_
   ```

2. **Evaluate Clustering Performance**:
   ```python
   from sklearn.metrics import silhouette_score
   silhouette_avg = silhouette_score(X_train, labels)
   print(f"Silhouette Score: {silhouette_avg:.4f}")
   ```

3. **Visualize Clusters**:
   ```python
   from sklearn.decomposition import PCA
   pca = PCA(n_components=2)
   X_pca = pca.fit_transform(X_train)
   plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
   plt.xlabel('Principal Component 1')
   plt.ylabel('Principal Component 2')
   plt.show()
   ```

---

## **4. Model Prediction**

### **Steps for Using the Model**:

1. **Load New Data**:
   ```python
   new_data = pd.read_csv('path/to/new_data.csv')
   ```

2. **Preprocess New Data**:
   - Make sure to preprocess the new data using the same scaling as before:
   ```python
   new_data[['air_temperature', 'dew_point', 'wind_spd']] = scaler.transform(new_data[['air_temperature', 'dew_point', 'wind_spd']])
   ```

3. **Make Predictions**:
   ```python
   new_predictions = gbr.predict(new_data)
   print(new_predictions)
   ```

---

## **Conclusion**

This guide covers setting up the project environment, processing data, training machine learning models using Gradient Boosting and K-Means Clustering, and making predictions.