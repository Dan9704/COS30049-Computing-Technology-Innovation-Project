You are right! I missed implementing the **Climate Change on Crop Model** section in the README. Below is the updated README file with the **Climate Change on Crop** model included:

---

# **Assignment 2 - Weather Wizard Team**

## **Project Setup**

This guide provides detailed steps to configure the project environment, process data, train the models (Gradient Boosting, Linear Regression, K-Means Clustering, and the Climate Change on Crop Classification), and use these models for prediction.

---

## **1. Environment Setup**

### **Using Conda**

We use Conda for setting up the environment to ensure proper dependency management:

1. **Install Conda**:
   - Ensure Conda is installed. You can download it from [Anaconda](https://www.anaconda.com/products/individual) or use Miniconda.

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
   - If the project is hosted on GitHub, clone it:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

---

## **2. Data Processing**

We preprocess the dataset to prepare it for model training and evaluation.

### **Steps for Data Processing**:

1. **Load the Dataset**:
   - Load the dataset using Pandas:
   ```python
   import pandas as pd
   df = pd.read_csv('dataset.csv')  # This is the raw file after combining the source files.
   ```

2. **Handle Missing Values**:
   - Remove rows with missing data:
   ```python
   df.dropna(inplace=True)
   ```

3. **Feature Selection & Scaling**:
   - Select important features and scale them using `StandardScaler`:
   ```python
   from sklearn.preprocessing import StandardScaler
   features = ['air_temperature', 'dew_point', 'wind_spd']
   scaler = StandardScaler()
   df[features] = scaler.fit_transform(df[features])
   ```

4. **Splitting Data for Training and Testing**:
   - We split the data into training and testing sets:
   ```python
   from sklearn.model_selection import train_test_split
   X = df[features]
   y = df['rel-humidity']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

---

## **3. Model Training**

We implement multiple models including **Gradient Boosting**, **Linear Regression**, **K-Means Clustering**, and **Climate Change on Crop Classification**.

### **Gradient Boosting (Melbourne Olympic Park & Cerberus Datasets)**

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
   print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
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

### **Linear Regression (Melbourne Olympic Park & Cerberus Datasets)**

1. **Train the Linear Regression Model**:
   ```python
   from sklearn.linear_model import LinearRegression
   model = LinearRegression()
   model.fit(X_train, y_train)
   ```

2. **Make Predictions and Evaluate**:
   ```python
   y_pred = model.predict(X_test)
   from sklearn.metrics import mean_squared_error, r2_score
   rmse = np.sqrt(mean_squared_error(y_test, y_pred))
   r2 = r2_score(y_test, y_pred)
   print(f"RMSE: {rmse}, R²: {r2}")
   ```

3. **Visualize Results**:
   ```python
   plt.figure(figsize=(14, 7))
   plt.plot(y_test.values, label='Actual Temperature', color='blue')
   plt.plot(y_pred, label='Predicted Temperature', color='red', linestyle='--')
   plt.xlabel('Time')
   plt.ylabel('Temperature')
   plt.legend()
   plt.show()
   ```

---

### **K-Means Clustering (Climate Change Dataset)**

1. **Train the K-Means Model**:
   ```python
   from sklearn.cluster import KMeans
   kmeans = KMeans(n_clusters=3, random_state=42)
   kmeans.fit(X_train)
   labels = kmeans.labels_
   ```

2. **Evaluate Clustering**:
   ```python
   from sklearn.metrics import silhouette_score
   silhouette_avg = silhouette_score(X_train, labels)
   print(f"Silhouette Score: {silhouette_avg}")
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

### **Climate Change on Crop Model**
#### Additional Dataset

1. **Feature Selection and Data Preparation**:
   ```python
   import pandas as pd
   from sklearn.preprocessing import StandardScaler
   from sklearn.model_selection import train_test_split

   # Load the dataset
   df = pd.read_csv('processed_climate_change_data.csv')

   # Select relevant features for classification or clustering
   features = ['Temperature', 'CO2 Levels', 'Precipitation']
   X = df[features]
   y = df['Extreme Weather Events']  # Target classification column

   # Normalize the data
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

2. **Clustering using K-Means**:
   ```python
   from sklearn.cluster import KMeans
   kmeans = KMeans(n_clusters=3, random_state=42)
   kmeans.fit(X_scaled)
   df['Cluster'] = kmeans.labels_

   from sklearn.metrics import silhouette_score
   silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
   print(f'Silhouette Score: {silhouette_avg}')
   ```

3. **Classification**:
   - We use the same scaled features for classification models like Random Forest or Logistic Regression to predict extreme weather events.
   ```python
   from sklearn.ensemble import RandomForestClassifier
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

   rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
   rf_model.fit(X_train, y_train)

   y_pred = rf_model.predict(X_test)

   from sklearn.metrics import classification_report
   print(classification_report(y_test, y_pred))
   ```

4. **Evaluate Clustering Scores**:
   ```python
   from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score
   homogeneity = homogeneity_score(df['Extreme Weather Events'], df['Cluster'])
   completeness = completeness_score(df['Extreme Weather Events'], df['Cluster'])

   print(f"Homogeneity Score: {homogeneity}")
   print(f"Completeness Score: {completeness}")
   ```

5. **Visualize the Clusters**:
   ```python
   from sklearn.decomposition import PCA
   pca = PCA(n_components=2)
   X_pca = pca.fit_transform(X_scaled)

   plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis')
   plt.xlabel('Principal Component 1')
   plt.ylabel('Principal Component 2')
   plt.title('PCA of Climate Change Clusters')
   plt.show()
   ```

---

## **Conclusion**

This README outlines the steps for configuring the project environment, processing data, training models using Gradient Boosting, Linear Regression, and K-Means Clustering, and making predictions on new datasets. Further improvements can be made by hyperparameter tuning and model evaluation on different datasets.

