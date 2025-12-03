# 🧩 Customer Segmentation using Machine Learning

Customer Segmentation is a machine learning project that groups customers based on their purchasing patterns, demographics, and behavioral data. By leveraging unsupervised learning, this project helps businesses understand customer clusters, personalize marketing strategies, and improve decision-making.

---

## 🚀 Features

- ✔ **Automated data preprocessing** (cleaning, scaling, feature engineering)  
- ✔ **Exploratory Data Analysis (EDA)** with interactive visualizations  
- ✔ **K-Means clustering model** for segmentation  
- ✔ **Elbow Method & Silhouette Score** for finding optimal clusters  
- ✔ **Streamlit dashboard** for real-time visualization  
- ✔ **Modular code structure** (`src/` for ML pipeline, `app/` for dashboard)  

---

## 📂 Project Structure

```

Customer-Segmentation/
│
├── app/
│   ├── dashboard.py          # Streamlit dashboard
│
├── src/
│   ├── preprocess.py         # Data loading, cleaning, feature engineering
│   ├── clustering.py         # ML model training & evaluation
│   ├── visualize.py          # Plots & charts
│
├── data/
│   └── customers.csv         # Dataset (if included)
│
├── README.md
└── requirements.txt

```

---

## 🛠️ Technologies Used

- **Python**
- **Pandas, NumPy**
- **Scikit-Learn**
- **Matplotlib, Seaborn, Plotly**
- **Streamlit**
- **Joblib** (model persistence)

---

## 📊 Methodology

1. **Data Preprocessing**  
   - Handling missing values  
   - Outlier removal  
   - Feature scaling using StandardScaler  
   - Creating engineered features (e.g., spending scores, income categories)

2. **Exploratory Data Analysis (EDA)**  
   - Correlation analysis  
   - Distribution plots  
   - Pair-plots  
   - Customer behaviour insights  

3. **Clustering Model**  
   - K-Means clustering  
   - Finding optimal `k` using Elbow Method  
   - Cluster visualization in 2D/3D  

4. **Dashboard Development**  
   - Segmentation results  
   - Real-time visualizations  
   - User-selectable parameters  

---

## 📈 Sample Visualizations

- Cluster heatmaps  
- Spending score vs income scatter plots  
- Cluster distribution charts  
- PCA-based cluster visualization  
---

## ▶️ How to Run

### **Clone the repository**
```

git clone [https://github.com/Pavithra329/Customer-Segmentation](https://github.com/Pavithra329/Customer-Segmentation)
cd Customer-Segmentation

```

### **Install dependencies**
```

pip install -r requirements.txt

```

### **Run the Streamlit app**
```

streamlit run app/dashboard.py

```
## 🙌 Author

**Pavithra S**  
---

