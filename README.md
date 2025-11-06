<!-- ---------------------------------------------------------------- -->
<!-- 🌟 PROJECT HEADER -->
<!-- ---------------------------------------------------------------- -->

<h1 align="center">
  <img src="https://raw.githubusercontent.com/github/explore/main/topics/python/python.png" width="60" height="60" alt="Python Logo">
  <img src="https://raw.githubusercontent.com/github/explore/main/topics/jupyter-notebook/jupyter-notebook.png" width="60" height="60" alt="Jupyter Logo">
  <br>
  ⚡ <span style="color:#2F80ED">Machine Learning-Based Electricity Market Forecasting</span> ⚡
</h1>

<h3 align="center">🔋 Forecasting Electricity Prices Using Support Vector Machines, Decision Trees and Random Forest</h3>

<p align="center">
  <em>An intelligent forecasting system leveraging Machine Learning to predict electricity market prices using energy and weather data.</em>
</p>


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-ML%20Models-brightgreen?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey?logo=windows&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-blueviolet">
</p>

---

## ⚡ Introduction

The **electricity market** plays a vital role in maintaining the stability, sustainability, and efficiency of modern power systems ⚙️.  
With the ever-increasing demand for electricity 🔌, the diversification of generation sources ⚡, and the rapid adoption of renewable energy 🌞🌬️, market prices have become highly dynamic and unpredictable.

Frequent fluctuations in electricity prices pose challenges to:
- 🏭 **Market participants**, who must plan trading and bidding strategies effectively  
- 🧮 **Policy makers**, who design long-term energy management systems  
- 💡 **Consumers**, who seek to minimize cost and optimize energy usage  

---

### 🤖 Why Machine Learning?

Traditional time-series models such as **ARIMA**, **VAR**, and regression-based approaches often fail to capture the **non-linear dependencies** among energy generation, consumption, and weather parameters.  
In contrast, **Machine Learning (ML)** models excel at learning from historical data and capturing complex relationships, enabling **highly accurate, data-driven predictions** 📈.

---

### 🧩 Project Overview

This project integrates two major datasets: an **Energy Dataset** and a **Weather Dataset**, to construct a **comprehensive forecasting framework**.  

#### 🔹 Process Flow:
1. **Data Preprocessing** – Cleaning, feature engineering, and normalization  
2. **Dataset Integration** – Combining energy and weather attributes into a single dataset  
3. **Model Training** – Building ML models including:
   - 🌳 **Decision Tree (DT)**
   - 🌲 **Random Forest (RF)**
   - 📈 **Support Vector Machine (SVM)**
4. **Evaluation** – Assessing accuracy using **MAE**, **MSE**, and **R² Score**  

These models are trained to **forecast electricity prices 24 hours in advance**, providing insights into feature importance and market behavior.

---

### 🌍 Impact & Significance

The outcomes of this project are designed to:
- ⚡ Support **efficient energy management**
- 💰 Enable **optimal pricing strategies**
- 🧠 Improve **market transparency and stakeholder decision-making**
- 🌱 Foster **renewable integration** in sustainable power systems  

Ultimately, this research demonstrates how **Machine Learning transforms energy forecasting**, bridging data-driven intelligence with practical energy economics.

---

<p align="center">
  <img src="img/1.gif" alt="Machine Learning-Based Electricity Market Forecasting Workflow" width="750">
</p>

---

## 📚 Literature Survey

Electricity price forecasting has long been a focus area in energy analytics due to its impact on **market optimization**, **demand-side management**, and **renewable integration**.  
Traditional approaches such as **ARIMA**, **VAR**, and **linear regression** have been widely utilized, but their inability to model **non-linear and multi-factor dependencies** has paved the way for **Machine Learning (ML)**-based techniques 🚀.

---

### 🧠 Evolution of Forecasting Approaches

Recent studies emphasize the superiority of ML over classical statistical models in capturing complex market behavior:

- **Roussis [3]** applied **Deep Neural Networks (DNNs)** on the Spanish Electricity Market dataset (Kaggle), demonstrating enhanced accuracy compared to traditional models.  
- **Oliver [4]** explored multiple ML methods for short-term electricity price prediction and found that **weather-related features** significantly influence pricing outcomes.  
- **Liu et al. [5]** leveraged ML-based forecasting within a **multi-agent control framework**, optimizing real-time energy allocation decisions.  
- **Ansari et al. [6]** benchmarked **foundation models** like Chronos for time-series forecasting, highlighting the dataset’s value as a standard benchmark for ML research.

---

### ⚙️ Comparative Model Insights

Machine Learning methods — including **Support Vector Regression (SVR)**, **Decision Trees (DT)**, **Random Forests (RF)**, and **XGBoost** — have consistently outperformed traditional models in forecasting volatility and uncertainty:

- **Bilal et al. [7]** compared **SVR**, **MLP**, and **ARIMA**, showing that SVR achieves **superior performance** in modeling complex, non-linear price fluctuations.  
- Ensemble-based approaches improved **robustness and generalization**, confirming ML’s role as a **state-of-the-art solution** for electricity market prediction.  

Together, these studies establish **ML forecasting models** as the foundation for **next-generation energy market analytics**.

---
