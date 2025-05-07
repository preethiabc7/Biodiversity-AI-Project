# README: ECO-AI - National Park Biodiversity

## Project Overview
**ECO-AI: National Park Biodiversity** is an AI-driven project designed to monitor biodiversity in U.S. national parks, identify endangered species, and guide conservation efforts. The project aligns with **Sustainable Development Goal (SDG) 15: Life on Land** by promoting sustainable ecosystem management through data-driven insights.

### Team Members
- Preethi.R  
- Mullai.A  
- Harini.R  
- Abhishree.D  

### Team ID
S4F_CP_TEAM_27050  

---

## Aim
To develop an AI system that monitors park biodiversity, flags endangered species, and guides conservation efforts for maintaining ecological balance in U.S. national parks.

---

## Problem Statement
Biodiversity monitoring in national parks faces challenges due to:
- Manual tracking methods.
- Scattered and unstructured data.
- Delayed responses to ecological threats.  

This project proposes an AI-based solution to streamline biodiversity assessment and enhance conservation planning.

---

## Objectives
1. **Analyze Species Data**: Process and analyze species data, conservation status, and ecological roles.
2. **Identify At-Risk Populations**: Use clustering techniques to detect vulnerable species in specific park regions.
3. **Generate Actionable Insights**: Provide recommendations for conservation strategies to support sustainable ecosystem management.

---

## Technology Stack
- **Dataset**: Kaggle Park Biodiversity dataset (`species.csv` and `parks.csv`).
- **Tools/Libraries**:
  - Python (Pandas, NumPy, Matplotlib, Seaborn).
  - Scikit-learn (K-Means clustering, Decision Trees, Random Forest).
- **Key Techniques**:
  - Data cleaning and preprocessing.
  - K-Means clustering for species categorization.
  - Classification models (Decision Tree, Random Forest) for vulnerability prediction.
  - Performance evaluation using silhouette scores and accuracy metrics.

---

## Code Implementation
The project involves the following steps:
1. **Data Loading and Merging**:
   ```python
   import pandas as pd
   species_df = pd.read_csv('species.csv')
   parks_df = pd.read_csv('parks.csv')
   merged_df = pd.merge(species_df, parks_df, on='Park Name', how='inner')
   ```

2. **Data Preprocessing**:
   - Handling missing values.
   - Label encoding categorical features (`Category`, `Conservation Status`).
   - Feature scaling.

3. **Clustering with K-Means**:
   ```python
   from sklearn.cluster import KMeans
   kmeans = KMeans(n_clusters=3, random_state=42)
   merged_df['Cluster'] = kmeans.fit_predict(X_scaled)
   ```

4. **Classification Models**:
   - Decision Tree and Random Forest for predicting species vulnerability.
   - Evaluation of model accuracy.

5. **Visualization**:
   - 2D and 3D plots for clustering results.
   - Feature importance analysis.

---

## Output
- **Top Parks by Species Vulnerability**: Identified parks with the highest average species vulnerability.
- **Conservation Strategies**: Recommended actions for high-risk parks (e.g., increased monitoring and protection).

Example Output:
```
Top Parks by Average Species Vulnerability:
1. Park A
2. Park B
3. Park C

Suggested Protection Strategies:
- Park A: Increase monitoring and protection of high-risk species.
- Park B: Implement habitat restoration programs.
```

---

## Conclusion
The ECO-AI system successfully:
- Monitors biodiversity in national parks.
- Detects at-risk species using clustering and classification techniques.
- Provides actionable insights for conservation efforts.  

Future enhancements may include real-time monitoring capabilities and integration with IoT sensors for dynamic data collection.

---

## How to Run the Code
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the Jupyter notebook or Python script:
   ```bash
   python eco_ai_biodiversity.py
   ```

---
