# IDFC Bank Credit Card Default Prediction

## Problem Statement
Develop a **"Behaviour Score"** predictive model to forecast the probability of default for existing credit card customers. The model uses historical data to predict the `bad_flag` (1 = default) and will aid in portfolio risk management.

---

## Datasets
- **Development Data**: 96,806 credit card records *with* target variable (`bad_flag`).
- **Validation Data**: 41,792 credit card records *without* target variable.

### Independent Variables
- **On Us Attributes**: Credit limit details.
- **Transaction Level Attributes**: Merchant-specific transaction behaviors.
- **Bureau Tradeline Attributes**: Product holdings & historical delinquencies.
- **Bureau Enquiry Attributes**: Recent credit inquiries.

---

## Data Preprocessing
### Missing Value Imputation (KNN)
Filled missing values using k-nearest neighbors averaging:
x_i(f) = (1/k) * Σ x_j(f) (j=1 to k)


### Class Imbalance Handling (SMOTE)
Generated synthetic minority class samples:
x_new = x₁ + λ(x₂ - x₁) (λ ∈ [0,1])


---

## Feature Engineering
### Bureau Attributes
- **Pattern**: `bureau_X + bureau_Y = bureau_Z`  
  *Example*: `bureau_74 + bureau_75 = bureau_76`
- **Interpretation**: Aggregates product holdings and delinquencies.

### Bureau Enquiry Attributes
- **Pattern**: `enquiry_A - enquiry_B = enquiry_C`  
  *Example*: `enquiry_1 - enquiry_2 = enquiry_3`
- **Interpretation**: Represents net approvals or risk metrics.

### Transaction Attributes
- **Pattern**: `Attribute_3 ≈ Attribute_1 × Attribute_2`  
  *Example*: Scaled transaction values over time.
- **Temporal Grouping**: 39 groups of 3 columns each, likely representing quarterly trends.

### PCA
Applied for dimensionality reduction on non-patterned features:
X' = X · V (V = eigenvectors)


---

## Modeling Approach
### Algorithms
1. **Decision Tree Classifier**  
   `Gini(t) = 1 − Σ p_i²`
2. **Random Forest Classifier**  
   Ensemble of decision trees.
3. **Neural Network**  
   - **Architecture**:
     - Input Layer: Features (size = 117)
     - Hidden Layers: 128 → 64 → 32 neurons (tanh activation, batch norm)
     - Output Layer: 1 neuron (sigmoid)

### Training Configuration
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam (`lr=0.001`)
- **Epochs**: 2500

---

## Results
### Performance Metrics
- **Accuracy**: 93%
- **Classification Report**:

- 
  
          precision  recall  f1-score  support
macro avg 0.93 0.93 0.93 28210
weighted avg 0.93 0.93 0.93 28210


## Conclusion
The neural network model achieved **93% accuracy** in predicting credit card defaults, enabling IDFC Bank to effectively manage portfolio risk through proactive customer behavior scoring.
