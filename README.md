IDFC Bank issues credit cards to eligible customers and deploys advanced machine learning models for decisions regarding eligibility, limits, and interest rates. These models
ensure that early risks are managed and profitability is optimized. IDFC Bank has now
decided to develop a robust risk management framework for its existing credit card customers.
To achieve this, the bank aims to create a ”Behaviour Score,” a predictive model to
forecast the likelihood of customers defaulting on their credit cards. The model is based
on customers whose credit cards are open and not overdue, and it predicts the probability
of defaults going forward. This model will help with various portfolio risk management
activities.
Problem Statement
Your objective is to develop a ”Behaviour Score” for IDFC Bank. The goal is to predict the probability of default for existing credit card customers using the development
dataset. The provided development data has a target variable called bad flag, where
bad flag = 1 denotes a default. The validation dataset contains similar features but lacks
the target variable, and the goal is to predict the default probability for this dataset.
Datasets
Development Data: Contains a random sample of 96,806 credit card details, including
features and the bad flag.
Validation Data: Contains 41,792 credit card details with the same features but
without the bad flag.
The independent variables include:
• On Us Attributes: Variables related to the customer’s credit limit.
• Transaction Level Attributes: Details regarding the customer’s transaction behaviors across various merchants.
• Bureau Tradeline Level Attributes: Information about product holdings and
historical delinquencies.
• Bureau Enquiry Level Attributes: Data about recent credit inquiries.
1
Data Preprocessing and Feature Engineering
Missing Value Imputation: KNN Imputation
The dataset contained missing values, which were addressed using KNN Imputation. This
technique fills missing values by finding the k-nearest neighbors of a given data point and
averaging their feature values.
KNN Imputation Equation:
xi(f) = 1
k
X
k
j=1
xj (f)
where xj (f) represents the f-th feature of the j-th nearest neighbor of xi
, and k is the
number of neighbors used for imputation.
Class Imbalance Handling: SMOTE
Since the dataset was imbalanced (with fewer defaults), we used SMOTE (Synthetic Minority Over-sampling Technique) for balancing the classes. SMOTE generates synthetic
samples by interpolating between existing minority class samples.
SMOTE Equation:
xnew = x1 + λ(x2 − x1)
where λ ∈ [0, 1] is a random value, and x1, x2 are the feature vectors of the original
minority class samples.
Feature Engineering
Bureau Attributes and Patterns
Bureau Attributes Representation: The bureau columns (e.g., bureau 74 to bureau 116 ) represent product holdings and possibly historical delinquencies. Summing
these columns provides insights into the total number of products or the combined metrics for a customer’s portfolio.
Observed Pattern:
• bureau 74 and bureau 75 are subsets or components of the total represented by
bureau 76.
• Similarly, bureau 84 + bureau 85 = bureau 86, and so on.
Possible Representation:
• bureau X (e.g., bureau 74 ) might represent current product holdings.
• bureau Y (e.g., bureau 75 ) might represent historical delinquencies or another product category.
• bureau Z (e.g., bureau 76 ) then becomes the aggregated total or a comprehensive
metric that sums the two.
2
Interpretation: This pattern aligns with the logic of combining metrics to create a
comprehensive feature. For example:
Current Holdings + Historical Delinquencies = Total Product Metric.
Alternatively, this could represent different product categories (e.g., secured + unsecured
products).
Conclusion: The observed pattern appears correct and reflects a well-structured
dataset. However, the exact correctness depends on understanding what each specific
column represents. If the domain logic supports these relationships, this aggregation is
consistent with best practices for creating meaningful features in financial datasets.
Bureau Enquiry Patterns
Bureau Enquiry Representation:
During the feature engineering phase, we identified a significant pattern in the bureau enquiry
columns. The values in some of these columns appear to be derived from operations between others. Specifically, we observed the following relationship:
• bureau enquiry 1 - bureau enquiry 2 = bureau enquiry 3
• bureau enquiry 11 - bureau enquiry 12 = bureau enquiry 13
• And so on...
This observation suggests several possible representations of the derived metric:
Net Attributes:
The difference between two columns could represent a net value or difference. For instance:
• bureau enquiry 1 may represent total inquiries or approvals.
• bureau enquiry 2 could represent denied or unapproved cases.
• bureau enquiry 3 would then be the net number of approved inquiries, or a similar
derived value.
Comparison or Change Analysis:
The difference could reflect a comparison between two time periods, products, or states.
For example:
• bureau enquiry 1 might reflect the initial state or total applications.
• bureau enquiry 2 could represent cancellations or rejections.
• bureau enquiry 3 would represent the effective or final outcome.
3
Risk or Performance Metrics:
The difference might represent a calculated performance metric or risk factor. For instance:
• bureau enquiry 1 could reflect expected defaults.
• bureau enquiry 2 could represent actual defaults.
• bureau enquiry 3 would then indicate the discrepancy or an over/underestimation
metric.
Segmentation or Filtering:
The difference could represent a filtered subset of data:
• bureau enquiry 1 could be a broad category (e.g., total inquiries).
• bureau enquiry 2 could represent a specific subset (e.g., non-eligible inquiries).
• bureau enquiry 3 would represent the eligible or remaining subset.
Conclusion:
Based on these observations, it is likely that this pattern represents a derived metric
or net result obtained by comparing or subtracting two related attributes. To further
confirm the exact meaning, it would be important to understand the specific role of each
column in the dataset or to review the domain logic associated with these columns (e.g.,
product inquiries, approvals, or risk assessment).
Transaction Attributes Pattern
Observations
Scaling Relationship:
Within the groups of attributes (e.g., Transaction attribute 1, Transaction attribute 2,
and Transaction attribute 3), a consistent scaling relationship is observed:
Transaction attribute 3 ≈ Transaction attribute 1 × Transaction attribute 2
This pattern suggests that Transaction attribute 2 acts as a scaling factor that transforms Transaction attribute 1 into Transaction attribute 3.
Grouping by Sets of 3 Columns:
The data is structured in groups of 3 consecutive columns:
• First block: Transaction attribute 1 to Transaction attribute 117 (39 groups
of 3 columns).
• Second block: Transaction attribute 234 to Transaction attribute 351 (another 39 groups of 3 columns).
These groups indicate systematic organization, potentially aligned with a recurring pattern.
4
Temporal Interpretation:
If the data is time-based, each group of 3 columns may represent data for 3 consecutive
months:
• Column 1: Represents the raw transaction values.
• Column 2: Acts as a scaling or normalization factor.
• Column 3: Represents scaled or normalized transaction values.
The two blocks may correspond to two separate time periods:
• Block 1: The first year or period (e.g., all months of the first year).
• Block 2: The second year or period, extending into the next year up to March.
Possible Use Case:
If the data indeed represents monthly transactions over multiple years, the structure
provides:
• Quarterly insights (3-month periods grouped together).
• The ability to compare trends across years or periods due to consistent scaling.
Conclusion
The scaling pattern and grouping into sets of 3 suggest that the data is organized for
temporal analysis, possibly representing quarterly summaries over multiple years. This
structure allows for systematic comparisons and insights into transaction behavior over
time.
Principal Component Analysis (PCA)
In cases where patterns were not easily identifiable, we applied PCA for dimensionality
reduction. PCA transforms the original features into a smaller number of uncorrelated
components while preserving the variance in the data.
PCA Equation:
X
′ = X · V
where V is the matrix of eigenvectors (principal components), and X′
is the transformed
feature set.
Modeling Approach
Algorithms Used
• Decision Tree Classifier:
Gini(t) = 1 −
X
C
i=1
p
2
i
5
• Random Forest Classifier:
yˆ =
1
N
X
N
i=1
Ti(x)
• Neural Network: Input Layer: Accepts the input features (size = number of
features in the dataset). Hidden Layer 1: 128 neurons, batch normalization, and
tanh activation.
• Hidden Layer 2: 64 neurons, batch normalization, and tanh activation.
• Hidden Layer 3: 32 neurons, batch normalization, and tanh activation.
• Output Layer: 1 neuron with sigmoid activation for binary classification.
Training Configuration
• Loss Function: Binary Cross-Entropy Loss
• Optimizer: Adam
• Learning Rate: 0.001
• Epochs: 2500
Model Performance
Accuracy: 93%
Classification Report:
accuracy 0.93 28210
macro avg 0.93 0.93 0.93 28210
weighted avg 0.93 0.93 0.93 28210
Conclusion
This project successfully developed a predictive model for estimating the risk of default
for credit card customers at IDFC Bank. The final neural network model demonstrated
strong performance with an accuracy of 93%, making it a reliable tool for portfolio risk
management.
