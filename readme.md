# MESSY CRIME INCIDENT DATASET

## Data Cleaning, Exploratory Data Analysis & Predictive Modeling

**Project Type:** Data Science / Machine Learning
**Dataset:** Synthetic Crime Incident Dataset
**Prediction Target:** Crime Severity
**Programming Language:** Python
**Primary Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

## 1. Project Overview

This project focuses on the complete data science workflow for a deliberately messy **synthetic crime incident dataset**.

The dataset was designed to contain common data-quality problems that can occur in real-world datasets, including:

* Missing values
* Duplicate records
* Duplicate incident IDs
* Inconsistent categorical values
* Spelling mistakes
* Abbreviations
* Inconsistent capitalization
* Extra spaces
* Incorrect numerical values
* Numerical values stored as text
* Invalid age values
* Negative values
* Missing dates
* Different representations of the same category

The project therefore goes beyond simply training a machine-learning model. It demonstrates the process of taking an imperfect dataset, understanding its problems, cleaning and validating it, exploring patterns, engineering additional features, and finally developing classification models.

The main machine-learning objective is to predict **crime severity** using information associated with a crime incident.

The four severity classes used in the notebook are:

* **Low**
* **Medium**
* **High**
* **Critical**

The dataset is synthetic and intended for educational purposes. It does **not** represent real crime records.

---

# 2. Problem Statement

Real-world datasets rarely arrive in a perfectly clean format.

Crime-related datasets can contain information collected from multiple sources, systems, officers, forms, or reporting mechanisms. As a result, the same category may be represented in several different ways.

For example, the notebook identifies variations such as:

* `asslt` and `assault`
* `burgiry` and `burglary`
* `roberry`, `robrry`, and `robbery`
* `kidnaping` and `kidnapping`
* `cyber crime` and `cybercrime`
* `drug offence` and `drug offense`
* `m`, `f`, `male`, and `female`
* `med` and `medium`
* `crit` and `critical`

If these inconsistencies are not addressed, they can cause the same real-world category to be treated as several different categories.

Missing values and invalid numerical observations can also affect analysis and machine-learning performance.

Therefore, this project aims to:

> Systematically inspect and clean a messy synthetic crime incident dataset, validate the resulting data, identify meaningful patterns through exploratory data analysis, engineer useful features, and build classification models capable of predicting crime severity.

---

# 3. Project Objectives

The notebook establishes the following objectives:

1. Inspect the raw crime incident dataset and understand its structure.
2. Identify major data-quality problems.
3. Detect missing values and duplicate records.
4. Identify inconsistent categorical values and spelling errors.
5. Correct invalid and inconsistent numerical values.
6. Handle missing numerical and categorical data.
7. Convert dates into useful datetime features.
8. Validate the cleaned dataset.
9. Perform exploratory data analysis.
10. Identify relationships and patterns within the crime data.
11. Engineer additional features.
12. Prepare numerical and categorical features for machine learning.
13. Build classification models for predicting crime severity.
14. Evaluate the models using accuracy, precision, recall, and F1-score.
15. Tune selected models using cross-validation and hyperparameter search.
16. Compare model performance.
17. Discuss the strengths and limitations of the resulting models.

---

# 4. Dataset Description

The dataset contains **5,250 rows and 37 columns** in the notebook's initial analysis.

The variables describe different aspects of crime incidents, including:

### Incident information

* `incident_id`
* `crime_type`
* `incident_datetime`
* `district`
* `city`
* `state`
* `address`
* `latitude`
* `longitude`

### Officer information

* `officer_id`
* `officer_first_name`
* `officer_last_name`
* `badge_number`

### Suspect information

* `suspect_id`
* `suspect_first_name`
* `suspect_last_name`
* `suspect_age`
* `suspect_gender`
* `suspect_race`

### Victim information

* `victim_id`
* `victim_first_name`
* `victim_last_name`
* `victim_age`
* `victim_gender`
* `victim_phone`

### Crime characteristics

* `weapon_used`
* `property_loss_usd`
* `reported_online`
* `severity`

### Investigation information

* `case_status`
* `resolution`
* `num_arrests`
* `notes`

### Engineered temporal variables

* `incident_year`
* `incident_month`
* `incident_day`
* `incident_hour`

---

# 5. Technologies and Libraries

The notebook uses Python and several commonly used data science and machine-learning libraries.

```text
Python
│
├── Pandas
├── NumPy
├── Matplotlib
├── Seaborn
└── Scikit-learn
    ├── train_test_split
    ├── StandardScaler
    ├── OneHotEncoder
    ├── ColumnTransformer
    ├── Pipeline
    ├── LogisticRegression
    ├── DecisionTreeClassifier
    ├── RandomForestClassifier
    ├── GridSearchCV
    └── RandomizedSearchCV
```

---

# 6. Project Workflow

The notebook follows a complete data science pipeline:

```text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Initial Data Exploration
     │
     ▼
Data Quality Assessment
     │
     ▼
Categorical Data Cleaning
     │
     ▼
Numerical Data Cleaning
     │
     ▼
Missing Value Treatment
     │
     ▼
Outlier Investigation
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Feature Engineering
     │
     ▼
Feature Selection / Leakage Removal
     │
     ▼
Train-Test Split
     │
     ▼
Numerical + Categorical Preprocessing
     │
     ▼
Machine Learning Pipelines
     │
     ├── Logistic Regression
     ├── Decision Tree
     └── Random Forest
     │
     ▼
Model Evaluation
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Model Comparison
```

---

# 7. Data Loading

The notebook loads the dataset from a ZIP archive using Python's `zipfile` module and Pandas.

The first file contained in the archive is read into a Pandas DataFrame.

Conceptually:

```python
with zipfile.ZipFile("archive (2).zip", "r") as zip_ref:
    file = zip_ref.namelist()[0]
    df = pd.read_csv(zip_ref.open(file))
```

Pandas is then used throughout the project for data inspection, transformation, aggregation, and analysis.

---

# 8. Initial Data Exploration

Before cleaning the dataset, the notebook performs an initial inspection.

The following checks are performed:

### First records

```python
df.head()
```

This provides an initial view of the dataset.

### Dataset dimensions

```python
df.shape
```

The dataset contains:

* **5,250 rows**
* **37 columns**

### Column names

The notebook examines the names of all variables to understand the structure of the dataset.

### Data types

```python
df.dtypes
df.info()
```

These checks help identify whether columns are stored as:

* integers
* floating-point numbers
* strings
* objects
* datetime values

### Missing values

```python
df.isnull().sum()
```

This identifies columns containing missing observations.

### Duplicate records

The notebook checks:

```python
df.duplicated().sum()
```

The result shows:

**200 duplicate rows**

It also checks duplicate incident IDs:

```python
df["incident_id"].duplicated().sum()
```

The result is:

**200 duplicate incident IDs**

These checks establish that duplication is one of the data-quality issues present in the dataset.

---

# 9. Missing Data Analysis

The notebook calculates both the number and percentage of missing values.

The resulting analysis identifies substantial missingness in several variables.

Examples include:

| Variable          | Missing Values | Missing % |
| ----------------- | -------------: | --------: |
| suspect_race      |          1,601 |    30.50% |
| suspect_age       |          1,430 |    27.24% |
| suspect_gender    |          1,413 |    26.91% |
| notes             |          1,069 |    20.36% |
| victim_phone      |          1,056 |    20.11% |
| victim_gender     |          1,000 |    19.05% |
| victim_age        |            970 |    18.48% |
| weapon_used       |            970 |    18.48% |
| resolution        |            951 |    18.11% |
| property_loss_usd |            781 |    14.88% |
| case_status       |            731 |    13.92% |
| incident_datetime |            720 |    13.71% |
| num_arrests       |            517 |     9.85% |
| reported_online   |            504 |     9.60% |
| severity          |            355 |     6.76% |

This analysis is important because missing data needs to be handled before machine-learning models can be trained reliably.

---

# 10. Categorical Data Cleaning

One of the major stages of the project is cleaning categorical variables.

The notebook standardizes categorical text by:

1. Converting values to strings.
2. Removing leading and trailing spaces.
3. Converting values to lowercase.
4. Correcting known spelling mistakes.
5. Standardizing abbreviations.
6. Combining variations that represent the same category.

---

## 10.1 Crime Type Cleaning

The `crime_type` column contains many inconsistent representations.

Examples found in the raw data include:

```text
asslt
burgiry
roberry
robrry
vandlism
kidnaping
cyber crime
drug offence
drug  offence
domestc violence
sexual assualt
burglry
homocide
d.u.i.
```

These are standardized using a correction dictionary.

Examples:

```text
asslt → assault
burgiry → burglary
roberry → robbery
robrry → robbery
vandlism → vandalism
kidnaping → kidnapping
cyber crime → cybercrime
domestc violence → domestic violence
drug offence → drug offense
```

The purpose is to prevent one crime type from being incorrectly represented as several separate categories.

---

# 11. District Cleaning

District values are standardized to lowercase and whitespace is removed.

Abbreviations are corrected, including:

```text
nor → north
sou → south
cen → central
mid → midtown
eas → east
wes → west
```

After cleaning, the district categories include:

* North
* South
* West
* Central
* Midtown
* East
* Southwest
* Northeast
* Northwest
* Southeast

---

# 12. Gender Cleaning

The suspect and victim gender fields are standardized.

The notebook converts:

```text
m → male
f → female
```

It also removes inconsistent capitalization and whitespace.

This prevents equivalent categories from being treated as different values.

---

# 13. Weapon Data Cleaning

The `weapon_used` variable is standardized by converting values to lowercase and removing unnecessary whitespace.

The notebook also standardizes the representation of:

```text
hands → hands/feet
```

The most frequent recorded weapon categories after cleaning include:

| Weapon       | Incidents |
| ------------ | --------: |
| unknown      |       970 |
| hands/feet   |       685 |
| knife        |       674 |
| blunt object |       661 |
| firearm      |       622 |
| unarmed      |       361 |
| rifle        |       346 |
| gun          |       318 |
| pistol       |       310 |
| bat          |       303 |

---

# 14. Severity Cleaning

Severity is the prediction target.

The notebook standardizes the target values by converting them to lowercase and mapping inconsistent representations.

Examples:

```text
1 → low
2 → medium
3 → high
4 → critical
med → medium
crit → critical
```

The final known severity categories are:

* low
* medium
* high
* critical

Before removing missing target values, the distribution is:

| Severity | Count |
| -------- | ----: |
| Medium   | 1,401 |
| Critical | 1,384 |
| Low      | 1,076 |
| High     | 1,034 |
| Missing  |   355 |

The most common known severity category is **Medium**.

---

# 15. Case Status and Resolution Cleaning

The notebook standardizes case status values and corrects typographical errors.

Examples include:

```text
pendng → pending
investgation → under investigation
```

Resolution values are also standardized:

```text
arres made → arrest made
warning → warning issued
dismissed → case dismissed
```

These transformations improve consistency for subsequent analysis.

---

# 16. Online Reporting Cleaning

The `reported_online` field contains several representations of Boolean values.

The notebook standardizes:

```text
yes
true
1
```

to:

```text
yes
```

and:

```text
no
false
0
```

to:

```text
no
```

The observed distribution includes:

* No: 2,405
* Yes: 2,341
* Missing: 504

---

# 17. Numerical Data Cleaning

The project also investigates numerical columns for invalid values and formatting problems.

Important numerical variables include:

* `latitude`
* `longitude`
* `suspect_age`
* `victim_age`
* `num_arrests`
* `property_loss_usd`

---

## 17.1 Property Loss

Property loss is initially cleaned because monetary values may contain:

* `$`
* commas
* spaces
* text formatting

The notebook removes dollar signs and commas before converting the variable to numeric format.

For example:

```text
$25,000
```

can be converted into:

```text
25000
```

Invalid values that cannot be converted are represented as missing values using `errors="coerce"`.

---

# 18. Invalid Numerical Values

The notebook checks for logically invalid observations.

Examples include:

### Suspect age

Values below 0 or above 100 are investigated.

### Victim age

Values below 0 or above 100 are investigated.

### Number of arrests

Negative arrest counts are considered invalid.

### Property loss

Negative monetary loss values are considered invalid.

Invalid observations are replaced with missing values rather than being allowed to remain as impossible values.

---

# 19. Datetime Processing

The `incident_datetime` field is converted to Pandas datetime format:

```python
df["incident_datetime"] = pd.to_datetime(
    df["incident_datetime"],
    errors="coerce"
)
```

The notebook then extracts useful temporal variables.

### Incident year

```python
df["incident_year"]
```

### Incident month

```python
df["incident_month"]
```

### Day of week

```python
df["incident_day"]
```

### Hour of day

```python
df["incident_hour"]
```

This transforms one complex datetime variable into multiple features that can be analyzed separately.

---

# 20. Missing Value Treatment

Missing values are handled differently depending on the type of variable.

---

## 20.1 Numerical Variables

For numerical variables such as:

* `suspect_age`
* `victim_age`
* `num_arrests`
* `property_loss_usd`

the notebook uses **median imputation**.

Conceptually:

```python
df[column] = df[column].fillna(df[column].median())
```

### Why median?

Median imputation is useful because the median is less affected by extreme values than the mean.

This is particularly useful when numerical variables may contain skewed distributions or outliers.

---

## 20.2 Categorical Variables

Missing categorical values are replaced with:

```text
unknown
```

This is preferable to inventing a category when the actual value was not recorded.

The approach preserves the information that the value was unavailable.

---

# 21. Outlier Investigation

The notebook does not automatically remove every extreme value.

Instead, it investigates potential outliers.

For property loss, the project calculates:

* Q1
* Q3
* IQR
* Lower limit
* Upper limit

using the standard IQR approach:

```text
IQR = Q3 - Q1

Lower limit = Q1 - 1.5 × IQR

Upper limit = Q3 + 1.5 × IQR
```

The notebook reports:

**Number of property-loss outliers: 0**

The project therefore investigates the values rather than automatically deleting observations simply because they are extreme.

---

# 22. Dataset Validation

After cleaning, the notebook checks:

* Number of rows
* Number of columns
* Duplicate rows
* Duplicate incident IDs
* Data types
* Missing values
* Numerical ranges
* Categorical consistency

At the validation stage, the DataFrame contains:

**5,250 rows and 37 columns**

The notebook still reports:

**200 duplicate rows**

and:

**200 duplicate incident IDs**

This is important to note because the notebook detects duplicates but does not subsequently remove them from the DataFrame.

---

# 23. Exploratory Data Analysis

After data cleaning, the project performs exploratory data analysis to understand the characteristics of the dataset.

The EDA examines:

* Crime types
* Severity
* Districts
* Time patterns
* Crime type vs severity
* Property loss
* Weapons
* Correlations
* Arrests
* Ages
* Yearly trends
* Monthly trends

---

# 24. Crime Type Distribution

The notebook finds that crime incidents are not evenly distributed across crime categories.

The ten most frequent crime types include:

| Crime Type     | Incidents |
| -------------- | --------: |
| Kidnapping     |       275 |
| Sexual assault |       270 |
| Robbery        |       246 |
| Cybercrime     |       237 |
| Vandalism      |       217 |
| Theft          |       215 |
| Trespassing    |       213 |
| Arson          |       205 |
| Assault        |       202 |
| Fraud          |       190 |

According to the notebook's EDA interpretation:

> Kidnapping is the most common crime type.

The analysis demonstrates why categorical standardization is important: without cleaning, variations such as different spellings or abbreviations could fragment the counts.

---

# 25. Crime Severity Distribution

The cleaned known target values are distributed as:

| Severity | Incidents |
| -------- | --------: |
| Medium   |     1,401 |
| Critical |     1,384 |
| Low      |     1,076 |
| High     |     1,034 |

Medium is the most common severity category.

The presence of four target classes makes this a **multiclass classification problem**.

---

# 26. Crime Distribution by District

Crime incidents vary across districts.

The observed distribution is:

| District  | Incidents |
| --------- | --------: |
| North     |       717 |
| South     |       696 |
| West      |       544 |
| Central   |       542 |
| Midtown   |       535 |
| East      |       473 |
| Southwest |       465 |
| Northeast |       449 |
| Northwest |       430 |
| Southeast |       399 |

The notebook identifies **North** as the district with the highest number of recorded incidents.

---

# 27. Crime by Hour

The notebook examines the distribution of incidents across the 24 hours of the day.

This allows the analysis to investigate whether crime frequency changes throughout the day.

The notebook's interpretation is that crime activity exhibits temporal patterns that may be useful for further analysis and prediction.

---

# 28. Crime by Year

The dataset covers the years **2018 through 2024**.

The number of recorded incidents by year is:

| Year | Incidents |
| ---- | --------: |
| 2018 |       622 |
| 2019 |       653 |
| 2020 |       632 |
| 2021 |       652 |
| 2022 |       669 |
| 2023 |       628 |
| 2024 |       674 |

The highest number of incidents in the notebook is recorded in **2024**, with 674 incidents.

---

# 29. Property Loss and Severity

The notebook calculates average property loss for each severity category.

| Severity | Average Property Loss (USD) |
| -------- | --------------------------: |
| Critical |                   25,111.38 |
| High     |                   25,226.94 |
| Low      |                   24,919.86 |
| Medium   |                   24,661.15 |

The differences are relatively small.

The notebook therefore treats property loss as a potentially useful descriptive and predictive variable but does not claim that property loss causes crime severity.

---

# 30. Arrests and Severity

Average arrests by severity are also examined.

| Severity | Average Arrests |
| -------- | --------------: |
| Critical |            2.42 |
| High     |            2.37 |
| Low      |            2.45 |
| Medium   |            2.43 |

The values are very similar across severity categories.

This suggests that the number of arrests does not show a large descriptive difference between the severity groups in this dataset.

---

# 31. Correlation Analysis

The notebook creates a correlation matrix for numerical variables including:

* suspect age
* victim age
* number of arrests
* property loss
* incident year
* incident month
* incident hour

The correlation coefficients are generally close to zero.

Examples:

```text
suspect_age vs victim_age       ≈ 0.00
suspect_age vs num_arrests      ≈ 0.02
victim_age vs property_loss     ≈ -0.01
num_arrests vs property_loss    ≈ -0.01
incident_year vs property_loss  ≈ -0.01
```

The notebook's interpretation is that there is generally little linear correlation between the numerical variables.

This does **not** mean that the variables are completely unrelated in every possible way. It specifically indicates weak linear relationships according to the calculated correlation coefficients.

---

# 32. Age Analysis

The cleaned dataset provides age information for suspects and victims.

Summary statistics include:

### Suspect age

* Mean: approximately 45.53
* Median: 46
* Minimum: 15
* Maximum: 75

### Victim age

* Mean: approximately 49.89
* Median: 50
* Minimum: 10
* Maximum: 90

The notebook also visualizes the distributions of suspect and victim ages.

---

# 33. Crime Type vs Severity

A cross-tabulation is used to examine the relationship between crime type and severity.

The notebook also calculates the percentage distribution of severity within each crime type.

This allows the project to identify whether certain crime categories have different severity profiles.

The notebook concludes:

> The proportion of severity levels varies across crime types.

This indicates that `crime_type` may contain useful information for predicting severity.

---

# 34. District vs Severity

The project examines severity across districts using a cross-tabulation.

This provides a descriptive view of whether severity categories are distributed differently across geographical districts.

The notebook observes that crime incidents and severity distributions vary across districts.

---

# 35. Feature Engineering

Feature engineering is performed to create additional variables that may be useful for analysis and prediction.

The notebook creates:

1. Suspect age groups
2. Victim age groups
3. Property loss bands
4. A weapon-related binary feature
5. A property-loss indicator

---

# 36. Suspect Age Groups

Suspect ages are grouped into categories:

```text
Minor
Young Adult
Adult
Middle Aged
Senior
```

The boundaries used in the notebook are:

```python
age_bins = [0, 17, 29, 44, 59, float("inf")]
```

The resulting distribution is:

| Group       | Count |
| ----------- | ----: |
| Middle Aged | 2,420 |
| Senior      |   993 |
| Adult       |   930 |
| Young Adult |   747 |
| Minor       |   160 |

---

# 37. Victim Age Groups

The same age-grouping strategy is applied to victims.

The resulting distribution is:

| Group       | Count |
| ----------- | ----: |
| Middle Aged | 1,771 |
| Senior      | 1,626 |
| Adult       |   798 |
| Young Adult |   619 |
| Minor       |   436 |

These engineered categorical variables provide a higher-level representation of age.

---

# 38. Property Loss Bands

Property loss is transformed into four categories:

```text
Low Loss
Moderate Loss
High Loss
Very High Loss
```

The notebook uses these boundaries:

```python
loss_bins = [-1, 999, 4999, 9999, float("inf")]
```

The resulting distribution is:

| Property Loss Band | Count |
| ------------------ | ----: |
| Very High Loss     | 4,371 |
| High Loss          |   454 |
| Moderate Loss      |   335 |
| Low Loss           |    90 |

This feature converts a continuous financial variable into interpretable categories.

---

# 39. Binary Feature Engineering

The notebook creates two binary variables.

## Weapon-related feature

```python
df["weapon_involved"] = (
    df["weapon_used"].isin(["unknown", "unarmed"])
).astype(int)
```

This produces:

* `0`: values not in `unknown` or `unarmed`
* `1`: values recorded as `unknown` or `unarmed`

**Implementation note:** Despite its name, the code assigns `1` to `unknown`/`unarmed`, so this feature should be interpreted according to the actual implementation rather than as a conventional "weapon present" indicator.

The resulting counts are:

* 0: 3,919
* 1: 1,331

## Property-loss indicator

```python
df["property_loss_recorded"] = (
    df["property_loss_usd"] > 0
).astype(int)
```

In the notebook, every observation has a positive property-loss value after processing:

```text
1 → 5,250
```

Therefore, this feature does not provide variation in the current dataset and is unlikely to be useful as a predictor.

---

# 40. Feature Selection for Machine Learning

The target variable is:

```text
severity
```

The notebook removes identifiers, personal information, investigation outcomes, and other fields that should not be used as predictive features.

Removed fields include:

```text
incident_id
officer_id
officer_first_name
officer_last_name
badge_number
suspect_id
suspect_first_name
suspect_last_name
victim_id
victim_first_name
victim_last_name
victim_phone
address
notes
case_status
num_arrests
resolution
```

The rationale is to avoid:

* using identifiers as predictors
* unnecessarily exposing personal information
* allowing investigation outcomes to influence severity prediction
* potential target leakage

---

# 41. Target Leakage

The notebook explicitly considers target leakage.

Variables such as:

```text
case_status
resolution
num_arrests
```

are removed from the modeling features in the later clean modeling stage.

This is important because some of these variables may represent information generated during or after the investigation rather than information available when severity needs to be predicted.

A predictive model should ideally use information that would be available at the time the prediction is intended to be made.

---

# 42. Final Modeling Dataset

After feature preparation and removal of records with missing target values, the notebook reports:

```text
X_model shape: (4895, 23)
y_model shape: (4895,)
```

Therefore, the final modeling dataset contains:

* **4,895 observations**
* **23 input features**
* **1 target variable: severity**

There are no missing target values in `y_model`.

---

# 43. Numerical Features

The modeling dataset contains the following numerical features:

```text
latitude
longitude
suspect_age
victim_age
property_loss_usd
incident_year
incident_month
incident_hour
```

These are processed using numerical preprocessing.

---

# 44. Categorical Features

The categorical features include:

```text
crime_type
district
city
state
suspect_gender
suspect_race
victim_gender
weapon_used
reported_online
incident_day
suspect_age_group
victim_age_group
property_loss_band
```

These variables require encoding before they can be used by the machine-learning models.

---

# 45. Train-Test Split

The notebook uses an **80/20 train-test split**.

```python
train_test_split(
    X_model,
    y_model,
    test_size=0.20,
    random_state=42,
    stratify=y_model
)
```

The use of `stratify=y_model` helps maintain the target-class distribution between training and testing datasets.

The resulting dimensions are:

```text
Training features: 3,916 rows
Testing features:    979 rows

Training target:    3,916 rows
Testing target:       979 rows
```

---

# 46. Data Preprocessing Pipeline

The project uses Scikit-learn pipelines to ensure that preprocessing is applied consistently.

---

## Numerical preprocessing

Numerical features are processed using:

1. Median imputation
2. Standard scaling

Conceptually:

```text
Numerical data
      │
      ▼
Median Imputation
      │
      ▼
StandardScaler
```

---

## Categorical preprocessing

Categorical variables are processed using:

1. Missing-value handling
2. One-hot encoding

The encoder uses:

```python
handle_unknown="ignore"
```

This prevents errors when a category appears in the test set that was not present in the training set.

---

# 47. One-Hot Encoding

Categorical variables cannot be directly used by most scikit-learn machine-learning estimators in their raw text form.

For example:

```text
crime_type = robbery
```

is transformed into numerical indicator variables.

One-hot encoding therefore converts categorical values into machine-readable numerical representations.

After preprocessing, the training and testing datasets contain:

```text
Processed training shape: (3916, 134)
Processed testing shape: (979, 134)
```

Thus, the 23 original modeling features expand to **134 encoded features** after preprocessing.

---

# 48. Machine Learning Models

The notebook evaluates three main classification algorithms:

1. Logistic Regression
2. Decision Tree
3. Random Forest

These are all implemented using Scikit-learn pipelines.

---

# 49. Logistic Regression

Logistic Regression is used as one of the baseline classification models.

Although logistic regression is often associated with binary classification, Scikit-learn can use it for multiclass classification.

The notebook evaluates its predictions using:

* Accuracy
* Precision
* Recall
* F1-score

### Initial Logistic Regression performance

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.250 |
| Precision | 0.240 |
| Recall    | 0.250 |
| F1 Score  | 0.239 |

The classification report shows relatively low performance across the four severity categories.

---

# 50. Decision Tree

A Decision Tree classifier is also trained.

The tree attempts to learn decision rules that separate the four severity classes.

### Decision Tree performance

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.324 |
| Precision | 0.324 |
| Recall    | 0.324 |
| F1 Score  | 0.323 |

The Decision Tree performs substantially better than Logistic Regression in the notebook's initial comparison.

The notebook also reports:

```text
Train Score: 1.00
Test Score: approximately 0.324
```

This indicates a substantial difference between training and test performance and is consistent with strong overfitting.

---

# 51. Random Forest

Random Forest is an ensemble model consisting of multiple decision trees.

The notebook initially trains a Random Forest with:

```python
n_estimators=100
```

### Initial Random Forest performance

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.317 |
| Precision | 0.316 |
| Recall    | 0.317 |
| F1 Score  | 0.297 |

The Random Forest performs better than Logistic Regression but slightly below the Decision Tree in this particular initial evaluation.

---

# 52. Model Evaluation Metrics

The notebook uses four primary evaluation metrics.

## Accuracy

Accuracy measures the proportion of predictions that are correct.

```text
Accuracy =
Correct Predictions / Total Predictions
```

It provides an overall measure of classification performance.

---

## Precision

Precision measures how many observations predicted as a class actually belong to that class.

A higher precision means fewer false-positive predictions.

---

## Recall

Recall measures how many actual observations belonging to a class were correctly identified.

Recall is particularly important when failing to identify a particular class has consequences.

---

## F1 Score

F1-score combines precision and recall into a single metric.

It is useful when a model needs a balance between correctly identifying classes and avoiding incorrect positive predictions.

The notebook uses **weighted averages** for the multiclass evaluation metrics.

---

# 53. Hyperparameter Tuning

The notebook does not rely only on default model settings.

It also explores hyperparameter tuning using:

* `GridSearchCV`
* `RandomizedSearchCV`

Cross-validation is used to evaluate different parameter combinations.

---

# 54. Logistic Regression Tuning

The notebook searches over:

```python
C = [0.01, 0.1, 1, 10, 100]
```

and:

```python
solver = ["lbfgs", "liblinear"]
```

using 5-fold cross-validation and weighted F1-score.

The tuned Logistic Regression model produces:

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.250 |
| Precision | 0.240 |
| Recall    | 0.250 |
| F1 Score  | 0.239 |

The tuning therefore does not materially improve the Logistic Regression performance in the notebook.

---

# 55. Random Forest Tuning

The notebook performs grid search over Random Forest parameters such as:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

The best parameters reported are:

```python
{
    'classifier__max_depth': None,
    'classifier__min_samples_leaf': 1,
    'classifier__min_samples_split': 5,
    'classifier__n_estimators': 100
}
```

The best cross-validation weighted F1-score reported is:

```text
0.300
```

The tuned Random Forest achieves:

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.337 |
| Precision | 0.350 |
| Recall    | 0.337 |
| F1 Score  | 0.316 |

This is the strongest model among the tuned models evaluated in the notebook.

---

# 56. Decision Tree Tuning

The notebook also performs Grid Search for the Decision Tree.

Parameters investigated include:

```text
max_depth
min_samples_split
min_samples_leaf
```

The best parameters reported are:

```python
{
    'classifier__max_depth': None,
    'classifier__min_samples_leaf': 1,
    'classifier__min_samples_split': 2
}
```

The best cross-validation weighted F1-score is:

```text
0.304
```

The tuned Decision Tree produces approximately the same test performance as the original Decision Tree:

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.324 |
| Precision | 0.324 |
| Recall    | 0.324 |
| F1 Score  | 0.323 |

---

# 57. Tuned Model Comparison

The notebook explicitly compares the tuned Logistic Regression and tuned Random Forest models.

| Model                     |  Accuracy | Precision |    Recall |        F1 |
| ------------------------- | --------: | --------: | --------: | --------: |
| Tuned Logistic Regression |     0.250 |     0.240 |     0.250 |     0.239 |
| Tuned Random Forest       | **0.337** | **0.350** | **0.337** | **0.316** |

Based on these results, the **Tuned Random Forest** is the strongest model among these tuned models.

---

# 58. Final Cleaned-Data Model Comparison

The notebook later retrains models on the cleaned modeling dataset.

The final reported results are:

| Model               |  Accuracy | Precision |    Recall |        F1 |
| ------------------- | --------: | --------: | --------: | --------: |
| Logistic Regression |     0.257 |     0.245 |     0.257 |     0.244 |
| Decision Tree       | **0.326** | **0.326** | **0.326** | **0.326** |
| Random Forest       |     0.308 |     0.311 |     0.308 |     0.286 |

These results are separate from the earlier tuned-model comparison and should not be mixed together.

---

# 59. Overall Modeling Interpretation

The model results show that predicting crime severity is difficult using the available variables.

The models achieve relatively modest performance:

* Logistic Regression performs the weakest.
* Decision Tree performs better than Logistic Regression.
* Random Forest performs well in some experiments but does not consistently outperform the Decision Tree in every notebook section.
* Hyperparameter tuning improves the Random Forest's test results in the earlier tuning experiment.
* The tuned Random Forest achieves the strongest reported result in the explicit tuned-model comparison.

The best tuned result reported is:

```text
Tuned Random Forest

Accuracy : 0.337
Precision: 0.350
Recall   : 0.337
F1 Score : 0.316
```

Therefore, based specifically on the tuned comparison, Random Forest is the preferred model in the notebook.

---

# 60. Confusion Matrices

The notebook generates confusion matrices for Logistic Regression and Random Forest models.

A confusion matrix allows the model's predictions to be examined class by class.

For this four-class problem, the matrix helps identify where the model confuses:

```text
Low
Medium
High
Critical
```

with one another.

This is more informative than accuracy alone because two models can have similar accuracy while making very different types of classification errors.

---

# 61. Important Modeling Considerations

There are several important observations from the notebook that should be considered when interpreting the results.

### 1. The dataset is synthetic

The dataset is explicitly described as synthetic and educational.

Therefore, the results should not be interpreted as evidence about real-world crime.

### 2. Predictive performance is modest

The reported accuracy values are around 25–34%.

This indicates that the available features do not provide a particularly strong predictive signal for severity in the current dataset.

### 3. Decision Tree overfitting

The notebook reports a Decision Tree training accuracy of approximately 100% while its test accuracy is approximately 32%.

This is a strong warning sign for overfitting.

### 4. Multiple experiments are present

The notebook contains both baseline and tuned model experiments.

Consequently, results should be reported together with the experiment configuration rather than selecting numbers without context.

### 5. Duplicate records

The notebook identifies 200 duplicate rows and 200 duplicate incident IDs.

The notebook detects these duplicates but does not remove them before the modeling stages.

This is an important limitation because duplicates can potentially influence model training and evaluation.

---

# 62. Data Leakage Considerations

The project makes an effort to remove potential leakage variables from the modeling data.

Variables such as:

```text
case_status
resolution
num_arrests
```

are excluded from the later modeling setup.

This is important because these variables may contain information that becomes available after an incident has been investigated.

Using such information to predict severity could produce an unrealistically optimistic model.

---

# 63. Privacy and Sensitive Information

The dataset contains fields representing personal or identifying information, including:

* names
* IDs
* badge numbers
* phone numbers
* addresses

These variables are removed from the machine-learning feature set.

This has two benefits:

1. They are not useful as meaningful generalizable predictors.
2. They should not be unnecessarily incorporated into a predictive model.

---

# 64. Key Findings

The major findings from the notebook are:

### Crime distribution

Crime incidents are not evenly distributed across crime types.

### Most common crime type

Kidnapping is the most frequent crime type in the cleaned EDA results.

### Severity

Medium is the most common severity category.

### Geography

North has the highest number of recorded incidents among the listed districts.

### Time

Crime incidents vary across hours, months, and years.

### Property loss

Average property loss varies only modestly between severity groups.

### Arrests

Average arrests are very similar across severity categories.

### Correlations

The numerical variables show generally weak linear correlations with each other.

### Feature engineering

Age groups, property-loss bands, and binary indicators were created to provide additional representations of the original data.

### Machine learning

The models demonstrate that crime severity is relatively difficult to predict using the available features.

### Best tuned model

The tuned Random Forest provides the strongest reported tuned performance:

```text
Accuracy : 0.337
Precision: 0.350
Recall   : 0.337
F1 Score : 0.316
```

---

# 65. Project Strengths

The project demonstrates several important data science practices.

### Comprehensive data cleaning

The notebook does not immediately jump into machine learning. It first investigates the quality of the data.

### Categorical standardization

Multiple spelling and formatting problems are explicitly identified and corrected.

### Numerical validation

Invalid ages, negative values, and numerical formatting problems are investigated.

### Missing-data strategy

Different strategies are used for numerical and categorical variables.

### Feature engineering

The project creates meaningful additional variables from existing information.

### Leakage awareness

Potentially problematic variables are removed from the modeling dataset.

### Reproducibility

The train-test split uses:

```python
random_state=42
```

which helps produce reproducible results.

### Pipelines

Scikit-learn pipelines are used to combine preprocessing and modeling steps.

### Hyperparameter tuning

Grid Search and cross-validation are used to investigate better model configurations.

### Multiple models

Three different classification approaches are compared rather than relying on a single algorithm.

---

# 66. Project Limitations

The notebook also reveals limitations that should be acknowledged.

## Duplicate observations

The dataset contains 200 duplicate rows and duplicate incident IDs, and these are detected but not removed in the notebook.

## Synthetic data

The dataset is synthetic and therefore may not represent real crime patterns.

## Limited predictive performance

The models achieve relatively low to moderate classification performance.

## Weak numerical correlations

The numerical variables show generally weak linear relationships.

## Potentially uninformative feature

`property_loss_recorded` is equal to 1 for all 5,250 records, meaning it has no variation in the processed dataset.

## Weapon indicator interpretation

The code used to construct `weapon_involved` assigns 1 to `unknown` and `unarmed`, so its name does not perfectly describe its implementation.

## Temporal representation

The notebook creates year, month, day, and hour features. The original `incident_datetime` is present in the modeling feature list but is not included among the numerical or categorical preprocessing feature lists, meaning it is not directly processed as a model feature by the shown preprocessing configuration.

## Moderate model performance

The relatively low scores indicate that additional information or improved modeling strategies would likely be required for stronger severity prediction.

---

# 67. Recommended Future Improvements

The following improvements could be considered in a future version of the project.

### 1. Remove duplicate records

Before modeling, duplicate incident records should be investigated and, where appropriate, removed.

For example:

```python
df = df.drop_duplicates()
```

However, duplicate handling should be based on the meaning of the data rather than automatically deleting records.

### 2. Investigate duplicate incident IDs

Duplicate IDs should be examined to determine whether they represent repeated records or multiple incidents sharing an identifier.

### 3. Improve feature selection

Features with no variation, such as `property_loss_recorded` in this dataset, should be removed.

### 4. Improve temporal modeling

Time features could be expanded to include:

* weekend vs weekday
* morning/afternoon/evening/night
* season
* cyclical representations of hour and month

### 5. Address class performance

Per-class precision, recall, and F1-score should be considered rather than relying only on weighted averages.

### 6. Explore additional models

Future work could investigate models such as:

* Gradient Boosting
* HistGradientBoosting
* XGBoost
* LightGBM
* Support Vector Machines

provided that the project environment supports them.

### 7. Class imbalance analysis

Although the severity classes are not extremely imbalanced, class-specific metrics should still be monitored.

### 8. Feature importance analysis

Feature importance or model interpretation methods could be used to identify which cleaned variables contribute most strongly to severity predictions.

### 9. Better validation

Future versions could use repeated cross-validation or a dedicated validation set to provide more robust estimates of generalization performance.

---

# 68. How to Run the Project

## Requirements

Install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Dataset

Place the dataset archive in the project directory.

The notebook expects:

```text
archive (2).zip
```

The notebook then reads the CSV contained inside the ZIP archive.

---

## Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Messy-crime-dataset.ipynb
```

Run the notebook cells sequentially.

---

# 69. Project Structure

A suggested repository structure is:

```text
messy-crime-dataset/
│
├── Messy-crime-dataset.ipynb
├── archive (2).zip
├── README.md
└── requirements.txt
```

A `requirements.txt` file can contain:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

# 70. Reproducibility

The notebook uses fixed random seeds in several machine-learning operations.

For example:

```python
random_state=42
```

is used in:

* train-test splitting
* Logistic Regression
* Decision Tree
* Random Forest
* randomized hyperparameter search

This makes the experiments more reproducible when run under the same software and data conditions.

---

# 71. Summary of the End-to-End Process

The project can be summarized as follows:

```text
1. Load the messy crime dataset
             ↓
2. Inspect rows, columns and data types
             ↓
3. Identify missing values
             ↓
4. Identify duplicate rows and IDs
             ↓
5. Clean categorical variables
             ↓
6. Correct spelling and formatting errors
             ↓
7. Clean numerical variables
             ↓
8. Correct invalid values
             ↓
9. Convert dates to datetime
             ↓
10. Extract year, month, day and hour
             ↓
11. Handle missing values
             ↓
12. Investigate outliers
             ↓
13. Perform exploratory data analysis
             ↓
14. Engineer age and property-loss features
             ↓
15. Define severity as the target
             ↓
16. Remove IDs, personal information and leakage variables
             ↓
17. Split data into training and testing sets
             ↓
18. Impute and scale numerical variables
             ↓
19. One-hot encode categorical variables
             ↓
20. Train Logistic Regression
             ↓
21. Train Decision Tree
             ↓
22. Train Random Forest
             ↓
23. Evaluate models
             ↓
24. Tune selected models
             ↓
25. Compare results
             ↓
26. Identify the strongest reported model
```

---

# 72. Final Conclusion

This project demonstrates a complete data science workflow using a deliberately messy synthetic crime incident dataset.

The most important part of the project is not simply the final machine-learning model. It is the process of transforming poor-quality data into a structured dataset suitable for analysis and predictive modeling.

The project demonstrates how to:

* Inspect raw data
* Detect missing values
* Identify duplicates
* Standardize categorical variables
* Correct spelling errors
* Validate numerical values
* Handle missing data
* Convert and engineer datetime features
* Investigate outliers
* Perform exploratory data analysis
* Engineer additional features
* Avoid potential target leakage
* Build preprocessing pipelines
* Train multiple classification models
* Evaluate classification performance
* Tune hyperparameters
* Compare machine-learning models

The modeling results indicate that predicting crime severity is challenging with the available features. Among the tuned models explicitly compared in the notebook, the **Tuned Random Forest** produced the strongest reported results, with an accuracy of approximately **33.7%** and a weighted F1-score of approximately **31.6%**.

However, these results should be interpreted carefully because the dataset is synthetic, duplicate records remain in the workflow, and the overall predictive performance is modest.

Overall, the project provides a practical demonstration of how a data scientist can move from a messy raw dataset through **data cleaning → exploration → feature engineering → preprocessing → machine learning → evaluation → model tuning**.

---

# 73. Technologies Used

| Technology       | Purpose                             |
| ---------------- | ----------------------------------- |
| Python           | Programming language                |
| Pandas           | Data manipulation and analysis      |
| NumPy            | Numerical operations                |
| Matplotlib       | Data visualization                  |
| Seaborn          | Statistical visualization           |
| Scikit-learn     | Machine learning and preprocessing  |
| Jupyter Notebook | Interactive development environment |

---

# 74. Author

**Project:** Messy Crime Incident Dataset
**Focus:** Data Cleaning, Exploratory Data Analysis and Crime Severity Prediction
**Dataset Type:** Synthetic / Educational

---

## License / Educational Disclaimer

This project uses a synthetic dataset for educational and demonstration purposes.

It should not be used to make decisions about real individuals, suspects, victims, police officers, communities, or real-world criminal incidents.

The machine-learning results are experimental and should not be interpreted as a validated crime-prediction system.
