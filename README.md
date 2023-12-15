# Background

As a burgeoning startup specializing in machine learning solutions within the European banking market, our focus spans diverse challenges—from fraud detection and sentiment classification to predicting and understanding customer intentions. A central aspect of our mission is developing a robust machine learning system that harnesses insights derived from call center data. Our overarching goal is to significantly enhance the success rate of calls made to customers for various client offerings. In pursuit of this objective, we're actively engaged in designing an ever-evolving machine learning product that not only delivers high success outcomes but also ensures interpretability, empowering our clients to make well-informed decisions.

# Data Description

Our dataset originates from the direct marketing efforts of a European banking institution. The marketing campaign involves making phone calls to customers, often multiple times, to secure product subscriptions, particularly term deposits. Term deposits, characterized by short-term maturities ranging from one month to a few years, necessitate customers to comprehend that funds can only be withdrawn after the term concludes. For privacy concerns, all customer information revealing personal details has been meticulously removed.

- **age:** Age of the customer (numeric)
- **job:** Type of job (categorical)
- **marital:** Marital status (categorical)
- **education:** Education level (categorical)
- **default:** Has credit in default? (binary)
- **balance:** Average yearly balance, in euros (numeric)
- **housing:** Has a housing loan? (binary)
- **loan:** Has a personal loan? (binary)
- **contact:** Contact communication type (categorical)
- **day:** Last contact day of the month (numeric)
- **month:** Last contact month of the year (categorical)
- **duration:** Last contact duration, in seconds (numeric)
- **campaign:** Number of contacts performed during this campaign and for this client (numeric, includes the last contact)
  
**Output (desired target):**
- **y:** Has the client subscribed to a term deposit? (binary)

# Methodology

- Conducte exploratory data analysis, balanced out the data using the SMOTE technique, and fille in missing values using the mode imputation method.
- Implement various machine learning models (Support Vector Machine, Random Forest, KNN, XGBoost).
- Apply Grid-Search for hyperparameter tuning.

# Summary

XGBoost demonstrated exceptional performance with 96% accuracy and a 0.96 F1-score.
