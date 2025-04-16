"use client"

import { useState } from "react"
import { Box, Container, Typography, Tabs, Tab, Paper, Card, CardContent, CardMedia, Grid } from "@mui/material"

const TheoryPage = () => {
  const [selectedTab, setSelectedTab] = useState(0)

  const handleTabChange = (event, newValue) => {
    setSelectedTab(newValue)
  }

  return (
    <Container maxWidth="lg">
      <Typography variant="h4" gutterBottom>
        Machine Learning Theory
      </Typography>
      <Typography variant="body1" paragraph>
        Understanding the mathematical foundations and practical applications of machine learning algorithms.
      </Typography>

      <Paper sx={{ mb: 4 }}>
        <Tabs
          value={selectedTab}
          onChange={handleTabChange}
          variant="fullWidth"
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab label="Linear Regression" />
          <Tab label="Random Forest" />
          <Tab label="Logistic Regression" />
        </Tabs>
      </Paper>

      {/* Linear Regression Content */}
      {selectedTab === 0 && (
        <Box>
          <Paper sx={{ p: 3, mb: 4 }}>
            <Typography variant="h5" gutterBottom>
              Linear Regression Theory
            </Typography>
            <Typography variant="body1" paragraph>
              Linear regression is a linear approach to modeling the relationship between a dependent variable and one
              or more independent variables. The case of one independent variable is called simple linear regression,
              while multiple independent variables is called multiple linear regression.
            </Typography>

            <Typography variant="h6" gutterBottom>
              Mathematical Formulation
            </Typography>
            <Typography variant="body1" paragraph>
              The linear regression model can be written as:
            </Typography>
            <Box sx={{ p: 2, bgcolor: "grey.100", borderRadius: 1, mb: 3 }}>
              <Typography variant="body1" sx={{ fontFamily: "monospace" }}>
                y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
              </Typography>
            </Box>
            <Typography variant="body1" paragraph>
              Where:
              <ul>
                <li>y is the dependent variable (target)</li>
                <li>x₁, x₂, ..., xₙ are the independent variables (features)</li>
                <li>β₀, β₁, β₂, ..., βₙ are the coefficients (parameters to be estimated)</li>
                <li>ε is the error term (residual)</li>
              </ul>
            </Typography>

            <Typography variant="h6" gutterBottom>
              Learning Process
            </Typography>
            <Typography variant="body1" paragraph>
              The goal of linear regression is to find the values of the coefficients that minimize the sum of squared
              residuals (SSR):
            </Typography>
            <Box sx={{ p: 2, bgcolor: "grey.100", borderRadius: 1, mb: 3 }}>
              <Typography variant="body1" sx={{ fontFamily: "monospace" }}>
                SSR = Σ(yᵢ - ŷᵢ)² = Σ(yᵢ - (β₀ + β₁x₁ᵢ + ... + βₙxₙᵢ))²
              </Typography>
            </Box>
            <Typography variant="body1" paragraph>
              This is typically solved using:
              <ul>
                <li>
                  <strong>Ordinary Least Squares (OLS)</strong>: Analytical solution for the coefficients
                </li>
                <li>
                  <strong>Gradient Descent</strong>: Iterative optimization algorithm
                </li>
              </ul>
            </Typography>

            <Typography variant="h6" gutterBottom>
              Implementation in Python
            </Typography>
            <Box sx={{ p: 2, bgcolor: "grey.100", borderRadius: 1, mb: 3, overflow: "auto" }}>
              <Typography variant="body1" sx={{ fontFamily: "monospace", whiteSpace: "pre" }}>
                {`import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Read the dataset
data = pd.read_csv("your_file.csv")  # Replace with your actual CSV file path

# Define features (X) and target (y)
X = data[["feature1", "feature2", "feature3"]]  # Replace with your actual feature column names
y = data["target"]  # Replace with your actual target column name

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Access the coefficients
coefficients = model.coef_
intercept = model.intercept_`}
              </Typography>
            </Box>
          </Paper>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6} sx={{ width: "48%" }}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Advantages
                  </Typography>
                  <ul>
                    <li>Simple and easy to understand</li>
                    <li>Computationally efficient</li>
                    <li>Provides clear interpretability of feature importance</li>
                    <li>Works well when the relationship is actually linear</li>
                  </ul>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6} sx={{ width: "48%" }}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Limitations
                  </Typography>
                  <ul>
                    <li>Assumes a linear relationship between variables</li>
                    <li>Sensitive to outliers</li>
                    <li>Can underfit complex relationships</li>
                    <li>Assumes independence of features</li>
                  </ul>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Box sx={{ mt: 4, textAlign: "center" }}>
            <Typography variant="h6" gutterBottom>
              Video Tutorial: Linear Regression
            </Typography>
            <iframe
              width="560"
              height="315"
              src="https://www.youtube.com/embed/zPG4NjIkCjc"
              title="YouTube video player"
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            ></iframe>
          </Box>
        </Box>
      )}

      {/* Random Forest Content */}
      {selectedTab === 1 && (
        <Box>
          <Paper sx={{ p: 3, mb: 4 }}>
            <Typography variant="h5" gutterBottom>
              Random Forest Theory
            </Typography>
            <Typography variant="body1" paragraph>
              Random Forest is an ensemble learning method that operates by constructing multiple decision trees during
              training and outputting the class that is the mode of the classes (classification) or mean prediction
              (regression) of the individual trees.
            </Typography>

            <Typography variant="h6" gutterBottom>
              Key Concepts
            </Typography>
            <Typography variant="body1" paragraph>
              <ul>
                <li>
                  <strong>Ensemble Learning</strong>: Combining multiple models to improve performance
                </li>
                <li>
                  <strong>Bagging (Bootstrap Aggregating)</strong>: Training each tree on a random subset of the data
                </li>
                <li>
                  <strong>Feature Randomness</strong>: Each tree considers a random subset of features at each split
                </li>
              </ul>
            </Typography>

            <Typography variant="h6" gutterBottom>
              Learning Process
            </Typography>
            <Typography variant="body1" paragraph>
              For a Random Forest with n_trees:
              <ol>
                <li>
                  For each tree i from 1 to n_trees:
                  <ul>
                    <li>Create a bootstrap sample D_i from the training data</li>
                    <li>Grow a decision tree T_i using D_i</li>
                    <li>At each node, randomly select m features (typically m = sqrt(total features))</li>
                    <li>Choose the best split among these m features</li>
                    <li>Grow the tree to the maximum depth (or until other stopping criteria)</li>
                  </ul>
                </li>
                <li>For regression: Average the predictions of all trees</li>
                <li>For classification: Take a majority vote from all trees</li>
              </ol>
            </Typography>

            <Typography variant="h6" gutterBottom>
              Implementation in Python
            </Typography>
            <Box sx={{ p: 2, bgcolor: "grey.100", borderRadius: 1, mb: 3, overflow: "auto" }}>
              <Typography variant="body1" sx={{ fontFamily: "monospace", whiteSpace: "pre" }}>
                {`import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Read the dataset
data = pd.read_csv("your_file.csv")  # Replace with actual file name

# Define features (X) and target (y)
X = data[["feature1", "feature2", "feature3"]]  # Replace with your actual feature columns
y = data["target"]  # Replace with your target column

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# For regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_reg = rf_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred_reg)

# For classification
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_clf = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_clf)

# Feature importance
feature_importance = rf_reg.feature_importances_  # or rf_clf.feature_importances_`}
              </Typography>

            </Box>
          </Paper>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6} sx={{ width: "48%" }}>
              <Card>

                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Advantages
                  </Typography>
                  <ul>
                    <li>Handles non-linear relationships well</li>
                    <li>Reduces overfitting compared to individual decision trees</li>
                    <li>Provides feature importance measures</li>
                    <li>Robust to outliers and noise</li>
                    <li>Works well with high-dimensional data</li>
                  </ul>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6} sx={{ width: "48%" }}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Limitations
                  </Typography>
                  <ul>
                    <li>More complex and computationally intensive than simple models</li>
                    <li>Less interpretable than linear models or single decision trees</li>
                    <li>Can be memory-intensive for large datasets</li>
                    <li>May overfit on noisy datasets</li>
                  </ul>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Box sx={{ mt: 4, textAlign: "center" }}>
            <Typography variant="h6" gutterBottom>
              Video Tutorial: Random Forest
            </Typography>
            <iframe
              width="560"
              height="315"
              src="https://www.youtube.com/embed/J4Wdy0Wc_xQ"
              title="YouTube video player"
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            ></iframe>
          </Box>
        </Box>
      )}

      {/* Logistic Regression Content */}
      {selectedTab === 2 && (
        <Box>
          <Paper sx={{ p: 3, mb: 4 }}>
            <Typography variant="h5" gutterBottom>
              Logistic Regression Theory
            </Typography>
            <Typography variant="body1" paragraph>
              Logistic regression is a statistical model that uses a logistic function to model a binary dependent
              variable. Despite its name, it's a classification algorithm rather than a regression algorithm.
            </Typography>

            <Typography variant="h6" gutterBottom>
              Mathematical Formulation
            </Typography>
            <Typography variant="body1" paragraph>
              The logistic function (sigmoid) is defined as:
            </Typography>
            <Box sx={{ p: 2, bgcolor: "grey.100", borderRadius: 1, mb: 3 }}>
              <Typography variant="body1" sx={{ fontFamily: "monospace" }}>
                P(y=1) = 1 / (1 + e^(-z))
              </Typography>
              <Typography variant="body1" sx={{ fontFamily: "monospace", mt: 1 }}>
                where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
              </Typography>
            </Box>
            <Typography variant="body1" paragraph>
              This transforms the output to a probability between 0 and 1. If P(y=1) ≥ 0.5, we classify as class 1,
              otherwise as class 0.
            </Typography>

            <Typography variant="h6" gutterBottom>
              Learning Process
            </Typography>
            <Typography variant="body1" paragraph>
              Logistic regression uses maximum likelihood estimation (MLE) to find the best-fitting parameters:
            </Typography>
            <Box sx={{ p: 2, bgcolor: "grey.100", borderRadius: 1, mb: 3 }}>
              <Typography variant="body1" sx={{ fontFamily: "monospace" }}>
                L(β) = Π P(y=1)^y * (1-P(y=1))^(1-y)
              </Typography>
            </Box>
            <Typography variant="body1" paragraph>
              In practice, we minimize the negative log-likelihood using optimization algorithms like gradient descent.
            </Typography>

            <Typography variant="h6" gutterBottom>
              Implementation in Python
            </Typography>
            <Box sx={{ p: 2, bgcolor: "grey.100", borderRadius: 1, mb: 3, overflow: "auto" }}>
              <Typography variant="body1" sx={{ fontFamily: "monospace", whiteSpace: "pre" }}>
                {`import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the dataset
data = pd.read_csv("your_file.csv")  # Replace with actual file name

# Define features (X) and target (y)
X = data[["feature1", "feature2", "feature3"]]  # Replace with your actual feature columns
y = data["target"]  # Replace with your target column

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)  # Probability estimates

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Access the coefficients
coefficients = model.coef_
intercept = model.intercept_`}
              </Typography>

            </Box>
          </Paper>

          <Grid container spacing={3}>
            <Grid item xs={12} md={6} sx={{ width: "48%" }}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Advantages
                  </Typography>
                  <ul>
                    <li>Provides probability scores for classifications</li>
                    <li>Highly interpretable (coefficients represent log-odds)</li>
                    <li>Less prone to overfitting in high-dimensional spaces</li>
                    <li>Computationally efficient</li>
                    <li>Works well for linearly separable classes</li>
                  </ul>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} md={6} sx={{ width: "48%" }}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Limitations
                  </Typography>
                  <ul>
                    <li>Assumes a linear decision boundary</li>
                    <li>Cannot solve non-linear problems without feature engineering</li>
                    <li>May underperform when there are complex relationships</li>
                    <li>Sensitive to imbalanced datasets</li>
                  </ul>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          <Box sx={{ mt: 4, textAlign: "center" }}>
            <Typography variant="h6" gutterBottom>
              Video Tutorial: Logistic Regression
            </Typography>
            <iframe
              width="560"
              height="315"
              src="https://www.youtube.com/embed/yIYKR4sgzI8"
              title="YouTube video player"
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            ></iframe>
          </Box>
        </Box>
      )}
    </Container>
  )
}

export default TheoryPage
