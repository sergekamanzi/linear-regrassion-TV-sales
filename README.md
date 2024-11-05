# TV Sales Prediction using Linear Regression

This project demonstrates the use of **linear regression** to predict sales based on the amount spent on TV advertising. It utilizes a dataset containing advertising expenses and corresponding sales figures to build a machine learning model, specifically using **Scikit-learn** for linear regression.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Description](#model-description)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The primary objective of this project is to predict sales based on TV advertising expenditure using a linear regression model. It provides insights into how spending on TV advertisements impacts sales and helps marketers allocate their budgets effectively.

## Data Description
The dataset contains the following columns:
- **TV**: Advertising budget spent on TV (in thousands of dollars)
- **Sales**: Product sales in thousands of units

### Sample Data
| TV    | Sales |
|-------|-------|
| 230.1 | 22.1  |
| 44.5  | 10.4  |
| 17.2  | 9.3   |

### Data Source
The dataset is fictional but commonly used in linear regression examples for educational purposes.

## Installation
To run this project locally, you will need Python 3.x installed along with several packages such as **numpy**, **pandas**, **matplotlib**, and **scikit-learn**.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tv-sales-prediction.git
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the model and visualize predictions, execute the following script:

```bash
python linear_regression_tv_sales.py
```

This script will:
- Load the dataset
- Train the linear regression model on TV advertising data
- Predict sales and plot the results

### Example Code:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load your data (assuming a pandas DataFrame)
data = pd.read_csv('advertising.csv')
X = data[['TV']]  # Feature
Y = data['Sales']  # Target

# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Visualize the predictions
plt.scatter(X_test, Y_test, color='black', label='Actual Sales')
plt.plot(X_test, Y_pred, color='blue', linewidth=2, label='Predicted Sales')
plt.xlabel('TV Ad Spending (in thousands)')
plt.ylabel('Sales (in thousands)')
plt.title('TV Ad Spending vs Sales Prediction')
plt.legend()
plt.show()
```


## Model Description
This project uses **Simple Linear Regression** to model the relationship between TV ad spending and sales. The linear regression equation is:

```
Sales = m * TV + b
```
Where:
- `m` is the slope (how much sales change for each unit increase in TV ad spending)
- `b` is the intercept (sales when TV ad spending is zero)

### Evaluation Metrics:
- **Mean Squared Error (MSE)**: Used to measure the average squared difference between actual sales and predicted sales.
- **R-squared**: Indicates how well the model explains the variance in sales.

## Results
After training the model, the results showed that **TV ad spending** had a significant positive impact on **sales**. The model was able to predict sales with a reasonable accuracy.

Sample plot of TV ad spending vs predicted sales:

![TV Sales Prediction Plot](plot.png)  

## Contributing
Contributions are welcome! If you have suggestions, please feel free to open an issue or submit a pull request.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
