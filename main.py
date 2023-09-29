import tkinter as tk
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Historical medal count, GDP, Per Capita, and Growth data with host countries
data = {
    'Year': [
        1951, 1954, 1958, 1962, 1966, 1970, 1974, 1978,
        1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010,
        2014, 2018
    ],
    'Medal_Count': [
        
        51, 17, 13, 33, 21, 25, 28, 28, 57, 37, 23, 23, 35, 36, 53, 65, 57, 70,
    ],
    'Host_Country': [
        'India', 'Philippines', 'Japan', 'Indonesia', 'Thailand', 'Thailand', 'Iran', 'Thailand',
        
        'India', 'South Korea', 'China', 'Japan', 'Thailand', 'South Korea', 'Qatar', 'China', 'South Korea', 'Indonesia', 'China', 'China'
    ],
    'GDP': [
        270.11, 288.21, 279.30, 327.28, 360.28, 392.90, 415.87, 421.35, 458.82, 468.39, 485.44, 514.94,
        607.70, 709.15, 820.38, 940.26, 1216.74, 1198.90, 1341.89, 1675.62, 1823.05, 1827.64, 1856.72, 2039.13,
        # Adding GDP data (update as needed)
        2103.59, 2294.80, 2651.47, 2702.93, 2835.61, 2671.60, 3150.31, 3385.09
    ],
    'Per_Capita': [
        304, 318, 302, 346, 374, 400, 415, 413, 441, 442, 450, 469, 544, 624, 711, 802, 1023, 994, 1097, 1351, 1438, 1450, 1434, 1438,
        # Adding Per Capita data (update as needed)
        1590, 1714, 1958, 1974, 2060, 2389, 2389
    ],
    'Growth': [
        1.06, 5.48, 4.75, 6.66, 7.57, 7.55, 4.05, 6.18, 8.85, 3.84, 4.82, 3.80, 7.86, 7.86, 8.06, 7.66, 3.09, 7.86, 7.92, 7.92, 7.86,
        4.05, 6.18, 7.55, 7.57, 6.66, 4.75, 5.48, 1.06, 5.53, 5.95, 9.63, 3.97, 4.78, 5.25, 3.82, 7.29, 3.48, 6.01, 6.74, -5.24,
        5.71, 7.25, 1.66, 9.15, 1.19, 3.30, -0.55, 1.64, 5.16, 6.54, 3.39, 7.83, - \
        0.06, - \
        2.64, 7.45, 5.99, 2.93, 3.72  # Adding Growth data (update as needed)
    ]
}

# Create a Polynomial Regression model
poly_degree = 3  # You can adjust the degree of the polynomial
poly_features = PolynomialFeatures(degree=poly_degree)
X_poly = poly_features.fit_transform(np.array(data['Year']).reshape(-1, 1))
model = LinearRegression()
model.fit(X_poly, data['Medal_Count'])

# Function to predict medal count for 2023


def predict_medals():
    year_2023 = np.array([[2023]])
    year_2023_poly = poly_features.transform(year_2023)
    # Add 30 to the predicted medals
    predicted_medals = int(model.predict(year_2023_poly)[0]) 

    # Display the output including all data
    host_country = data['Host_Country'][-1]
    gdp_2022 = data['GDP'][-2]  # GDP for 2022
    per_capita_2022 = data['Per_Capita'][-2]  # Per Capita for 2022
    growth_2022 = data['Growth'][-2]  # Growth for 2022

    label_result.config(text=f"Predicted Medal Count for 2023: {predicted_medals}\n"
                             f"Host Country: {host_country} (2023)\n"
                             f"GDP in 2022: ${gdp_2022:.2f}B\n"
                             f"Per Capita in 2022: ${per_capita_2022:.2f}\n"
                             f"Growth in 2022: {growth_2022:.2f}%")


# Create the GUI window
root = tk.Tk()
root.title("Asian Games Medal Prediction using Polynomial Regression")

# Create and configure widgets
label_title = tk.Label(
    root, text="Predict India's Medal Count in 2023 Asian Games", font=("Arial", 14, "bold"))
button_predict = tk.Button(
    root, text="Predict", command=predict_medals, font=("Arial", 12))
label_result = tk.Label(root, text="", font=("Arial", 12))

# Arrange widgets in the GUI
label_title.pack(pady=20)
button_predict.pack(pady=10)
label_result.pack()

# Start the GUI event loop
root.mainloop()
