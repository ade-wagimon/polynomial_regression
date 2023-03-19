# polynomial_regression

    # Import necessary libraries
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Create sample data
    x = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
    y = np.array([2, 4, 5, 4, 5])

    # Create polynomial features up to degree 2
    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(x)

    # Create a linear regression model
    model = LinearRegression()

    # Train the model on the polynomial features
    model.fit(x_poly, y)

    # Predict the output for new data points
    x_new = np.array([6, 7]).reshape((-1, 1))
    x_new_poly = poly.transform(x_new)
    y_new = model.predict(x_new_poly)

    # Print the predicted output
    print(y_new)



In this example, we first create some sample data consisting of input variable x and output variable y. We then use the PolynomialFeatures class from scikit-learn to create polynomial features up to degree 2 for the input variable x.

Next, we create an instance of the LinearRegression class and fit the model on the polynomial features using the fit() method.

Finally, we predict the output for new data points by transforming them into polynomial features using the transform() method of PolynomialFeatures and then passing the transformed features to the predict() method of the linear regression model.

I hope this example helps you understand how to implement polynomial regression in Python using scikit-learn!




