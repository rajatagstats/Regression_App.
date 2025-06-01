import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer

# Streamlit app
st.title("Upload an Excel File and Select Model Parameters")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Preview of the file:")
    st.dataframe(df)

    # Select frequency of data
    frequency = st.selectbox("Select data frequency", ["Daily", "Weekly", "Monthly", "Yearly"])

    # Select target variable
    target_var = st.selectbox("Select target variable", df.columns)

    # Select model
    model_choice = st.selectbox("Select a model", ["Linear Regression", "Random Forest", "Support Vector Machine"])

    # Prepare data (handling datetime columns)
    X = df.drop(columns=[target_var])

    for col in X.select_dtypes(include=['datetime64']):
        X[col] = pd.to_datetime(X[col], errors='coerce')  # Ensure proper format
        X[col] = X[col].fillna(pd.Timestamp('2000-01-01'))  # Handle missing values
        X[col] = X[col].apply(lambda x: x.toordinal() if pd.notnull(x) else None)  # Convert to ordinal safely

    y = df[target_var]

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")  # Replace NaNs with mean values
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Choose model
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor()
        elif model_choice == "Support Vector Machine":
            model = SVR()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = model.score(X_test, y_test)

        st.write(f"Model trained using {model_choice}. Accuracy: {score:.2f}")

        # Plot actual vs predicted values
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Perfect Fit')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs. Predicted Values")
        ax.legend()

        st.pyplot(fig)

