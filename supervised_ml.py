import numpy as np
<<<<<<< HEAD
from sklearn.linear_model import LinearRegression

# Sample data: Years of Experience vs. Salary
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([30000, 35000, 42000, 50000, 58000])

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Predict salary for 6 years of experience
predicted_salary = model.predict([[6]])
print(f"Supervised ML - Linear Regression Prediction:")
print(f"Predicted salary for 6 years of experience: ${predicted_salary[0]:,.2f}")
=======
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main():
    # 1. Prepare the dataset using Python lists
    hours_studied = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    exam_scores = [30, 40, 45, 55, 65, 70, 80, 85, 90, 95] 

    # 2. Separate input features (X) and target values (y)
    X = np.array(hours_studied).reshape(-1, 1)
    y = np.array(exam_scores)

    # 3. Create and train the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # 4. User Interaction & Prediction
    try:
        
        user_input = input("Enter the number of hours studied: ")
        hours = float(user_input)

        # Use .item() to safely pull out the raw single number (scalar)
        predicted_score = model.predict([[hours]]).item()
        print(f"Predicted exam score: {predicted_score:.2f}")
    except ValueError:
        print("Please enter a valid numeric value for hours.")
        return # Stop execution if input is bad


        # ==================== BONUS TASK: MATPLOTLIB GRAPH ====================
        print("\nGenerating graph... Close the window to exit the program.")

        # Plot the original dataset as red dots
        plt.scatter(X, y, color='red', label='Actual Data')

        # Generate predictions across the line to draw the best-fit line
        line_predictions = model.predict(X)
        plt.plot(X, line_predictions, color='blue', linewidth=2, label='Regression Line')

        # Optional: Plot the user's specific prediction point as a green star
        plt.scatter([[hours]], [[predicted_score]], color='green', marker='*', s=200, label='Your Prediction')

        # Add labels and styling
        plt.title('Relationship Between Study Hours and Exam Scores')
        plt.xlabel('Hours Studied')
        plt.ylabel('Exam Scores')
        plt.xlim(0, 11)
        plt.ylim(20, 110)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # Display the graph window
        plt.show()

# This block MUST be completely flush against the left wall (no indentation)
if __name__ == "__main__":
    main()












                
                



            


>>>>>>> 9adf13a821d0ab155de42d90a66e07bfa6687637
