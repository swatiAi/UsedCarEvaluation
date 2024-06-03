
# Used Car Evaluation Tool

Welcome to the Used Car Evaluation Tool! This tool is designed to provide an estimated value for used cars based on various input parameters such as make, model, year, mileage, and condition. It is developed using HTML and CSS for the front end, Python (Flask) for the backend, and includes a machine learning model for valuation.

## Features

- **User-Friendly Interface**: Intuitive and easy-to-use interface for inputting car details.
- **Accurate Valuation**: Provides an estimated value based on current market trends and historical data.
- **Database Management**: Efficiently handles car data using CSV files.
- **Machine Learning**: Utilizes a trained machine learning model for accurate car value predictions.
- **Responsive Design**: Works seamlessly on both desktop and mobile devices.

## File Structure

- **index.html**: The main HTML file for the front end.
- **app.py**: The main Python file for running the Flask server.
- **train.csv, carlistings.csv, result.csv, test.csv**: CSV files containing car data for training and testing the model.
- **model.pkl**: The serialized machine learning model.
- **scaler.pkl**: The serialized scaler used for data preprocessing.
- **UsedCarEvaluation.ipynb, UsedCarEvaluationtest.ipynb, FlaskSetup.ipynb**: Jupyter notebooks used for model training and testing.
- **Test.py**: A Python file for testing purposes.

## Installation

### Prerequisites

- **Python**: Version 3.7 or higher.
- **Flask**: A web framework for Python.
- **Scikit-learn**: For the machine learning model.

### Steps

1. **Clone the Repository**:
   \`\`\`bash
   git clone https://github.com/swatiAi/UsedCarEvaluation.git
   \`\`\`

2. **Navigate to the Project Directory**:
   \`\`\`bash
   cd UsedCarEvaluation
   \`\`\`

3. **Install the Required Packages**:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **Run the Flask Application**:
   \`\`\`bash
   python app.py
   \`\`\`

5. **Access the Application**:
   - Open your web browser and navigate to \`http://localhost:5000\`.

## Usage

1. **Input Car Details**:
   - Enter the car's make, model, year, mileage, and condition in the provided form on the main page.

2. **Get Valuation**:
   - Click on the "Evaluate" button to get an estimated value of the car.

## Contributing

We welcome contributions from the community! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (\`git checkout -b feature/your-feature\`).
3. Commit your changes (\`git commit -m 'Add some feature'\`).
4. Push to the branch (\`git push origin feature/your-feature\`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please open an issue or contact us at support@usedcarevaluation.com.
