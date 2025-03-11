import pickle

# Replace 'your_file.pkl' with your file path
with open('D:\FYPSeagullClassification01\Features_Analysis\Outputs\LBP_Features\lbp_features.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)  # This will print the content in a readable format
