name: Train Random Forest Weather Model

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]
  workflow_dispatch:

jobs:
  train-model:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pandas numpy scikit-learn matplotlib seaborn
        
    - name: Run training script
      run: |
        python train.py
        
    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: training-results
        path: |
          weather_analysis_results.png
          model_results.txt
          results.json
        retention-days: 30
        
    - name: Upload data file
      uses: actions/upload-artifact@v4
      with:
        name: original-data
        path: data.csv
        retention-days: 30
