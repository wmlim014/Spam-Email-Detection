[![Website](https://img.youtube.com/vi/<video-id>/0.jpg)](https://drive.google.com/drive/folders/17DYs4FJy2nYo5DIE90jqHRUeNrKGuuH5)

# Load app.py
```
cd SimplePage/webapp
```
```
myenv\Scripts\activate
```
```
python app.py 
```

# [Basic Setup in Terminal](https://blog.bolajiayodeji.com/how-to-deploy-a-machine-learning-model-to-the-web)
1. Install or upgrade pip
    ```
    python.exe -m pip install --upgrade --user pip
    ```
2. Python installation
    - Verified python versions
        ```
        python --version
        ```
    - Install python
        ```
        pip install python
        ```
3. Python environment setup
    - Go to folder stored webpage
        ```
        cd SimplePage/webapp
        ```
    - Create a python environment
        ```
        python -m venv myenv  
        ```
    - Access created environment
        ```
        myenv\Scripts\activate
        ```
    - Install package needed *(add more if needed)*
        ```
        pip install numpy scipy scikit-learn flask xgboost
        ```
    - Check package list and version
        ```
        pip list
        ```