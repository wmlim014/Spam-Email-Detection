### Refer to Basic Writing and Formatting Syntax for *.md files. [Click Here](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)*

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
        pip install numpy scipy scikit-learn flask
        ```
    - Check package list and version
        ```
        pip list
        ```