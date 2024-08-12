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
3. [Basic setup before Spacy installation](https://www.youtube.com/watch?v=Or5r9gg-bns)
    - Open an empty folder located in computer (prefered in thumb-drive) and open it in terminal
    - Use the following script step by step
        ```
        python -m venv spacy_venv
        ```

        * Create Temporarily Bypass Execution Policy
            ```
            powershell -ExecutionPolicy Bypass -File .\spacy_venv\Scripts\Activate.ps1
            ```

        * Change Execution Policy (current user only)
            ```
            Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
            ```
            
        ```
        .\spacy_venv\Scripts\activate
        ```