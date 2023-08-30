# AUTO CODEREVIEW

This is a project that the user uses chatGPT to automatically review gerrit commit records.

## Setup

1. If you don't have Python installed, [install it from here](https://www.python.org/downloads/)

2. If you don't have [Poetry](https://python-poetry.org/docs/)(Python package manager) installed, install it by the following command:
   
   ```bash
   $ curl -sSL https://install.python-poetry.org | python3 -
   ```
   
3. Install dependencies

   ```bash
   $ poetry install
   ```

4. Navigate to the run directory

   ```bash
   $ cd src
   ```

5. Fill in your gerrit address, username and password, and change number in the `main.py` file.

6. Add your [API key](https://beta.openai.com/account/api-keys) to the `reviewer.py` file.

7. Run the app

   ```bash
   $ python main.py
   ```