# CodeGeeX Proxy

This is a project that deploy ReplitLM model and the user uses grpc to communicate with ReplitLM. The project internally uses the [official ReplitLM](https://github.com/replit/ReplitLM).

## Setup
1. Install ReplitLM and download model weigths through this [link](https://github.com/replit/ReplitLM)

2. If you don’t have Python installed, [install it from here](https://www.python.org/downloads/)

3. If you don't have [Poetry](https://python-poetry.org/docs/)(Python package manager) installed, install it by the following command:
   
   ```bash
   $ curl -sSL https://install.python-poetry.org | python3 -
   ```
   
4. Install dependencies

   ```bash
   $ poetry install
   ```

5. Navigate to the run directory

   ```bash
   $ cd src
   ```

6. Run the app

   ```bash
   $ python main.py
   ```

You should now be able to access the app at grpc server( the port is `65091`)!
