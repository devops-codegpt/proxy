# ChatGPT Proxy

This is a project that the user uses grpc to communicate with chatGPT. The project internally uses the [official openAI API](https://github.com/openai/openai-python).

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

5. Make a copy of the example environment variables file

   ```bash
   $ cp .env.example .env
   ```

6. Add your [API key](https://beta.openai.com/account/api-keys) to the newly created `.env` file. `OPENAI_API_KEY` is required, other items are optional.

7. Run the app

   ```bash
   $ python main.py
   ```

You should now be able to access the app at grpc server( the port is `50051`)!