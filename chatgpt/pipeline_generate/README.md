# Pipeline Generate
Pipeflow application pipeline file generated by LLM.

## Installation
1. If you don't have Python installed, [install it from here](https://www.python.org/downloads/)

2. If you don't have [Poetry](https://python-poetry.org/docs/)(Python package manager) installed, install it by the following command:

     ```shell
     $ curl -sSL https://install.python-poetry.org | python3 -
     ```

3. Install dependencies

    ```shell
   $ poetry install
    ```

4. Navigate to the run directory

   ```bash
   $ cd src
   ```

5. Install the consul

    ```bash
    wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
    apt update
    apt install -y consul
    ```

    and run it in dev environment:

    ```bash
    consul agent -dev -ui -client=0.0.0.0
    ```

6. Fill in your consul(service register center) address and port in the `main.py` file.

7. Add your [API key](https://beta.openai.com/account/api-keys) to the `main.py` file.

8.  Run project
   ```shell
   python3 main.py
   ```