# Athena - Insights Service

# Setup
### Setting up Environment

To setup the environment please create the Python Virtual Environment if not already setup:
```
python3 -m venv .venv
```
If using Windows please use this command to activate venv:
```
.venv\Scripts\Activate.ps1
```
Else if on another platform please use this command:
```
source .venv/bin/activate
```
### Updating the dependencies

With the environment install any dependencies you want then update the requirements file

For example, lets call the module that we want to install in the venv ``dependency``
```
python3 -m pip install dependency
```
Then make sure to update the requirements.txt with all the dependencies from the venv. Remember that the venv should be activated by now. 
```
python3 -m pip freeze > requirements.txt
```
### Installing dependencies in the Virtual Environment

To install the dependencies from the requirements use this command
```
python3 -m pip install -r requirements.txt
```

### Deactivate virtual environment
To deactivate virtual environment run this command
```
deactivate
```

## Development

### Activating the Virtual Environment

Please active the virtual environment be following the steps above.

### Starting the server

Then to start the server run the command below:
```
python3 main.py
```
