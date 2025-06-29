import yaml

def cargar_parametros_modelos(path="parametros_modelos.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)