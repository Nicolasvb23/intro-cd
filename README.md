# Introducción a la Ciencia de Datos 2026

Este repositorio contiene las letras y notebooks de las tareas propuestas para la edición 2026 del curso de Introducción a la Ciencia de Datos.

## Entregables

### Tarea 1
- **Informe:** `Tarea_1/informe/main.pdf`
- **Notebook principal:** `Tarea_1/2026_tarea1.ipynb`
- **Script auxiliar:** `Tarea_1/build_chart_2a.py` — genera el gráfico `chart_2a`.
- **Slides:** `Tarea_1/slides/slides_grupo23.pdf`

### Tarea 2
- **Informe:** `Tarea_2/informe/main.pdf` (fuente en `main.tex`, figuras en `informe/imagenes/`).
- **Notebook principal:** `Tarea_2/2026_tarea2.ipynb` — incluye toda la implementación: limpieza y partición estratificada de los datos, representaciones Bag of Words y TF-IDF, visualización PCA, y la Parte 2 (Multinomial Naive Bayes, validación cruzada con GridSearchCV, modelo alternativo de regresión logística, cambio de medio de prensa y análisis de técnicas alternativas de extracción de features).

En general, el procedimiento para correr los notebooks es simplemente instalar los requerimientos y levantar el servidor de jupyter:
```sh
cd ruta/a/introCD/Tarea_1/
pip install -r requirements.txt
jupyter notebook
```

## [Opcional] Trabajando con entornos virtuales (virtualenvs)
Se recomienda utilizar entornos virtuales para instalar los requerimientos de cada proyecto en un entorno aislado, de forma que las dependencias no colisionen. Hay muchas formas de hacerlo, pero utilizando sólamente python estándar se puede ejecutar:
```sh
cd ruta/a/introCD/

# Solo una vez, crea el entorno virtual en la carpeta ruta/a/introCD/.venv/
python -m venv introcd .venv/ 

# Cada vez que ingrese a la consola, puede activar este entorno corriendo (desde la carpeta introCD/):
source .venv/bin/activate
```
La idea es siempre ejecutar los comandos `pip install ...`, o `python...` o `jupyter...` dentro de este entorno,
de forma que no se instale nada a nivel del sistema.

Si utiliza un IDE como [Visual Studio Code](https://code.visualstudio.com/) y corre el código en forma interactiva, este entorno virtual debería estar visible dentro de la lista de *kernels* cuando intente ejecutar python.

## [Opcional] Trabajando con múltiples versiones de python

Si está trabajando en muchos proyectos distintos que usan python, es muy probable que algunos requieran utilizar versiones particulares. En esos casos se recomienda utilizar [pyenv](https://github.com/pyenv/pyenv) para poder utilizar versiones de python distintas en la misma máquina.

Una vez instalado `pyenv`, puede instalar y cambiarse a cualquier version de python:
```sh
# Listar todas las versiones disponibles:
pyenv install --list

# Supongamos que quiero instalar la `3.10.9`:
pyenv install 3.10.9

# Ahora debería ver esta versión disponible en su computadora:
pyenv versions

# Para usar una versión determinada, tiene 3 opciones:

pyenv shell 3.10.9  # crea una variable de entorno `$PYENV_VERSION` que activa esta versión sólo en la terminal actual

pyenv local 3.10.9  # crea un archivo `.python-version` que hace activar esta versión cada vez que entra en la carpeta

pyenv global 3.10.9  # hace que todo el sistema utilice esta versión de python (puede generar problemas)

# Ahora debería ver un asterisco junto a la versión activa:
pyenv versions
```

**NOTA:** cuando crea un entorno virtual, **la versión de python activa en ese momento se copia dentro de la carpeta `.venv/`**, y cada vez que active dicho entorno, esa será la versión que se usará, sin importar lo que haga con `pyenv`. Si desea crear un entorno virtual con otra versión de python, puede borrar el entorno existente (`rm -r .venv/`), activar con pyenv la versión que desea usar, y luego volver a crear el entorno virtual.
