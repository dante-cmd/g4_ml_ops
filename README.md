# MLOps para la predicción de cantidad alumnos

Este proyecto tiene como objetivo desarrollar un sistema de MLOps para predecir la cantidad de alumnos a un nivel de granularidad específicado. Un nivel alto en la precisión de la predicción generará reducción en la cantidad de clases canceladas, lo que se traducirá en un aumento en la satisfacción de los estudiantes, carga operativa, reasignación de profesores, reasignación de agentes, etc.

Reducción en el tiempo de depliegue de modelos ML.


Componentes:
- Datos: procesos e infraestructura (recolección, limpieza, transformación, etc.). Qué pasa si se cae la fuente de datos? Dar alternativas.
- Algoritmos de Machine Learning o modelos (regresión, clasificación, etc.)
- Infraestructura para desplegar y mantener el modelo en producción (eficiiencia, escalabilidad, monitorización, etc.) Que sea suficiente robusta para soportar fallos.


Nivel de madurez de MLOPS
EL nivel de madures de que adopta depende del proyecto, la organización y el equipo. Algunos niveles comunes de madurez incluyen:
- Nivel 0: Sin MLOps (desarrollo manual y despliegue de modelos)
- Nivel 1: MLOps básico (automatización parcial de procesos)
- Nivel 2: MLOps intermedio (automatización completa de procesos, integración continua)
- Nivel 3: MLOps avanzado (monitorización y optimización continua, gobernanza de modelos)
- Nivel 4: MLOps empresarial (integración completa con sistemas empresariales, escalabilidad a nivel organizacional)

Los pilares fundamentales de un sistema MLOps incluyen:

- Data engineering
- *Model training* (extrategias de entrenamiento, validación, etc.), frecuencia de reentrenamiento (1 vez al mes, 1 vez a la semana, etc.)
- *Model deployment* (estrategias de despliegue, CI/CD, etc., en batch o en tiempo real depente del negocio)
- *Operations & monitoring* (monitorización del rendimiento del modelo, alertas, etc.). Es necesario definir métricas de rendimiento y establecer umbrales para alertas. la frecuencia de monitorización dependerá del caso de uso (por ejemplo, en tiempo real, diaria, semanal). Qué hacer cuando upgrade los sistemas? como mantener la continuidad del servicio anet ellos? mantenimientos del sistema (codigo e infraestructura).
- *Ethics & governance*: Explicar cómo se abordan los aspectos éticos y de gobernanza en el proyecto, como la privacidad de los datos, el sesgo del modelo y el cumplimiento normativo.


Granulariedad (output del modelo):

```python
(PERIODO: int
 SEDE: string 
 CURSO_ACTUAL: string
 HORARIO_ACTUAL: string
 PREDICT_ALUMNOS: int)
```


### Cookiecutter Data Science

Cookiecutter Data Science (CCDS) is a tool for setting up a data science project template that incorporates best practices. To learn more about CCDS's philosophy, visit the project homepage.

https://github.com/drivendataorg/cookiecutter-data-science


```text
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         {{ cookiecutter.module_name }} and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── {{ cookiecutter.module_name }}   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes {{ cookiecutter.module_name }} a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations   

```

# Versionamiento del proyecto

File: `version.yml`

- raw_version: default "v1"
- platinum_version: default "v1"
- features_version: default "v1"
- target_version: default "v1"
- experiment_name: default "test"

## 1. Raw

Se refiere a la version de *raw data* y se actualiza de version en caso de que se realicen cambios en `base_data/rawdata/`. Por ejemplo, cambios en la codificación de las cursos, nuevos horarios, etc.

## 2. Platinum 

Se refiere a la version de *platinum data* y se actualiza de version en caso de que se realicen cambios en el script `src/data/*.py`. Por ejemplo, crea un nuevo filtro, adiciona columnas, etc.

**Nota:** Actualizar la `raw_version` implica actualizar la `platinum_version`. raw_version $\Rightarrow$ platinum_version

## 3. Features y Target

Se refiere a la version de *features data* y se actualiza de version en caso de que se realicen cambios en el script `src/features/*.py`. Por ejemplo, se adiciona una nueva feature o target, se actualiza la lógica de cursos potenciales a crear (`get_idx`), etc.

**Nota:** Actualizar la `platinum_version` implica actualizar la `features_version` y la `target_version`. platinum_version $\Rightarrow$ features_version, target_version

## 4. Experiment name


