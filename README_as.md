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


