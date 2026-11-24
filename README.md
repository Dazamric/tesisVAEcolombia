# Medición del valor agregado en la educación superior colombiana

Código de la tesis de Maestría en Ciencia de Datos, Pontificia Universidad Javeriana Cali.

Los datos de Saber 11 y Saber Pro son del ICFES y no se incluyen en este repositorio.

## Orden de ejecución

1. `tabla_analitica_inicial.py`: construye la tabla analítica a partir de los cruces, Saber 11 y Saber Pro.
2. `HLM_Linea_Base.py`: modelo jerárquico lineal y VAE-HLM.
3. `ML_Predictivo.py`: modelos predictivos, SHAP y VAE-ML.
4. `OE5_Replica_ICFES.py`: réplica de la medición del ICFES por institución y núcleo básico de conocimiento.
5. `OE5_Nivel_IES.py`: comparación a nivel de IES entre la réplica y el VAE-ML.
6. `Capa3_DML_CausalForest.py`: DML y Causal Forests para el efecto del sector institucional.

Las rutas de los datos están al inicio de cada script y deben ajustarse.

## Dependencias

`pip install -r requirements.txt`
