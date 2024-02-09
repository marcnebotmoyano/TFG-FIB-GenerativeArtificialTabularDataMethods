# Métodos para la generación de datos artificiales tabulares

## Descripción
Este proyecto es una librería de Python diseñada para la generación de datos artificiales tabulares. Implementa varios métodos generativos, incluyendo VAE (Autoencoders Variacionales), GAN (Redes Generativas Antagónicas), Bit Diffusion y técnicas de Data Augmentation, para crear conjuntos de datos sintéticos que reflejen las propiedades estadísticas de conjuntos de datos reales.

## Motivación
La necesidad de datos sintéticos es cada vez más crítica en el campo de la ciencia de datos e inteligencia artificial, especialmente en situaciones donde los datos reales son escasos, sensibles o privados. Esta librería se creó para proporcionar una herramienta accesible y eficiente para generar datos tabulares que pueden ser utilizados para la formación de modelos, pruebas de concepto o la evaluación de algoritmos en un entorno controlado.

## Metodología
La librería incluye implementaciones de:
- **VAE**: Autoencoders Variacionales que aprenden a generar nuevos datos a partir de la distribución aprendida de los datos de entrada.
- **GAN**: Un modelo antagónico que mejora la calidad de los datos generados a través de un juego de suma cero entre un generador y un discriminador.
- **Bit Diffusion**: Una técnica que genera datos añadiendo ruido a los datos y aprendiendo a revertir este proceso.
- **Data Augmentation**: Métodos para aumentar la cantidad de datos mediante técnicas como el ruido aditivo y el enmascaramiento aleatorio.

## Instalación
Clona este repositorio y luego instala las dependencias necesarias con:

```bash
git clone https://github.com/marcnebotmoyano/TFG-FIB-GenerativeArtificialTabularDataMethods
cd TFG-FIB-GenerativeArtificialTabularDataMethods
pip install -r requirements.txt
```

## Cómo utilizarla
La librería se puede usar con el siguiente comando:

```bash
python demo.py --model_type [gan/vae/bitdiff/data_augmentation] --dataset_path ./path/to/your/dataset.csv
```

demo.py es solo un script ejemplificado del uso de la librería de PyTorch.

## Estructura de código
El repositorio está organizado de la siguiente manera:

models/: Contiene los modelos VAE, GAN y BitDiffusion.
trainers/: Scripts para entrenar los diferentes modelos.
data_loader/: Funciones para cargar y procesar los datos.
utils/: Funciones de utilidad como métricas de evaluación y visualización de datos.
main.py: Punto de entrada principal para ejecutar los modelos.
config.py: Configuración del input

## Resultados
Los resultados de los modelos se pueden visualizar con los scripts de utils/, los cuales incluyen PCA y t-SNE para comparaciones de alta dimensionalidad y gráficos de densidad para comparaciones univariadas.

## Memoria
[![Memoria.pdf]()]https://github.com/marcnebotmoyano/TFG-FIB-GenerativeArtificialTabularDataMethods/blob/main/Generative%20Artificial%20Tabular%20Data%20Methods%20-%20Marc%20Nebot%20i%20Moyano.pdf
Las contribuciones son bienvenidas. Por favor, crea una issue o un pull request con tus sugerencias.

## Licencia
Este proyecto se distribuye bajo la licencia MIT.

## Créditos
Desarrollado por Marc Nebot i Moyano
