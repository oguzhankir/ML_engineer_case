
---

## Folder Structure

- **notebooks**: Contains Jupyter Notebooks used for data analysis and model development.
  - **encoders**: Stores encoders used for preprocessing.
  - **models**: Stores trained models.
  - **config.json**: Configuration file for training data.
  - **model_development.ipynb**: Notebook for model development.
  - **data_analysis.ipynb**: Notebook for data analysis.
  
- **src**: Source code for the project.
  - **ml_engineer_case**: Main module.
    - **api**: Contains API routers and request DTOs.
      - **router**: Router implementations.
        - **\_\_init\_\_.py**: Initialization file.
        - **ml_model.py**: API endpoint for the machine learning model.
      - **\_\_init\_\_.py**: Initialization file.
      - **requests_dto.py**: Data transfer objects for API requests.
    - **artifacts**: Stores artifacts generated during model development.
      - **data**: Data used for training and testing.
      - **encoders**: Encoders used for preprocessing.
      - **models**: Trained models.
      - **config.json**: Configuration file for model training.
    - **operators**: Contains modules for various operations.
      - **\_\_init\_\_.py**: Initialization file.
      - **encode.py**: Encoding operations.
      - **feature_selection.py**: Feature selection operations.
      - **hyperparameters_search.py**: Hyperparameter search operations.
      - **inference.py**: Inference operations.
      - **model_manager.py**: Model management operations.
      - **preprocess.py**: Preprocessing operations.
    - **services**: Contains service implementations.
      - **\_\_init\_\_.py**: Initialization file.
      - **ml_model.py**: Machine learning model service.
    - **utils**: Contains utility modules.
      - **\_\_init\_\_.py**: Initialization file.
      - **exceptions.py**: Custom exceptions.
      - **logger.py**: Logging utility.
      - **utils.py**: General utility functions.
    - **\_\_init\_\_.py**: Initialization file.

- **tests**: Contains unit tests.
  - **test_inference.py**: Unit tests for inference operations.

- **.gitignore**: Git ignore file.
- **docker-compose.yml**: Docker Compose configuration.
- **Dockerfile**: Dockerfile for building the application.
- **main.py**: Main entry point for the application.
- **readme.md**: This file.
- **requirements.txt**: Python dependencies.

## Usage

To run the project, use the following command:

```bash
sudo docker compose down --remove-orphans && sudo docker compose up --build
```

---

