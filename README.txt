===============================================================
 PIPE-AND-FILTER ML PREPROCESSING PIPELINE
 Streamlit + Scikit-learn
===============================================================

This project demonstrates the Pipe-and-Filter architectural pattern through
a machine learning preprocessing pipeline.

The system allows users to upload a dataset and process it through a sequence
of independent preprocessing stages (filters), such as:

- Missing value imputation
- Categorical encoding
- Feature scaling
- Feature extraction (PCA)

Each stage operates independently and communicates only through the data pipe,
illustrating modularity, composability, and architectural transparency.

This README explains:
- Project structure
- Architectural design
- How the pipeline works
- Running locally
- Running with Docker

===============================================================


===============================================================
 1. TECHNOLOGIES
===============================================================

Language: Python 3.12
UI Framework: Streamlit
Data Processing: pandas
Machine Learning: scikit-learn
Architecture Pattern: Pipe-and-Filter
Containers (Optional): Docker + Docker Compose


===============================================================
 2. ARCHITECTURE OVERVIEW
===============================================================

This project follows the Pipe-and-Filter architectural pattern.

The architecture consists of three conceptual layers:

1. UI Layer — Streamlit
2. Pipeline Core — Pipe-and-Filter orchestration
3. Filter Modules — Independent preprocessing stages


--------------------
 UI LAYER (STREAMLIT)
--------------------

The Streamlit UI provides:

- CSV upload
- Pipeline configuration (enable/disable filters)
- Filter reordering
- Parameter configuration per filter
- Validation feedback (errors + warnings)
- Visualization of intermediate pipeline stages
- Download of final processed dataset

The UI does not perform preprocessing logic itself.
It delegates all processing to the pipeline core.


----------------------------
 PIPELINE CORE (PIPE)
----------------------------

The pipeline core is responsible for:

- Executing filters in sequence
- Passing the dataset between filters
- Tracking intermediate results
- Enforcing architectural validation rules
- Recording schema differences per stage

The pipeline operates on a DataPacket, which contains:
- The current dataset
- Metadata
- Execution history


----------------------------
 FILTERS (PROCESSING STAGES)
----------------------------

Each filter is an independent module implementing a single responsibility.

Implemented filters:

- Impute Missing Values
- One-Hot Encode Categoricals
- Scale Numeric Features
- PCA Feature Extraction

Filters:
- Do not depend on each other
- Communicate only through the pipeline
- Can be reordered or removed


===============================================================
 3. PIPELINE FLOW
===============================================================

Example pipeline execution:

Raw CSV
  ↓
[ Impute Missing Values ]
  ↓
[ One-Hot Encoding ]
  ↓
[ Scaling ]
  ↓
[ PCA Feature Extraction ]
  ↓
ML-Ready Feature Matrix

At each stage, the system records:
- Input/output shape
- Columns added or removed
- Columns modified in place
- Preview of the transformed dataset

This makes the Pipe-and-Filter behavior explicit and observable.


===============================================================
 4. VALIDATION LOGIC
===============================================================

The system performs two types of validation:

Structural Validation:
- Ensures valid filter ordering
- Prevents invalid pipeline configurations
- Example:
  - PCA must run after Scaling
  - Pipeline cannot be empty

Data-Aware Validation:
- Inspects the uploaded dataset
- Detects categorical vs numeric columns
- Enforces algorithm constraints
- Example:
  - PCA requires numeric features
  - n_components ≤ min(n_samples, n_features_at_PCA)

Invalid configurations are blocked before execution.


===============================================================
 5. SCHEMA-DIFF VISUALIZATION
===============================================================

For each filter stage, the UI displays a schema diff:

- Added columns
- Removed columns
- Modified columns
- Shape changes

This improves:
- Architectural observability
- Debugging
- Understanding of data transformations

Schema diff is computed in the pipeline core, not in the UI.


===============================================================
 6. PROJECT STRUCTURE
===============================================================

/
├── app/
│   └── ui_streamlit.py
├── pipeline/
│   ├── core.py
│   ├── registry.py
│   ├── validate.py
│   ├── validate_data.py
│   └── filters/
│       ├── impute.py
│       ├── encode.py
│       ├── scale.py
│       └── pca.py
├── data/
│   └── sample_customers.csv
├── tests/
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── .gitignore
└── README.txt


===============================================================
 7. RUNNING LOCALLY (WITHOUT DOCKER)
===============================================================

1) Create virtual environment:

python -m venv venv
venv\Scripts\activate   (Windows)

2) Install dependencies:

pip install -r requirements.txt

3) Run Streamlit app:

streamlit run app/ui_streamlit.py

Open in browser:

http://localhost:8501


===============================================================
 8. RUNNING WITH DOCKER (OPTIONAL)
===============================================================

This project includes:
- Dockerfile
- docker-compose.yaml

Build and run:

docker compose up --build

Access the app:

http://localhost:8501

Stop containers:

docker compose down

Docker ensures consistent execution across machines.


===============================================================
 9. SAMPLE DATASET
===============================================================

A sample dataset is provided in:

data/sample_customers.csv

The dataset includes:
- Numeric features
- Categorical features
- Missing values

It is designed to exercise all filters in the pipeline.


===============================================================
 10. NOTES
===============================================================

- This project is intended for academic/demo use
- Streamlit is used for rapid UI prototyping
- The pipeline is batch-oriented (not streaming)
- Architecture prioritizes clarity over performance
- The system can be extended with additional filters easily


===============================================================
 END OF FILE
===============================================================

