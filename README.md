# CNN-BiLSTM Model for Network Intrusion Detection (NSL-KDD)
Commande ECHO activ‚e.
End-to-end NIDS pipeline: preprocessing, baselines, CNN-BiLSTM, and a live web UI.
Commande ECHO activ‚e.
## Quickstart
- Create venv, install requirements, run app or training scripts.
Commande ECHO activ‚e.
## Structure
src/ (code), notebooks/, models/ (LFS), artifacts/ (LFS), app.py, requirements.txt
Commande ECHO activ‚e.
## Reproducibility
SMOTE train-only, One-Hot+StandardScaler serialized, fixed seeds, saved model (.h5).

### Run (recommended Python 3.10)
"C:\Users\MSI\AppData\Local\Programs\Python\Python310\python.exe" -m venv .venv310
.\.venv310\Scripts\activate
pip install -r requirements.txt
python app.py  # http://127.0.0.1:5000
