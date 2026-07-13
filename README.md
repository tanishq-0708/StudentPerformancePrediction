# 🎓 AI-Based Student Performance Prediction System

### Hybrid Stacking Ensemble Learning · Streamlit · Plotly · XGBoost

A modern, production-ready AI dashboard that predicts a student's final
exam score using a **Hybrid Stacking Regressor** (Linear Regression +
XGBoost → Linear Regression Meta-Learner), presented through an elegant
black & gold glassmorphism interface.

---

## 📖 Project Description

This system predicts a student's **Final Exam Score** based on three key
academic indicators:

| Feature | Range |
|---|---|
| Attendance (%) | 0 – 100 |
| Hours Studied (per Day) | 0 – 24 |
| Previous/Internal Score | 0 – 100 |

The trained model is loaded from pre-generated files (`student_model.pkl`
and `scaler.pkl`) and is used exclusively for prediction. The Streamlit
application performs **inference only**, without retraining the model.

The application provides:

- ✅ Predicted Final Exam Score
- ✅ Performance Category (Outstanding → Needs Improvement)
- ✅ Confidence-Based Performance Message
- ✅ Personalized Improvement Suggestions
- ✅ Four Interactive Plotly Visualizations (Bar, Gauge, Radar & Pie)

---

## 🧬 Machine Learning Pipeline

```text
Student Inputs
      ↓
Preprocessing (StandardScaler)
      ↓
Linear Regression   ─┐
                      ├──►  Linear Regression (Meta Learner)
XGBoost Regressor   ─┘
      ↓
Predicted Exam Score
```

**Base Learners:** Linear Regression, XGBoost Regressor

**Meta Learner:** Linear Regression

**Ensemble Technique:** Stacking (`sklearn.ensemble.StackingRegressor`)

The trained model achieved approximately **R² = 0.92** and **MAE ≈ 3.6**
on the testing dataset.

---

## 🗂️ Project Structure

```text
student-performance/
│
├── app.py                # Streamlit application (Prediction Dashboard)
├── train_model.py        # Offline model training script
├── student_model.pkl     # Trained Hybrid Stacking Model
├── scaler.pkl            # Saved StandardScaler
├── requirements.txt      # Required Python packages
├── assets/
│   └── images/           # Project screenshots and images
└── README.md             # Project documentation
```

---

## ⚙️ Installation

### 1. Clone or Extract the Project

```bash
cd student-performance
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The application will open automatically at:

```
http://localhost:8501
```

### (Optional) Train the Model Again

```bash
python train_model.py
```

This command regenerates:

- `student_model.pkl`
- `scaler.pkl`

The Streamlit application simply loads these saved files during prediction and does not retrain the model.

---

## 🖼️ Screenshots

> _Insert screenshots of the running application below._

- `assets/images/hero.png` — Home Dashboard
- `assets/images/result.png` — Prediction Results
- `assets/images/charts.png` — Interactive Analytics
- `assets/images/about.png` — About Project Section

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Frontend Dashboard | Streamlit (Black & Gold Glassmorphism UI) |
| Data Visualization | Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Model Storage | Pickle |
| Programming Language | Python 3.10+ |

---

## ✨ Key Features

- Modern dark-themed glassmorphism user interface
- Real-time prediction using a trained Hybrid Stacking Ensemble
- Automatic performance classification into six categories
- Personalized academic recommendations
- Four interactive Plotly visualizations
- Sidebar navigation (Prediction, About Project & Developer)
- Structured backend prediction output
- Fast model loading through caching
- Modular, scalable, and well-documented Python code

---

## 🔭 Future Scope

- Train the model using a larger real-world educational dataset
- Add secure login and individual student history tracking
- Support batch predictions using CSV uploads
- Integrate SHAP-based explainable AI visualizations
- Deploy the model as a FastAPI REST service
- Add multilingual recommendation support
- Track student performance trends over time

---

## 📜 License

This project is developed for educational, research, and demonstration purposes only.
