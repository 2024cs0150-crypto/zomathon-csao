# 🍽️ Zomathon – CSAO Recommendation System

## 📌 Problem Overview

Customers often miss relevant add-on items while ordering food.  
This project builds a **real-time Cart Super Add-On (CSAO) recommendation system** that:

- Suggests context-aware add-ons based on cart composition
- Improves Average Order Value (AOV)
- Maintains high acceptance rates
- Operates within strict latency constraints (≤300ms)

---

## 🧠 Approach Summary

Our solution includes:

- Expanded candidate pool (entire restaurant menu as candidates)
- 40+ engineered contextual features
- GroupShuffleSplit to prevent session leakage
- LightGBM with cost-sensitive learning
- Confidence-based production gating strategy

---

## 📊 Key Results

| Metric | Baseline | Final Model |
|--------|----------|------------|
| Precision@5 (Raw) | ~0.03 | Improved |
| Precision@5 (Filtered, 53% coverage) | – | **0.159** |
| Hit Rate@5 | – | 99%+ |
| Simulated AOV Lift | – | **+13.48%** |

---

## 🏗️ System Design Highlights

- Real-time inference under 250ms P99 latency
- Offline + online feature separation
- Cold-start fallback strategies
- Structured A/B testing framework
- Production-ready model artifact saving

---

## 📁 Project Structure


zomathon-csao/
│
├── csao_data/
│   ├── cart_events.csv
│   ├── menu_items.csv
│   ├── model_dataset.csv
│   ├── restaurants.csv
│   ├── sessions.csv
│   └── users.csv
│
├── csao_dataset_generator.py
├── csao_model_baseline.ipynb
└── README.md


---

🚀 How to Run

1️⃣ Generate Dataset

python csao_dataset_generator.py

This creates the synthetic dataset inside csao_data/.

2️⃣ Train & Evaluate Model

Open:

csao_model_baseline.ipynb

Run all cells to:

Train baseline model

Train final LightGBM model

Compute ranking metrics

Simulate business impact

⚠️ Note

This project uses synthetic data designed to mimic real-world food delivery dynamics.
All modeling decisions are structured for scalability and production-readiness.

👥 Team

Zomathon CSAO Team
-Fayaz Ahmed T
-Dhinesshvar V
-Apoorva GVL
Hackathon Submission 2026
