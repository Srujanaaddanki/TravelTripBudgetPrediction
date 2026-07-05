# TripAI — Technical Interview & Placement Guide

This guide compiles detailed architectural explanations, model details, and answers to common technical and HR questions to prepare you for software engineering, AI, and data science interviews.

---

## 🏗️ 1. Project Architecture & Flow

### System Workflow
```
User (Browser) 
      ↓ (Submit Form)
Streamlit UI (View Layer) 
      ↓ (Call Predictor & Orchestrator)
ML Prediction Layer (Random Forest model + encoders)
      ↓ (Compute Smart Budget & Cost Matrix)
Google Maps API / Caching Layer (3-tier Cache)
      ↓ (Fetch Current Weather Forecast)
Open-Meteo Weather API (Free current forecast)
      ↓ (Query Aggregate Satisfaction Ratings)
Dataset Intelligence Engine (historical survey CSV stats)
      ↓ (Fetch co-searches & popular destinations)
SQLite Database Layer (Search history log)
      ↓ (Assemble report components)
Travel Intelligence Engine (Orchestrator Controller)
      ↓ (Render Widgets)
SaaS Results Dashboard (View Presentation)
```

### MVC Concerns Separation
- **Model**: `final_model.pkl` (Random Forest model), `travel.db` (SQLite Database), `traveltripdata.csv` (Historical dataset)
- **View**: `ui_components.py` (landing/form views), `dashboard_components.py` (results cards), `styles.css` (dark SaaS theme rules)
- **Controller/Orchestration**: `travel_intelligence.py` (main entry orchestrator coordinating sub-services like `weather_service.py`, `budget_engine.py`, `confidence_engine.py`, `recommendation_engine.py`, and `report_exporter.py`).

---

## 🤖 2. Machine Learning Pipeline & Preprocessing

### Q: Why did you choose Random Forest Regressor over Linear Regression?
**A:** Linear Regression assumes linear relationships between features (like Trip Duration and Cost) and target budgets. However, travel costs depend on complex interactions between features (e.g. Luxury stay in Peak Season is exponentially more expensive than Budget stay in Off-season). 
Random Forest Regressor is an ensemble model composed of 200 Decision Trees. It handles non-linear relationships, multi-collinearity, and categorical encoders much better, yielding a high validation $R^2$ score of **~95%**.

### Q: How did you handle data preprocessing, encoding, and feature engineering?
**A:** 
1. **Preprocessing**: The raw budget strings contain currency symbols (₹), commas, and shorthand like "k" (e.g. "approx 15k"). We implemented a clean regex-based cost parsing engine (`_clean_cost`) to extract float values.
2. **Label Encoding**: Categorical variables (`Place`, `Month`, `Season`, `Trip_Type`, `Hotel_Quality`) are converted to numerical classes using `encoders.pkl`.
3. **Feature Engineering**: We map the travel month to its corresponding season (winter, summer, spring, rainy, autumn) dynamically using a month-season mapping dictionary to align user inputs with model columns.

---

## 🧠 3. Travel Intelligence & Service Modules

### Q: Explain the budget prediction workflow.
**A:**
1. **ML Model base cost**: Predicts the base budget from the destination, month, season, trip type, hotel quality, and days.
2. **Geographical distance adjustment**: `MapsService` computes the distance from the origin.
3. **Travel mode projection**: We multiply route distance by travel mode metrics (e.g., flight ticket flat rates vs. road fuel multipliers).
4. **Smart Budget scaling**: The orchestrator balances the base prediction with transit costs and stay duration weights to output a verified, recommended budget.

### Q: What is the Confidence Score and how is it calculated?
**A:** The **Confidence Score** is a multi-factor reliability index computed by `ConfidenceEngine`:
- **Model Accuracy weight (50%)**: Derived from validation $R^2$ score of model training.
- **Season/Month Alignment weight (25%)**: Matches weather predictions and crowds.
- **Dataset Sample Density weight (25%)**: Determined by the volume of similar historical records matching the requested destination.

### Q: How does the Recommendation Engine work?
**A:** The `RecommendationEngine` acts as a curated knowledge base mapping destinations to top local attractions, regional foods, hidden gems, and safety tips. It also filters historical profile preferences to suggest relevant activities (e.g. trekking for Adventure seekers, beach lounges for Relaxation).

---

## 💾 4. SQLite Caching & Database Persistence

### Q: How does your 3-tier caching strategy work?
**A:** Network latency and API quotas make direct Google Maps API requests expensive. To optimize, `MapsService` implements:
1. **SQLite Database Cache**: Search matching coordinates and distance matrices are queried from local SQLite first.
2. **Offline Data Fallback**: Maps matches to major Indian cities offline.
3. **Live API Query**: Queries the live Google Maps Matrix API and caches results for 30 days.

### Q: What does SQLite log?
**A:** We use SQLite (`travel.db`) to track user search logs, including: origin, destination, month, duration, travel mode, predicted cost, and timestamp. This powers the live dashboard analytics and enables trending place lookups.

---

## 🎯 5. Common HR & Recruiter Q&A

### Q: What was the biggest challenge you faced during this project?
**A:** Managing data sparsity for lesser-known destinations. If a user searched for a location not present in the ML model, the app could crash. I resolved this by designing a robust **graceful fallback mechanism** that bypasses the ML predictor and displays coordinates, weather forecasts, and route estimates instead.

### Q: How did you ensure code readability and maintainability?
**A:** We followed the **Single Responsibility Principle** (SRP). Every file is under 300 lines. We separated business logic from Streamlit's UI layer, added type hints to all method signatures, and fully documented functions with NumPy docstring templates.
