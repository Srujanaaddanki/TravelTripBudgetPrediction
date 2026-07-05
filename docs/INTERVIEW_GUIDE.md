# TripAI — Technical Interview & Placement Guide

This guide compiles essential project details, architectural walkthroughs, and answers to common technical and HR questions to prepare you for software engineering, AI, and data science interviews.

---

## 🏗️ 1. Project Architecture & Flow

### System Workflow
```
User (Browser) 
      ↓ (Submit Form)
Streamlit UI (View) 
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

## 🤖 2. Machine Learning Pipeline (Random Forest)

### Q: Why did you choose Random Forest Regressor over Linear Regression?
**A:** Linear Regression assumes linear relationships between features (like Trip Duration and Cost) and target budgets. However, travel costs depend on complex interactions between features (e.g. Luxury stay in Peak Season is exponentially more expensive than Budget stay in Off-season). 
Random Forest Regressor is an ensemble model composed of 200 Decision Trees. It handles non-linear relationships, multi-collinearity, and categorical encoders much better, yielding a high validation $R^2$ score of **~95%**.

### Q: How did you handle data preprocessing and encoding?
**A:** Categorical columns (`Place`, `Month`, `Season`, `Trip_Type`, `Hotel_Quality`) are mapped to numerical integer classes using **Label Encoders** (`encoders.pkl`). Numerical columns like `Days` (Duration) are converted to float. Costs are parsed using a regex parser (`_clean_cost`) to remove currency symbols (₹, commas, and shorthand like "k" representing thousands).

---

## 💾 3. SQLite Persistence & Caching Layer

### Q: Explain your 3-tier caching strategy.
**A:** API calls to Google Maps have networking latency and cost overhead. To optimize, we query `distance_cache` first. If missed, we query an offline static database mapping major Indian cities. If that also misses, we query the live Google Maps Distance Matrix API and cache the result with a TTL (Time-To-Live) of 30 days.

---

## 📈 4. Dataset Intelligence & Aggregations

### Q: How does the platform compute data-driven insights without using LLM API keys?
**A:** We use **Pandas** at runtime to filter the survey dataset (`traveltripdata.csv`). We extract similar travellers sharing the same Trip Type, Stay duration ($\pm 2$ days), and predicted cost range ($\pm 40\%$) to calculate average spending, favorite travel mode, and preferred stay grade. We also average satisfaction, hotel stay, and transport ratings to generate a unified **Destination Score** and progress indicator.

---

## 🎯 5. Commonly Asked Interview Questions

### Q: What are the limitations of the current system?
1. **Cold Start Problem**: If the user inputs a destination not covered in the survey dataset, the model cannot predict a budget. We handle this gracefully by falling back to geographical route mapping and weather insights only.
2. **Offline Weather Fallback**: If the Open-Meteo API fails due to no internet, the system falls back to season-based estimated values to prevent crashes.

### Q: How would you scale this platform in production?
**A:**
1. **Model Deployment**: Wrap the ML model in a microservice (FastAPI) containerized via Docker and deployed behind an Nginx load balancer.
2. **Caching**: Replace the local SQLite cache with Redis for distributed, high-throughput caching.
3. **Database**: Migrate from SQLite to PostgreSQL to handle concurrent transactional search tracking logs at scale.
