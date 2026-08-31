# 🌍 TraWell – AI-Powered Travel Budget Predictor

> **Travel far enough, you meet yourself.**

TraWell is an AI-powered travel planning application that helps users estimate their travel budget and plan trips using Machine Learning, travel APIs, weather information, route data, and destination intelligence.

The application allows users to enter both popular and lesser-known destinations and generates an approximate travel budget along with useful travel information.

---

## 🚀 Live Demo

🌐 **TraWell:**  
https://trawell-ai.streamlit.app/

---

## 📌 Project Overview

Planning a trip can be difficult when travelers need to consider transportation, accommodation, trip duration, weather, destination characteristics, and other expenses.

TraWell combines **Machine Learning + APIs + AI-based destination intelligence** to provide a smarter travel planning experience.

The user provides:

- Departure city
- Destination
- Month of travel
- Trip duration
- Trip type
- Travel mode
- Hotel quality

TraWell then processes the information and provides an estimated travel budget and destination-specific travel information.

---

## ✨ Key Features

### 🤖 Machine Learning Budget Prediction

TraWell uses a Machine Learning model to predict an estimated travel budget based on the user's trip details.

The prediction considers factors such as:

- Destination
- Travel month
- Trip duration
- Trip type
- Travel mode
- Hotel quality

---

### 🌍 Unknown Destination Support

TraWell is not limited to destinations available in a predefined list.

Users can enter destinations manually, including lesser-known destinations.

The application resolves destinations using its destination-resolution pipeline and can use geographical/API information when historical dataset information is unavailable.

This allows users to search for destinations such as:

- Annavaram
- Goa
- Doraha Fort
- Hampi
- Other valid destinations

---

### 🗺️ Interactive Route Map

TraWell provides an interactive route map between the departure city and the actual destination entered by the user.

The map uses the resolved destination information to calculate and display the route.

---

### 🌦️ Live Weather Information

TraWell provides weather information for the selected destination using weather APIs.

The weather information helps users understand the current conditions at their destination and supports travel planning.

---

### 🎒 Destination-Specific Packing Checklist

TraWell generates a packing checklist based on the user's destination and trip information.

The checklist can consider factors such as:

- Destination
- Weather
- Travel month
- Travel mode
- Trip type
- Destination characteristics

---

### ✅ Pre-Travel Checklist

TraWell also provides a pre-travel checklist to help users prepare before starting their journey.

Examples include:

- Government ID
- Travel tickets
- Hotel booking confirmation
- Important travel essentials

---

### 🧠 AI Travel Intelligence

TraWell uses AI-based destination intelligence to provide additional travel-related recommendations and information.

The application combines AI with API information to make the travel planning experience more useful and personalized.

---

## 💰 Budget Estimation

TraWell presents the estimated travel budget as a range rather than a single fixed value.

Example:

**Estimated Budget: ₹4,200 – ₹7,000**

The application can also provide an approximate daily budget based on the trip duration.

> Budget values are estimates and may vary depending on actual travel prices, availability, and destination conditions.

---

## 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │      User Input     │
                    │─────────────────────│
                    │ From City           │
                    │ Destination         │
                    │ Month               │
                    │ Duration            │
                    │ Trip Type           │
                    │ Travel Mode         │
                    │ Hotel Quality       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Destination         │
                    │ Resolution          │
                    │                     │
                    │ Cache / Alias       │
                    │ Exact Match         │
                    │ Geo APIs            │
                    │ AI Resolution       │
                    │ Fuzzy Matching      │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │ ML Model   │  │ Travel APIs│  │ Weather API│
        │            │  │            │  │            │
        │ Prediction │  │ Route/Geo  │  │ Weather    │
        └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   TraWell Report    │
                    │─────────────────────│
                    │ Budget Prediction   │
                    │ Weather             │
                    │ Route Map           │
                    │ Packing Checklist   │
                    │ Pre-Travel Checklist│
                    │ AI Recommendations  │
                    └─────────────────────┘

```

### Core Intelligence Components

* **Destination Resolution** — Identifies and validates destinations using cached, known, geocoded, AI, and fuzzy matching approaches.
* **ML Prediction** — Provides the base budget prediction from the trained Random Forest model.
* **Travel Intelligence** — Combines ML prediction, historical data, transportation, duration, and other travel signals.
* **Confidence Engine** — Evaluates the reliability and confidence level of the resulting estimate.
* **External Services** — Provide live geocoding, route mapping, weather forecasts, and AI intelligence.
* **SQLite Persistence** — Stores cache, unknown destination tracking, and user search history.

### Decoupled Folder Structure

```
TravelTripBudgetPrediction/
├── app.py                      # Application Entry Point & Service Bootstrap
├── train_model.py              # Random Forest Model Training Pipeline
├── requirements.txt            # Package Dependencies
├── runtime.txt                 # Deployment Runtime Version
├── LICENSE                     # Project License
├── .gitignore                  # Git Ignore Rules
├── config/                     # System Settings & Constants
├── data/                       # Historical Traveler Survey Dataset (traveltripdata.csv)
├── models/                     # Pickled Random Forest Model & Label Encoders
├── docs/                       # API Setup & Documentation Guides
├── tests/                      # Automated Verification & Integration Tests
├── src/                        # Core Source Code
│   ├── components/             # UI Components (Navbar, Hero, Charts, Maps, Checklists)
│   ├── data/                   # Maps Service, Database Schema & Search Tracking
│   ├── intelligence/           # Dataset Intelligence & Destination Knowledge Bases
│   ├── pages/                  # Streamlit Dashboard Views (Plan Trip, Analytics)
│   ├── services/               # Controllers (Weather, Budget, Gemini, Geo, Route, Confidence, Export)
│   └── styles/                 # Custom CSS Design Tokens & Responsive Stylesheet
```

---

## 🛠️ Technologies

* **Frontend & Presentation**: Streamlit, HTML5, Custom CSS
* **Geospatial & Visualization**: Folium, Streamlit-Folium, Plotly Express
* **Machine Learning & Preprocessing**: Scikit-Learn (Random Forest Regressor), Joblib, Pandas, NumPy
* **Database & Caching**: SQLite3 (WAL mode), SQLite Memory databases
* **APIs & Networks**: Open-Meteo API, Google Maps Distance Matrix API

---
## 🧠 Destination Resolution Flow
One of TraWell's important capabilities is handling destinations that are not available in the historical ML dataset.

                    User enters destination
                              │
                              ▼
                       Cache Lookup
                              │
                     ┌────────┴────────┐
                     │                 │
                   Found             Not Found
                     │                 │
                     ▼                 ▼
                Use cached       Known Data Check
                coordinates            │
                                 ┌─────┴─────┐
                                 │           │
                               Found       Not Found
                                 │           │
                                 ▼           ▼
                              Resolve      Geo API
                                           Lookup
                                              │
                                              ▼
                                      AI / Fuzzy Fallback
                                              │
                                              ▼
                                      Resolved Destination
                                      
The resolved destination is then used for destination-dependent services such as route information, weather, recommendations, and checklists.

---
## ⚡ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Srujanaaddanki/TravelTripBudgetPrediction.git
cd TravelTripBudgetPrediction
```

### 2. Install dependencies
Ensure you are using Python 3.9+ and run:
```bash
pip install -r requirements.txt
```

### 3. Verify the installation
Run the integration test suite to verify the database and caching system:
```bash
python tests/test_search_tracking.py
```

### 4. Launch the application locally
```bash
streamlit run app.py
```

---

## 🖼️ Screenshots

*(Screenshots of planning forms, interactive map routes, confidence metrics, and SaaS analytics dashboard here during portfolio hosting)*

---
## 🔗 Access Links

| Resource | Link |
|----------|------|
| 📊 Live Demo | https://trawell-ai.streamlit.app/ |
| 💻 GitHub Repository | https://github.com/Srujanaaddanki/TravelTripBudgetPrediction |
| 📁 Dataset | https://docs.google.com/spreadsheets/d/1j_kxCGl5NBICDFKOrWRxWOOJ3uc6MGg5/edit?gid=159424351#gid=159424351 |
| 💼 LinkedIn Post | https://www.linkedin.com/feed/update/urn:li:activity:7493453225753419776/ |

---

## 👤 Author

**Srujana Addanki**

Aspiring Data Analyst | Python | Machine Learning | Data Visualization

🔗 LinkedIn, GitHub, and project links are included for reference.

---

⭐ If you find this project useful, feel free to star the repository!

**TraWell — Travel far enough, you meet yourself.** 🌍
