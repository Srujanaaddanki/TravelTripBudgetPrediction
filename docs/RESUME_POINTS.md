# Resume / Placement Bullet Points for TripAI

These bullet points are optimized for applicant tracking systems (ATS) and tech recruiters. They highlight metrics, data size, caching efficiency, and system architecture.

---

## 🚀 Option 1: High-Impact & Metric-Oriented (Recommended)
*Best for general Software Engineering, Full-Stack, or general Data roles.*

* **Engineered** an AI-powered travel intelligence platform utilizing a **Random Forest Regressor** (200 decision trees) to estimate trip budgets across India, achieving a **~95% $R^2$ validation score**.
* **Developed** a high-performance **3-tier caching architecture** using **SQLite (WAL mode)** to store Google Maps API routes, significantly reducing external API requests and latency.
* **Designed** an interactive **Streamlit** dashboard integrating **Folium** for geospatial route visualization, **Open-Meteo API** for live weather predictions, and **Plotly** for budget distribution analytics.
* **Processed** and cleaned survey data by designing regex-based extraction pipelines, and implemented data augmentation (+/- 5% random noise injection) to expand training data to **45,000+ records** for improved model stability.

---

## 💻 Option 2: Software Engineering & System Architecture Focus
*Best if you are targeting pure Backend, Full-Stack, or Software Engineering roles.*

* **Built** a decoupled MVC (Model-View-Controller) travel planner application, separating the presentation layer (Streamlit views) from backend weather/budget orchestrator engines to ensure modularity.
* **Optimized** external API usage by building a custom caching engine inside SQLite, fallback routing mechanisms, and parallel weather fetching routines using the keyless Open-Meteo API.
* **Implemented** interactive geospatial mapping using Folium with FontAwesome custom markers, plotting route waypoints and transit-mode indicators dynamically.
* **Trained** and deployed a Random Forest Regression model with custom label encoders to predict trip costs for unseen destinations based on historical stay tiers and travel seasons.

---

## 📊 Option 3: Data Science & Machine Learning Focus
*Best if you are targeting Data Analyst, Data Scientist, or ML Engineer positions.*

* **Trained and evaluated** multiple regression algorithms (Linear Regression, Decision Trees, Random Forest) to model destination spending patterns, selecting Random Forest for its **95% accuracy**.
* **Implemented** data preprocessing pipelines: standardized column headers, resolved data anomalies via regex-based currency parsing, eliminated extreme budget outliers, and engineered features like seasonality, stay quality tiers, and duration.
* **Engineered** data augmentation pipelines generating synthetic data variations, expanding the training dataset size by **50x** to resolve class imbalance and boost model generalization.
* **Created** exploratory data analysis (EDA) charts (Seaborn/Plotly) highlighting correlations between stay quality, month-wise seasonal variations, trip durations, and travel budgets.
