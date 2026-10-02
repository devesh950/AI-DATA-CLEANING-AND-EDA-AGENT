# Metriva — Executive Data Cleaning & EDA Platform

> **Turn messy business spreadsheets into executive intelligence in seconds.**  
> Zero SQL, zero complex BI setups, 100% cloud-ready (Vercel, Render, Railway, AWS).

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/backend-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Plotly](https://img.shields.io/badge/visuals-Plotly.js-informational.svg)](https://plotly.com/javascript/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Overview

**Metriva** is a lightweight, standalone web application that replaces bloated BI tools and sleep-prone notebook interfaces. Business owners, finance teams, and operators can drag-and-drop CSV or Excel exports to immediately uncover revenue drivers, detect integrity anomalies, calculate margin profiles, and generate board-ready reports.

---

## 🚀 Key Features

- **⚡ Heuristic Schema Auto-Detection**: Automatically detects revenue columns, date/time periods, client/category dimensions, and expense fields without manual configuration.
- **🛡️ Automated Data Quality & Anomaly Audit**: Analyzes dataset completeness, checks missing rates per column, and flags dirty data risks.
- **📈 Revenue & Margin Intelligence**: Calculates gross margin automatically and warns against customer concentration risks (e.g. if the top 3 clients generate >60% of revenue).
- **💡 Plain-English Strategic Insights**: Generates plain-language executive takeaways categorized by growth drivers, profitability alerts, and quality audits.
- **📊 Interactive Visualizations**: Pre-configured Plotly charts including category contribution bars, timeline trends, category share donuts, and cost-vs-revenue correlation.
- **🖨️ 1-Click Executive PDF Export**: Built-in print stylesheet formats your analysis into a clean executive report for leadership meetings.
- **🔒 In-Memory Private Processing**: Ephemeral memory execution ensures confidential financial spreadsheets are never retained on persistent disks.

---

## 🏗️ Architecture & Tech Stack

- **Backend**: Python 3.9+, [Flask](https://flask.palletsprojects.com/)
- **Data Engine**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [OpenPyXL](https://openpyxl.readthedocs.io/)
- **Frontend**: Clean Vanilla HTML5, CSS3 design tokens (`Plus Jakarta Sans`), [Plotly.js](https://plotly.com/javascript/)
- **Production Server**: [Gunicorn](https://gunicorn.org/)

---

## 📁 Repository Structure

```text
├── app.py              # Flask server, data cleaning engine, and REST API
├── templates/
│   ├── index.html      # High-converting SaaS landing page
│   └── app.html        # Dedicated interactive analysis workspace
├── demo_data/          # Sample CSV datasets for testing
├── Procfile            # Cloud WSGI deployment command (Gunicorn)
├── vercel.json         # Vercel serverless configuration
├── requirements.txt    # Lean production dependencies
├── .gitignore          # Git exclusion rules
└── README.md           # Documentation
```

---

## 💻 Local Setup & Development

### 1. Clone the repository
```bash
git clone https://github.com/devesh950/AI-DATA-CLEANING-AND-EDA-AGENT.git
cd AI-DATA-CLEANING-AND-EDA-AGENT
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.
- **Landing Page**: `http://localhost:5000/`
- **Interactive Workspace**: `http://localhost:5000/app`

---

## ☁️ Cloud Deployment

### Deploy on Vercel
This repository includes a `vercel.json` file. Connect your GitHub repository to Vercel and it will automatically deploy the serverless Python functions.

### Deploy on Render / Railway
1. Create a new **Web Service** and connect this repository.
2. The included `Procfile` will automatically launch Gunicorn:
   ```bash
   gunicorn app:app
   ```
3. Set your environment variables if needed (`PORT=5000`).

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
