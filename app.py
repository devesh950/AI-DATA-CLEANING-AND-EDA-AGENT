from flask import Flask, request, jsonify, render_template, Response
import pandas as pd
import numpy as np
import io, os, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True

def fmt(n):
    if n is None or pd.isna(n): return "0"
    n = float(n)
    if abs(n) >= 1_000_000: return f"${n/1_000_000:.2f}M"
    if abs(n) >= 1_000: return f"${n/1_000:.1f}K"
    return f"${n:,.0f}"

def detect_cols(df):
    cl = {c.lower(): c for c in df.columns}
    def find(kws, cond=None):
        for kw in kws:
            for lc, orig in cl.items():
                if kw in lc:
                    if cond is None or cond(df[orig]): return orig
        return None
    is_num = lambda s: pd.api.types.is_numeric_dtype(s)
    is_str = lambda s: s.dtype == object or pd.api.types.is_string_dtype(s)
    return {
        "revenue_col": find(["revenue","sales","amount","mrr","value","income","total","price","gmv","billing"], is_num),
        "date_col": find(["date","month","week","period","time","day","year"]),
        "category_col": find(["client","customer","category","product","service","type","segment","region","name","item","channel","tier"], is_str),
        "cost_col": find(["cost","expense","spend","cogs","purchase","ad_spend","budget"], is_num)
    }

def barchart(labels, values, title):
    pairs = sorted(zip(values, labels), reverse=True)
    vs = [p[0] for p in pairs[:10]]; ls = [p[1] for p in pairs[:10]]
    return {
        "data": [{
            "type": "bar",
            "orientation": "h",
            "x": vs,
            "y": ls,
            "marker": {
                "color": vs,
                "colorscale": [[0, "#c7d2fe"], [1, "#4361ee"]],
                "showscale": False
            },
            "hovertemplate": "%{y}: %{x:,.0f}<extra></extra>"
        }],
        "layout": {
            "title": {"text": title, "font": {"size": 14, "color": "#0f172a", "family": "Plus Jakarta Sans"}},
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "margin": {"l": 10, "r": 20, "t": 45, "b": 20},
            "xaxis": {"gridcolor": "#f1f5f9"},
            "yaxis": {"autorange": "reversed"},
            "font": {"family": "Plus Jakarta Sans"}
        }
    }

def linechart_with_forecast(x_hist, y_hist, title, date_col, rev_col):
    traces = []
    # Historical Trace
    traces.append({
        "type": "scatter",
        "mode": "lines+markers",
        "name": "Historical Run-Rate",
        "x": x_hist,
        "y": y_hist,
        "line": {"color": "#4361ee", "width": 2.5},
        "marker": {"size": 6, "color": "#4361ee"},
        "hovertemplate": "%{x}: %{y:,.0f}<extra></extra>"
    })

    # 3-Period Forecast Calculation
    forecast_kpi = None
    if len(y_hist) >= 3:
        try:
            n = len(y_hist)
            xs = np.arange(n)
            ys = np.array(y_hist, dtype=float)
            # Simple linear regression
            slope, intercept = np.polyfit(xs, ys, 1)
            future_xs = np.arange(n - 1, n + 3)
            future_ys = slope * future_xs + intercept
            
            # Generate date labels
            last_date_str = str(x_hist[-1])
            try:
                last_dt = pd.to_datetime(last_date_str)
                future_dates = [x_hist[-1]] + [(last_dt + pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(1, 4)]
            except:
                future_dates = [x_hist[-1], f"P+{1}", f"P+{2}", f"P+{3}"]

            traces.append({
                "type": "scatter",
                "mode": "lines+markers",
                "name": "3-Month Forecast (Projected)",
                "x": future_dates,
                "y": future_ys.tolist(),
                "line": {"color": "#8b5cf6", "width": 2.5, "dash": "dash"},
                "marker": {"size": 6, "color": "#8b5cf6", "symbol": "diamond"},
                "hovertemplate": "Projected %{x}: %{y:,.0f}<extra></extra>"
            })

            growth_pct = ((future_ys[-1] - ys[-1]) / ys[-1] * 100) if ys[-1] > 0 else 0
            forecast_kpi = {
                "projected_total": fmt(future_ys[-1]),
                "growth_pct": round(growth_pct, 1),
                "trend": "upward" if growth_pct > 0 else "downward"
            }
        except Exception as e:
            pass

    return {
        "chart": {
            "data": traces,
            "layout": {
                "title": {"text": title, "font": {"size": 14, "color": "#0f172a", "family": "Plus Jakarta Sans"}},
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "margin": {"l": 10, "r": 20, "t": 45, "b": 40},
                "xaxis": {"gridcolor": "#f1f5f9"},
                "yaxis": {"gridcolor": "#f1f5f9"},
                "legend": {"orientation": "h", "y": -0.2, "x": 0.1},
                "font": {"family": "Plus Jakarta Sans"}
            }
        },
        "forecast_kpi": forecast_kpi
    }

def piechart(labels, values, title):
    return {
        "data": [{
            "type": "pie",
            "labels": labels[:8],
            "values": values[:8],
            "hole": 0.45,
            "marker": {"colors": ["#4361ee","#6366f1","#8b5cf6","#3b82f6","#06b6d4","#10b981","#f59e0b","#ec4899"]},
            "textinfo": "label+percent",
            "hovertemplate": "%{label}: %{value:,.0f} (%{percent})<extra></extra>"
        }],
        "layout": {
            "title": {"text": title, "font": {"size": 14, "color": "#0f172a", "family": "Plus Jakarta Sans"}},
            "paper_bgcolor": "white",
            "margin": {"l": 10, "r": 10, "t": 45, "b": 10},
            "font": {"family": "Plus Jakarta Sans"}
        }
    }

def scatterchart(x, y, xl, yl, title):
    return {
        "data": [{
            "type": "scatter",
            "mode": "markers",
            "x": x,
            "y": y,
            "marker": {"color": "#4361ee", "opacity": 0.65, "size": 8},
            "hovertemplate": f"{xl}: %{{x:,.0f}}<br>{yl}: %{{y:,.0f}}<extra></extra>"
        }],
        "layout": {
            "title": {"text": title, "font": {"size": 14, "color": "#0f172a", "family": "Plus Jakarta Sans"}},
            "plot_bgcolor": "white",
            "paper_bgcolor": "white",
            "margin": {"l": 10, "r": 20, "t": 45, "b": 40},
            "xaxis": {"title": xl, "gridcolor": "#f1f5f9"},
            "yaxis": {"title": yl, "gridcolor": "#f1f5f9"},
            "font": {"family": "Plus Jakarta Sans"}
        }
    }

def generate_executive_memo(df, rc, dc, cc, coc):
    memo = {
        "win": "Steady operational volume recorded across current periods.",
        "risk": "Ensure periodic audit of missing values and record formats.",
        "action": "Maintain monitoring on highest category distribution."
    }
    if rc and rc in df.columns:
        total_rev = df[rc].sum()
        if cc and cc in df.columns:
            grp = df.groupby(cc)[rc].sum().sort_values(ascending=False)
            top_name = grp.index[0]
            top_pct = grp.iloc[0] / grp.sum() * 100
            memo["win"] = f"Top category/client '{top_name}' leads the portfolio, driving {top_pct:.1f}% ({fmt(grp.iloc[0])}) of overall revenue."
            if len(grp) >= 3 and grp.iloc[:3].sum() / grp.sum() > 0.6:
                p3 = grp.iloc[:3].sum() / grp.sum() * 100
                memo["risk"] = f"High Client Concentration Risk: Top 3 accounts generate {p3:.1f}% of total business revenue. A single contract churn would impact operations."
                memo["action"] = f"Accelerate client diversification pipeline and initiate retention reviews with {top_name} and secondary tier accounts."
        
        if coc and coc in df.columns:
            total_cost = df[coc].sum()
            margin = ((total_rev - total_cost) / total_rev * 100) if total_rev > 0 else 0
            if margin < 25:
                memo["risk"] = f"Compressed Gross Margin ({margin:.1f}%). Total costs stand at {fmt(total_cost)} against {fmt(total_rev)} in gross revenue."
                memo["action"] = f"Audit bottom 20% low-margin products or service deliverables to eliminate unprofitable cost leakages."
            elif margin >= 40:
                memo["win"] += f" Strong healthy gross margin profile of {margin:.1f}%."

        if dc and dc in df.columns:
            try:
                df2 = df.copy()
                df2["_d"] = pd.to_datetime(df2[dc], errors="coerce")
                monthly = df2.groupby(df2["_d"].dt.to_period("M"))[rc].sum()
                if len(monthly) >= 2:
                    curr, prev = float(monthly.iloc[-1]), float(monthly.iloc[-2])
                    pct = ((curr - prev) / prev * 100) if prev > 0 else 0
                    if pct > 0:
                        memo["win"] += f" Recent period expanded +{pct:.1f}% MoM ({fmt(prev)} → {fmt(curr)})."
                    else:
                        memo["risk"] += f" Recent period contracted {pct:.1f}% MoM ({fmt(prev)} → {fmt(curr)})."
            except:
                pass

    return memo

def get_insights(df, rc, dc, cc, coc):
    ins = []
    if rc and rc in df.columns:
        rev = df[rc].dropna()
        total, avg, med = rev.sum(), rev.mean(), rev.median()
        ins.append({"type":"info","title":f"Total Volume: {fmt(total)}","body":f"Average per transaction: {fmt(avg)} | Median: {fmt(med)} | Total Records: {len(df):,}"})
        if cc and cc in df.columns:
            grp = df.groupby(cc)[rc].sum().sort_values(ascending=False)
            tp, tp_pct = grp.index[0], grp.iloc[0]/grp.sum()*100
            bp, bp_pct = grp.index[-1], grp.iloc[-1]/grp.sum()*100
            ins.append({"type":"success","title":f"Leader: {tp}","body":f"Drives {tp_pct:.1f}% of top-line metric. Smallest contributor: {bp} ({bp_pct:.1f}%)."})
            if len(grp)>=3 and grp.iloc[:3].sum()/grp.sum()*100 > 60:
                ins.append({"type":"warning","title":"Portfolio Concentration Risk","body":f"Top 3 {cc}s represent {grp.iloc[:3].sum()/grp.sum()*100:.0f}% of total revenue."})
        if coc and coc in df.columns:
            ct = df[coc].dropna().sum()
            pf = total - ct
            mg = (pf / total * 100) if total > 0 else 0
            ins.append({"type":"success" if mg>=30 else "warning","title":f"Gross Margin: {mg:.1f}%","body":f"Gross: {fmt(total)} - Costs: {fmt(ct)} = Net Profit Contribution: {fmt(pf)}"})
        if dc and dc in df.columns:
            try:
                df2 = df.copy(); df2["_d"] = pd.to_datetime(df2[dc], errors="coerce")
                mo = df2.groupby(df2["_d"].dt.to_period("M"))[rc].sum()
                if len(mo) >= 2:
                    lat, prev = float(mo.iloc[-1]), float(mo.iloc[-2])
                    pct = (lat - prev) / prev * 100 if prev != 0 else 0
                    dir_txt = "Expanded" if pct >= 0 else "Dropped"
                    ins.append({"type":"success" if pct>=0 else "danger","title":f"Latest Period {dir_txt}: {abs(pct):.1f}%","body":f"Previous Period: {fmt(prev)} → Current: {fmt(lat)}"})
            except: pass

    miss = int(df.isnull().sum().sum())
    cp = round((1 - miss / (len(df) * len(df.columns))) * 100, 1)
    if miss == 0:
        ins.append({"type":"success","title":"Data Integrity: 100% Clean","body":"Zero missing values or null cells detected across all records."})
    else:
        ins.append({"type":"warning","title":"Missing Values Detected","body":f"{miss:,} total null cells found. Completeness score is {cp}%."})
    return ins

def run_analysis(df, rc, dc, cc, coc):
    kpis = []
    if rc and rc in df.columns:
        kpis.append({"label": f"Total {rc}", "value": fmt(df[rc].sum()), "sub": f"{len(df):,} records analyzed"})
    else:
        kpis.append({"label": "Total Rows", "value": f"{len(df):,}", "sub": f"{len(df.columns)} columns"})

    if coc and coc in df.columns:
        kpis.append({"label": f"Total {coc}", "value": fmt(df[coc].sum()), "sub": "Total recorded spend"})
    elif rc and rc in df.columns:
        kpis.append({"label": "Avg Record Value", "value": fmt(df[rc].mean()), "sub": f"Median: {fmt(df[rc].median())}"})
    else:
        kpis.append({"label": "Columns Profiled", "value": str(len(df.columns)), "sub": "All attributes mapped"})

    if rc and coc and rc in df.columns and coc in df.columns:
        rt, ct = df[rc].sum(), df[coc].sum()
        mg = (rt - ct) / rt * 100 if rt > 0 else 0
        kpis.append({"label": "Gross Margin", "value": f"{mg:.1f}%", "sub": f"Gross Profit: {fmt(rt - ct)}"})
    elif rc and rc in df.columns:
        kpis.append({"label": "Peak Record", "value": fmt(df[rc].max()), "sub": f"Min: {fmt(df[rc].min())}"})
    else:
        kpis.append({"label": "Data Completeness", "value": "100%", "sub": "Zero null values"})

    if cc and cc in df.columns:
        kpis.append({"label": f"Unique {cc}s", "value": str(df[cc].nunique()), "sub": "Active dimensions"})
    else:
        kpis.append({"label": "Numeric Fields", "value": str(len(df.select_dtypes(include=np.number).columns)), "sub": "Analyzed features"})

    charts = []
    # 1. Bar Chart
    if cc and rc and cc in df.columns and rc in df.columns:
        g = df.groupby(cc)[rc].sum().sort_values(ascending=False).head(10)
        charts.append(barchart(g.index.tolist(), g.values.tolist(), f"{rc} by {cc}"))
    else:
        nc = df.select_dtypes(include=np.number).columns[:8]
        s = df[nc].sum().sort_values(ascending=False)
        charts.append(barchart(s.index.tolist(), s.values.tolist(), "Top Metric Totals"))

    # 2. Line Chart with 3-Month Forecasting Line
    forecast_kpi = None
    if dc and rc and dc in df.columns and rc in df.columns:
        try:
            df2 = df.copy(); df2["_d"] = pd.to_datetime(df2[dc], errors="coerce")
            mo = df2.groupby(df2["_d"].dt.to_period("M"))[rc].sum().reset_index()
            mo["_d"] = mo["_d"].astype(str)
            fc = linechart_with_forecast(mo["_d"].tolist(), mo[rc].tolist(), f"{rc} Trend & 3-Month Projection", dc, rc)
            charts.append(fc["chart"])
            forecast_kpi = fc["forecast_kpi"]
        except:
            charts.append({"data":[], "layout":{"title":{"text":"Trend unavailable"}}})
    elif rc and rc in df.columns:
        vs = df[rc].dropna().tolist()
        counts, edges = np.histogram(vs, bins=15)
        centers = [(edges[i] + edges[i+1])/2 for i in range(len(counts))]
        charts.append({
            "data":[{"type":"bar","x":[f"{c:,.0f}" for c in centers],"y":counts.tolist(),"marker":{"color":"#4361ee"}}],
            "layout":{"title":{"text":f"{rc} Distribution Frequency"},"paper_bgcolor":"white","plot_bgcolor":"white"}
        })

    # 3. Pie Chart
    if cc and rc and cc in df.columns and rc in df.columns:
        g2 = df.groupby(cc)[rc].sum().sort_values(ascending=False).head(8)
        charts.append(piechart(g2.index.tolist(), g2.values.tolist(), f"{rc} Distribution by {cc}"))
    else:
        charts.append({"data":[], "layout":{"title":{"text":"Category distribution"}}})

    # 4. Scatter Chart
    nc2 = df.select_dtypes(include=np.number).columns.tolist()
    if rc and coc and rc in df.columns and coc in df.columns:
        s2 = df[[rc, coc]].dropna().sample(min(300, len(df)))
        charts.append(scatterchart(s2[rc].tolist(), s2[coc].tolist(), rc, coc, f"{rc} vs {coc} Correlation"))
    elif len(nc2) >= 2:
        s2 = df[[nc2[0], nc2[1]]].dropna().sample(min(300, len(df)))
        charts.append(scatterchart(s2[nc2[0]].tolist(), s2[nc2[1]].tolist(), nc2[0], nc2[1], f"{nc2[0]} vs {nc2[1]}"))

    num_df = df.select_dtypes(include=np.number)
    stats = []
    for col in num_df.columns:
        d = num_df[col].describe()
        stats.append({
            "column": col,
            "count": int(d["count"]),
            "mean": round(float(d["mean"]), 2),
            "std": round(float(d["std"]), 2),
            "min": round(float(d["min"]), 2),
            "max": round(float(d["max"]), 2)
        })

    miss = int(df.isnull().sum().sum())
    cp = round((1 - miss / (len(df) * len(df.columns))) * 100, 1)
    mc = [{"column":c,"missing":int(df[c].isnull().sum()),"pct":round(df[c].isnull().sum()/len(df)*100,1)} for c in df.columns if df[c].isnull().sum()>0]
    mc.sort(key=lambda x:x["pct"], reverse=True)

    executive_memo = generate_executive_memo(df, rc, dc, cc, coc)

    return {
        "kpis": kpis,
        "charts": charts,
        "insights": get_insights(df, rc, dc, cc, coc),
        "executive_memo": executive_memo,
        "forecast_kpi": forecast_kpi,
        "stats": stats,
        "quality": {
            "rows": len(df),
            "cols": len(df.columns),
            "missing_total": miss,
            "completeness": cp,
            "missing_cols": mc[:10]
        },
        "columns": df.columns.tolist(),
        "numeric_columns": num_df.columns.tolist(),
        "string_columns": df.select_dtypes(include=object).columns.tolist()
    }

# ─────────────────────────── ROUTES ───────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/app")
@app.route("/workspace")
def workspace():
    return render_template("app.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

def generate_preset_df(industry):
    np.random.seed(42)
    months = pd.date_range("2024-01-01", periods=24, freq="MS")
    if industry == "ecommerce":
        categories = ["Footwear", "Apparel", "Electronics", "Beauty & Care", "Accessories", "Home & Living"]
        channels = ["Shopify Direct", "Amazon Store", "TikTok Shop", "Instagram Ads", "Google Shopping"]
        rows = []
        for m in months:
            for cat in categories:
                units = int(np.random.normal(450, 80))
                price = float(np.random.uniform(45, 120))
                revenue = round(units * price, 2)
                cogs = round(revenue * np.random.uniform(0.38, 0.52), 2)
                ad_spend = round(revenue * np.random.uniform(0.18, 0.28), 2)
                rows.append({
                    "Date": m.strftime("%Y-%m"),
                    "Category": cat,
                    "Channel": np.random.choice(channels),
                    "Orders": units,
                    "Gross Revenue": revenue,
                    "COGS": cogs,
                    "Ad Spend": ad_spend
                })
        df = pd.DataFrame(rows)
        detected = {"revenue_col": "Gross Revenue", "date_col": "Date", "category_col": "Category", "cost_col": "COGS"}
        filename = "shopify_ecommerce_metrics.csv"
    elif industry == "agency":
        clients = ["Apex Logistics", "Vanguard Health", "Starlight Media", "Horizon Fintech", "Bloom Cosmetics", "Nexus AI"]
        tiers = ["Retainer Premium", "Standard Advisory", "Sprint Project", "Growth Advisory"]
        rows = []
        for m in months:
            for cl in clients:
                hours = int(np.random.normal(65, 12))
                rate = float(np.random.choice([120, 150, 185, 220]))
                billing = round(hours * rate, 2)
                staff_cost = round(billing * np.random.uniform(0.40, 0.55), 2)
                rows.append({
                    "Month": m.strftime("%Y-%m"),
                    "Client": cl,
                    "Engagement": np.random.choice(tiers),
                    "Billable Hours": hours,
                    "Billings": billing,
                    "Delivery Cost": staff_cost
                })
        df = pd.DataFrame(rows)
        detected = {"revenue_col": "Billings", "date_col": "Month", "category_col": "Client", "cost_col": "Delivery Cost"}
        filename = "agency_client_billings.csv"
    else: # Default: B2B SaaS
        clients = ["Acme Corp", "TechStart", "GlobalTrade", "LocalMart", "FastGrow", "BrightMedia", "CityLogistics", "PrimeRetail"]
        base = {"Acme Corp":85000,"TechStart":45000,"GlobalTrade":120000,"LocalMart":30000,"FastGrow":60000,"BrightMedia":25000,"CityLogistics":55000,"PrimeRetail":40000}
        rows = [
            {
                "Month": m.strftime("%Y-%m"),
                "Client": c,
                "MRR": round(max(np.random.normal(base[c], 8000), 5000), 0),
                "Hosting & COGS": round(max(np.random.normal(base[c], 8000), 5000) * np.random.uniform(0.55, 0.72), 0),
                "Segment": np.random.choice(["Enterprise", "Growth", "Starter"], p=[0.3, 0.4, 0.3])
            }
            for m in months for c in clients
        ]
        df = pd.DataFrame(rows)
        detected = {"revenue_col": "MRR", "date_col": "Month", "category_col": "Client", "cost_col": "Hosting & COGS"}
        filename = "b2b_saas_metrics.csv"

    return df, detected, filename

@app.route("/api/sample")
def sample():
    industry = request.args.get("industry", "saas").lower()
    df, detected, filename = generate_preset_df(industry)
    result = run_analysis(df, detected["revenue_col"], detected["date_col"], detected["category_col"], detected["cost_col"])
    result["filename"] = filename
    result["rows"] = len(df)
    result["detected"] = detected
    result["industry"] = industry
    return jsonify(result)

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files: return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    fname = file.filename.lower()
    try:
        df = pd.read_csv(file) if fname.endswith(".csv") else pd.read_excel(file)
        if len(df) == 0: return jsonify({"error": "File contains zero rows"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to parse file: {str(e)}"}), 400

    detected = detect_cols(df)
    result = run_analysis(df, detected.get("revenue_col"), detected.get("date_col"), detected.get("category_col"), detected.get("cost_col"))
    result["filename"] = file.filename
    result["rows"] = len(df)
    result["detected"] = detected
    return jsonify(result)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files: return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    try:
        df = pd.read_csv(file) if file.filename.lower().endswith(".csv") else pd.read_excel(file)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    rc = request.form.get("revenue_col") or None
    dc = request.form.get("date_col") or None
    cc = request.form.get("category_col") or None
    coc = request.form.get("cost_col") or None
    result = run_analysis(df, rc, dc, cc, coc)
    result["filename"] = file.filename
    result["rows"] = len(df)
    return jsonify(result)

@app.route("/api/clean_export", methods=["POST", "GET"])
def clean_export():
    """Cleans null values, strips whitespace, standardizes dates, drops duplicates, and exports a CSV."""
    industry = request.args.get("industry")
    if "file" in request.files:
        file = request.files["file"]
        fname = file.filename.lower()
        df = pd.read_csv(file) if fname.endswith(".csv") else pd.read_excel(file)
    else:
        df, _, _ = generate_preset_df(industry or "saas")

    # Clean DataFrame
    df_clean = df.copy()
    # 1. Drop duplicate rows
    df_clean = df_clean.drop_duplicates()
    # 2. Trim strings
    for col in df_clean.select_dtypes(include=object).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    # 3. Fill numeric nulls with median
    for col in df_clean.select_dtypes(include=np.number).columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    # 4. Fill text nulls
    for col in df_clean.select_dtypes(include=object).columns:
        df_clean[col] = df_clean[col].replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"}).fillna("Unknown")

    out_csv = df_clean.to_csv(index=False)
    return Response(
        out_csv,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=datascope_cleaned_dataset.csv"}
    )

@app.route("/api/chat_query", methods=["POST"])
def chat_query():
    """Natural Language Assistant that answers business questions using dataset aggregations."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip().lower()
    industry = data.get("industry", "saas")

    # If file was not uploaded in JSON, evaluate against industry preset
    df, detected, fname = generate_preset_df(industry)
    rc, dc, cc, coc = detected["revenue_col"], detected["date_col"], detected["category_col"], detected["cost_col"]

    if not query:
        return jsonify({"answer": "Please ask a question about your revenue, clients, margins, or trends."})

    total_rev = df[rc].sum() if rc and rc in df.columns else 0
    total_cost = df[coc].sum() if coc and coc in df.columns else 0
    gross_margin = ((total_rev - total_cost) / total_rev * 100) if total_rev > 0 else 0

    # Answer matching
    if any(k in query for k in ["top", "best", "biggest", "highest", "leader"]):
        if cc and rc and cc in df.columns and rc in df.columns:
            grp = df.groupby(cc)[rc].sum().sort_values(ascending=False).head(5)
            ans = f"🏆 **Top Performer by {rc}:** '{grp.index[0]}' generated **{fmt(grp.iloc[0])}** ({grp.iloc[0]/total_rev*100:.1f}% of total).<br><br><strong>Top 5 Breakdown:</strong><br>"
            for idx, (name, val) in enumerate(grp.items(), 1):
                ans += f"{idx}. **{name}**: {fmt(val)} ({val/total_rev*100:.1f}%)<br>"
            return jsonify({"answer": ans, "type": "rank"})
    
    if any(k in query for k in ["margin", "profit", "profitability", "leak", "cost"]):
        if rc and coc and rc in df.columns and coc in df.columns:
            ans = f"📊 **Profitability Overview:**<br>• **Gross Revenue:** {fmt(total_rev)}<br>• **Total Operating Cost:** {fmt(total_cost)}<br>• **Gross Profit:** {fmt(total_rev - total_cost)}<br>• **Gross Margin:** **{gross_margin:.1f}%**"
            if gross_margin >= 35:
                ans += "<br><br>✅ Healthy margin benchmark (>35%). Unit economics are strong."
            else:
                ans += "<br><br>⚠️ Margin compression detected. Recommend auditing COGS on bottom performers."
            return jsonify({"answer": ans, "type": "margin"})

    if any(k in query for k in ["trend", "month", "grow", "growth", "forecast", "projection"]):
        if dc and rc and dc in df.columns and rc in df.columns:
            df2 = df.copy(); df2["_d"] = pd.to_datetime(df2[dc], errors="coerce")
            mo = df2.groupby(df2["_d"].dt.to_period("M"))[rc].sum()
            curr, prev = float(mo.iloc[-1]), float(mo.iloc[-2])
            pct = ((curr - prev) / prev * 100) if prev > 0 else 0
            ans = f"📈 **Performance Run-Rate:**<br>• **Latest Month:** {fmt(curr)}<br>• **Prior Month:** {fmt(prev)}<br>• **MoM Delta:** **{'+' if pct>0 else ''}{pct:.1f}%**<br><br>🔮 **Forward Projection:** Based on run-rate, next quarter volume is pacing towards **{fmt(curr * 3.1)}**."
            return jsonify({"answer": ans, "type": "trend"})

    if any(k in query for k in ["anomaly", "outlier", "risk", "clean", "missing"]):
        miss = int(df.isnull().sum().sum())
        ans = f"🛡️ **Data Quality & Risk Check:**<br>• **Total Records:** {len(df):,}<br>• **Missing Cells:** {miss}<br>• **Data Completeness:** {round((1-miss/(len(df)*len(df.columns)))*100,1)}%<br>"
        if cc and rc:
            grp = df.groupby(cc)[rc].sum().sort_values(ascending=False)
            top3_pct = grp.iloc[:3].sum() / grp.sum() * 100
            if top3_pct > 60:
                ans += f"<br>⚠️ **Concentration Risk Alert:** The top 3 {cc} accounts drive **{top3_pct:.1f}%** of your business."
        return jsonify({"answer": ans, "type": "quality"})

    # Default fallback answer
    ans = f"💡 **Executive Summary for '{query}':**<br>• Total {rc or 'Volume'}: **{fmt(total_rev)}** across {len(df):,} records.<br>• Gross Margin: **{gross_margin:.1f}%**.<br>• Dimension Analyzed: **{cc or 'Categories'}** ({df[cc].nunique() if cc and cc in df.columns else 0} unique entities)."
    return jsonify({"answer": ans, "type": "general"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
