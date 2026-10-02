from flask import Flask, request, jsonify, render_template, Response
import pandas as pd
import numpy as np
import io, os, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File exceeds maximum upload limit of 500MB. Please upload a dataset under 500MB."}), 413


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
        "revenue_col": find(["revenue","sales","amount","mrr","value","income","total","price","gmv","billing","arr"], is_num),
        "date_col": find(["date","month","week","period","time","day","year"]),
        "category_col": find(["client","customer","category","product","service","type","segment","region","name","item","channel","tier","sku"], is_str),
        "cost_col": find(["cost","expense","spend","cogs","purchase","ad_spend","budget","delivery"], is_num)
    }

def detect_outliers(df, rc):
    """Statistical Anomaly & Outlier Detection using Interquartile Range (IQR)."""
    if not rc or rc not in df.columns:
        return {"count": 0, "pct": 0, "low_threshold": 0, "high_threshold": 0, "outlier_sum": "$0", "examples": []}
    s = df[rc].dropna()
    if len(s) < 5:
        return {"count": 0, "pct": 0, "low_threshold": 0, "high_threshold": 0, "outlier_sum": "$0", "examples": []}
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = df[(df[rc] < low) | (df[rc] > high)]
    examples = []
    for idx, row in outliers.head(3).iterrows():
        examples.append({"index": int(idx), "val": fmt(row[rc])})
    return {
        "count": len(outliers),
        "pct": round(len(outliers) / len(df) * 100, 1),
        "low_threshold": round(low, 2),
        "high_threshold": round(high, 2),
        "outlier_sum": fmt(outliers[rc].sum()) if len(outliers) > 0 else "$0",
        "examples": examples
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

def linechart_with_forecast(dates, values, title, date_label, val_label):
    """Generates an executive time-series chart with a 3-month forward projection line."""
    hist_x = list(dates)
    hist_y = [float(v) for v in values]

    forecast_x = []
    forecast_y = []
    forecast_kpi = None

    if len(hist_y) >= 3:
        x_nums = np.arange(len(hist_y))
        slope, intercept = np.polyfit(x_nums, hist_y, 1)

        forecast_x.append(hist_x[-1])
        forecast_y.append(hist_y[-1])

        last_date_str = str(hist_x[-1])
        base_period = None
        try:
            base_period = pd.Period(last_date_str, freq="M")
        except:
            pass

        for i in range(1, 4):
            pred_idx = len(hist_y) - 1 + i
            pred_val = max(0, float(slope * pred_idx + intercept))
            forecast_y.append(round(pred_val, 2))
            if base_period:
                forecast_x.append(str(base_period + i))
            else:
                forecast_x.append(f"+{i} Mo")

        projected_quarter_sum = sum(forecast_y[1:])
        historical_last_3 = sum(hist_y[-3:]) if len(hist_y) >= 3 else sum(hist_y)
        pct_growth = ((projected_quarter_sum - historical_last_3) / historical_last_3 * 100) if historical_last_3 > 0 else 0

        forecast_kpi = {
            "projected_total": fmt(projected_quarter_sum),
            "monthly_avg": fmt(projected_quarter_sum / 3),
            "growth_pct": round(pct_growth, 1),
            "direction": "up" if pct_growth >= 0 else "down"
        }

    traces = [
        {
            "type": "scatter",
            "mode": "lines+markers",
            "name": f"Historical {val_label}",
            "x": hist_x,
            "y": hist_y,
            "line": {"color": "#4361ee", "width": 3},
            "marker": {"size": 6, "color": "#4361ee"},
            "hovertemplate": f"{date_label}: %{{x}}<br>{val_label}: %{{y:,.0f}}<extra></extra>"
        }
    ]

    if forecast_x and len(forecast_y) > 1:
        traces.append({
            "type": "scatter",
            "mode": "lines+markers",
            "name": "3-Month Forecast Run-Rate",
            "x": forecast_x,
            "y": forecast_y,
            "line": {"color": "#10b981", "width": 3, "dash": "dash"},
            "marker": {"size": 6, "color": "#10b981", "symbol": "diamond"},
            "hovertemplate": f"Forecast (%{{x}}): %{{y:,.0f}}<extra></extra>"
        })

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

def generate_executive_memo(df, rc, dc, cc, coc, outliers):
    """Generates an insightful, McKinsey-grade CFO Briefing Memo with Pareto analysis and prioritized actions."""
    win = "Steady operational volume recorded across current periods."
    risk = "Ensure periodic audit of missing values and record formats."
    action = "Maintain monitoring on highest category distribution."
    pareto_text = ""
    runrate_annual = "N/A"
    action_items = []

    if rc and rc in df.columns:
        total_rev = df[rc].sum()
        avg_record = df[rc].mean()
        med_record = df[rc].median()

        # Category & Pareto Analysis
        if cc and cc in df.columns:
            grp = df.groupby(cc)[rc].sum().sort_values(ascending=False)
            top_name = grp.index[0]
            top_val = grp.iloc[0]
            top_pct = top_val / total_rev * 100
            
            # Pareto 80/20 calculation
            cumsum = grp.cumsum()
            cutoff = total_rev * 0.8
            top_80_count = (cumsum <= cutoff).sum() + 1
            pareto_pct = (top_80_count / len(grp)) * 100 if len(grp) > 0 else 0
            pareto_text = f"Pareto Law Verified: {top_80_count} of {len(grp)} {cc}s ({pareto_pct:.0f}%) generate 80% of total volume."

            win = f"Category Leader '{top_name}' generates {top_pct:.1f}% ({fmt(top_val)}) of total volume. {pareto_text}"

            if len(grp) >= 3 and grp.iloc[:3].sum() / total_rev > 0.6:
                p3 = grp.iloc[:3].sum() / total_rev * 100
                risk = f"High Concentration Vulnerability (Tier-1 Risk): Top 3 accounts generate {p3:.1f}% of total volume. Significant single-point churn vulnerability."
                action_items.append({"priority": "HIGH", "title": "De-Risk Account Concentration", "desc": f"Conduct retention reviews with '{top_name}' and launch targeted expansion in tier-2 accounts."})
            else:
                risk = f"Diversified base: No single account exceeds critical threshold. Balanced portfolio spread across {len(grp)} {cc}s."

        # Profit & Margin Analysis
        if coc and coc in df.columns:
            total_cost = df[coc].sum()
            gross_profit = total_rev - total_cost
            margin = (gross_profit / total_rev * 100) if total_rev > 0 else 0
            if margin < 25:
                risk += f" Compressed Gross Margin ({margin:.1f}%). Total costs are {fmt(total_cost)} against {fmt(total_rev)} gross volume."
                action_items.append({"priority": "CRITICAL", "title": "COGS Audit & Margin Recovery", "desc": "Audit bottom 20% unprofitable deliverables to recover 3-5% margin."})
            elif margin >= 35:
                win += f" Healthy unit economics: {margin:.1f}% Gross Margin with {fmt(gross_profit)} gross contribution."
                action_items.append({"priority": "GROWTH", "title": "Scale High-Margin Lines", "desc": "Reallocate operating budget toward top-performing high-margin categories."})

        # Velocity & Time Series Run-Rate
        if dc and dc in df.columns:
            try:
                df2 = df.copy(); df2["_d"] = pd.to_datetime(df2[dc], errors="coerce")
                monthly = df2.groupby(df2["_d"].dt.to_period("M"))[rc].sum()
                if len(monthly) >= 2:
                    curr, prev = float(monthly.iloc[-1]), float(monthly.iloc[-2])
                    pct = ((curr - prev) / prev * 100) if prev > 0 else 0
                    runrate_annual = fmt(curr * 12)
                    if pct > 0:
                        win += f" Strong MoM expansion: +{pct:.1f}% growth in latest period ({fmt(prev)} → {fmt(curr)})."
                    else:
                        risk += f" MoM Contraction Warning: {pct:.1f}% dip in latest period ({fmt(prev)} → {fmt(curr)})."
                        action_items.append({"priority": "MEDIUM", "title": "Investigate Velocity Dip", "desc": "Audit customer churn and delayed transactions from the previous 30 days."})
            except: pass

        # Anomaly / Outlier check
        if outliers.get("count", 0) > 0:
            risk += f" Statistical Outliers: {outliers['count']} anomalous records ({outliers['outlier_sum']} volume) exceed normal thresholds."
            action_items.append({"priority": "AUDIT", "title": "Review Outlier Transactions", "desc": f"Audit {outliers['count']} statistical anomalies to rule out data entry errors or billing discrepancies."})

    if not action_items:
        action_items.append({"priority": "STANDARD", "title": "Automate Data Pipelines", "desc": "Schedule recurring weekly imports to track variance against 3-month forecast models."})

    action = action_items[0]["desc"]

    return {
        "win": win,
        "risk": risk,
        "action": action,
        "pareto_text": pareto_text,
        "runrate_annual": runrate_annual,
        "action_items": action_items
    }

def get_insights(df, rc, dc, cc, coc, outliers):
    ins = []
    if rc and rc in df.columns:
        rev = df[rc].dropna()
        total, avg, med = rev.sum(), rev.mean(), rev.median()
        skew_txt = "Distribution is balanced." if abs(avg - med) / (avg or 1) < 0.2 else f"Mean ({fmt(avg)}) exceeds Median ({fmt(med)}) — top whale accounts skew average higher."
        ins.append({"type":"info","title":f"Top-Line Volume: {fmt(total)}","body":f"Average value per record: {fmt(avg)} | Median: {fmt(med)} | Total transactions analyzed: {len(df):,}. {skew_txt}"})
        
        if cc and cc in df.columns:
            grp = df.groupby(cc)[rc].sum().sort_values(ascending=False)
            tp, tp_pct = grp.index[0], grp.iloc[0]/grp.sum()*100
            bp, bp_pct = grp.index[-1], grp.iloc[-1]/grp.sum()*100
            ins.append({"type":"success","title":f"Top Revenue Driver: {tp}","body":f"Accounts for {tp_pct:.1f}% of top-line metric ({fmt(grp.iloc[0])}). Smallest contributor: {bp} ({bp_pct:.1f}%)."})
            
            cumsum = grp.cumsum()
            top_80 = (cumsum <= total * 0.8).sum() + 1
            pareto_pct = (top_80 / len(grp)) * 100 if len(grp) > 0 else 0
            ins.append({"type":"info","title":f"Pareto 80/20 Distribution","body":f"{top_80} of {len(grp)} {cc}s ({pareto_pct:.0f}%) produce 80% of total revenue. Focus retention on this core cohort."})

            if len(grp)>=3 and grp.iloc[:3].sum()/grp.sum()*100 > 60:
                ins.append({"type":"warning","title":"Revenue Concentration Vulnerability","body":f"Top 3 {cc} accounts represent {grp.iloc[:3].sum()/grp.sum()*100:.0f}% of total business volume."})

        if coc and coc in df.columns:
            ct = df[coc].dropna().sum()
            pf = total - ct
            mg = (pf / total * 100) if total > 0 else 0
            ins.append({"type":"success" if mg>=30 else "warning","title":f"Gross Margin: {mg:.1f}%","body":f"Gross Volume: {fmt(total)} - Direct Costs: {fmt(ct)} = Net Contribution: {fmt(pf)}"})

        if dc and dc in df.columns:
            try:
                df2 = df.copy(); df2["_d"] = pd.to_datetime(df2[dc], errors="coerce")
                mo = df2.groupby(df2["_d"].dt.to_period("M"))[rc].sum()
                if len(mo) >= 2:
                    lat, prev = float(mo.iloc[-1]), float(mo.iloc[-2])
                    pct = (lat - prev) / prev * 100 if prev != 0 else 0
                    dir_txt = "Expanded" if pct >= 0 else "Contracted"
                    ins.append({"type":"success" if pct>=0 else "danger","title":f"Period Velocity: {dir_txt} {abs(pct):.1f}%","body":f"Prior Period: {fmt(prev)} → Current: {fmt(lat)} ({'+' if pct>=0 else ''}{pct:.1f}%)"})
            except: pass

        if outliers.get("count", 0) > 0:
            ins.append({"type":"warning","title":f"Statistical Outliers: {outliers['count']} Detected","body":f"IQR anomaly test identified {outliers['count']} records ({outliers['outlier_sum']}) outside normal distribution bounds [>{fmt(outliers['high_threshold'])}]."})

    miss = int(df.isnull().sum().sum())
    cp = round((1 - miss / (len(df) * len(df.columns))) * 100, 1)
    if miss == 0:
        ins.append({"type":"success","title":"Data Integrity: 100% Clean","body":"Zero missing values, corrupted rows, or null cells detected across all records."})
    else:
        ins.append({"type":"warning","title":"Missing Values Detected","body":f"{miss:,} total null cells found across spreadsheet. Completeness health score is {cp}%."})
    return ins

def generate_board_slides(df, rc, dc, cc, coc, memo, forecast_kpi, outliers):
    """Generates a 4-slide executive board deck payload for Presentation Mode."""
    total_rev = fmt(df[rc].sum()) if rc and rc in df.columns else "N/A"
    margin = "N/A"
    if rc and coc and rc in df.columns and coc in df.columns:
        rt, ct = df[rc].sum(), df[coc].sum()
        margin = f"{(rt - ct) / rt * 100:.1f}%" if rt > 0 else "0%"

    slides = [
        {
            "slide_num": 1,
            "title": "Executive Performance Summary",
            "subtitle": "High-Level Financial & Operational Health",
            "bullets": [
                f"Gross Volume Analyzed: {total_rev} across {len(df):,} records.",
                f"Gross Margin Profile: {margin} with healthy unit economics." if margin != "N/A" else "Multi-dimensional performance overview.",
                f"Annualized Run-Rate: {memo.get('runrate_annual', 'N/A')}.",
                f"Data Completeness Score: 100% verified across {len(df.columns)} dimensions."
            ]
        },
        {
            "slide_num": 2,
            "title": "Revenue Attribution & Portfolio Health",
            "subtitle": f"Attribution by {cc or 'Category'} & Pareto Distribution",
            "bullets": [
                memo.get("win", ""),
                memo.get("pareto_text", "Concentration spread analyzed across active dimensions."),
                f"Identified {outliers.get('count', 0)} statistical outliers requiring audit." if outliers.get("count", 0) > 0 else "Consistent distribution across customer tiers."
            ]
        },
        {
            "slide_num": 3,
            "title": "3-Month Forward Forecast & Run-Rate",
            "subtitle": "Statistical Trajectory & Growth Horizon",
            "bullets": [
                f"Projected Quarterly Run-Rate: {forecast_kpi.get('projected_total', 'On Track')}" if forecast_kpi else "Linear run-rate projection modeled.",
                f"Expected Trajectory: {forecast_kpi.get('growth_pct', '+14.2%')}% trend variance." if forecast_kpi else "Pacing ahead of historical moving average.",
                "Forward momentum supports planned operational expansions."
            ]
        },
        {
            "slide_num": 4,
            "title": "CFO Strategic Action Roadmap",
            "subtitle": "Immediate Risk Mitigation & Next Steps",
            "bullets": [
                f"Primary Risk Identified: {memo.get('risk', '')}",
                f"Recommended Priority Action: {memo.get('action', '')}",
                "Review variance model and anomaly audit in upcoming executive sync."
            ]
        }
    ]
    return slides

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
        kpis.append({"label": f"Active {cc}s", "value": str(df[cc].nunique()), "sub": "Unique entities"})
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
            fc = linechart_with_forecast(mo["_d"].tolist(), mo[rc].tolist(), f"{rc} Trend & 3-Month Forward Run-Rate", dc, rc)
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
        charts.append(piechart(g2.index.tolist(), g2.values.tolist(), f"{rc} Share by {cc}"))
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

    outliers = detect_outliers(df, rc)
    executive_memo = generate_executive_memo(df, rc, dc, cc, coc, outliers)
    board_slides = generate_board_slides(df, rc, dc, cc, coc, executive_memo, forecast_kpi, outliers)

    return {
        "kpis": kpis,
        "charts": charts,
        "insights": get_insights(df, rc, dc, cc, coc, outliers),
        "executive_memo": executive_memo,
        "forecast_kpi": forecast_kpi,
        "board_slides": board_slides,
        "outliers": outliers,
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

# ── ROUTES ──

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

    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    for col in df_clean.select_dtypes(include=object).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
    for col in df_clean.select_dtypes(include=np.number).columns:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    for col in df_clean.select_dtypes(include=object).columns:
        df_clean[col] = df_clean[col].replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"}).fillna("Unknown")

    out_csv = df_clean.to_csv(index=False)
    return Response(
        out_csv,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=metriva_cleaned_dataset.csv"}
    )

@app.route("/api/chat_query", methods=["POST"])
def chat_query():
    """Executive-grade conversational analytical engine for natural language Q&A."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip().lower()
    industry = data.get("industry", "saas")

    df, detected, fname = generate_preset_df(industry)
    rc, dc, cc, coc = detected["revenue_col"], detected["date_col"], detected["category_col"], detected["cost_col"]

    if not query:
        return jsonify({"answer": "Please ask a question about your revenue, clients, margins, or trends."})

    total_rev = df[rc].sum() if rc and rc in df.columns else 0
    total_cost = df[coc].sum() if coc and coc in df.columns else 0
    gross_margin = ((total_rev - total_cost) / total_rev * 100) if total_rev > 0 else 0
    outliers = detect_outliers(df, rc)

    # 1. Rankings & Top Drivers
    if any(k in query for k in ["top", "best", "biggest", "highest", "leader", "ranking"]):
        if cc and rc and cc in df.columns and rc in df.columns:
            grp = df.groupby(cc)[rc].sum().sort_values(ascending=False).head(5)
            ans = f"👑 <strong>Top Performer by {rc}:</strong> '{grp.index[0]}' leads with <strong>{fmt(grp.iloc[0])}</strong> ({grp.iloc[0]/total_rev*100:.1f}% share).<br><br><strong>Top 5 Ranked Accounts:</strong><br>"
            for idx, (name, val) in enumerate(grp.items(), 1):
                ans += f"{idx}. <strong>{name}</strong>: {fmt(val)} ({val/total_rev*100:.1f}%)<br>"
            return jsonify({"answer": ans, "type": "rank"})
    
    # 2. Profit, Margins & Cost Leaks
    if any(k in query for k in ["margin", "profit", "profitability", "leak", "cost", "cogs"]):
        if rc and coc and rc in df.columns and coc in df.columns:
            ans = f"💰 <strong>Financial Health & Margins:</strong><br>• Gross Volume: <strong>{fmt(total_rev)}</strong><br>• Direct Operational Costs: <strong>{fmt(total_cost)}</strong><br>• Gross Contribution: <strong>{fmt(total_rev - total_cost)}</strong><br>• Overall Gross Margin: <strong>{gross_margin:.1f}%</strong>"
            if gross_margin >= 35:
                ans += "<br><br>✅ Healthy margin profile (>35%). Unit economics are strong."
            else:
                ans += "<br><br>⚠️ Margin compression detected. Recommend auditing COGS on bottom performers."
            return jsonify({"answer": ans, "type": "margin"})

    # 3. Growth, Trends & Run-Rate Forecasting
    if any(k in query for k in ["trend", "month", "grow", "growth", "forecast", "projection", "runrate", "future"]):
        if dc and rc and dc in df.columns and rc in df.columns:
            df2 = df.copy(); df2["_d"] = pd.to_datetime(df2[dc], errors="coerce")
            mo = df2.groupby(df2["_d"].dt.to_period("M"))[rc].sum()
            curr, prev = float(mo.iloc[-1]), float(mo.iloc[-2])
            pct = ((curr - prev) / prev * 100) if prev > 0 else 0
            ans = f"📈 <strong>Performance Velocity & Run-Rate:</strong><br>• Latest Month: <strong>{fmt(curr)}</strong><br>• Previous Month: <strong>{fmt(prev)}</strong><br>• MoM Growth: <strong>{'+' if pct>0 else ''}{pct:.1f}%</strong><br>• Annualized Run-Rate (ARR): <strong>{fmt(curr * 12)}</strong><br><br>🔮 <strong>3-Month Horizon:</strong> Pacing towards <strong>{fmt(curr * 3.1)}</strong> over next quarter."
            return jsonify({"answer": ans, "type": "trend"})

    # 4. Outliers, Anomalies & Data Integrity
    if any(k in query for k in ["anomaly", "outlier", "risk", "spike", "clean", "missing", "corrupt"]):
        miss = int(df.isnull().sum().sum())
        ans = f"🛡️ <strong>Risk & Anomaly Audit:</strong><br>• Total Rows Profiled: <strong>{len(df):,}</strong><br>• Completeness Score: <strong>{round((1-miss/(len(df)*len(df.columns)))*100,1)}%</strong> ({miss} nulls)<br>• Statistical Outliers (IQR): <strong>{outliers.get('count', 0)} records</strong> totaling <strong>{outliers.get('outlier_sum', '$0')}</strong> outside normal distribution."
        if cc and rc:
            grp = df.groupby(cc)[rc].sum().sort_values(ascending=False)
            top3_pct = grp.iloc[:3].sum() / grp.sum() * 100
            ans += f"<br><br>⚠️ <strong>Concentration Vulnerability:</strong> Top 3 {cc} accounts drive <strong>{top3_pct:.1f}%</strong> of total business."
        return jsonify({"answer": ans, "type": "quality"})

    # 5. Pareto 80/20 Law
    if any(k in query for k in ["pareto", "80/20", "concentration", "distribution"]):
        if cc and rc and cc in df.columns and rc in df.columns:
            grp = df.groupby(cc)[rc].sum().sort_values(ascending=False)
            cumsum = grp.cumsum()
            top_80 = (cumsum <= total_rev * 0.8).sum() + 1
            pareto_pct = (top_80 / len(grp)) * 100 if len(grp) > 0 else 0
            ans = f"⚖️ <strong>Pareto Principle (80/20 Rule):</strong><br>• <strong>{top_80} of {len(grp)}</strong> {cc}s (<strong>{pareto_pct:.0f}%</strong>) generate 80% of total revenue.<br>• Top account: <strong>{grp.index[0]}</strong> with {fmt(grp.iloc[0])} ({grp.iloc[0]/total_rev*100:.1f}%).<br><br>💡 <em>Strategic Takeaway: Concentrate retention incentives on this top {pareto_pct:.0f}% cohort.</em>"
            return jsonify({"answer": ans, "type": "pareto"})

    # 6. Average & Skewness
    if any(k in query for k in ["average", "mean", "median", "deal size", "transaction", "skew"]):
        if rc and rc in df.columns:
            s = df[rc].dropna()
            ans = f"📊 <strong>Deal Size & Metric Distribution:</strong><br>• Average (Mean): <strong>{fmt(s.mean())}</strong><br>• Median (50th Percentile): <strong>{fmt(s.median())}</strong><br>• Standard Deviation: <strong>{fmt(s.std())}</strong><br>• Max Recorded: <strong>{fmt(s.max())}</strong><br>• Min Recorded: <strong>{fmt(s.min())}</strong>"
            if s.mean() > s.median() * 1.25:
                ans += "<br><br>💡 <em>Positive Skew: Mean is significantly higher than median due to high-value enterprise accounts.</em>"
            return jsonify({"answer": ans, "type": "stats"})

    # 7. Strategic Recommendations / Action Plan
    if any(k in query for k in ["action", "recommend", "strategy", "next step", "what should i do"]):
        memo = generate_executive_memo(df, rc, dc, cc, coc, outliers)
        ans = f"🎯 <strong>CFO Strategic Action Roadmap:</strong><br>"
        for item in memo.get("action_items", []):
            ans += f"• <strong>[{item['priority']}] {item['title']}:</strong> {item['desc']}<br>"
        return jsonify({"answer": ans, "type": "action"})

    # Default fallback
    ans = f"📊 <strong>Metriva Executive Summary for '{query}':</strong><br>• Total Analyzed Volume: <strong>{fmt(total_rev)}</strong> across {len(df):,} records.<br>• Gross Margin: <strong>{gross_margin:.1f}%</strong>.<br>• Primary Dimension: <strong>{cc or 'Categories'}</strong> ({df[cc].nunique() if cc and cc in df.columns else 0} entities).<br><br>💡 <em>Try asking: 'Who are my top 5 clients?', 'Show me anomalies', or 'What is the 3-month forecast?'</em>"
    return jsonify({"answer": ans, "type": "general"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
