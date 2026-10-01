from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

def fmt(n):
    if n is None: return "0"
    n = float(n)
    if abs(n) >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if abs(n) >= 1_000: return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"

def detect_cols(df):
    cl = {c.lower(): c for c in df.columns}
    def find(kws, cond=None):
        for kw in kws:
            for lc, orig in cl.items():
                if kw in lc:
                    if cond is None or cond(df[orig]): return orig
        return None
    is_num = lambda s: pd.api.types.is_numeric_dtype(s)
    is_str = lambda s: s.dtype == object
    return {
        "revenue_col": find(["revenue","sales","amount","value","income","total","price","gmv","billing"], is_num),
        "date_col": find(["date","month","week","period","time","day","year"]),
        "category_col": find(["client","customer","category","product","service","type","segment","region","name","item","channel"], is_str),
        "cost_col": find(["cost","expense","spend","cogs","purchase"], is_num)
    }

def barchart(labels, values, title):
    pairs = sorted(zip(values,labels),reverse=True)
    vs=[p[0] for p in pairs]; ls=[p[1] for p in pairs]
    return {"data":[{"type":"bar","orientation":"h","x":vs,"y":ls,"marker":{"color":vs,"colorscale":[[0,"#dbeafe"],[1,"#4361ee"]],"showscale":False},"hovertemplate":"%{y}: %{x:,.0f}<extra></extra>"}],"layout":{"title":{"text":title,"font":{"size":14,"color":"#1a1a2e","family":"Inter"}},"plot_bgcolor":"white","paper_bgcolor":"white","margin":{"l":10,"r":20,"t":50,"b":20},"xaxis":{"gridcolor":"#f3f4f6"},"yaxis":{"autorange":"reversed"},"font":{"family":"Inter"}}}

def linechart(x, y, title):
    return {"data":[{"type":"scatter","mode":"lines+markers","x":x,"y":y,"line":{"color":"#4361ee","width":2.5},"marker":{"size":6,"color":"#4361ee"},"hovertemplate":"%{x}: %{y:,.0f}<extra></extra>"}],"layout":{"title":{"text":title,"font":{"size":14,"color":"#1a1a2e","family":"Inter"}},"plot_bgcolor":"white","paper_bgcolor":"white","margin":{"l":10,"r":20,"t":50,"b":40},"xaxis":{"gridcolor":"#f3f4f6"},"yaxis":{"gridcolor":"#f3f4f6"},"font":{"family":"Inter"}}}

def piechart(labels, values, title):
    return {"data":[{"type":"pie","labels":labels,"values":values,"hole":0.45,"marker":{"colors":["#4361ee","#7c8ef7","#b8c0fb","#3a52d4","#2d3fae","#1e2d88","#c5caff","#e8eafc"]},"textinfo":"label+percent","hovertemplate":"%{label}: %{value:,.0f} (%{percent})<extra></extra>"}],"layout":{"title":{"text":title,"font":{"size":14,"color":"#1a1a2e","family":"Inter"}},"paper_bgcolor":"white","margin":{"l":10,"r":10,"t":50,"b":10},"font":{"family":"Inter"}}}

def scatterchart(x, y, xl, yl, title):
    return {"data":[{"type":"scatter","mode":"markers","x":x,"y":y,"marker":{"color":"#4361ee","opacity":0.6,"size":7},"hovertemplate":"%{x:,.0f} / %{y:,.0f}<extra></extra>"}],"layout":{"title":{"text":title,"font":{"size":14,"color":"#1a1a2e","family":"Inter"}},"plot_bgcolor":"white","paper_bgcolor":"white","margin":{"l":10,"r":20,"t":50,"b":40},"xaxis":{"title":xl,"gridcolor":"#f3f4f6"},"yaxis":{"title":yl,"gridcolor":"#f3f4f6"},"font":{"family":"Inter"}}}

def get_insights(df, rc, dc, cc, coc):
    ins = []
    if rc and rc in df.columns:
        rev=df[rc].dropna(); total=rev.sum(); avg=rev.mean(); med=rev.median()
        ins.append({"type":"info","title":f"Total {rc}: {fmt(total)}","body":f"Average: {fmt(avg)} | Median: {fmt(med)} | Records: {len(df):,}"})
        if cc and cc in df.columns:
            grp=df.groupby(cc)[rc].sum().sort_values(ascending=False)
            tp=grp.index[0]; tp_pct=grp.iloc[0]/grp.sum()*100
            bp=grp.index[-1]; bp_pct=grp.iloc[-1]/grp.sum()*100
            ins.append({"type":"success","title":f"Top Performer: {tp}","body":f"Drives {tp_pct:.1f}% of {rc}. Bottom: {bp} at {bp_pct:.1f}%."})
            if len(grp)>=3 and grp.iloc[:3].sum()/grp.sum()*100>60:
                p3=grp.iloc[:3].sum()/grp.sum()*100
                ins.append({"type":"warning","title":"Revenue Concentration Risk","body":f"Top 3 {cc}s drive {p3:.0f}% of revenue. High dependency."})
        if coc and coc in df.columns:
            ct=df[coc].dropna().sum(); pf=total-ct; mg=(pf/total*100) if total>0 else 0
            ins.append({"type":"success" if mg>20 else "warning","title":f"Gross Margin: {mg:.1f}%","body":f"Revenue: {fmt(total)} - Costs: {fmt(ct)} = Profit: {fmt(pf)}"})
        if dc and dc in df.columns:
            try:
                df2=df.copy(); df2["_d"]=pd.to_datetime(df2[dc],errors="coerce")
                mo=df2.groupby(df2["_d"].dt.to_period("M"))[rc].sum()
                if len(mo)>=2:
                    lat,prev=float(mo.iloc[-1]),float(mo.iloc[-2]); pct=(lat-prev)/prev*100 if prev!=0 else 0
                    d="up" if pct>0 else "down"
                    ins.append({"type":"success" if pct>0 else "danger","title":f"Latest Period: {d} {abs(pct):.1f}% vs previous","body":f"Previous: {fmt(prev)} -> Latest: {fmt(lat)}"})
            except: pass
    miss=int(df.isnull().sum().sum()); cp=round((1-miss/(len(df)*len(df.columns)))*100,1)
    bad=[c for c in df.columns if df[c].isnull().sum()/len(df)>0.1]
    if bad: ins.append({"type":"warning","title":"Missing Data Detected","body":f"Columns >10% missing: {chr(44).join(bad[:3])}"})
    else: ins.append({"type":"success","title":"Data Quality: Good","body":f"Completeness: {cp}%. No major missing values."})
    return ins

def run_analysis(df, rc, dc, cc, coc):
    kpis = []
    if rc and rc in df.columns: kpis.append({"label":f"Total {rc}","value":fmt(df[rc].sum()),"sub":f"{len(df):,} records"})
    else: kpis.append({"label":"Total Rows","value":f"{len(df):,}","sub":""})
    if coc and coc in df.columns: kpis.append({"label":f"Total {coc}","value":fmt(df[coc].sum()),"sub":""})
    elif rc and rc in df.columns: kpis.append({"label":"Avg per Record","value":fmt(df[rc].mean()),"sub":""})
    else: kpis.append({"label":"Columns","value":str(len(df.columns)),"sub":""})
    if rc and coc and rc in df.columns and coc in df.columns:
        rt=df[rc].sum(); ct=df[coc].sum(); mg=(rt-ct)/rt*100 if rt>0 else 0
        kpis.append({"label":"Gross Margin","value":f"{mg:.1f}%","sub":f"Profit: {fmt(rt-ct)}"})
    elif rc and rc in df.columns: kpis.append({"label":"Maximum","value":fmt(df[rc].max()),"sub":""})
    else: kpis.append({"label":"Missing Values","value":str(int(df.isnull().sum().sum())),"sub":""})
    if cc and cc in df.columns: kpis.append({"label":f"Unique {cc}s","value":str(df[cc].nunique()),"sub":""})
    else: kpis.append({"label":"Numeric Cols","value":str(len(df.select_dtypes(include=np.number).columns)),"sub":""})

    charts = []
    if cc and rc and cc in df.columns and rc in df.columns:
        g=df.groupby(cc)[rc].sum().sort_values(ascending=False).head(10)
        charts.append(barchart(g.index.tolist(),g.values.tolist(),f"{rc} by {cc}"))
    else:
        nc=df.select_dtypes(include=np.number).columns[:8]; s=df[nc].sum().sort_values(ascending=False)
        charts.append(barchart(s.index.tolist(),s.values.tolist(),"Column Totals"))

    if dc and rc and dc in df.columns and rc in df.columns:
        try:
            df2=df.copy(); df2["_d"]=pd.to_datetime(df2[dc],errors="coerce")
            mo=df2.groupby(df2["_d"].dt.to_period("M"))[rc].sum().reset_index()
            mo["_d"]=mo["_d"].astype(str)
            charts.append(linechart(mo["_d"].tolist(),mo[rc].tolist(),f"{rc} Over Time"))
        except: charts.append({"data":[],"layout":{"title":{"text":"Trend unavailable"},"paper_bgcolor":"white"}})
    elif rc and rc in df.columns:
        vs=df[rc].dropna().tolist(); counts,edges=np.histogram(vs,bins=20)
        centers=[(edges[i]+edges[i+1])/2 for i in range(len(counts))]
        charts.append({"data":[{"type":"bar","x":[f"{c:,.0f}" for c in centers],"y":counts.tolist(),"marker":{"color":"#4361ee","opacity":0.8}}],"layout":{"title":{"text":f"{rc} Distribution","font":{"size":14,"color":"#1a1a2e","family":"Inter"}},"plot_bgcolor":"white","paper_bgcolor":"white","margin":{"l":10,"r":20,"t":50,"b":60},"xaxis":{"tickangle":-45},"font":{"family":"Inter"}}})
    else: charts.append({"data":[],"layout":{"title":{"text":"No numeric data"},"paper_bgcolor":"white"}})

    if cc and rc and cc in df.columns and rc in df.columns:
        g2=df.groupby(cc)[rc].sum().sort_values(ascending=False).head(8)
        charts.append(piechart(g2.index.tolist(),g2.values.tolist(),f"{rc} Share by {cc}"))
    else: charts.append({"data":[],"layout":{"title":{"text":"No category data"},"paper_bgcolor":"white"}})

    nc2=df.select_dtypes(include=np.number).columns.tolist()
    if rc and coc and rc in df.columns and coc in df.columns:
        s2=df[[rc,coc]].dropna().sample(min(300,len(df)))
        charts.append(scatterchart(s2[rc].tolist(),s2[coc].tolist(),rc,coc,f"{rc} vs {coc}"))
    elif len(nc2)>=2:
        s2=df[[nc2[0],nc2[1]]].dropna().sample(min(300,len(df)))
        charts.append(scatterchart(s2[nc2[0]].tolist(),s2[nc2[1]].tolist(),nc2[0],nc2[1],f"{nc2[0]} vs {nc2[1]}"))
    else: charts.append({"data":[],"layout":{"title":{"text":"Insufficient data"},"paper_bgcolor":"white"}})

    num_df=df.select_dtypes(include=np.number); stats=[]
    for col in num_df.columns:
        d=num_df[col].describe()
        stats.append({"column":col,"count":int(d["count"]),"mean":round(float(d["mean"]),2),"std":round(float(d["std"]),2),"min":round(float(d["min"]),2),"max":round(float(d["max"]),2)})

    miss=int(df.isnull().sum().sum()); cp=round((1-miss/(len(df)*len(df.columns)))*100,1)
    mc=[{"column":c,"missing":int(df[c].isnull().sum()),"pct":round(df[c].isnull().sum()/len(df)*100,1)} for c in df.columns if df[c].isnull().sum()>0]
    mc.sort(key=lambda x:x["pct"],reverse=True)
    return {"kpis":kpis,"charts":charts,"insights":get_insights(df,rc,dc,cc,coc),"stats":stats,"quality":{"rows":len(df),"cols":len(df.columns),"missing_total":miss,"completeness":cp,"missing_cols":mc[:10]},"columns":df.columns.tolist(),"numeric_columns":num_df.columns.tolist(),"string_columns":df.select_dtypes(include=object).columns.tolist()}

@app.route("/")
def index(): return render_template("index.html")

@app.route("/app")
@app.route("/workspace")
def workspace(): return render_template("app.html")

@app.route("/health")
def health(): return jsonify({"status":"ok"})

@app.route("/api/sample")
def sample():
    np.random.seed(42)
    months=pd.date_range("2024-01-01",periods=24,freq="MS")
    clients=["Acme Corp","TechStart","GlobalTrade","LocalMart","FastGrow","BrightMedia","CityLogistics","PrimeRetail"]
    base={"Acme Corp":85000,"TechStart":45000,"GlobalTrade":120000,"LocalMart":30000,"FastGrow":60000,"BrightMedia":25000,"CityLogistics":55000,"PrimeRetail":40000}
    rows=[{"Month":m.strftime("%Y-%m"),"Client":c,"Revenue":round(max(np.random.normal(base[c],8000),5000),0),"Cost":round(max(np.random.normal(base[c],8000),5000)*np.random.uniform(0.55,0.72),0),"Segment":np.random.choice(["Enterprise","Growth","Starter"],p=[0.3,0.4,0.3])} for m in months for c in clients]
    df=pd.DataFrame(rows)
    result=run_analysis(df,"Revenue","Month","Client","Cost")
    result["filename"]="sample_sales.csv"; result["rows"]=len(df); result["detected"]={"revenue_col":"Revenue","date_col":"Month","category_col":"Client","cost_col":"Cost"}
    return jsonify(result)

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    file=request.files["file"]; fname=file.filename.lower()
    try:
        df=pd.read_csv(file) if fname.endswith(".csv") else pd.read_excel(file)
        if len(df)==0: return jsonify({"error":"Empty file"}),400
    except Exception as e: return jsonify({"error":str(e)}),400
    detected=detect_cols(df)
    result=run_analysis(df,detected.get("revenue_col"),detected.get("date_col"),detected.get("category_col"),detected.get("cost_col"))
    result["filename"]=file.filename; result["rows"]=len(df); result["detected"]=detected
    return jsonify(result)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    file=request.files["file"]
    try: df=pd.read_csv(file) if file.filename.lower().endswith(".csv") else pd.read_excel(file)
    except Exception as e: return jsonify({"error":str(e)}),400
    rc=request.form.get("revenue_col") or None; dc=request.form.get("date_col") or None
    cc=request.form.get("category_col") or None; coc=request.form.get("cost_col") or None
    result=run_analysis(df,rc,dc,cc,coc)
    result["filename"]=file.filename; result["rows"]=len(df)
    return jsonify(result)

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port,debug=os.environ.get("DEBUG","false")=="true")
