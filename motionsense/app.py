from flask import Flask, render_template, request, jsonify
import joblib, numpy as np, os

app = Flask(__name__)
MODEL_PATH = "modelo_har_movil.pkl"
bundle = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

ACT_ES = {
    "WALKING":"Caminar",
    "WALKING_UPSTAIRS":"Subir escaleras",
    "WALKING_DOWNSTAIRS":"Bajar escaleras",
    "SITTING":"Sentarse",
    "STANDING":"Estar de pie",
    "LAYING":"Estar acostado",
}
ACT_ICON = {
    "WALKING":"🚶",
    "WALKING_UPSTAIRS":"⬆️",
    "WALKING_DOWNSTAIRS":"⬇️",
    "SITTING":"🪑",
    "STANDING":"🧍",
    "LAYING":"🛏️",
}

WINDOW_MS = 2560
STEP_MS = 1280
TARGET_SAMPLES = 128

def stats_1d(x):
    x=np.asarray(x,float)
    return [np.mean(x),np.std(x),np.min(x),np.max(x),np.median(x),
            np.percentile(x,25),np.percentile(x,75),
            np.sqrt(np.mean(x**2)),np.mean(np.abs(np.diff(x))),
            np.percentile(x,75)-np.percentile(x,25)]

def resample_window(samples,start_ms,end_ms,n=TARGET_SAMPLES):
    ss=[s for s in samples if start_ms <= s["t"] <= end_ms]
    if len(ss)<20: return None,None
    t=np.array([s["t"] for s in ss],float); t=t-t[0]
    if t[-1]<=0: return None,None
    target=np.linspace(0,t[-1],n)
    vals={}
    for key in ["ax","ay","az","gx","gy","gz"]:
        v=np.array([s.get(key,0.0) for s in ss],float)
        vals[key]=np.interp(target,t,v)
    acc=np.column_stack([vals["ax"],vals["ay"],vals["az"]])/9.80665
    gyro=np.deg2rad(np.column_stack([vals["gx"],vals["gy"],vals["gz"]]))
    return acc,gyro

def extract_features(acc,gyro):
    acc_mag=np.linalg.norm(acc,axis=1)
    gyro_mag=np.linalg.norm(gyro,axis=1)
    channels=[acc[:,0],acc[:,1],acc[:,2],gyro[:,0],gyro[:,1],gyro[:,2],acc_mag,gyro_mag]
    feats=[]
    for ch in channels: feats.extend(stats_1d(ch))
    return np.asarray(feats,float).reshape(1,-1)

def predict_window(X,model_name):
    if model_name=="tree":
        model=bundle["tree"]; pred=model.predict(X)[0]; proba=model.predict_proba(X)[0]
    else:
        model=bundle["knn"]; X=bundle["scaler"].transform(X); pred=model.predict(X)[0]; proba=model.predict_proba(X)[0]
    return str(pred),np.asarray(proba,float),[str(c) for c in model.classes_]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify(ok=bundle is not None)

@app.route("/predict",methods=["POST"])
def predict():
    if bundle is None:
        return jsonify(ok=False,error="El modelo no está disponible."),500

    data=request.get_json(force=True)
    samples=data.get("samples",[])
    model_name=data.get("model","knn")

    if len(samples)<50:
        return jsonify(ok=False,error=f"Se recibieron pocas muestras ({len(samples)})."),400

    try:
        t0=float(samples[0]["t"])
        norm=[]
        for s in samples:
            ns=dict(s); ns["t"]=float(s["t"])-t0; norm.append(ns)

        duration=float(norm[-1]["t"])
        if duration<WINDOW_MS:
            return jsonify(ok=False,error="La captura fue demasiado corta."),400

        results=[]; start=0.0
        while start+WINDOW_MS<=duration:
            acc,gyro=resample_window(norm,start,start+WINDOW_MS)
            if acc is not None:
                X=extract_features(acc,gyro)
                pred,proba,classes=predict_window(X,model_name)
                results.append((pred,proba,classes))
            start += STEP_MS

        if not results:
            return jsonify(ok=False,error="No se formaron ventanas válidas."),400

        classes=results[0][2]
        mean_proba=np.mean(np.vstack([r[1] for r in results]),axis=0)
        final_idx=int(np.argmax(mean_proba))
        final_pred=classes[final_idx]

        votes={}
        for pred,_,_ in results:
            es=ACT_ES.get(pred,pred)
            votes[es]=votes.get(es,0)+1

        probs={ACT_ES.get(c,c):float(p) for c,p in zip(classes,mean_proba)}
        gy=np.array([[s.get("gx",0),s.get("gy",0),s.get("gz",0)] for s in norm],float)
        gyro_detected=bool(np.mean(np.linalg.norm(gy,axis=1))>0.01)

        return jsonify(
            ok=True,
            prediction=final_pred,
            prediction_es=ACT_ES.get(final_pred,final_pred),
            prediction_icon=ACT_ICON.get(final_pred,"🤖"),
            confidence=float(mean_proba[final_idx]),
            probabilities=probs,
            samples_received=len(samples),
            windows_used=len(results),
            votes=votes,
            gyro_detected=gyro_detected
        )
    except Exception as e:
        return jsonify(ok=False,error=str(e)),500

if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)),debug=False)
