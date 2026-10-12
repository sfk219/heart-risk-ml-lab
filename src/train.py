"""Educational experiment with synthetic mathematical data only; NOT a medical predictor."""
import argparse,json
from pathlib import Path
import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score
def experiment(n=600):
    if n<100:raise ValueError("Use at least 100 synthetic samples")
    x,y=make_classification(n_samples=n,n_features=8,n_informative=5,n_redundant=1,random_state=42)
    xt,xv,yt,yv=train_test_split(x,y,stratify=y,test_size=.25,random_state=42)
    candidates={"logistic":make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)),
                "forest":RandomForestClassifier(n_estimators=80,random_state=42)}
    scores={k:cross_val_score(m,xt,yt,cv=StratifiedKFold(3,shuffle=True,random_state=42),scoring="roc_auc").mean() for k,m in candidates.items()}
    selected=max(scores,key=scores.get);model=candidates[selected].fit(xt,yt)
    probability=model.predict_proba(xv)[:,1];predicted=model.predict(xv)
    metrics={"data":"synthetic nonclinical","model":selected,"train_rows":len(yt),"test_rows":len(yv),
             "roc_auc":round(float(roc_auc_score(yv,probability)),3),
             "f1":round(float(f1_score(yv,predicted)),3),
             "precision":round(float(precision_score(yv,predicted)),3),
             "recall":round(float(recall_score(yv,predicted)),3)}
    return model,metrics
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",default="artifacts");args=p.parse_args()
    model,metrics=experiment();dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,dest/"model.joblib");(dest/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
