import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ---------- TEXT MODEL ----------
sms = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1", "v2"]]
sms.columns = ["label", "text"]
sms["y"] = (sms.label == "spam").astype(int)
sms = sms[["text", "y"]]

ind = pd.read_csv("data/indian_scam.csv").rename(columns={"label": "y"})
ind_train, ind_test = train_test_split(ind, test_size=0.4, stratify=ind.y, random_state=42)

sms_train, sms_test = train_test_split(sms, test_size=0.2, stratify=sms.y, random_state=42)
train = pd.concat([sms_train, ind_train, ind_train])  # Indian data repeated to give it weight

text_model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
text_model.fit(train.text, train.y)

print("=== SMS test set (general) ===")
p = text_model.predict(sms_test.text)
print(classification_report(sms_test.y, p)); print(confusion_matrix(sms_test.y, p))

print("=== Indian test set ===")
p = text_model.predict(ind_test.text)
print(classification_report(ind_test.y, p)); print(confusion_matrix(ind_test.y, p))
joblib.dump(text_model, "models/text_model.pkl")

# ---------- URL MODEL ----------
u = pd.read_csv("data/phishing_site_urls.csv")
u["y"] = (u.Label == "bad").astype(int)
u = u.sample(100000, random_state=42)  # keeps training fast
utr, ute = train_test_split(u, test_size=0.2, stratify=u.y, random_state=42)

url_model = make_pipeline(
    TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=200000),
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
url_model.fit(utr.URL, utr.y)
print("=== URL test set ===")
p = url_model.predict(ute.URL)
print(classification_report(ute.y, p)); print(confusion_matrix(ute.y, p))
joblib.dump(url_model, "models/url_model.pkl")