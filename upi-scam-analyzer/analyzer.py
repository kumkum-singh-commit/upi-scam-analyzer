import re, joblib, numpy as np, cv2, tldextract
from urllib.parse import urlparse, parse_qs
_tld = tldextract.TLDExtract(suffix_list_urls=())

text_model = joblib.load("models/text_model.pkl")
url_model = joblib.load("models/url_model.pkl")

RULES = [
    (r"\b(otp|upi pin|mpin|cvv|pin)\b", 20, "Mentions OTP/PIN/CVV (banks never ask you to share these)"),
    (r"kyc|account.{0,30}(block|suspend|freeze)|(block|suspend|freeze).{0,30}account", 20, "Fake KYC / account-blocked pattern"),
    (r"urgent|immediately|turant|within \d+ ?(hours|hrs|minutes)|last chance|expire", 10, "Creates urgency"),
    (r"collect request|approve.{0,20}request|enter.{0,20}pin.{0,20}receive", 30, "Collect-request trick: PIN is for PAYING, never receiving"),
    (r"lottery|you (have )?won|cashback|reward|prize|congratulations", 15, "Too-good-to-be-true reward"),
    (r"anydesk|teamviewer|quicksupport|\.apk|install.{0,15}app", 25, "Asks you to install a remote-access/unknown app"),
    (r"electricity.{0,40}(disconnect|cut)|(disconnect|cut).{0,40}electricity", 20, "Electricity-disconnection scam pattern"),
    (r"part[- ]?time|work from home|earn.{0,15}(per day|/day|daily)|telegram", 15, "Fake job/task offer pattern"),
]
SHORTENERS = ("bit.ly", "tinyurl.com", "cutt.ly", "rb.gy", "is.gd", "t.co", "shorturl.at")
BRANDS = {"sbi": "sbi.co.in", "hdfc": "hdfcbank.com", "icici": "icicibank.com",
          "paytm": "paytm.com", "phonepe": "phonepe.com", "irctc": "irctc.co.in",
          "axis": "axisbank.com", "gpay": "pay.google.com"}
BAD_TLDS = {"xyz", "top", "click", "online", "site", "icu", "live", "shop", "cfd", "buzz"}

def find_urls(text):
    return re.findall(r"(?:https?://|www\.)[^\s]+|\b[\w-]+\.(?:xyz|top|click|in|com|co|site|icu|live)\S*", text, re.I)

def analyze_url(url):
    pts, why = 0, []
    p = urlparse(url if "://" in url else "http://" + url)
    host = (p.netloc or "").lower()
    ext = tldextract.extract(url)
    reg = f"{ext.domain}.{ext.suffix}"
    prob = float(url_model.predict_proba([url])[0][1])
    if prob > 0.7:
        pts += 25; why.append(f"URL pattern looks like known phishing links (model: {prob:.0%})")
    if re.fullmatch(r"[\d.]+(:\d+)?", host):
        pts += 25; why.append("Uses a raw IP address instead of a domain")
    if p.scheme == "http":
        pts += 8; why.append("Not using HTTPS")
    if ext.suffix.split(".")[-1] in BAD_TLDS:
        pts += 15; why.append(f"Suspicious domain ending .{ext.suffix}")
    if "@" in url:
        pts += 15; why.append("Contains '@' (hides the real destination)")
    if host in SHORTENERS or reg in SHORTENERS:
        pts += 15; why.append("Shortened link hides the real destination")
    for b, real in BRANDS.items():
        if b in host and reg != real:
            pts += 30; why.append(f"Pretends to be '{b}' but domain is {reg}, not {real}"); break
    if host.count("-") >= 3 or len(url) > 90:
        pts += 8; why.append("Unusually long or hyphen-heavy URL")
    return pts, why

def analyze_upi(payload):
    pts, why = 0, []
    q = parse_qs(urlparse(payload).query)
    pa, pn, am = q.get("pa", [""])[0], q.get("pn", [""])[0], q.get("am", [""])[0]
    why.append(f"UPI payment QR -> pays to '{pa}' (name: '{pn}'). This QR makes YOU pay.")
    pts += 15
    if am:
        pts += 15; why.append(f"Amount Rs {am} is pre-filled. Scammers use this in 'receive money' tricks")
    if not pn:
        pts += 10; why.append("No payee name shown")
    return pts, why

def analyze(text="", qr_payload=None):
    points, reasons = 0, []
    ml = 0.0
    if text.strip():
        ml = float(text_model.predict_proba([text])[0][1])
        if ml > 0.5:
            reasons.append(f"Message wording is similar to known scam messages (model: {ml:.0%})")
        for pat, w, msg in RULES:
            if re.search(pat, text, re.I):
                points += w; reasons.append(msg)
        for u in find_urls(text):
            pt, wy = analyze_url(u); points += pt; reasons += [f"[{u}] {r}" for r in wy]
    if qr_payload:
        if qr_payload.lower().startswith("upi://"):
            pt, wy = analyze_upi(qr_payload)
        else:
            pt, wy = analyze_url(qr_payload)
        points += pt; reasons += wy
    score = int(min(100, 60 * ml + points))
    label = "SCAM" if score >= 60 else "SUSPICIOUS" if score >= 30 else "LIKELY SAFE"
    return score, label, reasons

def decode_qr(pil_image):
    img = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    data, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
    return data or None