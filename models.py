import joblib
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

def load_naive_bayes(model_path="naive_bayes_model.pkl", vectorizer_path="tfidf_vectorizer.pkl"):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_naive_bayes(texts, model, vectorizer):
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    confidences = [round(max(prob) * 100, 2) for prob in probs]
    return list(preds), confidences

def load_indobert(model_dir="indobert_sentiment_model_full"):
    tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained(f"{model_dir}/model")
    return model, tokenizer

def predict_indobert(texts, model, tokenizer):
    labels_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    results = []
    confidences = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        
        results.extend([labels_map[p.item()] for p in preds])
        confidences.extend([round(p.item() * 100, 2) for p in max_probs])
        
    return results, confidences

def get_candidate_gemini_models(api_key):
    genai.configure(api_key=api_key)
    candidates = []
    try:
        supported = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        priorities = ['gemini-2.5-flash', 'gemini-3.5-flash']
        
        for p in priorities:
            for s in supported:
                if p in s.lower() and s not in candidates:
                    candidates.append(s)
                    
        #for s in supported:
        #    if s not in candidates:
        #        candidates.append(s)
    except Exception as e:
        print(f"Error checking available models: {e}")
        
    #fallbacks = ('gemini-3.5-flash')
    #for fb in fallbacks:
    #    if fb not in candidates:
    #        candidates.append(fb)
    if not candidates:
        candidates = ['models/gemini-3.5-flash']
            
    return candidates

def predict_gemini(texts, api_key):
    genai.configure(api_key=api_key)
    candidates = get_candidate_gemini_models(api_key)
    
    working_model_name = None
    results = []
    if not texts:
        return results
        
    for text in texts:
        prompt = f"""
        Anda adalah asisten AI yang ahli dalam analisis sentimen, khususnya untuk ulasan aplikasi mobile.
        Tugas Anda adalah mengklasifikasikan sentimen dari ulasan aplikasi JKN Mobile berikut.
        Label yang diizinkan hanya: Positif, Netral, atau Negatif.
        Jawab hanya dengan salah satu dari label tersebut, tanpa teks tambahan.
        
        Ulasan: "{text}"
        Sentimen:
        """
        if working_model_name is None:
            success = False
            for model_name in candidates:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    working_model_name = model_name
                    pred = response.text.strip()
                    if "positif" in pred.lower():
                        results.append("Positif")
                    elif "negatif" in pred.lower():
                        results.append("Negatif")
                    elif "netral" in pred.lower():
                        results.append("Netral")
                    else:
                        results.append("Netral")
                    success = True
                    break
                except Exception as e:
                    print(f"Kandidat {model_name} gagal ({e}), mencoba model berikutnya...")
                    continue
            if not success:
                results.append("Error")
        else:
            try:
                model = genai.GenerativeModel(working_model_name)
                response = model.generate_content(prompt)
                pred = response.text.strip()
                if "positif" in pred.lower():
                    results.append("Positif")
                elif "negatif" in pred.lower():
                    results.append("Negatif")
                elif "netral" in pred.lower():
                    results.append("Netral")
                else:
                    results.append("Netral")
            except Exception as e:
                print(f"Error predicting with working model {working_model_name}: {e}")
                results.append("Error")
                time.sleep(1)
                
    return results

def predict_gemini_single_with_reasoning(text, api_key):
    genai.configure(api_key=api_key)
    candidates = get_candidate_gemini_models(api_key)
    
    prompt = f"""
Anda adalah asisten AI yang ahli dalam analisis sentimen, khususnya untuk ulasan aplikasi mobile JKN Mobile.
Tugas Anda adalah:
1. Mengklasifikasikan sentimen dari ulasan aplikasi berikut ke dalam salah satu label: Positif, Netral, atau Negatif.
2. Memberikan penjelasan/alasan singkat mengapa ulasan tersebut diklasifikasikan ke dalam label tersebut berdasarkan kata-kata atau makna yang terkandung di dalamnya.

Format keluaran HARUS persis seperti berikut (tanpa markdown atau karakter tambahan di luar format):
Sentimen: [Positif/Netral/Negatif]
Alasan: [Penjelasan singkat 1-2 kalimat]

Ulasan: "{text}"
"""
    last_error = None
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            pred_text = response.text.strip()
            
            sentiment = "Netral"
            reasoning = "Alasan tidak dapat diparse dengan baik, namun analisis berhasil dilakukan."
            
            for line in pred_text.split('\n'):
                line_clean = line.strip()
                if line_clean.lower().startswith("sentimen:"):
                    val = line_clean.split(":", 1)[1].strip()
                    if "positif" in val.lower():
                        sentiment = "Positif"
                    elif "negatif" in val.lower():
                        sentiment = "Negatif"
                    elif "netral" in val.lower():
                        sentiment = "Netral"
                elif line_clean.lower().startswith("alasan:"):
                    reasoning = line_clean.split(":", 1)[1].strip()
                    
            if reasoning == "Alasan tidak dapat diparse dengan baik, namun analisis berhasil dilakukan." and len(pred_text.split('\n')) > 1:
                other_lines = [l.strip() for l in pred_text.split('\n') if not l.strip().lower().startswith("sentimen:")]
                if other_lines:
                    reasoning = " ".join(other_lines).replace("Alasan:", "").strip()
                    
            reasoning += f"\n\n*(Model API: {model_name})*"
            return sentiment, reasoning
        except Exception as e:
            last_error = e
            print(f"Kandidat {model_name} gagal ({e}), mencoba model berikutnya...")
            continue
            
    return "Error", f"Gagal memanggil semua kandidat model Gemini. Error terakhir: {str(last_error)}"



