import streamlit as st
import pandas as pd
import time
import io
import plotly.express as px
import plotly.graph_objects as go
import importlib
import models
importlib.reload(models)
from models import load_naive_bayes, predict_naive_bayes, load_indobert, predict_indobert, predict_gemini, predict_gemini_single_with_reasoning

st.set_page_config(page_title="JKN Sentiment AI", page_icon="🔮", layout="wide")

# Increase base font size slightly (Streamlit requires CSS for this specific tweak)
st.markdown("""
<style>
    html, body, [class*="st-"] {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_nb_model():
    return load_naive_bayes()

@st.cache_resource
def get_indobert_model():
    return load_indobert()

def convert_df(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    processed_data = output.getvalue()
    return processed_data

def main():
    st.title("🔮 JKN Mobile Sentiment Analyzer")
    st.markdown("Analyze user feedback efficiently with state-of-the-art AI models.")
    
    tab1, tab2 = st.tabs(["📁 Batch Excel Analysis", "💬 Single Review Analysis"])
    
    with tab1:
        with st.container(border=True):
            st.markdown("### ⚙️ Configuration")
            uploaded_file = st.file_uploader("Upload Excel File (must contain 'id' & 'comment')", type=['xlsx'])
            
            model_choice = st.selectbox("Select Model", ["Naive Bayes", "IndoBERT"])
            run_button = st.button("🚀 Run Analysis")

        with st.container(border=True):
            st.markdown("### 📊 Results Preview")
            
            if uploaded_file is not None:
                df = pd.read_excel(uploaded_file)
                
                if 'id' not in df.columns or 'comment' not in df.columns:
                    st.error("Error: The uploaded Excel file must contain 'id' and 'comment' columns.")
                    return
                
                st.dataframe(df.head(), width="stretch")
                
                if run_button:
                    texts = df['comment'].astype(str).tolist()
                    
                    with st.status(f"Analyzing sentiments using {model_choice}...", expanded=True) as status:
                        start_time = time.time()
                        if model_choice == "Naive Bayes":
                            nb_model, vectorizer = get_nb_model()
                            st.write("Extracting features and predicting...")
                            preds, confs, _ = predict_naive_bayes(texts, nb_model, vectorizer)
                            
                        elif model_choice == "IndoBERT":
                            bert_model, tokenizer = get_indobert_model()
                            st.write("Running Transformer inference...")
                            preds, confs, _ = predict_indobert(texts, bert_model, tokenizer)
                            
                        df['sentiment_result'] = preds
                        df['confidence'] = confs
                        end_time = time.time()
                        inference_time = end_time - start_time
                        throughput = len(texts) / inference_time if inference_time > 0 else 0
                        avg_confidence = sum(confs)/len(confs) if confs else 0
                        status.update(label="Analysis Complete!", state="complete", expanded=False)
                    
                    st.toast(f"Analysis Complete in {inference_time:.2f} seconds!", icon="✅")
                    
                    st.markdown("### ⏱️ Performance Metrics")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Total Reviews", f"{len(texts)}")
                    col_m2.metric("Avg Confidence", f"{avg_confidence:.2f}%")
                    col_m3.metric("Throughput", f"{throughput:.1f} rows/s")
                    st.markdown("---")
                    
                    st.dataframe(
                        df.head(10), 
                        width="stretch",
                        column_config={
                            "confidence": st.column_config.ProgressColumn(
                                "Confidence",
                                format="%.2f%%",
                                min_value=0,
                                max_value=100,
                            ),
                            "sentiment_result": st.column_config.TextColumn(
                                "Sentiment"
                            )
                        }
                    )
                    
                    # Visualizations
                    st.markdown("### 📈 Data Distribution")
                    
                    # Bar Chart
                    sentiment_counts = df['sentiment_result'].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']
                    
                    fig_bar = px.bar(
                        sentiment_counts, 
                        x='Sentiment', 
                        y='Count', 
                        color='Sentiment',
                        color_discrete_map={
                            'Positif': '#10b981',
                            'Netral': '#64748b',
                            'Negatif': '#ef4444',
                            'Error': '#f59e0b'
                        },
                        title="Sentiment Count"
                    )
                    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_bar, width="stretch")
                    
                    # Pie Chart
                    fig_pie = px.pie(
                        sentiment_counts, 
                        values='Count', 
                        names='Sentiment',
                        color='Sentiment',
                        color_discrete_map={
                            'Positif': '#10b981',
                            'Netral': '#64748b',
                            'Negatif': '#ef4444',
                            'Error': '#f59e0b'
                        },
                        hole=0.4,
                        title="Sentiment Distribution"
                    )
                    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                    st.plotly_chart(fig_pie, width="stretch")
                    
                    # Confidence Distribution
                    if 'confidence' in df.columns:
                        st.markdown("### 🎯 Confidence Breakdown")
                        
                        df['Confidence Level'] = pd.cut(df['confidence'], bins=[0, 50, 80, 100], labels=['Low', 'Medium', 'High'])
                        fig_tree = px.treemap(
                            df, 
                            path=['sentiment_result', 'Confidence Level'], 
                            title="Treemap of Sentiment by Confidence",
                            color='sentiment_result',
                            color_discrete_map={
                                'Positif': '#10b981',
                                'Netral': '#64748b',
                                'Negatif': '#ef4444',
                                '(?)': '#ffffff'
                            }
                        )
                        fig_tree.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                        st.plotly_chart(fig_tree, width="stretch")
                        
                        st.markdown("### 📊 Confidence Score Distribution")
                        fig_hist = px.histogram(
                            df, 
                            x="confidence", 
                            color="sentiment_result",
                            nbins=20,
                            color_discrete_map={
                                'Positif': '#10b981',
                                'Netral': '#64748b',
                                'Negatif': '#ef4444',
                                'Error': '#f59e0b'
                            }
                        )
                        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                        st.plotly_chart(fig_hist, width="stretch")
                    
                    # Download
                    st.markdown("### 💾 Export")
                    excel_data = convert_df(df)
                    st.download_button(
                        label="Download Analyzed Data (Excel)",
                        data=excel_data,
                        file_name="sentiment_analysis_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            else:
                st.info("Upload an Excel file to get started.")

    with tab2:
        st.markdown("### 💬 Single Review Sentiment Prediction & Reasoning")
        st.markdown("Test individual JKN Mobile user feedback directly. When using Gemini, you will also receive an AI-generated reasoning explaining why the review received its classification.")
        
        with st.container(border=True):
            single_text = st.text_area("Input Review Text", height=150, placeholder="Contoh: Aplikasi JKN Mobile sangat membantu untuk daftar antrean faskes 1, sukses selalu!")
            single_model_choice = st.selectbox("Select Model for Single Text", ["Naive Bayes", "IndoBERT", "Gemini 2.5 Flash"], key="single_model_choice")
            
            single_api_key = ""
            if single_model_choice == "Gemini 2.5 Flash":
                single_api_key = st.text_input("Enter Gemini API Key", type="password", key="single_api_key")
                
            single_run_button = st.button("🚀 Analyze Single Review")
            
        with st.container(border=True):
            st.markdown("### 🏷️ Prediction Result")
            if single_run_button:
                if not single_text.strip():
                    st.warning("Please enter some text to analyze.")
                elif single_model_choice == "Gemini 2.5 Flash" and not single_api_key:
                    st.warning("Please enter your Gemini API Key.")
                else:
                    with st.status(f"Analyzing with {single_model_choice}...", expanded=True) as status:
                        start_time = time.time()
                        res_confidence = None
                        res_all_probs = None
                        if single_model_choice == "Naive Bayes":
                            st.write("Predicting...")
                            nb_model, vectorizer = get_nb_model()
                            preds, confs, all_probs = predict_naive_bayes([single_text], nb_model, vectorizer)
                            res_sentiment = preds[0]
                            res_confidence = confs[0]
                            res_all_probs = all_probs[0]
                            res_reasoning = "Model Naive Bayes mengklasifikasikan sentimen ini berdasarkan probabilitas kata-kata (berbobot TF-IDF) yang muncul dalam ulasan."
                        elif single_model_choice == "IndoBERT":
                            st.write("Running transformer...")
                            bert_model, tokenizer = get_indobert_model()
                            preds, confs, all_probs = predict_indobert([single_text], bert_model, tokenizer)
                            res_sentiment = preds[0]
                            res_confidence = confs[0]
                            res_all_probs = all_probs[0]
                            res_reasoning = "Model IndoBERT mengklasifikasikan sentimen ini berdasarkan representasi kontekstual mendalam dari arsitektur transformer pre-trained pada bahasa Indonesia."
                        elif single_model_choice == "Gemini 2.5 Flash":
                            st.write("Calling Gemini API...")
                            res_sentiment, res_reasoning = predict_gemini_single_with_reasoning(single_text, single_api_key)
                            
                        end_time = time.time()
                        inference_time = end_time - start_time
                        status.update(label="Analysis Complete!", state="complete", expanded=False)
                        st.toast(f"Analysis complete in {inference_time:.3f}s", icon="✅")
                            
                        # Display result badge
                        if res_sentiment == "Positif":
                            st.success(f"**{res_sentiment}**")
                        elif res_sentiment == "Negatif":
                            st.error(f"**{res_sentiment}**")
                        elif res_sentiment == "Error":
                            st.warning(f"**{res_sentiment}**")
                        else:
                            st.info(f"**{res_sentiment}**")
                            
                        st.markdown("#### ⏱️ Latency")
                        st.info(f"Model inference took {inference_time:.3f} seconds.")
                        
                        st.markdown("#### 💡 Reasoning / Alasan")
                        st.info(res_reasoning)
                        
                        col_res1, col_res2 = st.columns(2)
                        
                        with col_res1:
                            if res_confidence is not None:
                                fig_gauge = go.Figure(go.Indicator(
                                    mode = "gauge+number",
                                    value = res_confidence,
                                    domain = {'x': [0, 1], 'y': [0, 1]},
                                    title = {'text': "Confidence Score (%)"},
                                    gauge = {
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "#3b82f6"},
                                        'steps': [
                                            {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.2)"},
                                            {'range': [50, 80], 'color': "rgba(245, 158, 11, 0.2)"},
                                            {'range': [80, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                                        ]
                                    }
                                ))
                                fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                                st.plotly_chart(fig_gauge, width="stretch")
                                
                        with col_res2:
                            if res_all_probs is not None:
                                categories = list(res_all_probs.keys())
                                values = list(res_all_probs.values())
                                
                                # Close the radar chart loop
                                categories.append(categories[0])
                                values.append(values[0])
                                
                                fig_radar = go.Figure()
                                fig_radar.add_trace(go.Scatterpolar(
                                      r=values,
                                      theta=categories,
                                      fill='toself',
                                      name='Probability',
                                      line_color='#8b5cf6',
                                      fillcolor='rgba(139, 92, 246, 0.4)'
                                ))
                                fig_radar.update_layout(
                                    title=dict(text="Probability Distribution", x=0.5, xanchor='center', font=dict(size=16)),
                                    polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.1)')),
                                    showlegend=False,
                                    height=250,
                                    margin=dict(l=10, r=10, t=40, b=10),
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='white'
                                )
                                st.plotly_chart(fig_radar, width="stretch")
            else:
                st.info("Enter review text on the left and click 'Analyze Single Review' to view results and reasoning.")

if __name__ == "__main__":
    main()
