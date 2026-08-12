import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model and tokenizer from HuggingFace Hub
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("utkarshgarg29/amazon-bert-sentiment")
    model = DistilBertForSequenceClassification.from_pretrained("utkarshgarg29/amazon-bert-sentiment")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def predict_sentiment(text):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][prediction].item()
    label = "Positive" if prediction == 1 else "Negative"
    return label, confidence

# UI
st.title("Amazon Review Sentiment Classifier")
st.write("Fine-tuned DistilBERT model — enter a product review to see its predicted sentiment.")

review = st.text_area("Enter a review:", height=120)

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        label, confidence = predict_sentiment(review)
        if label == "Positive":
            st.success(f"**{label}** ({confidence:.1%} confidence)")
        else:
            st.error(f"**{label}** ({confidence:.1%} confidence)")
