import os
import streamlit as st
import yfinance as yf
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from groq import Groq
import plotly.graph_objects as go
from dotenv import load_dotenv
from datetime import datetime, timedelta

#load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# use news API to get news articles about the company
def get_analyst_news(company_name):
    today = datetime.now()
    one_week_ago = today - timedelta(days=7)
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f'"{company_name}"',  
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "from": one_week_ago.strftime("%Y-%m-%d"),
        "to": today.strftime("%Y-%m-%d"),
    }
    response = requests.get(url, params=params)
    articles = response.json().get("articles", [])

    # filter relevant articles
    relevant_articles = [
        article
        for article in articles
        if company_name.lower() in ((article.get("title") or "") + (article.get("description") or "")).lower()
    ]
    return relevant_articles

# use Groq Llama 3.3 70b to summarize news articles
def summarize_analyst_findings(articles):
    if not articles:
        return "No analyst-related articles found."
    combined_text = "\n".join(
        f"Title: {article['title']}\nDescription: {article.get('description', '')}"
        for article in articles
    )
    prompt = (
        f"Summarize the following articles discussing the performance of the stock:\n\n{combined_text}"
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# make stock graph with hover feature
def plot_stock_prices_interactive(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="6mo")  

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name=f"{ticker} Stock Price",
            hovertemplate="<b>Date:</b> %{x}<br><b>Price:</b> $%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"{ticker} Stock Price (Last 6 Months)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        showlegend=True,
        hovermode="x unified"  
    )

    st.plotly_chart(fig)
st.title("📈 Stock Financials and Analyst News Analyzer")

ticker = st.text_input("Enter a stock ticker (e.g., AAPL):", value="NVDA", key="stock_ticker_input")
company_name = st.text_input("Enter company name (e.g., Nvidia):", value="Nvidia", key="company_name_input")

if st.button("Analyze"):
    st.subheader("1️⃣ Stock Price Graph")
    plot_stock_prices_interactive(ticker)

    st.subheader("2️⃣ Analyst-Related News Articles")
    analyst_articles = get_analyst_news(company_name)

    if analyst_articles:
        st.write(f"Found {len(analyst_articles)} relevant analyst articles.")
        for article in analyst_articles[:5]:  
            st.write(f"**📌 Title:** {article['title']}")
            st.write(f"**📰 Source:** {article['source']['name']}")
            st.write(f"**📅 Published At:** {article['publishedAt']}")
            st.write(f"**🔗 URL:** [Read more]({article['url']})")

        st.subheader("3️⃣ Summary of Analyst Findings")
        summary = summarize_analyst_findings(analyst_articles)
        st.write(summary)
    else:
        st.write(f"⚠️ No relevant articles found for {company_name}.")
