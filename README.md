# LLM Supported SEC Report Analyzer Chat Assistant

## Project Overview

The **LLM Supported SEC Report Analyzer Chat Assistant** is designed to simplify the analysis of critical financial documents, such as earning call transcripts, SEC filings, and financial statements. This project aims to assist financial analysts and investors by providing a tool that can quickly extract and analyze data from these resources.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Goal](#goal)
- [User Interface](#user-interface)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Tools and Features](#tools-and-features)
  - [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
  - [Financial Data Search Tool](#financial-data-search-tool)
  - [Earning Call Analyzer Tool](#earning-call-analyzer-tool)
  - [Text Embedding](#text-embedding)
  - [Stock Price Prediction Tool](#stock-price-prediction-tool)
  - [Sentiment Scoring](#sentiment-scoring)
  - [Stock Price Visualization Tools](#stock-price-visualization-tools)
  - [Generate Report Tool](#generate-report-tool)
- [Tool Testing and Evaluation](#tool-testing-and-evaluation)

## Problem Statement

Analyzing financial documents such as earning call transcripts, SEC filings, and financial statement tables is complex and inconvenient. There is a need for a tool that can assist in extracting and analyzing crucial data from these sources efficiently.

## Goal

The primary goal of our application is to help financial analysts and investors by:
- Quickly bringing desired data from financial statement tables and earning call transcripts.
- Assisting users in gaining insights from SEC filings and earnings call reports.
- Providing stock price visualizations and predictions.

## User Interface
![image](https://github.com/selimcavas/LLM-SEC-Report-Analyzer/assets/92586913/0ca52ce5-b97c-46ff-ac57-9047f22b3eec)

Our application features a conversation-based chatbot with a graphical user interface (GUI), enabling users to extract data from the following resources:
- **Yahoo Finance Data:** Income statement, Balance Sheet, and Cash Flow.
- **SEC Filings:** 10-K and 10-Q Reports.
- **Earning Call Transcripts:** Transcript texts from NASDAQ 100 companies.

## System Architecture
![System Architecture](https://github.com/selimcavas/LLM-SEC-Report-Analyzer/assets/92586913/de5fff29-d560-4495-b9fb-fac24e75eff8)


## Technology Stack

- Python
- Streamlit
- Pinecone
- LangChain
- Tensorflow
- Mixtral 8x7B Instruct LLM

## Tools and Features

### Retrieval Augmented Generation (RAG)

Since LLM models have a knowledge cutoff and cannot access up-to-date information, we use the RAG approach to enable the LLM to work on recent data.

### Financial Data Search Tool

The text2SQL tool enables the LLM to analyze and execute queries over the SQL Database.

### Earning Call Analyzer Tool

This tool enables the LLM to quickly bring crucial insights over the transcripts retrieved by similarity search.

### Text Embedding

Each earning call transcript file is split into chunks and embedded using OpenAI’s “text-embedding-ada-002” model, then stored in the Pinecone Index.

### Stock Price Prediction Tool

Helps users understand trends and future movements of stock prices by using a sentiment score-supported LSTM.

### Sentiment Scoring

The FinBERT-tone model, a fine-tuned version of FinBERT, analyzes the sentiment of financial text, classifying it as positive, negative, or neutral based on 10,000 manually annotated sentences from analyst reports.

### Stock Price Visualization Tools

- Visualizes stock prices in the desired range and analyzes changes.
- Return Comparison tool compares returns of multiple tickers in one graph with an analysis comment.

### Generate Report Tool

Combines all chat histories and produces a concise PDF report, analyzing and providing insights about the future of a company.

## Tool Testing and Evaluation

- **Challenges:** Our LLM may sometimes hallucinate, causing drops in accuracy. Response times may increase due to the longevity of generated answers or connections made.
- **Evaluation Metrics:** 
  - **Only LSTM:** Uses stock prices only.
  - **Sentiment Supported LSTM:** Uses both stock prices and sentiment scores.
    

  | Metric  | Only LSTM | Sentiment Supported LSTM |
  |---------|------------|---------------------------|
  | Loss    | 0.010256   | 0.003475                  |
  | MAE     | 0.030057   | 0.040420                  |
  | MAPE    | 0.081722   | 0.083317                  |
  | R2      | 0.822231   | 0.676072                  |

---

Project by:
- Salih Metin Arkanöz
- Selim Çavaş
