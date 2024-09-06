import json
import os
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

# Função para buscar o preço de uma ação usando a API do Yahoo Finance
def fetch_stock_price(ticket):
    try:
        stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
        if stock.empty:
            raise ValueError("Nenhum dado encontrado para o código de ação fornecido.")
        return stock
    except Exception as e:
        st.error(f"Erro ao buscar dados para {ticket}: {str(e)}")
        return None

# Função para plotar os dados históricos de preços das ações para comparação
def plot_stock_price_comparison(stock_data_list, tickets):
    plt.figure(figsize=(10, 5))
    for stock_data, ticket in zip(stock_data_list, tickets):
        plt.plot(stock_data.index, stock_data['Close'], label=f'{ticket} Preço de Fechamento')
    
    plt.title(f"Comparação de Preços Históricos de Ações")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)  # Exibe o gráfico no Streamlit

# Verificação se a chave da API OpenAI está carregada
if "OPENAI_API_KEY" not in st.secrets:
    st.error("A chave 'OPENAI_API_KEY' não foi encontrada em secrets.toml.")
else:
    # Definindo o modelo de linguagem LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=st.secrets['OPENAI_API_KEY'])

    # Configuração do agente para análise de preços de ações
    stockPriceAnalyst = Agent(
        role="Analista Sênior de Preços de Ações",
        goal="Encontre o preço da ação {ticket} e analise tendências",
        backstory="Você é altamente experiente em analisar o preço de uma ação específica e fazer previsões sobre seu preço futuro.",
        verbose=True,
        llm=llm,
        max_iter=5,
        memory=True,
        tools=[yahoo_finance_tool],
        allow_delegation=False
    )

    # Configuração da interface do Streamlit para entrada de dados do usuário
    with st.sidebar:
        st.header('Digite o código das ações para pesquisa')

        with st.form(key='research_form'):
            # Adicionar múltiplas seleções
            stock_symbols = st.multiselect(
                "Selecione os códigos das ações para comparar",
                options=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],  # Você pode adicionar mais opções aqui
                default=["AAPL"]
            )
            submit_button = st.form_submit_button(label="Pesquisar")

    if submit_button:
        if not stock_symbols:
            st.error("Por favor, selecione pelo menos uma ação")
        else:
            stock_data_list = []
            valid_symbols = []
            
            # Busca os dados de cada ação selecionada
            for symbol in stock_symbols:
                stock_data = fetch_stock_price(symbol)
                if stock_data is not None:
                    stock_data_list.append(stock_data)
                    valid_symbols.append(symbol)

            # Exibe o gráfico comparativo se houver dados válidos
            if valid_symbols:
                st.subheader(f"Comparação de Preços Históricos para {', '.join(valid_symbols)}")
                plot_stock_price_comparison(stock_data_list, valid_symbols)
            else:
                st.error("Nenhum dado encontrado para os códigos de ações fornecidos.")
