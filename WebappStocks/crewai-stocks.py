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

# Função para plotar os dados históricos de preços das ações
def plot_stock_price(stock_data, ticket):
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data.index, stock_data['Close'], label='Preço de Fechamento')
    plt.title(f"Preço Histórico de Fechamento para {ticket}")
    plt.xlabel("Data")
    plt.ylabel("Preço")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)  # Exibe o gráfico no Streamlit

# Criação de uma ferramenta que usa a função fetch_stock_price para buscar dados de ações
yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Obtém preços de ações para {ticket} do último ano sobre uma empresa específica da API Yahoo Finance",
    func=lambda ticket: fetch_stock_price(ticket)
)

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

    # Configuração da tarefa para análise de preço das ações
    getStockPrice = Task(
        description="Analise o histórico de preços da ação {ticket} e crie uma análise de tendência - alta, baixa ou lateral",
        expected_output="Especifique a tendência atual do preço da ação - alta, baixa ou lateral. Exemplo: ação = 'AAPL, preço ALTA'",
        agent=stockPriceAnalyst
    )

    # Configuração da ferramenta de busca DuckDuckGo para análise de notícias
    search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

    # Configuração do agente para análise de notícias de ações
    newsAnalyst = Agent(
        role="Analista de Notícias de Ações",
        goal="Crie um resumo curto das notícias de mercado relacionadas à empresa da ação {ticket}. Especifique a tendência atual - alta, baixa ou lateral com o contexto das notícias. Para cada ativo solicitado, especifique um número entre 0 e 100, onde 0 é medo extremo e 100 é ganância extrema.",
        backstory="Você é altamente experiente em analisar as tendências do mercado e notícias, e acompanha ativos há mais de 10 anos.",
        verbose=True,
        llm=llm,
        max_iter=10,
        memory=True,
        tools=[search_tool],
        allow_delegation=False
    )

    # Configuração da tarefa para análise de notícias e criação de relatórios
    get_news = Task(
        description=f"""Tome a ação e sempre inclua BTC (se não for solicitado). Use a ferramenta de busca para pesquisar cada uma individualmente. A data atual é {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. Compile os resultados em um relatório útil""",
        expected_output="""Um resumo geral do mercado e um resumo de uma frase para cada ativo solicitado. Inclua uma pontuação de medo/ganância para cada ativo com base nas notícias. Use o formato: <ATIVO> <RESUMO BASEADO NAS NOTÍCIAS> <PREDIÇÃO DE TENDÊNCIA> <PONTUAÇÃO DE MEDO/GANÂNCIA>""",
        agent=newsAnalyst
    )

    # Configuração do agente para escrever relatórios analíticos
    stockAnalystWrite = Agent(
        role="Redator Sênior de Análise de Ações",
        goal="Analise as tendências de preço e notícias e escreva um informativo perspicaz, atraente e informativo de 3 parágrafos com base na ação {ticket}",
        backstory="Você é amplamente aceito como o melhor analista de ações do mercado.",
        verbose=True,
        llm=llm,
        max_iter=5,
        memory=True,
        allow_delegation=True
    )

    # Configuração da tarefa para escrever análises e relatórios
    writeAnalyses = Task(
        description="Use a tendência do preço da ação e o relatório de notícias da ação para criar uma análise e escrever um informativo sobre a empresa {ticket} que seja breve e destaque os pontos mais importantes.",
        expected_output="""Um informativo eloquente de 3 parágrafos formatado como markdown de forma fácil de ler. Deve conter:
        - Resumo executivo em 3 pontos
        - Introdução - estabeleça o cenário geral e desperte o interesse
        - Parte principal fornecendo a essência da análise, incluindo o resumo das notícias e as pontuações de medo/ganância
        - Resumo - principais fatos e previsão concreta de tendência futura - alta, baixa ou lateral.""",
        agent=stockAnalystWrite,
        context=[getStockPrice, get_news]
    )

    # Configuração da equipe (Crew) que gerencia os agentes e tarefas
    crew = Crew(
        agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
        tasks=[getStockPrice, get_news, writeAnalyses],
        verbose=2,
        process=Process.hierarchical,
        full_output=True,
        share_crew=False,
        manager_llm=llm,
        max_iter=15
    )

    # Inicia o processo de execução das tarefas configuradas para a equipe
    results = crew.kickoff(inputs={"ticket": "AAPL", "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

    # Exibe o resultado final após a execução de todas as tarefas
    final_output = results['final_output']

    # Configuração da interface do Streamlit para entrada de dados do usuário
    with st.sidebar:
        st.header('Digite o código da ação para pesquisa')

        with st.form(key='research_form'):
            topic = st.text_input("Selecione o código da ação", value="AAPL")  # Valor padrão AAPL
            submit_button = st.form_submit_button(label="Pesquisar")

    if submit_button:
        if not topic:
            st.error("Por favor, preencha o campo com o código da ação")
        else:
            # Busca os dados da ação
            stock_data = fetch_stock_price(topic)
            
            if stock_data is not None:
                # Exibe o gráfico dos dados históricos
                st.subheader(f"Dados Históricos para {topic}")
                plot_stock_price(stock_data, topic)
                
                # Inicia o processo com o código da ação fornecido pelo usuário
                with st.spinner('Buscando dados e realizando análises, por favor aguarde...'):
                    results = crew.kickoff(inputs={"ticket": topic, "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

                # Exibe os resultados da pesquisa para o usuário
                st.subheader("Resultados da sua pesquisa:")
                st.markdown(results['final_output'])


