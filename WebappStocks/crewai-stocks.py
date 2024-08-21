import json
import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# Função para buscar o preço de uma ação usando a API do Yahoo Finance
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock

# Criação de uma ferramenta que usa a função fetch_stock_price para buscar dados de ações
yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Fetches stocks prices for {ticket} from the last year about a specific company from Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket)
)

# Configurando a chave da API OpenAI a partir das variáveis de ambiente (método seguro de gerenciamento de chaves)
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Definindo o modelo de linguagem LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=st.secrets['OPENAI_API_KEY'])

# Configuração do agente para análise de preços de ações
stockPriceAnalyst = Agent(
    role="Senior stock price Analyst",  # Definindo o papel do agente
    goal="Find the {ticket} stock price and analyses trends",  # Definindo o objetivo do agente
    backstory="""You're highly experienced in analyzing the price of a specific stock and make predictions about its future price.""",  # Contexto para o agente
    verbose=True,  # Exibe informações detalhadas durante a execução
    llm=llm,  # Modelo de linguagem utilizado pelo agente
    max_iter=5,  # Número máximo de iterações que o agente pode realizar
    memory=True,  # O agente deve lembrar de informações de sessões anteriores
    tools=[yahoo_finance_tool],  # Ferramentas disponíveis para o agente
    allow_delegation=False  # O agente não pode delegar tarefas a outros agentes
)

# Configuração da tarefa para análise de preço das ações
getStockPrice = Task(
    description="Analyze the stock {ticket} price history and create a trend analysis of up, down or sideways",  # Descrição da tarefa
    expected_output="""Specify the current trend stock price - up, down or sideways.
    eg. stock= 'APPL, price UP'""",  # Resultado esperado da tarefa
    agent=stockPriceAnalyst  # Agente responsável por executar a tarefa
)

# Configuração da ferramenta de busca DuckDuckGo para análise de notícias
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

# Configuração do agente para análise de notícias de ações
newsAnalyst = Agent(
    role="Stock News Analyst",  # Definindo o papel do agente
    goal="""Create a short summary of the market news related to the stock {ticket} company. Specify the current trend – up, down or sideways with the news context. For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",  # Objetivo do agente
    backstory="""You're highly experienced in analyzing the market trends and news and have tracked assets for more than 10 years.

    You're also master level analyst in the traditional markets and have deep understanding of human psychology.

    You understand news, their titles and information, but you look at these with a healthy dose of skepticism. You consider also the source of the news articles.""",  # Contexto para o agente
    verbose=True,  # Exibe informações detalhadas durante a execução
    llm=llm,  # Modelo de linguagem utilizado pelo agente
    max_iter=10,  # Número máximo de iterações que o agente pode realizar
    memory=True,  # O agente deve lembrar de informações de sessões anteriores
    tools=[search_tool],  # Ferramentas disponíveis para o agente
    allow_delegation=False  # O agente não pode delegar tarefas a outros agentes
)

# Configuração da tarefa para análise de notícias e criação de relatórios
get_news = Task(
    description=f"""Take the stock and always include BTC to it (if not requested).
    Use the search tool to search each one individually.
    
    The current date is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    Compose the results into a helpful report""",  # Descrição da tarefa
    expected_output="""A summary of the overall market and one sentence summary for each request asset.
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,  # Resultado esperado da tarefa
    agent=newsAnalyst  # Agente responsável por executar a tarefa
)

# Configuração do agente para escrever relatórios analíticos
stockAnalystWrite = Agent(
    role="Senior Stock Analyst Writer",  # Definindo o papel do agente
    goal="""Analyze the trends price and news and write an insightful, compelling and informative 3 paragraph long newsletter based on the stock""",  # Objetivo do agente
    backstory="""You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories and narratives that resonate with wider audiences. You understand macro factors and combine multiple theories - eg, cycle theory and fundamental analyses. You're able to hold multiple opinions when analyzing anything.""",  # Contexto para o agente
    verbose=True,  # Exibe informações detalhadas durante a execução
    llm=llm,  # Modelo de linguagem utilizado pelo agente
    max_iter=5,  # Número máximo de iterações que o agente pode realizar
    memory=True,  # O agente deve lembrar de informações de sessões anteriores
    allow_delegation=True  # O agente pode delegar tarefas a outros agentes
)

# Configuração da tarefa para escrever análises e relatórios
writeAnalyses = Task(
    description="""Use the stock price trend and the stock news report to create an analysis and write the newsletter about the {ticket} company that is brief and highlights the most important points.
    Focus on the stock price trend, news, and fear/greed score. What are the near future considerations? Include the previous analyses of stock trend and news summary.""",  # Descrição da tarefa
    expected_output="""An eloquent 3 paragraphs newsletter formatted as markdown in an easy readable manner. It should contain:
    - 3 bullets executive summary
    - Introduction - set the overall picture and spike up the interest
    - main part provides the meat of the analysis including the news summary and fear/greed scores
    - summary - key facts and concrete future trend prediction - up, down or sideways.
    """,  # Resultado esperado da tarefa
    agent=stockAnalystWrite,  # Agente responsável por executar a tarefa
    context=[getStockPrice, get_news]  # Tarefas que fornecem contexto para essa tarefa
)

# Configuração da equipe (Crew) que gerencia os agentes e tarefas
crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWrite],  # Lista de agentes na equipe
    tasks=[getStockPrice, get_news, writeAnalyses],  # Lista de tarefas a serem executadas
    verbose=2,  # Nível de detalhamento das informações exibidas
    process=Process.hierarchical,  # Processo de execução das tarefas (hierárquico)
    full_output=True,  # Exibe a saída completa das tarefas
    share_crew=False,  # Define se a equipe pode ser compartilhada
    manager_llm=llm,  # Modelo de linguagem utilizado para gerenciar a equipe
    max_iter=15  # Número máximo de iterações para o processo completo
)

# Inicia o processo de execução das tarefas configuradas para a equipe
results = crew.kickoff(inputs={"ticket": "APPL", "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# Exibe o resultado final após a execução de todas as tarefas
results['final_output']

# Configuração da interface do Streamlit para entrada de dados do usuário
with st.sidebar:
    st.header('Enter the Stock to Research')  # Cabeçalho da barra lateral

    # Formulário para o usuário inserir o ticket da ação que deseja pesquisar
    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")  # Campo para entrada do ticket da ação
        submit_button = st.form_submit_button(label="Run Research")  # Botão para submeter o formulário

# Verifica se o botão de submissão foi clicado
if submit_button:
    # Exibe um erro se o campo de ticket estiver vazio
    if not topic:
        st.error("Please fill the ticket field")
    else:
        # Inicia o processo com o ticket fornecido pelo usuário
        results = crew.kickoff(inputs={"ticket": topic, "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

        # Exibe os resultados da pesquisa para o usuário
        st.subheader("Results of your research:")
        st.write(results['final_output'])
