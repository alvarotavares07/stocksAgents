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
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock

# Função para plotar os dados históricos de preços das ações
def plot_stock_price(stock_data, ticket):
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data.index, stock_data['Close'], label='Close Price')
    plt.title(f"Historical Close Price for {ticket}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)  # Exibe o gráfico no Streamlit

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
    role="Senior stock price Analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory="You're highly experienced in analyzing the price of a specific stock and make predictions about its future price.",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    tools=[yahoo_finance_tool],
    allow_delegation=False
)

# Configuração da tarefa para análise de preço das ações
getStockPrice = Task(
    description="Analyze the stock {ticket} price history and create a trend analysis of up, down or sideways",
    expected_output="Specify the current trend stock price - up, down or sideways.\neg. stock= 'APPL, price UP'",
    agent=stockPriceAnalyst
)

# Configuração da ferramenta de busca DuckDuckGo para análise de notícias
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

# Configuração do agente para análise de notícias de ações
newsAnalyst = Agent(
    role="Stock News Analyst",
    goal="Create a short summary of the market news related to the stock {ticket} company. Specify the current trend – up, down or sideways with the news context. For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.",
    backstory="You're highly experienced in analyzing the market trends and news and have tracked assets for more than 10 years.",
    verbose=True,
    llm=llm,
    max_iter=10,
    memory=True,
    tools=[search_tool],
    allow_delegation=False
)

# Configuração da tarefa para análise de notícias e criação de relatórios
get_news = Task(
    description=f"""Take the stock and always include BTC to it (if not requested).
    Use the search tool to search each one individually.
    
    The current date is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    Compose the results into a helpful report""",
    expected_output="""A summary of the overall market and one sentence summary for each request asset.
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
    agent=newsAnalyst
)

# Configuração do agente para escrever relatórios analíticos
stockAnalystWrite = Agent(
    role="Senior Stock Analyst Writer",
    goal="Analyze the trends price and news and write an insightful, compelling and informative 3 paragraph long newsletter based on the stock",
    backstory="You're widely accepted as the best stock analyst in the market.",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True
)

# Configuração da tarefa para escrever análises e relatórios
writeAnalyses = Task(
    description="Use the stock price trend and the stock news report to create an analysis and write the newsletter about the {ticket} company that is brief and highlights the most important points.",
    expected_output="""An eloquent 3 paragraphs newsletter formatted as markdown in an easy readable manner. It should contain:
    - 3 bullets executive summary
    - Introduction - set the overall picture and spike up the interest
    - main part provides the meat of the analysis including the news summary and fear/greed scores
    - summary - key facts and concrete future trend prediction - up, down or sideways.
    """,
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
results = crew.kickoff(inputs={"ticket": "APPL", "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

# Exibe o resultado final após a execução de todas as tarefas
final_output = results['final_output']

# Configuração da interface do Streamlit para entrada de dados do usuário
with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label="Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        # Busca os dados da ação
        stock_data = fetch_stock_price(topic)
        
        # Exibe o gráfico dos dados históricos
        st.subheader(f"Historical Data for {topic}")
        plot_stock_price(stock_data, topic)
        
        # Inicia o processo com o ticket fornecido pelo usuário
        results = crew.kickoff(inputs={"ticket": topic, "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

        # Exibe os resultados da pesquisa para o usuário
        st.subheader("Results of your research:")
        st.write(results['final_output'])

