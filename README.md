# Stock Analysis AI Agent

## Descrição

Este projeto é uma aplicação de análise de ações que utiliza agentes de Inteligência Artificial para coletar, analisar e gerar relatórios sobre o preço das ações e notícias de mercado. A aplicação é construída utilizando a biblioteca Streamlit para a interface do usuário, `yfinance` para obter dados financeiros, e `LangChain` para integração com modelos de linguagem.

## Funcionalidades

- **Análise de Preço das Ações:** O agente analisa o histórico de preços de uma ação específica e identifica a tendência atual (alta, baixa ou lateral).
- **Análise de Notícias:** O agente coleta e analisa notícias relevantes para a ação em questão, fornecendo uma visão geral do sentimento do mercado.
- **Geração de Relatórios:** O sistema gera um relatório conciso com base nas análises de preço e notícias, incluindo uma previsão de tendência futura.

## Tecnologias Utilizadas

- **Python 3.x**
- **Streamlit:** Interface web interativa.
- **yfinance:** Coleta de dados históricos de ações.
- **LangChain:** Integração com modelos de linguagem para execução das tarefas.
- **OpenAI GPT:** Modelo de linguagem para análise e geração de relatórios.

## Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/alvarotavares07/stocksAgents.git
   cd stocksAgents/WebappStocks

2. Crie e ative um ambiente virtual:

  python -m venv venv
  source venv/bin/activate  # Para Linux/Mac
  venv\Scripts\activate  # Para Windows

3. Instale as dependências:  

  pip install -r requirements.txt

4. Crie um arquivo secrets.toml na pasta .streamlit para armazenar sua chave da API OpenAI:

  [general]
  OPENAI_API_KEY = "sua-chave-api-aqui"

Como Usar

1. Inicie a aplicação Streamlit:
  streamlit run crewai-stocks.py

2. Acesse a aplicação no navegador na URL fornecida (geralmente http://localhost:8501).

3. Na interface da aplicação, insira o ticket da ação que deseja analisar e clique em "Run Research".

4. Veja os resultados da análise, incluindo a tendência de preço, um resumo das notícias e o relatório gerado.




