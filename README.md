# 🤖 Desvendando obstáculos com IA e ML: 
Explorando a simulação de Dados de Sensores IoT

Sabe aquele projetinho que voce começa fazer e que acaba virando UM MOSTRO (no bom sentido)???

Este projeto foi feito para a IMERSÂO ALURA + GOOGLE GEMINI 2025

---

## ✨ Visão Geral do Projeto

Mergulhe em um projeto fascinante que une o poder do **Machine Learning (ML)** e da **Inteligência Artificial (IA)** com a simulação de dados dados de **sensores do ecosistema Arduino** para detectar e classificar diferentes tipos de padrões. 

Desenvolvido integralmente no ambiente **Google Colab**, este projeto é uma demonstração prática de como construir um pipeline de IA/ML, desde a geração flexível de dados até a análise aprofundada dos resultados, utilizando as capacidades avançadas do **Google Gemini**.

Mesmo sem a necessidade de hardware físico imediato (Arduino), este projeto permite focar nos desafios de **pré-processamento de dados de séries temporais multi-feature**, **construção e treinamento de modelos de ML** para classificação, e uma **aplicação inovadora do Gemini** tanto na fase de configuração quanto na análise e interpretação dos resultados.

É a combinação perfeita para quem busca aprender sobre ML, IA, dados de sensores e orquestração de projetos de Data Science de ponta a ponta em um ambiente prático e acessível.


## 🚀 O Que Torna Este Projeto Único?

* **Abordagem Multidisciplinar:** Conecta os mundos de Sensores (simulados), Processamento de Dados, Machine Learning e Inteligência Artificial Avançada.
* **Uso Inovador do Google Gemini:**
    * **Coleta de Requisitos atravé de um chat (Etapa 1):** Utilize o Gemini em um chat interativo no notebook Colab para guiar a definição das características do sensor, tipos de obstáculos e parâmetros de simulação. A IA atua como uma persona "desenvolvedor entrevistador", e o resultado da conversa é a configuração estruturada do projeto (JSON)!
    * **Análise Textual Aprofundada (Etapa 4):** o Gemini e alimentado com as métricas de avaliação e exemplos de erros do modelo de ML para obter insights em linguagem natural sobre o desempenho, pontos fortes, fracos e possíveis causas de erros, elevando a análise além dos números.
* **Pipeline Completo em Google Colab:** Todo o desenvolvimento, treinamento e análise são realizados em notebooks Colab, facilitando a execução e o compartilhamento.
* **Flexibilidade de Dados:** A estrutura do projeto é construída para lidar com dados de séries temporais com *múltiplas características* (features), tornando-o adaptável conceitualmente a diversos tipos de sensores (distância, acelerômetro, giroscópio, etc.) apenas respondendo as perguntas da Gemini-IA e toda a mágica acontece.
* **Reprodutibilidade e Compartilhamento:** Projetado para ser facilmente replicável por outros usuários utilizando Google Drive para persistência de arquivos e Secreate API do Colab para manuseio seguro da chave API.

## 🧠 O Que Você Vai Aprender (Ou Colocar em Prática)

* Simulação e Geração de Dados de Séries Temporais PARA QUALQUER TIPO DE SENSOR.
* Técnicas de Pré-processamento para Dados Sequenciais (Escalamento, Janelamento).
* Construção e Treinamento de Modelos de Classificação com Keras/TensorFlow (Ex: LSTMs, GRUs, Conv1D).
* Avaliação Rigorosa de Modelos de ML (Acurácia, Precisão, Recall, F1, Matriz de Confusão).
* Engenharia de Prompts e Interação com a API Google Gemini para tarefas específicas.
* Utilização de Segredos do Google Colab para gerenciar chaves de API de forma segura.
* Orquestração de um pipeline de Data Science em múltiplos notebooks.
* Gerenciamento de configuração e persistência de dados/modelos via Google Drive.

* EXEMPLO DE USO
  ## Não se esqueça de utilizar as stop words "xpto" para sair do chat 

Responda quando pergunatado pela IA que:
 trata-se de um sensor modelo SR-HC04 que acoplado a uma bengala guia para deficientes visuais. Desejo criar um modelo de maquina para detecção de OBJETOS, AEREOS, FIXOS, ACIMA DA LINHA DA CINTURA para informar ao usuário se existe algum obstáculo a frente.
 - para uso em ambiente urbano;  a bengala estará em movimento constante
 - a taxa de amostragem a cada 500 milisegundos;
 - o sensor ficar em angulo que ajude a detectar objetos acima da linha da cintura
 - Detectar:
     - a presença de pessoas à frente
     - Obstáculos Fixos: placas, marquises
     - obstáculos Suspensos: galhos de árvores, toldos, etc.
     -
 - apos estas respostas a aplicação vai gerar todas as informações para se construir um modelo de ML para a detecção de objetos
   - gerar dados (mock)

## 🏗️ Estrutura do Projeto

O projeto é em Colab pensado para ser executados em sequência:

1.  `Etapa 1`: **Configuração Interativa e Geração de Dados.** Converse com a IA para definir os parâmetros, gere os dados simulados multi-feature e salve o arquivo de configuração (`project_config.json`) e o dataset (`simulated_sensor_data.csv`) no seu Google Drive.
2.  `Etapa 2`: **Pré-processamento e Treinamento do Modelo.** Carrega a configuração e os dados, pré-processa as séries temporais, constrói e treina o modelo de classificação ML (ex: baseado em LSTM) com os dados gerados na Etapa 1. Salva o modelo treinado e os dados de teste no GDrive.
3.  `Etapa 3`: **Avaliação Numérica.** Carrega o modelo e os dados de teste, calcula métricas de desempenho detalhadas (matriz de confusão, precision, recall, etc.) e salva os resultados numéricos e informações básicas sobre os erros no GDrive.
4.  `Etapa 4`: **Análise Textual com Gemini.** Carrega os resultados numéricos da avaliação e os dados de teste. Utiliza a API Gemini para realizar uma análise em linguagem natural sobre o desempenho do modelo e especular sobre as causas dos erros, interagindo diretamente no notebook.

## ▶️ Como Executar o Projeto

Para rodar este projeto, você precisará de:

* Uma Conta Google.
* Acesso ao Google Colab e Google Drive.
* Uma Chave de API do Google Gemini. Você pode obtê-la gratuitamente no [Google AI Studio](https://aistudio.google.com/app/apikey).

Siga os passos abaixo:

1.  **Faça o download do arquivo `imClassSensIotIA.ipynb`:**  para o seu ambiente local.
2.  **Abra os Notebooks no Google Colab:** Faça upload ou abra os arquivos `.ipynb` diretamente no Google Colab.
3.  **Configure a Chave API Gemini:** No **Google Colab**, no menu lateral esquerdo, clique no ícone de 'chave' (🔑 Segredos). Adicione um **novo segredo** com o nome **`GEMINI_API_KEY`** e cole sua chave de API Gemini no campo 'Valor'. Marque a caixa 'Acesso ao notebook'. **Este passo é crucial para a Etapa 1 e Etapa 4.**
4.  **Execute os Notebooks em Sequência:**
    * Abra e execute as células do notebook. Siga as instruções no notebook para montar seu Google GDrive e interagir com o chat Gemini para definir a configuração e gerar os dados.

Siga as instruções e explicações contidas nas células Markdown de cada notebook.


