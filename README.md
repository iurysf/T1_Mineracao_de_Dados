#  CineScope 🔮 - Previsão de Sucesso em Filmes com IA

<div align="center">
<img width="419" height="806" alt="image" src="https://github.com/user-attachments/assets/77785062-3f57-4947-b207-fa5e08a8889c" />
</div>

**CineScope** é uma aplicação desktop desenvolvida em Python que utiliza Machine Learning para analisar e prever o sucesso de filmes. A ferramenta não apenas classifica um filme como "Sucesso" ou "Fracasso", mas também oferece insights sobre os fatores que mais influenciam essa decisão e encontra filmes similares com base nas características fornecidas.

Este projeto foi desenvolvido como um trabalho acadêmico para a disciplina de Mineração de Dados, demonstrando um pipeline completo de Data Science, desde a limpeza e pré-processamento dos dados até o treinamento, avaliação e implantação interativa de múltiplos modelos de IA.

---

## ✨ Funcionalidades Principais

*   **Previsão de Sucesso:** Classifica um filme com base em seus atributos (ano, orçamento, duração, etc.).
*   **Seleção Dinâmica de Modelos:** Permite ao usuário escolher entre diferentes algoritmos de Machine Learning (`Decision Tree`, `Random Forest`, `KNN`) e comparar suas métricas de performance em tempo real.
*   **Explicabilidade (XAI - "O Porquê?"):** Mostra os 3 principais fatores que mais influenciaram a decisão do modelo, tornando a IA menos "caixa-preta".
*   **Sistema de Recomendação Simples:** Para o modelo KNN, a aplicação encontra e exibe os 3 filmes mais similares do dataset com base nos dados inseridos.
*   **Interface Gráfica Intuitiva:** Uma interface amigável construída com `Tkinter` e `ttkthemes` para uma experiência de usuário agradável.
*   **Preenchimento Aleatório:** Um botão "🎲" para gerar dados aleatórios, facilitando testes e demonstrações rápidas.

---

## 🚀 Como Executar o Projeto

Para rodar o CineScope em sua máquina, siga os passos abaixo.

### 1. Pré-requisitos

Certifique-se de ter o Python 3.8+ instalado.

### 2. Baixar o Banco de Dados

O modelo foi treinado com um dataset público de filmes do IMDb. É **essencial** que você baixe este dataset para que o script de treinamento possa ser executado.

*   **Fonte:** [Base dos Dados](https://basedosdados.org/dataset/6ba4745d-f131-4f8e-9e55-e8416199a6af?table=79de8c5e-9c21-4398-a9fb-bc40e6d6e77f)
*   **Ação:** Baixe o arquivo csv, troque o nome para `imdb_filmes.csv` e coloque-o na **raiz do diretório do projeto**.

### 3. Clonar o Repositório

```bash
git clone https://github.com/iurysf/T1_Mineracao_de_Dados.git
cd T1_Mineracao_de_Dados
```

### 4. Instalar as Dependências

Crie um ambiente virtual e instale as bibliotecas necessárias.

```bash
# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instalar as bibliotecas
pip install -r requirements.txt```

### 5. Treinar os Modelos

Antes de executar a interface, você precisa treinar os modelos e gerar os artefatos (modelos salvos, scaler, etc.). Execute o script de treinamento:

```bash
python main.py
```
Este script irá processar o `imdb_filmes.csv`, treinar os modelos e criar uma pasta chamada `artefatos_modelo` com todos os arquivos necessários.

### 6. Executar a Aplicação

Com os modelos treinados, você pode iniciar a interface gráfica:

```bash
python program.py
```

---

## 🛠️ Estrutura do Projeto

*   `main.py`: Script de **treinamento**. Responsável pela limpeza dos dados, engenharia de features, treinamento e avaliação dos modelos, e salvamento dos artefatos.
*   `program.py`: Script da **aplicação principal**. Contém a interface gráfica (Tkinter) que carrega os artefatos e realiza as previsões interativas.
*   `artefatos_modelo/`: Pasta criada pelo `main.py` que contém:
    *   `todos_os_modelos.joblib`: Os três modelos de classificação treinados.
    *   `scaler.joblib`: O `StandardScaler` ajustado.
    *   `*.json`: Arquivos com as listas de gêneros, idiomas, métricas e outras informações necessárias para a UI.
*   `icons/`: Pasta com os ícones usados na interface.
*   `requirements.txt`: Lista de dependências Python.
*   `imdb_filmes.csv`: **(Precisa ser baixado)** O dataset bruto.

---

## 💡 Conceitos Aplicados

*   **Ciência de Dados:** Limpeza de dados (Regex, tratamento de nulos), Engenharia de Features (One-Hot Encoding).
*   **Machine Learning:**
    *   **Classificação:** `Decision Tree`, `Random Forest`, `K-Nearest Neighbors`.
    *   **Avaliação de Modelos:** Divisão Treino-Teste Estratificada, Acurácia, Precisão, F1-Score.
    *   **Explicabilidade (XAI):** Feature Importances.
*   **Desenvolvimento de Software:** Programação Orientada a Objetos (POO) em Python, Desenvolvimento de GUI com Tkinter.

---

Feito com ❤️ por iurysf
