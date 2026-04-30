import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go  # NOVA: Para o gráfico de velocímetro bonitão!

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA WEB
# ==========================================
st.set_page_config(page_title="Calculadora de Diabetes", page_icon="🩺", layout="centered")

st.title("🩺 Calculadora de Risco de Diabetes")
st.write("Ajuste os controles abaixo para ver o cálculo do risco reagir em **tempo real**.")
st.divider()

# ==========================================
# 2. CARREGAMENTO DOS DADOS (Direto do GitHub)
# ==========================================
# Lê o arquivo que está na mesma pasta que o app.py no GitHub
try:
    dados = pd.read_csv('diabetes.csv')
except FileNotFoundError:
    st.error("⚠️ Erro: O arquivo 'diabetes.csv' não foi encontrado. Verifique se ele está no seu GitHub!")
    st.stop() # Faz o site parar aqui se não achar os dados

# ==========================================
# 3. PREPARAR DADOS E TREINAR MODELOS
# ==========================================
entrada = dados.drop("Resultado", axis=1)
saida = dados["Resultado"]
nomes_colunas = entrada.columns.tolist() # Guarda os nomes exatos para não dar erro depois

entrada_treino, entrada_teste, saida_treino, saida_teste = train_test_split(
    entrada, saida, test_size=0.2, random_state=42
)

normalizador = StandardScaler()
entrada_treino_norm = normalizador.fit_transform(entrada_treino)
entrada_teste_norm = normalizador.transform(entrada_teste)

# Treinando
modelo_logistico = LogisticRegression(max_iter=1000)
modelo_logistico.fit(entrada_treino_norm, saida_treino)

modelo_arvore = DecisionTreeClassifier(max_depth=4) # Evitando Overfitting
modelo_arvore.fit(entrada_treino, saida_treino)

# ==========================================
# 4. INTERFACE: ENTRADA DINÂMICA
# ==========================================
st.subheader("📋 Dados do Paciente")

col1, col2 = st.columns(2)

with col1:
    glicose = st.slider("Glicose (mg/dL)", min_value=0.0, max_value=250.0, value=127.0)
    idade = st.slider("Idade", min_value=1, max_value=120, value=30)
    gravidezes = st.number_input("Número de gravidezes", min_value=0, max_value=20, value=1)
    pele = st.slider("Espessura da pele", min_value=0.0, max_value=100.0, value=20.0)

with col2:
    imc = st.slider("IMC", min_value=0.0, max_value=70.0, value=25.0)
    historico = st.slider("Histórico familiar", min_value=0.0, max_value=3.0, value=0.5)
    pressao = st.slider("Pressão arterial (mmHg)", min_value=0.0, max_value=180.0, value=72.0)
    insulina = st.slider("Insulina", min_value=0.0, max_value=900.0, value=80.0)

# ==========================================
# 5. CÁLCULO E GRÁFICO PRINCIPAL (VELOCÍMETRO)
# ==========================================
# Array na ordem exata do CSV
novo_paciente = np.array([[gravidezes, glicose, pressao, pele, insulina, imc, historico, idade]])

# Transforma em DataFrame com os nomes corretos para sumir com o Aviso (Warning) do StandardScaler
novo_paciente_df = pd.DataFrame(novo_paciente, columns=nomes_colunas)
probabilidade = modelo_logistico.predict_proba(normalizador.transform(novo_paciente_df))[0][1]
prob_porcentagem = probabilidade * 100

st.divider()

# Definindo cores e textos baseado no risco
if probabilidade < 0.3:
    cor = "#00CC96" # Verde
    risco_texto = "Risco BAIXO"
    recomendacao = "Recomendação: Continue com hábitos saudáveis!"
elif probabilidade < 0.7:
    cor = "#FFA15A" # Laranja/Amarelo
    risco_texto = "Risco MÉDIO"
    recomendacao = "Recomendação: Melhorar alimentação e praticar exercícios físicos."
else:
    cor = "#EF553B" # Vermelho
    risco_texto = "Risco ALTO"
    recomendacao = "Recomendação: Procure um médico imediatamente e realize exames."

# Criação do Gráfico de Velocímetro estilo App
fig_velocimetro = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prob_porcentagem,
    number = {'suffix': "%", 'valueformat': '.1f', 'font': {'size': 50, 'color': cor}},
    title = {'text': risco_texto, 'font': {'size': 24, 'color': cor}},
    gauge = {
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': cor},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 30], 'color': "rgba(0, 204, 150, 0.1)"},
            {'range': [30, 70], 'color': "rgba(255, 161, 90, 0.1)"},
            {'range': [70, 100], 'color': "rgba(239, 85, 59, 0.1)"}],
    }
))

fig_velocimetro.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))

# Exibe o gráfico centralizado
st.plotly_chart(fig_velocimetro, use_container_width=True)

# Exibe a recomendação
st.markdown(f"<h5 style='text-align: center; color: gray;'>{recomendacao}</h5>", unsafe_allow_html=True)


# ==========================================
# 6. GRÁFICOS DE EXPLICAÇÃO DA IA
# ==========================================
st.write("---")
st.subheader("📈 Entendendo a Inteligência Artificial")
st.write("Abaixo você pode ver como o modelo 'pensa' e onde ele acerta e erra.")

colA, colB = st.columns(2)

with colA:
    st.write("**1. Fatores mais importantes**")
    importancias = modelo_arvore.feature_importances_
    
    fig_barras, ax_barras = plt.subplots(figsize=(5, 4))
    # 'hue=nomes_colunas' adicionado para remover o Aviso (Warning) do Seaborn
    sns.barplot(x=importancias, y=nomes_colunas, hue=nomes_colunas, legend=False, palette="viridis", ax=ax_barras)
    plt.xlabel("Peso de Importância")
    plt.tight_layout()
    st.pyplot(fig_barras)

with colB:
    st.write("**2. Matriz de Confusão (Teste)**")
    previsoes_teste = modelo_logistico.predict(entrada_teste_norm)
    cm = confusion_matrix(saida_teste, previsoes_teste)
    
    fig_matriz, ax_matriz = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_matriz,
                xticklabels=['Saudável', 'Diabetes'],
                yticklabels=['Saudável', 'Diabetes'])
    plt.ylabel('Realidade')
    plt.xlabel('Previsão da IA')
    plt.tight_layout()
    st.pyplot(fig_matriz)
