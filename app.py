import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURAÇÃO DA PÁGINA WEB
# ==========================================
st.set_page_config(page_title="Calculadora de Diabetes", page_icon="🩺", layout="centered")

st.title("🩺 Calculadora de Risco de Diabetes")
st.write("Baseado em um modelo de Inteligência Artificial para análise de dados de saúde.")
st.divider()

# ==========================================
# 2. CARREGAMENTO DOS DADOS (Upload)
# ==========================================
st.sidebar.header("📁 Base de Dados")
st.sidebar.write("Faça o upload do seu arquivo `diabetes.csv` para treinar a IA.")
arquivo = st.sidebar.file_uploader("", type=["csv"])

if arquivo is None:
    st.warning("⚠️ Aguardando o upload do arquivo `diabetes.csv` na barra lateral...")
else:
    # Lê os dados
    dados = pd.read_csv(arquivo)
    st.sidebar.success("Base carregada com sucesso!")

    # ==========================================
    # 3. PREPARAR DADOS E TREINAR MODELOS
    # ==========================================
    # Aqui assumimos que a coluna se chama 'Resultado' no seu CSV
    entrada = dados.drop("Resultado", axis=1)
    saida = dados["Resultado"]

    entrada_treino, entrada_teste, saida_treino, saida_teste = train_test_split(
        entrada, saida, test_size=0.2, random_state=42
    )

    normalizador = StandardScaler()
    entrada_treino_norm = normalizador.fit_transform(entrada_treino)
    
    # Treinando os modelos silenciosamente
    modelo_logistico = LogisticRegression(max_iter=1000)
    modelo_logistico.fit(entrada_treino_norm, saida_treino)

    modelo_arvore = DecisionTreeClassifier(max_depth=4) # Árvore limitada
    modelo_arvore.fit(entrada_treino, saida_treino)

    # ==========================================
    # 4. INTERFACE PARA ENTRADA DO USUÁRIO
    # ==========================================
    st.subheader("📋 Dados do Novo Paciente")
    
    # Organizando as barras de deslizar em duas colunas para ficar bonito
    col1, col2 = st.columns(2)

    with col1:
        gravidezes = st.number_input("Número de gravidezes", min_value=0, max_value=20, value=0)
        glicose = st.slider("Glicose", min_value=0.0, max_value=250.0, value=100.0)
        pressao = st.slider("Pressão arterial", min_value=0.0, max_value=180.0, value=75.0)
        pele = st.slider("Espessura da pele", min_value=0.0, max_value=100.0, value=20.0)

    with col2:
        insulina = st.slider("Insulina", min_value=0.0, max_value=900.0, value=80.0)
        imc = st.slider("IMC", min_value=0.0, max_value=70.0, value=25.0)
        historico = st.slider("Histórico familiar", min_value=0.0, max_value=3.0, value=0.5)
        idade = st.slider("Idade", min_value=1, max_value=120, value=30)

    # Botão para calcular
    if st.button("🔬 Analisar Risco de Diabetes"):
        
        # ==========================================
        # 5. PREVISÃO E RESULTADO
        # ==========================================
        novo_paciente = np.array([[gravidezes, glicose, pressao, pele, insulina, imc, historico, idade]])
        novo_paciente_norm = normalizador.transform(novo_paciente)

        # Usando a regressão logística para a previsão de probabilidade
        probabilidade = modelo_logistico.predict_proba(novo_paciente_norm)[0][1]
        
        st.divider()
        st.subheader("📊 Resultado da Análise")

        # Formatando as mensagens com cores baseadas no risco
        if probabilidade < 0.3:
            st.success(f"**Risco BAIXO**: {probabilidade * 100:.1f}% de chance")
            st.write("Recomendação: Continue com hábitos saudáveis!")
        elif probabilidade < 0.7:
            st.warning(f"**Risco MÉDIO**: {probabilidade * 100:.1f}% de chance")
            st.write("Recomendação: Melhorar alimentação e praticar exercícios físicos.")
        else:
            st.error(f"**Risco ALTO**: {probabilidade * 100:.1f}% de chance")
            st.write("Recomendação: Procure um médico imediatamente e realize exames detalhados.")

        # ==========================================
        # 6. GRÁFICO DE FATORES IMPORTANTES (BÔNUS)
        # ==========================================
        st.write("---")
        st.write("**Fatores que mais influenciaram esse modelo:**")
        
        importancias = modelo_arvore.feature_importances_
        df_importancias = pd.DataFrame({
            "Fatores": entrada.columns,
            "Peso": importancias
        }).sort_values(by="Peso", ascending=True) # Ascendente para o gráfico ficar certo
        
        # Cria um gráfico de barras nativo do próprio Streamlit!
        st.bar_chart(df_importancias.set_index("Fatores"))