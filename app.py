# =========================
# 1. IMPORTAR BIBLIOTECAS
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt       # Para desenhar os gráficos
import seaborn as sns                 # Para deixar os gráficos bonitos

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# =========================
# 2. CARREGAR DADOS
# =========================
# O Streamlit lê o arquivo que está na mesma pasta do GitHub direto!
try:
    dados = pd.read_csv('diabetes.csv')
    st.success("Base de dados carregada com sucesso!")
except FileNotFoundError:
    st.error("Erro: O arquivo 'diabetes.csv' não foi encontrado no seu GitHub.")
    st.stop() # Para o aplicativo aqui se não achar o arquivo

# =========================
# 3. MOSTRAR DADOS
# =========================
print("\nPrimeiras linhas do dataset:")
print(dados.head())

# =========================
# 4. PREPARAR DADOS
# =========================
entrada = dados.drop("Resultado", axis=1)
saida = dados["Resultado"]

entrada_treino, entrada_teste, saida_treino, saida_teste = train_test_split(
    entrada, saida, test_size=0.2, random_state=42
)

# =========================
# 5. NORMALIZAÇÃO
# =========================
normalizador = StandardScaler()

entrada_treino_norm = normalizador.fit_transform(entrada_treino)
entrada_teste_norm = normalizador.transform(entrada_teste)

# =========================
# 6. TREINAR MODELOS
# =========================
# Limitamos a profundidade da árvore para 4 para evitar "Overfitting" (decorar os dados)
modelo_arvore = DecisionTreeClassifier(max_depth=4)
modelo_arvore.fit(entrada_treino, saida_treino)

modelo_logistico = LogisticRegression(max_iter=1000)
modelo_logistico.fit(entrada_treino_norm, saida_treino)

# =========================
# 7. AVALIAÇÃO
# =========================
acuracia_arvore = modelo_arvore.score(entrada_teste, saida_teste)
acuracia_logistico = modelo_logistico.score(entrada_teste_norm, saida_teste)

print("\n=== AVALIAÇÃO DOS MODELOS ===")
print("Acurácia Árvore:", round(acuracia_arvore, 2))
print("Acurácia Regressão:", round(acuracia_logistico, 2))

# =========================
# 8. CLASSIFICAÇÃO DE RISCO
# =========================
def classificar_risco(prob):
    if prob < 0.3:
        return "BAIXO"
    elif prob < 0.7:
        return "MÉDIO"
    else:
        return "ALTO"

# =========================
# 9. ENTRADA DO USUÁRIO
# =========================
print("\n=== DIGITE OS DADOS DO PACIENTE ===")

gravidezes = int(input("Número de gravidezes: "))
glicose = float(input("Glicose: "))
pressao = float(input("Pressão arterial: "))
pele = float(input("Espessura da pele: "))
insulina = float(input("Insulina: "))
imc = float(input("IMC: "))
historico = float(input("Histórico familiar: "))
idade = int(input("Idade: "))

novo_paciente = np.array([[gravidezes, glicose, pressao, pele, insulina, imc, historico, idade]])

# =========================
# 10. PREVISÃO
# =========================
probabilidade = modelo_logistico.predict_proba(
    normalizador.transform(novo_paciente)
)[0][1]

risco = classificar_risco(probabilidade)

print("\n=== RESULTADO ===")
print("Probabilidade de diabetes:", round(probabilidade, 2))
print("Classificação de risco:", risco)

# =========================
# 11. EXPLICAÇÃO E RECOMENDAÇÃO
# =========================
print("\n=== RECOMENDAÇÃO ===")

if risco == "ALTO":
    print("Alto risco. Procure um médico e realize exames.")
elif risco == "MÉDIO":
    print("Risco moderado. Melhorar alimentação e praticar exercícios.")
else:
    print("Baixo risco. Continue com hábitos saudáveis.")

# =========================
# 12. EXIBIÇÃO DE GRÁFICOS
# =========================
print("\n=== GERANDO GRÁFICOS DE ANÁLISE ===")
print("No Colab, os gráficos aparecerão logo abaixo desta célula.")

# Pegando as importâncias e os nomes para o gráfico de barras
importancias = modelo_arvore.feature_importances_
nomes = entrada.columns

# --- Gráfico 1: Fatores Mais Importantes ---
plt.figure(figsize=(8, 5))
sns.barplot(x=importancias, y=nomes, hue=nomes, legend=False, palette="viridis")
plt.title("Fatores Mais Importantes para o Risco de Diabetes")
plt.xlabel("Nível de Importância (Peso)")
plt.ylabel("Fatores")
plt.tight_layout()
plt.show() 

# --- Gráfico 2: Matriz de Confusão ---
# Primeiro, pedimos para a Regressão tentar prever os dados de teste
previsoes_teste = modelo_logistico.predict(entrada_teste_norm)

# Depois calculamos a matriz comparando a previsão com a realidade
matriz = confusion_matrix(saida_teste, previsoes_teste)

plt.figure(figsize=(6, 4))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Prev: Saudável', 'Prev: Diabetes'],
            yticklabels=['Real: Saudável', 'Real: Diabetes'])
plt.title("Matriz de Confusão - Acertos e Erros da IA")
plt.show()
