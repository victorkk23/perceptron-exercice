"""
Tutorial Prático: Implementando um Perceptron do Zero em Python
Disciplina: Inteligência Artificial
Aula 02: Do Perceptron ao Aprendizado de Máquina
Professor: Alexandre "Montanha" de Oliveira
"""

# =============================================================================
# PARTE 2.1: Importar Bibliotecas
# =============================================================================
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica (salva em arquivo)
import matplotlib.pyplot as plt
import os

# Configurar visualizações
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Fixar seed para reprodutibilidade
np.random.seed(42)
print("✅ Bibliotecas carregadas com sucesso!")

# Diretório para salvar os gráficos
os.makedirs("graficos", exist_ok=True)


# =============================================================================
# PARTE 2.2: Implementação da Classe Perceptron
# =============================================================================
class Perceptron:
    """
    Implementação do Perceptron de Rosenblatt (1958)
    Um neurônio artificial que aprende a classificar dados linearmente separáveis.
    """

    def __init__(self, learning_rate=0.1, n_iterations=100):
        """
        Inicializa o perceptron.

        Parâmetros:
        ----------
        learning_rate : float
            Taxa de aprendizado (η) - controla o tamanho dos ajustes nos pesos.
            Valores típicos: 0.01 a 1.0
        n_iterations : int
            Número máximo de épocas (passagens completas pelos dados de treino)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None        # Pesos (w) - serão inicializados no treino
        self.bias = None           # Bias (b) - será inicializado no treino
        self.errors_history = []   # Histórico de erros por época (para análise)

    def step_function(self, z):
        """
        Função de ativação: Função Degrau (Step Function)
        Converte a soma ponderada em uma decisão binária.

        Parâmetro:
        ---------
        z : float ou array
            Soma ponderada (produto escalar entre entradas e pesos + bias)

        Retorna:
        -------
        int ou array : 1 se z >= 0, caso contrário 0
        """
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        """
        Faz previsões para novos dados.

        Processo:
        1. Calcula o produto escalar entre entradas (X) e pesos (w)
        2. Adiciona o bias (b)
        3. Aplica a função de ativação

        Parâmetro:
        ---------
        X : array de shape (n_amostras, n_features)
            Dados de entrada para classificar

        Retorna:
        -------
        array : Previsões (0 ou 1) para cada amostra

        Fórmula: ŷ = f(X · w + b)  onde f é a função degrau
        """
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = self.step_function(linear_output)
        return predictions

    def fit(self, X, y):
        """
        Treina o perceptron usando os dados fornecidos.

        Algoritmo de Treinamento:
        1. Inicializar pesos e bias com zeros
        2. Para cada época:
           a. Para cada exemplo (x, y):
              - Calcular previsão ŷ
              - Calcular erro: e = y - ŷ
              - Atualizar pesos: w = w + η × e × x
              - Atualizar bias: b = b + η × e
        3. Repetir até convergir ou atingir n_iterations

        Parâmetros:
        ----------
        X : array de shape (n_amostras, n_features)
            Dados de treinamento (entradas)
        y : array de shape (n_amostras,)
            Rótulos verdadeiros (saídas esperadas: 0 ou 1)

        Retorna:
        -------
        self : objeto Perceptron treinado
        """
        n_samples, n_features = X.shape

        # Passo 1: Inicializar pesos e bias com zeros
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.errors_history = []

        print(f"🎯 Iniciando treinamento...")
        print(f"   Pesos iniciais: {self.weights}")
        print(f"   Bias inicial: {self.bias}")
        print(f"   Taxa de aprendizado: {self.learning_rate}")
        print(f"   Número de amostras: {n_samples}")
        print("-" * 50)

        # Passo 2: Loop de treinamento (épocas)
        for epoch in range(self.n_iterations):
            errors = 0  # Contador de erros nesta época

            for idx, x_i in enumerate(X):
                # Fazer previsão para este exemplo
                prediction = self.predict(x_i.reshape(1, -1))[0]

                # Calcular erro
                error = y[idx] - prediction

                # Atualizar pesos se houve erro
                if error != 0:
                    update = self.learning_rate * error
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1

            # Armazenar número de erros desta época
            self.errors_history.append(errors)

            # Mostrar progresso a cada 10 épocas
            if (epoch + 1) % 10 == 0:
                print(f"Época {epoch + 1:3d} | Erros: {errors:2d} | "
                      f"Pesos: {self.weights} | Bias: {self.bias:.4f}")

            # Critério de parada: se não houver erros, convergiu
            if errors == 0:
                print(f"\n✅ Convergência alcançada na época {epoch + 1}!")
                print(f"   Pesos finais: {self.weights}")
                print(f"   Bias final: {self.bias:.4f}")
                break

        # Se saiu do loop sem convergir
        if self.errors_history[-1] > 0:
            print(f"\n⚠  Treinamento finalizado sem convergência completa")
            print(f"   Ainda havia {self.errors_history[-1]} erros na última época")

        return self


print("\n" + "=" * 50)
print("CLASSE PERCEPTRON IMPLEMENTADA COM SUCESSO!")
print("=" * 50)


# =============================================================================
# PARTE 3: Função auxiliar para visualizar fronteira de decisão
# =============================================================================
def plot_decision_boundary(X, y, perceptron, title, filename):
    """
    Plota os pontos de dados e a fronteira de decisão do perceptron.
    A fronteira de decisão é a linha que separa as duas classes.
    Matematicamente: w₁x₁ + w₂x₂ + b = 0
    """
    plt.figure(figsize=(10, 7))

    # Classe 0 (círculos vermelhos)
    plt.scatter(X[y == 0, 0], X[y == 0, 1],
                color='red', marker='o', s=200,
                edgecolors='black', linewidths=2,
                label='Classe 0', alpha=0.7)

    # Classe 1 (estrelas verdes)
    plt.scatter(X[y == 1, 0], X[y == 1, 1],
                color='green', marker='*', s=400,
                edgecolors='black', linewidths=2,
                label='Classe 1', alpha=0.7)

    # Calcular e plotar fronteira de decisão
    # Fronteira: w₁x₁ + w₂x₂ + b = 0  →  x₂ = -(w₁x₁ + b) / w₂
    x1_boundary = np.linspace(-0.5, 1.5, 100)
    if perceptron.weights[1] != 0:
        x2_boundary = -(perceptron.weights[0] * x1_boundary + perceptron.bias) / perceptron.weights[1]
        plt.plot(x1_boundary, x2_boundary, 'b-', linewidth=2, label='Fronteira de Decisão')

    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('x₁ (Entrada 1)', fontsize=14, fontweight='bold')
    plt.ylabel('x₂ (Entrada 2)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)

    equation = (f'Fronteira: {perceptron.weights[0]:.2f}x₁ + '
                f'{perceptron.weights[1]:.2f}x₂ + {perceptron.bias:.2f} = 0')
    plt.text(0.02, 0.98, equation, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"graficos/{filename}", dpi=150)
    plt.close()
    print(f"   📊 Gráfico salvo em: graficos/{filename}")


# =============================================================================
# PARTE 3: Experimento 1 — Porta Lógica AND
# =============================================================================
print("\n" + "=" * 60)
print("EXPERIMENTO 1: PORTA LÓGICA AND")
print("=" * 60)

X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_and = np.array([0, 0, 0, 1])

print("\n📊 Dataset AND:")
print("Entradas (X):")
print(X_and)
print("\nSaídas esperadas (y):")
print(y_and)
print("\n" + "-" * 60)

# Criar e treinar o perceptron
perceptron_and = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron_and.fit(X_and, y_and)

print("\n" + "-" * 60)
print("📈 RESULTADOS DO TREINAMENTO")
print("-" * 60)

predictions_and = perceptron_and.predict(X_and)
print("\n🎯 Comparação: Esperado vs Previsto")
print("-" * 40)
for i in range(len(X_and)):
    correto = '✅ CORRETO' if y_and[i] == predictions_and[i] else '❌ ERRADO'
    print(f"Entrada: {X_and[i]} | Esperado: {y_and[i]} | "
          f"Previsto: {predictions_and[i]} | {correto}")

accuracy_and = np.mean(predictions_and == y_and) * 100
print(f"\n🎯 Acurácia: {accuracy_and:.2f}%")

# Fronteira de decisão
plot_decision_boundary(X_and, y_and, perceptron_and,
                       'Perceptron - Porta Lógica AND',
                       'and_fronteira.png')

# Curva de aprendizado AND
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(perceptron_and.errors_history) + 1),
         perceptron_and.errors_history,
         marker='o', linewidth=2, markersize=8, color='blue')
plt.xlabel('Época', fontsize=14, fontweight='bold')
plt.ylabel('Número de Erros', fontsize=14, fontweight='bold')
plt.title('Curva de Aprendizado - AND', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graficos/and_curva_aprendizado.png", dpi=150)
plt.close()
print("   📊 Gráfico salvo em: graficos/and_curva_aprendizado.png")

print("\n📊 Análise da Curva de Aprendizado:")
print("-" * 60)
print(f"• Número total de épocas: {len(perceptron_and.errors_history)}")
print(f"• Erros na época 1: {perceptron_and.errors_history[0]}")
print(f"• Erros na última época: {perceptron_and.errors_history[-1]}")
convergiu = 'SIM ✅' if perceptron_and.errors_history[-1] == 0 else 'NÃO ❌'
print(f"• Convergência: {convergiu}")
print("\n💡 O que isso significa:")
print("  O perceptron começou errando algumas previsões,")
print("  mas aprendeu rapidamente e chegou a zero erros.")
print("  Isso acontece porque AND é linearmente separável!")


# =============================================================================
# PARTE 4: Experimento 2 — Porta Lógica OR
# =============================================================================
print("\n" + "=" * 60)
print("EXPERIMENTO 2: PORTA LÓGICA OR")
print("=" * 60)

X_or = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_or = np.array([0, 1, 1, 1])

print("\n📊 Dataset OR:")
print("Entradas (X):")
print(X_or)
print("\nSaídas esperadas (y):")
print(y_or)
print("\n" + "-" * 60)

perceptron_or = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron_or.fit(X_or, y_or)

print("\n" + "-" * 60)
print("📈 RESULTADOS DO TREINAMENTO")
print("-" * 60)

predictions_or = perceptron_or.predict(X_or)
print("\n🎯 Comparação: Esperado vs Previsto")
print("-" * 40)
for i in range(len(X_or)):
    correto = '✅ CORRETO' if y_or[i] == predictions_or[i] else '❌ ERRADO'
    print(f"Entrada: {X_or[i]} | Esperado: {y_or[i]} | "
          f"Previsto: {predictions_or[i]} | {correto}")

accuracy_or = np.mean(predictions_or == y_or) * 100
print(f"\n🎯 Acurácia: {accuracy_or:.2f}%")

plot_decision_boundary(X_or, y_or, perceptron_or,
                       'Perceptron - Porta Lógica OR',
                       'or_fronteira.png')

# Comparação AND vs OR
print("\n" + "=" * 60)
print("COMPARAÇÃO: AND vs OR")
print("=" * 60)
print("\n📊 Pesos Finais:")
print("-" * 40)
print(f"AND - Peso 1: {perceptron_and.weights[0]:.4f}")
print(f"AND - Peso 2: {perceptron_and.weights[1]:.4f}")
print(f"AND - Bias:   {perceptron_and.bias:.4f}")
print()
print(f"OR  - Peso 1: {perceptron_or.weights[0]:.4f}")
print(f"OR  - Peso 2: {perceptron_or.weights[1]:.4f}")
print(f"OR  - Bias:   {perceptron_or.bias:.4f}")
print("\n💡 Interpretação:")
print("-" * 60)
print("• Para AND: pesos maiores e bias mais negativo")
print("  → Precisa de MAIS evidência para ativar (ambas entradas)")
print()
print("• Para OR: pesos menores e bias menos negativo")
print("  → Precisa de MENOS evidência para ativar (qualquer entrada)")
print()
print("• Ambos convergem rapidamente porque são linearmente separáveis!")


# =============================================================================
# PARTE 5: Experimento 3 — Porta Lógica XOR (Falha Esperada)
# =============================================================================
print("\n" + "=" * 60)
print("EXPERIMENTO 3: PORTA LÓGICA XOR")
print("=" * 60)

X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_xor = np.array([0, 1, 1, 0])

print("\n📊 Dataset XOR:")
print("Entradas (X):")
print(X_xor)
print("\nSaídas esperadas (y):")
print(y_xor)
print("\n⚠  ATENÇÃO: XOR é NÃO linearmente separável!")
print("   O perceptron NÃO deve conseguir aprender este padrão.")
print("\n" + "-" * 60)

perceptron_xor = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron_xor.fit(X_xor, y_xor)

print("\n" + "-" * 60)
print("📈 RESULTADOS DO TREINAMENTO")
print("-" * 60)

predictions_xor = perceptron_xor.predict(X_xor)
print("\n🎯 Comparação: Esperado vs Previsto")
print("-" * 40)
for i in range(len(X_xor)):
    correto = '✅ CORRETO' if y_xor[i] == predictions_xor[i] else '❌ ERRADO'
    print(f"Entrada: {X_xor[i]} | Esperado: {y_xor[i]} | "
          f"Previsto: {predictions_xor[i]} | {correto}")

accuracy_xor = np.mean(predictions_xor == y_xor) * 100
print(f"\n🎯 Acurácia: {accuracy_xor:.2f}%")
if accuracy_xor < 100:
    print("\n⚠  FALHA ESPERADA!")
    print("   O perceptron não conseguiu aprender XOR perfeitamente.")

plot_decision_boundary(X_xor, y_xor, perceptron_xor,
                       'Perceptron - Porta Lógica XOR (FALHA ESPERADA)',
                       'xor_fronteira.png')

# Curva de aprendizado XOR
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(perceptron_xor.errors_history) + 1),
         perceptron_xor.errors_history,
         marker='o', linewidth=2, markersize=8, color='red')
plt.xlabel('Época', fontsize=14, fontweight='bold')
plt.ylabel('Número de Erros', fontsize=14, fontweight='bold')
plt.title('Curva de Aprendizado - XOR (Não Converge)',
          fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("graficos/xor_curva_aprendizado.png", dpi=150)
plt.close()
print("   📊 Gráfico salvo em: graficos/xor_curva_aprendizado.png")

print("\n" + "=" * 60)
print("POR QUE O PERCEPTRON FALHA NO XOR?")
print("=" * 60)
print("\n🔍 Análise Geométrica:")
print("-" * 60)
print("1. O perceptron só consegue desenhar LINHAS RETAS")
print("2. XOR precisa de uma separação em FORMATO DE CRUZ ou CURVA")
print("3. Matematicamente: XOR é NÃO linearmente separável")
print()
print("💡 Solução Histórica:")
print("  • 1969: Minsky & Papert publicam 'Perceptrons'")
print("  • Provam matematicamente a limitação")
print("  • Causa o primeiro 'inverno da IA'")
print("  • Solução: Redes Neurais Multicamadas (1986)")
print("\n📊 Análise da Curva:")
print("-" * 60)
print("• Note que o número de erros NÃO chega a zero")
print("• O perceptron fica 'preso' tentando aprender")
print("• Não há configuração de pesos que resolva XOR com uma linha")
print("• Este foi um resultado CRUCIAL na história da IA")


# =============================================================================
# PARTE 6: Experimento 4 — Impacto da Taxa de Aprendizado
# =============================================================================
print("\n" + "=" * 60)
print("EXPERIMENTO 4: IMPACTO DA TAXA DE APRENDIZADO")
print("=" * 60)

learning_rates = [0.01, 0.1, 1.0]
colors = ['blue', 'green', 'red']

plt.figure(figsize=(14, 5))
for idx, lr in enumerate(learning_rates):
    print(f"\n{'=' * 60}")
    print(f"Taxa de Aprendizado: {lr}")
    print(f"{'=' * 60}")

    perceptron_lr = Perceptron(learning_rate=lr, n_iterations=100)
    perceptron_lr.fit(X_and, y_and)

    plt.subplot(1, 3, idx + 1)
    plt.plot(range(1, len(perceptron_lr.errors_history) + 1),
             perceptron_lr.errors_history,
             marker='o', linewidth=2, markersize=6, color=colors[idx])
    plt.xlabel('Época', fontsize=12, fontweight='bold')
    plt.ylabel('Erros', fontsize=12, fontweight='bold')
    plt.title(f'η = {lr}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 5)

plt.tight_layout()
plt.savefig("graficos/comparacao_taxas_aprendizado.png", dpi=150)
plt.close()
print("\n   📊 Gráfico salvo em: graficos/comparacao_taxas_aprendizado.png")

print("\n" + "=" * 60)
print("ANÁLISE DOS RESULTADOS")
print("=" * 60)
print("\n🐌 Taxa Baixa (η = 0.01):")
print("-" * 60)
print("• Convergência LENTA")
print("• Passos muito pequenos")
print("• Seguro, mas ineficiente")
print("• Use quando: dados ruidosos ou instabilidade")
print("\n⚖  Taxa Moderada (η = 0.1):")
print("-" * 60)
print("• EQUILÍBRIO ideal")
print("• Convergência rápida e estável")
print("• Valor PADRÃO recomendado")
print("• Use quando: situação típica")
print("\n🚀 Taxa Alta (η = 1.0):")
print("-" * 60)
print("• Convergência muito rápida OU instabilidade")
print("• Passos grandes podem 'pular' a solução")
print("• Pode oscilar sem convergir em problemas complexos")
print("• Use quando: problema simples e quer velocidade")
print("\n💡 Regra Prática:")
print("-" * 60)
print("• Comece com η = 0.1 (valor padrão)")
print("• Se não convergir: reduza (ex: 0.01)")
print("• Se convergir muito devagar: aumente (ex: 0.5)")
print("• Problemas difíceis: taxas menores (0.001 - 0.1)")
print("• Problemas simples: taxas maiores (0.1 - 1.0)")


# =============================================================================
# PARTE 6.2: Tabela Comparativa Detalhada
# =============================================================================
print("\n" + "=" * 60)
print("TABELA COMPARATIVA: IMPACTO DA TAXA DE APRENDIZADO")
print("=" * 60)

results = []
for lr in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    perceptron_test = Perceptron(learning_rate=lr, n_iterations=100)
    # Suprimir saída do fit para a tabela
    import io, sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    perceptron_test.fit(X_and, y_and)
    sys.stdout = old_stdout

    results.append({
        'Taxa': lr,
        'Épocas': len(perceptron_test.errors_history),
        'Convergiu': perceptron_test.errors_history[-1] == 0,
        'Erros_Finais': perceptron_test.errors_history[-1]
    })

print("\n{:<10} {:<10} {:<12} {:<15}".format(
    "Taxa (η)", "Épocas", "Convergiu?", "Erros Finais"
))
print("-" * 60)
for r in results:
    print("{:<10} {:<10} {:<12} {:<15}".format(
        r['Taxa'],
        r['Épocas'],
        "✅ Sim" if r['Convergiu'] else "❌ Não",
        r['Erros_Finais']
    ))
print("\n" + "=" * 60)


# =============================================================================
# PARTE 7: Conceitos Avançados e Conexão com IA Moderna
# =============================================================================
print("\n" + "=" * 60)
print("DO PERCEPTRON AO DEEP LEARNING")
print("=" * 60)
print("""
📜 LINHA DO TEMPO E EVOLUÇÃO:

1943 - McCulloch & Pitts
│    Primeiro modelo matemático de neurônio

1958 - Rosenblatt
│    PERCEPTRON (o que implementamos hoje!)
│    ✅ Aprende padrões lineares
│    ❌ Não resolve XOR

1969 - Minsky & Papert
│    Livro "Perceptrons" → Primeiro Inverno da IA
│    Provam limitações matemáticas

1986 - Rumelhart, Hinton & Williams
│    BACKPROPAGATION + Redes Multicamadas
│    ✅ Resolve XOR com camada oculta
│    ✅ Funções não-lineares

2012 - Krizhevsky (AlexNet)
│    Deep Learning vence ImageNet
│    Revolução da IA moderna

2017-2024 - Era Transformer
│    GPT, BERT, ChatGPT, Gemini
│    Bilhões de parâmetros
│    Mas o PRINCÍPIO é o mesmo! ⚡
""")

print("🧠 Como o Perceptron se Conecta com Redes Modernas:")
print("-" * 60)
print()
print("1. NEURÔNIO BÁSICO (Perceptron):")
print("   ŷ = f(w·x + b)")
print()
print("2. REDE NEURAL MULTICAMADAS:")
print("   • Empilhar múltiplos perceptrons")
print("   • Adicionar camadas ocultas")
print("   • Resolver problemas não-lineares (XOR!)")
print()
print("3. DEEP LEARNING:")
print("   • Muitas camadas (10, 50, 100+)")
print("   • Milhões/bilhões de pesos")
print("   • Mesma regra de atualização (gradient descent)")
print()
print("💡 CONCLUSÃO:")
print("   O perceptron é o 'átomo' das redes neurais modernas!")
print("   Tudo que você aprendeu hoje escala para GPT, BERT, etc.")


# =============================================================================
# PARTE 8: Checklist de Aprendizado
# =============================================================================
print("\n" + "=" * 60)
print("CHECKLIST: O QUE VOCÊ APRENDEU HOJE")
print("=" * 60)

checklist = [
    ("Conceito de neurônio artificial",
     "Unidade básica que processa informação"),
    ("Arquitetura do perceptron",
     "Entradas, pesos, bias, soma ponderada, ativação"),
    ("Função de ativação (step function)",
     "Converte números em decisões binárias"),
    ("Regra de aprendizado do perceptron",
     "w = w + η × erro × x"),
    ("Implementação do zero em Python",
     "Classe completa sem bibliotecas de ML"),
    ("Problemas linearmente separáveis",
     "AND, OR - perceptron RESOLVE"),
    ("Problemas não-linearmente separáveis",
     "XOR - perceptron FALHA"),
    ("Taxa de aprendizado (η)",
     "Controla velocidade e estabilidade"),
    ("Fronteira de decisão",
     "Linha que separa classes"),
    ("Convergência",
     "Quando o modelo para de errar"),
    ("Limitação histórica",
     "Inverno da IA (1969) por causa do XOR"),
    ("Conexão com Deep Learning",
     "Perceptron é o bloco básico de redes modernas")
]

for i, (concept, description) in enumerate(checklist, 1):
    print(f"\n{i:2d}. ✅ {concept}")
    print(f"    → {description}")

print("\n" + "=" * 60)
print("\n✨ 'A jornada de mil camadas começa com um único perceptron.' ✨")
print("=" * 60)
print("\n📁 Gráficos salvos na pasta: graficos/")
print("   - and_fronteira.png")
print("   - and_curva_aprendizado.png")
print("   - or_fronteira.png")
print("   - xor_fronteira.png")
print("   - xor_curva_aprendizado.png")
print("   - comparacao_taxas_aprendizado.png")
