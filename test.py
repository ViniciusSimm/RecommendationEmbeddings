import matplotlib.pyplot as plt

def criar_e_abrir_grafico():
    # Dados para o gráfico
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 11]

    # Criar o gráfico
    plt.figure()
    plt.plot(x, y, marker='o', linestyle='-', color='b', label='Exemplo de Dados')

    # Configurar títulos e rótulos
    plt.title("Exemplo de Gráfico")
    plt.xlabel("Eixo X")
    plt.ylabel("Eixo Y")
    plt.legend()

    # Exibir o gráfico
    plt.show()

# Chamada da função
criar_e_abrir_grafico()
