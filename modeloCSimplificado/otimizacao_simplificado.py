import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import os
import time
import pygmo as pg
import modelo_simplificado

output_dir = 'resultados/simulacao_simplificada_1'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ============================
# DADOS EXPERIMENTAIS
# ============================
dias = [8.12, 10.13, 12.17, 14.21, 15.21, 17.22, 19.25, 20.20, 21.19, 22.27, 23.26, 24.29,
        25.31, 27.42, 29.45, 31.52, 32.55, 34.62, 38.75, 40.77, 45.90, 47.97, 50]
dados_experimentais_tp = np.array([
    7636.25, 88895.83, 168841.06, 335570.60, 503181.18, 833156.53,
    1412119.79, 2240213.76, 2903183.02, 2654052.30, 2407554.13, 2155810.16,
    1906687.17, 1825163.16, 1657158.86, 1161529.97, 831157.03, 414157.12,
    251120.69, 169608.27, 85069.57, 84799.37, 0.00
])
t = np.linspace(0, 60, 600)

# ============================
# NOVOS PARÂMETROS
# ============================
parametros = {
    'H0': 1e8, 
    'SI0': 1e5,      # NK0
    'alfa_H': 0.001,   
    'alfa_SI': 0.01,   
    'delta_T': 0.5,    
    'delta_I': 0.1,    
    'delta_SI': 0.1,   # delta_NK
}

param_names = [
    'pi_release', # Liberação de T por I
    'beta_inf',   # Infecção de H por T
    'beta_SI',    # Morte de T por SI
    'evasao',     # Fator de evasão
    'beta_SI_2',  # Morte de I por SI
    'beta',       # Ativação de SI por T
    'c',          # Coeficiente
    'c2',         # Coeficiente 
]

# ============================
# CONDIÇÕES INICIAIS
# ============================
condicoes_iniciais = [1e3, 0, parametros['H0'], parametros['SI0']]

# ============================
# LIMITES DE BUSCA 
# ============================
div = 0.5
mult = 5

bounds_list = [
    (1e0 * div , 2e1 * mult ),      # pi_release (modelo antigo)
    (1e-9 * div , 1e-5 * mult ),    # beta_inf (modelo antigo)
    (1e-7 * div , 1e-4 * mult ),    # beta_SI           |
    (0.0 , 1.0 ),                   # evasao            |
    (1e-7 * div , 1e-4 * mult ),    # beta_SI_2         |  ---> CHUTES
    (1e-3 * div , 1e0 * mult ),     # beta              |
    (1e-3 * div , 1e0 * mult ),     # c                 |   
    (1e4 * div , 1e8 * mult )       # c2                |
]
# Arrays por causa do pagmo
bounds = (
    [b[0] for b in bounds_list], # Limites inferiores
    [b[1] for b in bounds_list]  # Limites superiores
)

# ============================
# CLASSE DO PROBLEMA
# ============================
class ChagasProblemSimplificado:
    def __init__(self, p_base, param_keys, dias_exp, dados_exp, t_eval, cond_iniciais, bounds):
        self.p_base = p_base
        self.param_keys = param_keys
        self.dias_exp = dias_exp
        self.dados_exp_brutos = dados_exp
        self.t_eval = t_eval
        self.cond_iniciais = cond_iniciais
        self.bounds = bounds

    def fitness(self, params_vector):
        p = self.p_base.copy()
        for i, key in enumerate(self.param_keys):
            p[key] = params_vector[i]

        sol = integrate.odeint(modelo_simplificado.modelo_simplificado_cy, 
                               self.cond_iniciais, 
                               self.t_eval, 
                               args=(p,))
        
        Tp_model = np.maximum(0.0, np.interp(self.dias_exp, self.t_eval, sol[:, 0]))
            
        erro = np.mean((np.log(Tp_model + 1) - np.log(np.array(self.dados_exp_brutos) + 1))**2)
        
        return [erro] # Pagmo ESPERA um tupla/lista de retorno

    def get_bounds(self):
        return self.bounds

# ============================
# RODAR SIMULAÇÃO
# ============================
if __name__ == '__main__':
    prob = pg.problem(ChagasProblemSimplificado(
        p_base=parametros,
        param_keys=param_names,
        dias_exp=dias,
        dados_exp=dados_experimentais_tp,
        t_eval=t,
        cond_iniciais=condicoes_iniciais,
        bounds=bounds
    ))

    populacao = 50
    iteracoes = 1500

    algo = pg.algorithm(pg.sade(gen=iteracoes, memory=True)) 

    pop = pg.population(prob, size=populacao)

    print(f"Iniciando otimização (SADE) com {populacao} indivíduos por {iteracoes} gerações...")
    start_time = time.time()
    
    pop = algo.evolve(pop)
    
    end_time = time.time()
    print(f"\nOtimização concluída em {end_time - start_time:.2f} segundos.")
    
    # Resultados
    p_otim_values = pop.champion_x
    erro = pop.champion_f[0]

    p_otim = parametros.copy()
    
    output_text = "Parametros da Simulacao (PAGMO/SADE - Modelo SIMPLIFICADO):\n"
    output_text += "---------------------------------------------------\n"
    output_text += f"{'Populacao (populacao)':<25} = {populacao}\n"
    output_text += f"{'Iteracoes (iteracoes)':<25} = {iteracoes}\n"
    output_text += f"{'Erro final (MSLE)':<25} = {erro:.6f}\n"
    output_text += "---------------------------------------------------\n"
    output_text += "Parametros Otimizados:\n"

    print("\n--- Configuração da Simulação ---")
    print(f"{'População (populacao)':<25} = {populacao}")
    print(f"{'Iterações (iteracoes)':<25} = {iteracoes}")
    print(f"{'Erro final (MSLE)':<25} = {erro:.6f}")
    print("--- Parâmetros Otimizados Encontrados ---")
    
    for name, value in zip(param_names, p_otim_values):
        p_otim[name] = value
        if 'e' in f"{value:.10e}":
            print(f"{name:<16} = {value:.10e}")
            output_text += f"{name:<16} = {value:.10e}\n"
        else:
            print(f"{name:<16} = {value:.6f}")
            output_text += f"{name:<16} = {value:.6f}\n"

    with open(os.path.join(output_dir, "parametros_otimizados.txt"), "w") as f:
        f.write(output_text)

    # Gerar o gráfico final
    sol_otim = integrate.odeint(modelo_simplificado.modelo_simplificado_cy, condicoes_iniciais, t, args=(p_otim,))
    Tp_curva = sol_otim[:, 0]

    plt.figure(figsize=(9, 5))
    plt.plot(dias, dados_experimentais_tp, 'o', label='Dados experimentais')
    plt.plot(t, Tp_curva, '-', label='Modelo (SADE)')
    plt.xlabel('Dias')
    plt.ylabel('Trypomastigotas (Tp)')
    plt.title(f'Ajuste do modelo simplificado (Erro: {erro:.6f})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "grafico_ajuste_final.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nResultados salvos em: {os.path.abspath(output_dir)}")

