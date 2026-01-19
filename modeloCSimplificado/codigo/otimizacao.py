import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
import os
import time
import pygmo as pg
import modelo_simplificado

#output_dir = 'resultados/variando_pi_release/simulacao_5'
output_dir = 'resultados/ajuste_usando_intervalo_confianca/simulacao_2'



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

# ===============
# DESVIOS PADRAO
# ===============
desvios_por_ponto = np.array([
    10000, 10000, 10000, 10000, 140000, 160000,
    200000, 200000, 250000, 200000, 230000, 100000,
    150000, 100000, 130000, 200000, 150000, 120000,
    140000, 110000, 100000, 100000, 10000
])

# Limites
limite_superior = dados_experimentais_tp + desvios_por_ponto
limite_inferior = dados_experimentais_tp - desvios_por_ponto

# =============
#   ZSCORE 
# =============

media_exp = np.mean(dados_experimentais_tp)
desvio_exp = np.std(dados_experimentais_tp)
dados_exp_zscore = (dados_experimentais_tp - media_exp) / desvio_exp

# ============================
# PARÂMETROS BASE
# ============================
parametros = {
    'H0': 1e8, 
    'SI0': 1e5,      
    'alfa_H': 0.001,   
    'alfa_SI': 0.01,   
    'evasao': 1e-01,      # Evasão imune
    'beta_SI_2': 1e-05,   # letalidade do SI na segunda fase
    
    'pi_release': 8.3474902837e+06,  # Altura do pico
    'beta_inf': 2.2982256469e-09,    # Velocidade da subida
    'K_hill_T': 6.1465811073e+05,    # Constante de hill, metade do máximo
    'beta': 9.5315248335e-08,        # Ativação do SI pelo parasita
    'beta_SI': 1.2410342764e-06,     # letalidade do SI
    'delta_T': 5.9670116539e-01,     # Morte natural
    'delta_I': 9.7153206125e-01,     # Morte natural do I
    'delta_SI': 2.2079343248e-02,    # Morte natural do SI

}

otimizados = [
    'pi_release', 
    'beta_inf', 
    #'K_hill_T', 
    'beta',       
    #'beta_SI', 
    #'delta_T',
    #'delta_I',
    #'delta_SI',
]

# ============================
# CONDIÇÕES INICIAIS
# ============================
condicoes_iniciais = [1e3, 0, parametros['H0'], parametros['SI0']]
#condicoes_iniciais = [T0, I0, H0, SI0]
# ============================
# LIMITES DE BUSCA 
# ============================

multi = 1.1

bounds_list = [
    (8.3474902837e+05, 8.3474902837e+07),      # pi_release
    (2.5e-9, 3.1e-9),       # beta_inf 
    #(4e7, 5e7),         # K_hill_T 
    (8.5e-8, 10.5e-8),       # beta
    #(1e-6, 1.5e-6),       # beta_SI 
    #(5.4e-1, 6.6e-1),         # delta_T
    #(8.7e-1, 10.7e-1),         # delta_I
    #(2.0e-2, 2.4e-2),         # delta_SI
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
    '''def __init__(self, p_base, param_keys, dias_exp, dados_exp_zscore, t_eval, cond_iniciais, bounds, media_exp, desvio_exp):
    #def __init__(self, p_base, param_keys, dias_exp, dados_exp, t_eval, cond_iniciais, bounds):
        self.p_base = p_base
        self.param_keys = param_keys
        self.dias_exp = dias_exp
        #self.dados_exp_brutos = dados_exp

        #ZSCORE
        self.dados_exp_zscore = dados_exp_zscore
        self.media_exp = media_exp
        self.desvio_exp = desvio_exp


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
        #ZSCORE
        Tp_model_zscore = (Tp_model - self.media_exp) / self.desvio_exp

        #erro = np.mean((np.log(Tp_model + 1) - np.log(np.array(self.dados_exp_brutos) + 1))**2)
        #ZSCORE
        erro = np.mean((Tp_model_zscore - self.dados_exp_zscore)**2)
        
        return [erro] # Pagmo ESPERA um tupla/lista de retorno
    '''


    'Usa desvio padrao para definir um intervalo de confiança'
    def __init__(self, p_base, param_keys, dias_exp, t_eval, cond_iniciais, bounds, 
                 media_exp, desvio_exp, limite_superior, limite_inferior):
        
        self.p_base = p_base
        self.param_keys = param_keys
        self.dias_exp = dias_exp
        
        self.media_exp = media_exp
        self.desvio_exp = desvio_exp
        
        # Transformando os limites para a escala Z-SCORE (para manter a consistência da sua otimização)
        self.upper_z = (limite_superior - media_exp) / desvio_exp
        self.lower_z = (limite_inferior - media_exp) / desvio_exp

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
        
        # Converte o modelo para Z-Score também
        Tp_model_zscore = (Tp_model - self.media_exp) / self.desvio_exp

        # LÓGICA DO INTERVALO:
        # Se Tp_model_zscore > upper_z: erro é (Tp - upper)
        # Se Tp_model_zscore < lower_z: erro é (lower - Tp)
        # Se estiver no meio: erro é 0
        
        diff_upper = np.maximum(0.0, Tp_model_zscore - self.upper_z) # Só pega valor se estourar pra cima
        diff_lower = np.maximum(0.0, self.lower_z - Tp_model_zscore) # Só pega valor se estourar pra baixo
        
        # O erro total é a soma quadrática dessas violações
        # Se a curva estiver toda dentro do tubo, o erro será 0.0
        erro = np.mean(diff_upper**2 + diff_lower**2)
        
        return [erro]

    def get_bounds(self):
        return self.bounds

# ============================
# RODAR SIMULAÇÃO
# ============================
if __name__ == '__main__':
    '''prob = pg.problem(ChagasProblemSimplificado(
        p_base=parametros, 
        param_keys=otimizados, 
        dias_exp=dias, 
        dados_exp_zscore=dados_exp_zscore, # Passa os dados transformados
        t_eval=t, 
        cond_iniciais=condicoes_iniciais, 
        bounds=bounds,
        media_exp=media_exp,   # Passa a média
        desvio_exp=desvio_exp  # Passa o desvio padrão
    ))'''

    'Usando intervalo de confiança'
    prob = pg.problem(ChagasProblemSimplificado(
        p_base=parametros, 
        param_keys=otimizados, 
        dias_exp=dias, 
        # dados_exp_zscore=dados_exp_zscore, # Não precisa mais passar a média pontual direta se não quiser
        t_eval=t, 
        cond_iniciais=condicoes_iniciais, 
        bounds=bounds,
        media_exp=media_exp,   
        desvio_exp=desvio_exp,
        limite_superior=limite_superior, # NOVO
        limite_inferior=limite_inferior  # NOVO
    ))

    # ===============================
    # PARAMETROS DE OTIMIZAÇÃO
    # ===============================
    populacao = 150
    iteracoes = 1000
    # ===============================
    # ===============================


    algo = pg.algorithm(pg.sade(gen=iteracoes, memory=True)) 

    pop = pg.population(prob, size=populacao)

    print(f"Iniciando otimização com {populacao} indivíduos por {iteracoes} gerações...")
    start_time = time.time()
    
    pop = algo.evolve(pop)
    
    end_time = time.time()
    print(f"\nOtimização concluída em {end_time - start_time:.2f} segundos.")
    
    # Resultados
    p_otim_values = pop.champion_x
    erro = pop.champion_f[0]

    p_otim = parametros.copy()
    
    output_text = "Parametros da Simulacao:\n"
    output_text += "---------------------------------------------------\n"
    output_text += f"{'Populacao (populacao)':<25} = {populacao}\n"
    output_text += f"{'Iteracoes (iteracoes)':<25} = {iteracoes}\n"
    output_text += f"{'Erro final (MSLE)':<25} = {erro:.6f}\n"
    output_text += "---------------------------------------------------\n"

    
    output_text += "Condicoes Iniciais:\n"
    ci_names = ['Tp0', 'I0', 'H0', 'SI0']
    for name, value in zip(ci_names, condicoes_iniciais):
        if 'e' in f"{value:.10e}":
            output_text += f"{name:<16} = {value:.10e}\n"
        else:
            output_text += f"{name:<16} = {value:.6f}\n"
    output_text += "---------------------------------------------------\n"

    output_text += "Limites de Busca:\n"
    for i, name in enumerate(otimizados):
        lower, upper = bounds_list[i]
        if 'e' in f"{lower:.10e}":
            output_text += f"{name}_min     = {lower:.10e}\n"
        else:
            output_text += f"{name}_min     = {lower:.6f}\n"
        if 'e' in f"{upper:.10e}":
            output_text += f"{name}_max     = {upper:.10e}\n"
        else:
            output_text += f"{name}_max     = {upper:.6f}\n"
    output_text += "---------------------------------------------------\n"

    output_text += "Parametros Base:\n"

    for name, value in parametros.items():
        if 'e' in f"{value:.10e}":
            output_text += f"{name:<16} = {value:.10e}\n"
        else:
            output_text += f"{name:<16} = {value:.6f}\n"

    output_text += "---------------------------------------------------\n"
    output_text += "Parametros Otimizados:\n"

    print("\n--- Configuração da Simulação ---")
    print(f"{'População (populacao)':<25} = {populacao}")
    print(f"{'Iterações (iteracoes)':<25} = {iteracoes}")
    print(f"{'Erro final (MSLE)':<25} = {erro:.6f}")
    print("--- Parâmetros Otimizados Encontrados ---")

    for name, value in zip(otimizados, p_otim_values):
        p_otim[name] = value
        if 'e' in f"{value:.10e}":
            print(f"{name:<16} = {value:.10e}")
            output_text += f"{name:<16} = {value:.10e}\n"
        else:
            print(f"{name:<16} = {value:.6f}")
            output_text += f"{name:<16} = {value:.6f}\n"

    output_text += "---------------------------------------------------\n"
    output_text += "Observacoes\n"
    output_text += "Valores do melhor resultado (simulacao 12 do usar_dados_melhor_grafico_e_hill)\n"
    output_text += "---------------------------------------------------\n"
    
    with open(os.path.join(output_dir, "parametros_otimizados.txt"), "w") as f:
        f.write(output_text)

    # Gerar o gráfico final
    sol_otim = integrate.odeint(modelo_simplificado.modelo_simplificado_cy, condicoes_iniciais, t, args=(p_otim,))
    Tp_curva = sol_otim[:, 0]

    plt.figure(figsize=(9, 5))
    plt.plot(dias, dados_experimentais_tp, 'o', label='Dados experimentais')
    plt.plot(t, Tp_curva, '-', label='Modelo')

    #=================
    #plt.yscale('log')
    #=================

    plt.xlabel('Dias')
    plt.ylabel('Trypomastigotas (Tp)')
    plt.title(f'Ajuste do modelo simplificado (Erro: {erro:.6f})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "grafico_ajuste_final.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nResultados salvos em: {os.path.abspath(output_dir)}")