"""
simulacao_computacional.py - SIFSM / Aurora Siger
Simulação do estado de carga (SoC) da bateria da colônia em 48 h marcianas.

Conceitos matemáticos usados:
  - Integração numérica (regra do trapézio): energia = integral da potência no tempo (área sob a curva)
  - Média de função: P_medio = (1/T) * integral de P(t) dt
  - Estado de carga: SoC(t) = SoC(0) + integral de (geração - consumo) dt, limitado entre 0 e a capacidade

Cenários comparados:
  1. Normal               -> geração solar plena
  2. Tempestade de poeira -> geração solar cai para 30%
  3. Tempestade + economia -> mesma tempestade, com corte de 25% no consumo total da colônia

Limitações: modelo simplificado (sem degradação da bateria, sem efeito da temperatura na
capacidade, perfil solar senoidal ideal, consumo com variação diária fixa).
Uso:  python simulacao_computacional.py   (gera graficos_ou_imagens/simulacao_bateria.png)
"""
import os

import matplotlib

matplotlib.use("Agg")  # salva em arquivo, sem abrir janela (funciona em qualquer computador)
import matplotlib.pyplot as plt
import numpy as np

PASTA = os.path.dirname(os.path.abspath(__file__))
PASTA_GRAFICOS = os.path.join(PASTA, "graficos_ou_imagens")

# Parâmetros da colônia (valores simulados)
PICO_SOLAR_KW = 700.0       # potência máxima dos painéis ao meio-dia
CAPACIDADE_KWH = 4000.0     # capacidade total das baterias
SOC_INICIAL = 0.80          # começa com 80% de carga
LIMITE_CRITICO = 0.20       # abaixo de 20% a colônia entra em modo crítico
HORAS = 48.0
PASSO_H = 0.1               # passo de tempo (6 minutos)


def geracao_solar_kw(t, fator=1.0):
    """Geração solar: senoide positiva entre 6 h e 18 h de cada ciclo de 24 h; zero à noite."""
    hora = t % 24
    onda = np.sin(np.pi * (hora - 6) / 12)
    return PICO_SOLAR_KW * fator * np.where((hora >= 6) & (hora <= 18), onda, 0.0)


def consumo_kw(t, base_kw, fator=1.0):
    """Consumo: valor médio (base) com variação de +-15% ao longo do dia (pico à tarde)."""
    return base_kw * fator * (1 + 0.15 * np.sin(2 * np.pi * (t - 8) / 24))


def integral_trapezio(y, t):
    """Integral numérica pela regra do trapézio: soma de (y[i]+y[i+1])/2 * dt."""
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(t)))


def simular(nome, fator_solar, base_kw, fator_consumo=1.0):
    """Executa um cenário e devolve um dicionário com séries e indicadores."""
    t = np.arange(0.0, HORAS + PASSO_H, PASSO_H)
    ger = geracao_solar_kw(t, fator_solar)
    con = consumo_kw(t, base_kw, fator_consumo)

    soc = np.zeros_like(t)
    soc[0] = SOC_INICIAL * CAPACIDADE_KWH
    deficit = 0.0  # energia que faltou (bateria vazia)
    for i in range(1, len(t)):
        liquida = (ger[i - 1] + ger[i]) / 2 - (con[i - 1] + con[i]) / 2  # potência líquida média no passo
        novo = soc[i - 1] + liquida * PASSO_H
        if novo < 0:
            deficit += -novo
            novo = 0.0
        soc[i] = min(novo, CAPACIDADE_KWH)

    energia_ger = integral_trapezio(ger, t)
    energia_con = integral_trapezio(con, t)
    soc_pct = 100 * soc / CAPACIDADE_KWH
    return {
        "nome": nome,
        "t": t,
        "geracao": ger,
        "consumo": con,
        "soc_pct": soc_pct,
        "energia_gerada_kwh": energia_ger,
        "energia_consumida_kwh": energia_con,
        "consumo_medio_kw": energia_con / HORAS,   # média de função
        "soc_minimo_pct": float(soc_pct.min()),
        "horas_modo_critico": float(np.sum(soc_pct < 100 * LIMITE_CRITICO) * PASSO_H),
        "deficit_kwh": deficit,
    }


def rodar_cenarios(base_kw):
    """Roda os 3 cenários com o consumo-base informado (kW)."""
    return [
        simular("Normal", 1.0, base_kw),
        simular("Tempestade de poeira", 0.30, base_kw),
        simular("Tempestade + economia (-25% total)", 0.30, base_kw, fator_consumo=0.75),
    ]


def salvar_grafico(cenarios, caminho=None):
    """Gera o gráfico comparando o SoC dos cenários e o balanço de potência do cenário normal."""
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    caminho = caminho or os.path.join(PASTA_GRAFICOS, "simulacao_bateria.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    n = cenarios[0]
    ax1.plot(n["t"], n["geracao"], label="Geração solar (normal)", color="#e0a100")
    ax1.plot(n["t"], n["consumo"], label="Consumo da colônia", color="#2a6fb0")
    ax1.fill_between(n["t"], n["geracao"], n["consumo"], where=n["geracao"] > n["consumo"],
                     alpha=0.2, color="green", label="Excedente (carrega bateria)")
    ax1.set_ylabel("Potência (kW)")
    ax1.set_title("Balanço de potência - a área sob cada curva é a energia (integral)")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    cores = ["#2a9d4a", "#c0392b", "#8e44ad"]
    for c, cor in zip(cenarios, cores):
        ax2.plot(c["t"], c["soc_pct"], label=c["nome"], color=cor, linewidth=2)
    ax2.axhline(100 * LIMITE_CRITICO, color="gray", linestyle="--", label="Limite crítico (20%)")
    ax2.set_ylabel("Estado de carga (%)")
    ax2.set_xlabel("Tempo (horas marcianas)")
    ax2.set_title("Estado de carga da bateria por cenário")
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
    return caminho


def imprimir_resumo(cenarios):
    print(f"{'Cenário':<38}{'Média kW':>10}{'SoC mín %':>11}{'h crítico':>11}{'Déficit kWh':>13}")
    for c in cenarios:
        print(f"{c['nome']:<38}{c['consumo_medio_kw']:>10.1f}{c['soc_minimo_pct']:>11.1f}"
              f"{c['horas_modo_critico']:>11.1f}{c['deficit_kwh']:>13.0f}")


if __name__ == "__main__":
    resultados = rodar_cenarios(base_kw=195.0)  # consumo-base de exemplo (kW)
    imprimir_resumo(resultados)
    print("Gráfico salvo em:", salvar_grafico(resultados))