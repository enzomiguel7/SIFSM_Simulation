# -*- coding: utf-8 -*-
"""
codigo_fonte.py - SIFSM: Sistema Integrado de Futuro Sustentável em Marte (Aurora Siger)

Equipe: [PREENCHER: nomes dos integrantes]

Arquivo principal do sistema. Execução:  python codigo_fonte.py

O que o sistema faz (resumo):
  1. Banco de dados SQLite com CRUD de sensores e leituras (tratamento de erros com try/except/else/finally)
  2. Regra booleana de alerta: ALERTA = (consumo_alto AND modulo_critico) OR (NOT sensor_valido)
  3. Estruturas de dados: fila de prioridade (heap), pilha (histórico/desfazer), tabela hash (dict), grafo (BFS)
  4. Simulação da bateria (arquivo simulacao_computacional.py) com integração numérica
  5. Regressão linear simples e múltipla (consumo x temperatura x ocupação)
  6. Agentes de IA simulados (triador, especialistas, ferramentas, handoff, guardrails, sessão, RAG simulado)
  7. Demonstração de arquitetura computacional (CPU sequencial x vetorizado) e reflexão socioambiental
  8. Opção 13: análise integrada da sustentabilidade da colônia (encadeia tudo)
"""
import csv
import heapq
import itertools
import os
import sqlite3
import sys
import time
import unicodedata
from collections import deque
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # gráficos são salvos em arquivo (não abrem janela)
import matplotlib.pyplot as plt
import numpy as np

import simulacao_computacional as sim

# ----------------------------------------------------------------------------
# Caminhos e constantes
# ----------------------------------------------------------------------------
PASTA = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PASTA, "banco_aurora_siger.db")
SQL_PATH = os.path.join(PASTA, "script_banco.sql")
CSV_PATH = os.path.join(PASTA, "dados_aurora_siger.csv")
PASTA_GRAFICOS = os.path.join(PASTA, "graficos_ou_imagens")
JANELA_CICLOS = 15  # análises usam os últimos 15 ciclos de leitura


# ============================================================================
# 1. BANCO DE DADOS E CRUD
# ============================================================================
def conectar():
    """Abre conexão com o banco e liga a verificação de chaves estrangeiras."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def banco_valido():
    """True se o arquivo do banco existe e já contém a tabela de leituras."""
    if not os.path.exists(DB_PATH):
        return False
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        r = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'leituras'").fetchone()
        return r[0] == 1
    except sqlite3.Error:
        return False
    finally:
        if conn:
            conn.close()


def inicializar_banco(forcar=False):
    """Cria as tabelas (script_banco.sql) e carrega as leituras do CSV na primeira execução."""
    if banco_valido() and not forcar:
        return
    conn = None
    falhou = False
    try:
        with open(SQL_PATH, encoding="utf-8") as f:
            script = f.read()
        conn = sqlite3.connect(DB_PATH)
        conn.executescript(script)  # cria tabelas + módulos + sensores
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            dados = [
                (r["codigo_sensor"], int(r["ciclo"]), float(r["consumo_kw"]),
                 float(r["temperatura_c"]), int(r["ocupacao"]), int(r["sensor_valido"]))
                for r in csv.DictReader(f)
            ]
        conn.executemany(
            "INSERT INTO leituras (codigo_sensor, ciclo, consumo_kw, temperatura_c, ocupacao, sensor_valido) "
            "VALUES (?, ?, ?, ?, ?, ?)", dados)
        conn.commit()
    except FileNotFoundError as e:
        falhou = True
        print(f"[ERRO] Arquivo necessário não encontrado: {e.filename}")
    except (sqlite3.Error, ValueError, KeyError) as e:
        falhou = True
        print(f"[ERRO] Falha ao criar o banco: {e}")
    else:
        print(f"Banco criado e {len(dados)} leituras carregadas de dados_aurora_siger.csv.")
    finally:
        if conn:
            conn.close()  # sempre fecha a conexão
        if falhou and os.path.exists(DB_PATH):
            os.remove(DB_PATH)  # não deixa um banco pela metade bloquear a próxima execução


def consultar(sql, params=()):
    """SELECT seguro: devolve lista de tuplas (ou lista vazia se houver erro)."""
    conn = None
    try:
        conn = conectar()
        return conn.execute(sql, params).fetchall()
    except sqlite3.Error as e:
        print(f"[ERRO] Falha na consulta ao banco: {e}")
        return []
    finally:
        if conn:
            conn.close()


def traduzir_erro(e):
    """Converte mensagens técnicas do SQLite em mensagens compreensíveis."""
    texto = str(e)
    if "UNIQUE" in texto:
        return "já existe um registro com esse código."
    if "FOREIGN KEY" in texto:
        return "o registro está relacionado a outros dados (ou o módulo/sensor informado não existe)."
    if "CHECK" in texto:
        return "algum valor está fora das regras do banco (ex.: número negativo)."
    return texto


def executar_escrita(sql, params=()):
    """
    INSERT/UPDATE/DELETE com try/except/else/finally.
    Devolve (ok, mensagem, linhas_afetadas).
    """
    conn = None
    try:
        conn = conectar()
        cur = conn.execute(sql, params)
        conn.commit()
    except sqlite3.IntegrityError as e:  # violação de regra (duplicado, FK, CHECK)
        if conn:
            conn.rollback()
        return False, traduzir_erro(e), 0
    except sqlite3.Error as e:  # qualquer outra falha do banco
        if conn:
            conn.rollback()
        return False, f"falha no banco de dados: {e}", 0
    else:  # só executa se NÃO houve erro
        return True, "operação concluída", cur.rowcount
    finally:  # sempre executa, com ou sem erro
        if conn:
            conn.close()


# ---------------- Entrada de dados com validação ----------------
def pedir(texto):
    return input(texto).strip()


def ler_float(texto, minimo=None, maximo=None):
    valor = float(pedir(texto).replace(",", "."))  # ValueError se não for número
    if minimo is not None and valor < minimo:
        raise ValueError(f"o valor deve ser maior ou igual a {minimo}")
    if maximo is not None and valor > maximo:
        raise ValueError(f"o valor deve ser menor ou igual a {maximo}")
    return valor


def ler_int(texto, minimo=None):
    valor = int(pedir(texto))  # ValueError se não for inteiro
    if minimo is not None and valor < minimo:
        raise ValueError(f"o valor deve ser maior ou igual a {minimo}")
    return valor


def ler_sim_nao(texto):
    resp = pedir(texto).lower()
    if resp not in ("s", "n"):
        raise ValueError("responda apenas com s ou n")
    return resp == "s"


# ---------------- ESTRUTURA: PILHA (histórico com desfazer) ----------------
HISTORICO = []  # pilha: último a entrar é o primeiro a sair (LIFO)


def empilhar(descricao, sql_desfazer, params):
    HISTORICO.append({"descricao": descricao, "sql": sql_desfazer, "params": params,
                      "hora": datetime.now().strftime("%H:%M:%S")})


def desfazer_ultima():
    if not HISTORICO:
        print("Histórico vazio: não há nada para desfazer.")
        return
    item = HISTORICO[-1]  # topo da pilha
    ok, msg, _ = executar_escrita(item["sql"], item["params"])
    if ok:
        HISTORICO.pop()
        print(f"Desfeito: {item['descricao']}")
    else:
        print(f"[ERRO] Não foi possível desfazer ({msg}). A ação continua no topo da pilha.")


def mostrar_historico():
    print("\n--- Histórico de ações (topo da pilha primeiro) ---")
    if not HISTORICO:
        print("(vazio)")
    for item in reversed(HISTORICO):
        print(f"  {item['hora']}  {item['descricao']}")


# ---------------- CRUD de SENSORES ----------------
def listar_sensores():
    print("\n--- Sensores cadastrados ---")
    linhas = consultar(
        "SELECT s.codigo, m.sigla, s.tipo, s.ativo, "
        "(SELECT COUNT(*) FROM leituras l WHERE l.codigo_sensor = s.codigo) "
        "FROM sensores s JOIN modulos m ON m.id_modulo = s.id_modulo ORDER BY s.codigo")
    print(f"{'Código':<12}{'Módulo':<8}{'Tipo':<14}{'Ativo':<7}{'Leituras':>8}")
    for cod, sigla, tipo, ativo, n in linhas:
        print(f"{cod:<12}{sigla:<8}{tipo:<14}{'sim' if ativo else 'não':<7}{n:>8}")
    print(f"Total: {len(linhas)} sensor(es).")


def buscar_sensor(codigo):
    """Busca um sensor pelo código no banco (retorna tupla ou None)."""
    r = consultar("SELECT codigo, id_modulo, tipo, ativo FROM sensores WHERE codigo = ?", (codigo,))
    return r[0] if r else None


def cadastrar_sensor():
    print("\n--- Cadastrar sensor ---")
    try:
        codigo = pedir("Código do sensor (ex.: TEMP-HAB): ").upper()
        if not codigo or len(codigo) > 20 or " " in codigo:
            raise ValueError("código inválido (use até 20 caracteres, sem espaços)")
        sigla = pedir("Sigla do módulo (ENE/HAB/AGR/LAB/MED/COM): ").upper()
        tipo = pedir("Tipo do sensor (consumo/temperatura/umidade/...): ").lower()
        if not tipo:
            raise ValueError("o tipo não pode ficar vazio")
    except ValueError as e:
        print(f"[ERRO] Entrada inválida: {e}")
        return
    modulo = consultar("SELECT id_modulo FROM modulos WHERE sigla = ?", (sigla,))
    if not modulo:
        print(f"[ERRO] Módulo '{sigla}' não existe.")
        return
    ok, msg, _ = executar_escrita(
        "INSERT INTO sensores (codigo, id_modulo, tipo, ativo) VALUES (?, ?, ?, 1)",
        (codigo, modulo[0][0], tipo))
    if ok:
        print(f"Sensor {codigo} cadastrado com sucesso.")
        empilhar(f"Cadastro do sensor {codigo}", "DELETE FROM sensores WHERE codigo = ?", (codigo,))
    else:
        print(f"[ERRO] Não foi possível cadastrar: {msg}")


def alterar_sensor():
    print("\n--- Alterar sensor ---")
    codigo = pedir("Código do sensor a alterar: ").upper()
    atual = buscar_sensor(codigo)
    if atual is None:
        print(f"[ERRO] Sensor '{codigo}' não encontrado.")
        return
    _, id_modulo, tipo_atual, ativo_atual = atual
    try:
        novo_tipo = pedir(f"Novo tipo [{tipo_atual}] (Enter mantém): ").lower() or tipo_atual
        resp = pedir(f"Sensor ativo? s/n [{'s' if ativo_atual else 'n'}] (Enter mantém): ").lower()
        if resp not in ("", "s", "n"):
            raise ValueError("responda s, n ou Enter")
        novo_ativo = ativo_atual if resp == "" else int(resp == "s")
    except ValueError as e:
        print(f"[ERRO] Entrada inválida: {e}")
        return
    ok, msg, _ = executar_escrita("UPDATE sensores SET tipo = ?, ativo = ? WHERE codigo = ?",
                                  (novo_tipo, novo_ativo, codigo))
    if ok:
        print(f"Sensor {codigo} alterado com sucesso.")
        empilhar(f"Alteração do sensor {codigo}", "UPDATE sensores SET tipo = ?, ativo = ? WHERE codigo = ?",
                 (tipo_atual, ativo_atual, codigo))
    else:
        print(f"[ERRO] Não foi possível alterar: {msg}")


def excluir_sensor():
    print("\n--- Excluir sensor ---")
    codigo = pedir("Código do sensor a excluir: ").upper()
    atual = buscar_sensor(codigo)
    if atual is None:
        print(f"[ERRO] Sensor '{codigo}' não encontrado.")
        return
    try:
        if not ler_sim_nao(f"Confirma a exclusão de {codigo}? (s/n): "):
            print("Exclusão cancelada.")
            return
    except ValueError as e:
        print(f"[ERRO] {e}. Exclusão cancelada.")
        return
    ok, msg, _ = executar_escrita("DELETE FROM sensores WHERE codigo = ?", (codigo,))
    if ok:
        print(f"Sensor {codigo} excluído.")
        empilhar(f"Exclusão do sensor {codigo}",
                 "INSERT INTO sensores (codigo, id_modulo, tipo, ativo) VALUES (?, ?, ?, ?)", atual)
    else:
        print(f"[ERRO] Não foi possível excluir: {msg}")
        print("       (Dica: sensores que já possuem leituras não podem ser excluídos.)")


# ---------------- CRUD de LEITURAS ----------------
def cadastrar_leitura():
    print("\n--- Cadastrar leitura ---")
    try:
        codigo = pedir("Código do sensor: ").upper()
        if buscar_sensor(codigo) is None:
            raise ValueError(f"sensor '{codigo}' não existe")
        ciclo = ler_int("Ciclo (hora simulada, inteiro >= 0): ", 0)
        consumo = ler_float("Consumo em kW (>= 0): ", 0)
        temp = ler_float("Temperatura externa em °C (-150 a 40): ", -150, 40)
        ocup = ler_int("Ocupação (pessoas no módulo, >= 0): ", 0)
        valido = ler_sim_nao("O sensor está funcionando corretamente? (s/n): ")
    except ValueError as e:
        print(f"[ERRO] Entrada inválida: {e}")
        return
    ok, msg, _ = executar_escrita(
        "INSERT INTO leituras (codigo_sensor, ciclo, consumo_kw, temperatura_c, ocupacao, sensor_valido) "
        "VALUES (?, ?, ?, ?, ?, ?)", (codigo, ciclo, consumo, temp, ocup, int(valido)))
    print("Leitura cadastrada com sucesso." if ok else f"[ERRO] Não foi possível cadastrar: {msg}")


def listar_leituras():
    print("\n--- Listar leituras ---")
    codigo = pedir("Código do sensor (Enter = todos): ").upper()
    try:
        qtd = int(pedir("Quantas leituras mostrar? [10]: ") or 10)
        if qtd < 1:
            raise ValueError
    except ValueError:
        print("[ERRO] Quantidade inválida; usando 10.")
        qtd = 10
    filtro = "WHERE codigo_sensor = ?" if codigo else ""
    params = (codigo, qtd) if codigo else (qtd,)
    linhas = consultar(
        f"SELECT id_leitura, codigo_sensor, ciclo, consumo_kw, temperatura_c, ocupacao, sensor_valido "
        f"FROM leituras {filtro} ORDER BY ciclo DESC, id_leitura DESC LIMIT ?", params)
    print(f"{'ID':>5} {'Sensor':<10}{'Ciclo':>6}{'kW':>9}{'Temp °C':>9}{'Ocup':>6}  Válido")
    for i, cod, ciclo, kw, t, oc, v in linhas:
        print(f"{i:>5} {cod:<10}{ciclo:>6}{kw:>9.2f}{t:>9.2f}{oc:>6}  {'sim' if v else 'NÃO'}")
    if not linhas:
        print("Nenhuma leitura encontrada.")


def alterar_leitura():
    print("\n--- Alterar leitura (consumo) ---")
    try:
        id_l = ler_int("ID da leitura: ", 1)
        novo = ler_float("Novo consumo em kW (>= 0): ", 0)
    except ValueError as e:
        print(f"[ERRO] Entrada inválida: {e}")
        return
    ok, msg, n = executar_escrita("UPDATE leituras SET consumo_kw = ? WHERE id_leitura = ?", (novo, id_l))
    if not ok:
        print(f"[ERRO] {msg}")
    elif n == 0:
        print(f"[ERRO] Leitura {id_l} não encontrada.")
    else:
        print(f"Leitura {id_l} alterada.")


def excluir_leitura():
    print("\n--- Excluir leitura ---")
    try:
        id_l = ler_int("ID da leitura: ", 1)
        if not ler_sim_nao(f"Confirma excluir a leitura {id_l}? (s/n): "):
            print("Exclusão cancelada.")
            return
    except ValueError as e:
        print(f"[ERRO] Entrada inválida: {e}")
        return
    ok, msg, n = executar_escrita("DELETE FROM leituras WHERE id_leitura = ?", (id_l,))
    if not ok:
        print(f"[ERRO] {msg}")
    elif n == 0:
        print(f"[ERRO] Leitura {id_l} não encontrada.")
    else:
        print(f"Leitura {id_l} excluída.")


def menu_leituras():
    while True:
        print("\n=== Leituras dos sensores ===\n1) Cadastrar  2) Listar  3) Alterar  4) Excluir  0) Voltar")
        op = pedir("Opção: ")
        if op == "0":
            return
        {"1": cadastrar_leitura, "2": listar_leituras, "3": alterar_leitura,
         "4": excluir_leitura}.get(op, lambda: print("Opção inválida."))()


def reiniciar_banco():
    print("\n--- Recarregar dados do CSV ---")
    try:
        if not ler_sim_nao("Isso APAGA as alterações feitas e recria o banco. Continuar? (s/n): "):
            print("Cancelado.")
            return
    except ValueError as e:
        print(f"[ERRO] {e}. Cancelado.")
        return
    inicializar_banco(forcar=True)
    HISTORICO.clear()


# ============================================================================
# 2. REGRA BOOLEANA E CIRCUITO LÓGICO
# ============================================================================
def regra_alerta(consumo_alto, modulo_critico, sensor_valido):
    """
    ALERTA = (consumo_alto AND modulo_critico) OR (NOT sensor_valido)
    No circuito: uma porta AND, uma NOT e uma OR (arquivo circuito_logico.png).
    """
    return (consumo_alto and modulo_critico) or (not sensor_valido)


def mostrar_tabela_verdade():
    print("\nExpressão booleana:  ALERTA = (A . B) + V'")
    print("  A = consumo_alto   B = modulo_critico   V = sensor_valido   (V' = NOT V)\n")
    print(" A  B  V | A.B  V' | ALERTA")
    print("---------+---------+-------")
    for a, b, v in itertools.product([0, 1], repeat=3):
        print(f" {a}  {b}  {v} |  {int(a and b)}   {int(not v)}  |   {int(regra_alerta(a, b, v))}")
    print("\nCircuito: portas AND(A,B) e NOT(V) -> OR -> LED/alerta. Veja circuito_logico.png e link_tinkercad.txt.")


def testar_combinacao():
    print("\n--- Testar a regra com valores informados ---")
    try:
        a = ler_sim_nao("Consumo está alto? (s/n): ")
        b = ler_sim_nao("O módulo é crítico? (s/n): ")
        v = ler_sim_nao("O sensor está válido? (s/n): ")
    except ValueError as e:
        print(f"[ERRO] {e}")
        return
    r = regra_alerta(a, b, v)
    print("Resultado: ALERTA ACIONADO (LED aceso)" if r else "Resultado: operação normal (LED apagado)")


def ciclo_inicial_janela(janela=JANELA_CICLOS):
    """Primeiro ciclo da janela = menor dos últimos N ciclos DISTINTOS (robusto a ciclos digitados com saltos)."""
    r = consultar("SELECT MIN(ciclo) FROM (SELECT DISTINCT ciclo FROM leituras ORDER BY ciclo DESC LIMIT ?)",
                  (janela,))
    return r[0][0] if r and r[0][0] is not None else 0


def avaliar_leituras(janela=JANELA_CICLOS):
    """Aplica a regra booleana às leituras recentes. Devolve (lista_de_alertas, ciclo_inicial)."""
    ini = ciclo_inicial_janela(janela)
    linhas = consultar(
        "SELECT l.id_leitura, l.codigo_sensor, m.sigla, m.critico, m.limite_consumo_kw, "
        "l.ciclo, l.consumo_kw, l.sensor_valido "
        "FROM leituras l JOIN sensores s ON s.codigo = l.codigo_sensor "
        "JOIN modulos m ON m.id_modulo = s.id_modulo WHERE l.ciclo >= ? ORDER BY l.ciclo", (ini,))
    alertas = []
    for id_l, cod, sigla, crit, lim, ciclo, cons, valido in linhas:
        # leitura de sensor inválido não é confiável: não conta como consumo alto
        consumo_alto = bool(valido) and cons > lim
        if regra_alerta(consumo_alto, bool(crit), bool(valido)):
            if not valido:
                prio, desc = 2, f"Sensor {cod} inválido no ciclo {ciclo} (módulo {sigla}) - verificar manutenção"
            else:
                prio, desc = 1, (f"Consumo alto em módulo crítico {sigla}: {cons:.1f} kW "
                                 f"(limite {lim:.0f} kW) no ciclo {ciclo}")
            alertas.append({"id_leitura": id_l, "sigla": sigla, "ciclo": ciclo, "prioridade": prio,
                            "descricao": desc, "excesso": (cons / lim - 1) if valido else 0.0})
    return alertas, ini


def gravar_alertas(alertas):
    """Grava os alertas na tabela alertas (substitui os anteriores) em uma única transação."""
    conn = None
    try:
        conn = conectar()
        conn.execute("DELETE FROM alertas")
        agora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.executemany("INSERT INTO alertas (id_leitura, prioridade, descricao, criado_em) VALUES (?, ?, ?, ?)",
                         [(a["id_leitura"], a["prioridade"], a["descricao"], agora) for a in alertas])
        conn.commit()
    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        print(f"[ERRO] Não foi possível gravar os alertas: {e}")
        return False
    else:
        return True
    finally:
        if conn:
            conn.close()


def aplicar_regra_nas_leituras():
    alertas, ini = avaliar_leituras()
    print(f"\nRegra aplicada às leituras dos ciclos {ini} em diante: {len(alertas)} alerta(s) gerado(s).")
    if alertas and gravar_alertas(alertas):
        print("Alertas gravados na tabela 'alertas'. Veja a ordem de prioridade em Estruturas de dados > Fila.")


def menu_regra():
    while True:
        print("\n=== Regra lógica de decisão ===")
        print("1) Tabela verdade, expressão e circuito  2) Testar combinação  3) Aplicar nas leituras  0) Voltar")
        op = pedir("Opção: ")
        if op == "0":
            return
        {"1": mostrar_tabela_verdade, "2": testar_combinacao,
         "3": aplicar_regra_nas_leituras}.get(op, lambda: print("Opção inválida."))()


# ============================================================================
# 3. ESTRUTURAS DE DADOS
# ============================================================================
# ---- Fila de prioridade (heap): o alerta mais urgente sai primeiro ----
def montar_fila(alertas):
    fila = []
    for a in alertas:
        # chave: (prioridade, maior excesso primeiro, ciclo, id) -> o dict nunca é comparado
        heapq.heappush(fila, (a["prioridade"], -a["excesso"], a["ciclo"], a["id_leitura"], a))
    return fila


def demonstrar_fila():
    alertas, ini = avaliar_leituras()
    fila = montar_fila(alertas)
    print(f"\n--- Fila de prioridade de alertas (ciclos {ini}+): {len(fila)} na fila ---")
    n = 0
    while fila and n < 8:
        prio, _, _, _, a = heapq.heappop(fila)  # sempre retira o mais urgente
        n += 1
        print(f"  {n}. [P{prio}] {a['descricao']}")
    if fila:
        print(f"  ... e mais {len(fila)} alerta(s) na fila.")
    if not alertas:
        print("  Nenhum alerta no momento.")


# ---- Tabela hash (dict): busca por código em tempo constante ----
def demonstrar_hash():
    print("\n--- Tabela hash: busca de sensores por código ---")
    linhas = consultar("SELECT codigo, id_modulo, tipo, ativo FROM sensores")
    indice = {cod: (mod, tipo, ativo) for cod, mod, tipo, ativo in linhas}  # código -> dados
    print(f"Índice criado com {len(indice)} sensores reais.")
    # comparação com 200 mil sensores fictícios para o efeito ficar visível
    grande = [f"SENS-{i:06d}" for i in range(200_000)]
    indice_grande = {c: i for i, c in enumerate(grande)}
    alvo = "SENS-199999"
    t0 = time.perf_counter()
    achou = next(i for i, c in enumerate(grande) if c == alvo)  # busca linear: percorre a lista
    t_lista = time.perf_counter() - t0
    t0 = time.perf_counter()
    achou_h = indice_grande[alvo]  # busca hash: vai direto
    t_hash = time.perf_counter() - t0
    print(f"Busca de {alvo} entre 200.000 sensores:")
    print(f"  lista (busca linear): {t_lista * 1000:8.3f} ms")
    print(f"  tabela hash (dict)  : {t_hash * 1000:8.5f} ms  (mesma posição: {achou == achou_h})")
    cod = pedir("Digite um código real para buscar (ex.: CONS-HAB): ").upper()
    dado = indice.get(cod)
    print(f"  Encontrado: {dado}" if dado else f"  Sensor '{cod}' não existe no índice.")


# ---- Grafo: dependências entre módulos (aresta A -> B: A fornece algo a B) ----
GRAFO = {
    "ENE": ["HAB", "AGR", "LAB", "MED", "COM"],  # energia alimenta todos
    "AGR": ["HAB"],                              # alimentos para a habitação
    "COM": ["MED", "LAB"],                       # telemedicina e dados do laboratório
    "HAB": [], "LAB": [], "MED": [],
}


def modulos_afetados(origem):
    """Busca em largura (BFS): quais módulos são afetados se 'origem' falhar."""
    visitados = {origem}
    fila = deque([(origem, 0)])
    resultado = []
    while fila:
        atual, nivel = fila.popleft()
        for vizinho in GRAFO.get(atual, []):
            if vizinho not in visitados:
                visitados.add(vizinho)
                resultado.append((vizinho, nivel + 1))
                fila.append((vizinho, nivel + 1))
    return resultado


def demonstrar_grafo():
    print("\n--- Grafo de dependências entre módulos ---")
    for origem, destinos in GRAFO.items():
        print(f"  {origem} -> {', '.join(destinos) if destinos else '(nenhum)'}")
    sigla = pedir("Qual módulo falhou? (ENE/HAB/AGR/LAB/MED/COM): ").upper()
    if sigla not in GRAFO:
        print(f"[ERRO] Módulo '{sigla}' não existe.")
        return
    afetados = modulos_afetados(sigla)
    if afetados:
        print(f"Se {sigla} falhar, são afetados: " + ", ".join(f"{m} (nível {n})" for m, n in afetados))
    else:
        print(f"{sigla} não tem módulos dependentes: a falha fica isolada.")


def menu_estruturas():
    while True:
        print("\n=== Estruturas de dados ===")
        print("1) Fila de prioridade (alertas)  2) Pilha (histórico/desfazer)  3) Tabela hash  4) Grafo  0) Voltar")
        op = pedir("Opção: ")
        if op == "0":
            return
        if op == "2":
            mostrar_historico()
            if HISTORICO and ler_sim_nao_seguro("Desfazer a última ação? (s/n): "):
                desfazer_ultima()
        else:
            {"1": demonstrar_fila, "3": demonstrar_hash,
             "4": demonstrar_grafo}.get(op, lambda: print("Opção inválida."))()


def ler_sim_nao_seguro(texto):
    try:
        return ler_sim_nao(texto)
    except ValueError as e:
        print(f"[ERRO] {e}")
        return False


# ============================================================================
# 4. SIMULAÇÃO COMPUTACIONAL
# ============================================================================
def consumo_base_do_banco(janela=JANELA_CICLOS):
    """Soma do consumo médio (kW) de cada módulo nas leituras válidas recentes."""
    ini = ciclo_inicial_janela(janela)
    linhas = consultar(
        "SELECT m.sigla, AVG(l.consumo_kw) FROM leituras l "
        "JOIN sensores s ON s.codigo = l.codigo_sensor JOIN modulos m ON m.id_modulo = s.id_modulo "
        "WHERE l.sensor_valido = 1 AND s.tipo = 'consumo' AND l.ciclo >= ? "
        "GROUP BY m.sigla ORDER BY m.id_modulo", (ini,))
    return sum(v for _, v in linhas), linhas


def executar_simulacao(verbose=True):
    total, linhas = consumo_base_do_banco()
    if total <= 0:
        total = 195.0
        print("Sem leituras válidas no banco; usando consumo-base padrão de 195 kW.")
    if verbose:
        print(f"\n--- Simulação da bateria (48 h) ---\nConsumo-base vindo do banco: {total:.1f} kW "
              f"(soma da média de {len(linhas)} módulos)\n")
    cenarios = sim.rodar_cenarios(total)
    sim.imprimir_resumo(cenarios)
    caminho = sim.salvar_grafico(cenarios)
    if verbose:
        print(f"\nGráfico salvo em: {caminho}")
        print("Interpretação: energia = área sob a curva de potência (integral pela regra do trapézio);")
        print("consumo médio = integral / 48 h (média de função). Limitações: modelo simplificado (sem")
        print("degradação da bateria, sem efeito da temperatura, geração solar senoidal ideal).")
    return cenarios


# ============================================================================
# 5. REGRESSÃO LINEAR
# ============================================================================
def dados_regressao(sigla):
    """Lê temperatura, ocupação e consumo do módulo (somente leituras válidas: limpeza de dados)."""
    linhas = consultar(
        "SELECT l.temperatura_c, l.ocupacao, l.consumo_kw FROM leituras l "
        "JOIN sensores s ON s.codigo = l.codigo_sensor JOIN modulos m ON m.id_modulo = s.id_modulo "
        "WHERE m.sigla = ? AND s.tipo = 'consumo' AND l.sensor_valido = 1 ORDER BY l.ciclo", (sigla,))
    return np.array(linhas, dtype=float)


def ajustar_mqo(X, y):
    """Mínimos quadrados ordinários: devolve [intercepto, coef1, coef2...]."""
    Xb = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return beta


def prever(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def metricas(y, y_prev):
    ss_res = float(np.sum((y - y_prev) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot, float(np.sqrt(ss_res / len(y)))  # R², RMSE


def executar_regressao(sigla="HAB", verbose=True, salvar=True):
    """Regressão simples (temperatura) e múltipla (temperatura + ocupação) do consumo de um módulo."""
    dados = dados_regressao(sigla)
    if len(dados) < 10:
        print(f"[ERRO] Poucos dados válidos para o módulo {sigla} ({len(dados)}). São necessários ao menos 10.")
        return None
    temp, ocup, y = dados[:, 0], dados[:, 1], dados[:, 2]
    n = len(y)

    # separação treino (80%) / teste (20%) com semente fixa
    rng = np.random.default_rng(7)
    idx = rng.permutation(n)
    n_teste = max(2, int(0.2 * n))
    te, tr = idx[:n_teste], idx[n_teste:]

    b1 = ajustar_mqo(temp[tr].reshape(-1, 1), y[tr])
    b2 = ajustar_mqo(np.column_stack([temp[tr], ocup[tr]]), y[tr])
    r2_1_tr, _ = metricas(y[tr], prever(b1, temp[tr].reshape(-1, 1)))
    r2_1_te, rmse_1 = metricas(y[te], prever(b1, temp[te].reshape(-1, 1)))
    r2_2_tr, _ = metricas(y[tr], prever(b2, np.column_stack([temp[tr], ocup[tr]])))
    r2_2_te, rmse_2 = metricas(y[te], prever(b2, np.column_stack([temp[te], ocup[te]])))

    corr = np.corrcoef(np.column_stack([temp, ocup, y]).T)
    r_temp_ocup = corr[0, 1]
    vif = 1 / (1 - r_temp_ocup ** 2)
    resid = y[tr] - prever(b2, np.column_stack([temp[tr], ocup[tr]]))
    outliers = int(np.sum(np.abs(resid) > 2 * resid.std()))

    if verbose:
        print(f"\n--- Regressão linear: consumo do módulo {sigla} ---")
        print(f"Observações válidas: {n} (treino {len(tr)}, teste {len(te)})  |  leituras inválidas ficam de fora")
        print("\nAnálise exploratória (correlação de Pearson):")
        print(f"  consumo x temperatura : {corr[2, 0]:+.2f}")
        print(f"  consumo x ocupação    : {corr[2, 1]:+.2f}")
        print(f"  temperatura x ocupação: {corr[0, 1]:+.2f}  (fator de inflação da variância VIF = {vif:.2f})")
        print(f"  consumo médio {y.mean():.1f} kW | mín {y.min():.1f} | máx {y.max():.1f} | desvio {y.std():.1f}")
        print("\nModelo simples   consumo = b0 + b1*temperatura")
        print(f"  b0 = {b1[0]:.2f} kW   b1 = {b1[1]:+.3f} kW/°C   R² treino {r2_1_tr:.3f} | R² teste {r2_1_te:.3f} | RMSE teste {rmse_1:.2f} kW")
        print("Modelo múltiplo  consumo = b0 + b1*temperatura + b2*ocupação")
        print(f"  b0 = {b2[0]:.2f} kW   b1 = {b2[1]:+.3f} kW/°C   b2 = {b2[2]:+.3f} kW/pessoa")
        print(f"  R² treino {r2_2_tr:.3f} | R² teste {r2_2_te:.3f} | RMSE teste {rmse_2:.2f} kW")
        print("\nInterpretação:")
        print(f"  - Cada 1 °C mais frio aumenta o consumo em cerca de {abs(b2[1]):.2f} kW (aquecimento), mantida a ocupação.")
        print(f"  - Cada pessoa adicional no módulo aumenta o consumo em cerca de {b2[2]:.2f} kW, mantida a temperatura.")
        print(f"  - O modelo múltiplo explica {100 * r2_2_te:.0f}% da variação no teste (simples: {100 * r2_1_te:.0f}%).")
        print(f"  - Resíduos fora de 2 desvios: {outliers}. VIF {vif:.2f}: " +
              ("colinearidade baixa: coeficientes bem interpretáveis." if vif < 2.5
               else "colinearidade moderada (VIF < 5): coeficientes interpretáveis, mas com cautela." if vif < 5
               else "colinearidade alta: interpretar os coeficientes individuais com muito cuidado."))
        if abs(r_temp_ocup) > 0.5 and abs(b1[1]) > 1.2 * abs(b2[1]):
            print(f"  - O efeito da temperatura mudou de {b1[1]:+.3f} (modelo simples) para {b2[1]:+.3f} (múltiplo): parte do "
                  "efeito aparente do frio\n    era da ocupação (variável omitida), pois frio e ocupação andam juntos.")
        print("  - Limitação: dados SIMULADOS e poucos pontos; correlação não prova causa; só vale na faixa observada.")

    if salvar:
        os.makedirs(PASTA_GRAFICOS, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.scatter(temp, y, c=ocup, cmap="viridis", s=35)
        xs = np.linspace(temp.min(), temp.max(), 50)
        ax1.plot(xs, b1[0] + b1[1] * xs, color="#c0392b", label="Regressão simples")
        ax1.set_xlabel("Temperatura externa (°C)")
        ax1.set_ylabel(f"Consumo do módulo {sigla} (kW)")
        ax1.set_title("Consumo x temperatura (cor = ocupação)")
        ax1.legend()
        ax1.grid(alpha=0.3)
        prev_tr = prever(b2, np.column_stack([temp[tr], ocup[tr]]))
        ax2.scatter(prev_tr, resid, s=30, color="#2a6fb0")
        ax2.axhline(0, color="gray", linestyle="--")
        ax2.set_xlabel("Consumo previsto (kW)")
        ax2.set_ylabel("Resíduo (kW)")
        ax2.set_title("Resíduos do modelo múltiplo")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        caminho = os.path.join(PASTA_GRAFICOS, "regressao_consumo.png")
        fig.savefig(caminho, dpi=130)
        plt.close(fig)
        if verbose:
            print(f"\nGráfico salvo em: {caminho}")
    return {"sigla": sigla, "b1": b1, "b2": b2, "r2_simples": r2_1_te, "r2_multiplo": r2_2_te,
            "rmse_multiplo": rmse_2, "vif": vif, "n": n}


def menu_regressao():
    sigla = pedir("Módulo para analisar (ENE/HAB/AGR/LAB/MED/COM) [HAB]: ").upper() or "HAB"
    executar_regressao(sigla)


# ============================================================================
# 6. AGENTES DE IA (SIMULADOS, DETERMINÍSTICOS, SEM API EXTERNA)
# ============================================================================
# Arquitetura: guardrail -> triador -> agente especialista (usa ferramentas) -> handoff opcional
# Um LLM real entraria no lugar das regras de palavras-chave, mas as ferramentas e guardrails seriam os mesmos.
def normalizar(texto):
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.lower()


MODULO_PALAVRAS = {
    "HAB": ["habitacao", "habitat", "alojamento"],
    "AGR": ["agricultura", "estufa", "plantacao"],
    "LAB": ["laboratorio"],
    "MED": ["medico", "medica", "enfermaria"],
    "COM": ["comunicacao", "radio"],
    "ENE": ["gerador", "painel", "modulo de energia"],
}

# Base de conhecimento simulada (RAG): o agente "recupera" o procedimento por palavra-chave
BASE_CONHECIMENTO = {
    "bateria": "P-01: manter o estado de carga acima de 20%. Abaixo disso, ativar modo economia e cortar cargas "
               "não críticas (laboratório e agricultura) antes de afetar habitação, suporte médico e comunicação.",
    "poeira": "P-02: em tempestade de poeira a geração solar cai cerca de 70%. Ativar o modo economia antes da "
              "chegada e priorizar suporte médico, comunicação e habitação.",
    "sensor": "P-03: leitura inválida (sensor_valido = 0) não deve ser usada. Abrir chamado de manutenção "
              "prioridade 2 e conferir o sensor presencialmente.",
    "temperatura": "P-04: temperaturas externas muito baixas elevam o consumo de aquecimento. Verificar isolamento "
                   "dos módulos e revisar a previsão de consumo noturno.",
}

GUARDRAILS_ACOES = ["abrir eclusa", "desligar suporte", "desativar alarme", "desligar oxigenio", "cortar oxigenio",
                    "apagar historico", "apagar registros", "ignorar regras"]
GUARDRAILS_PRIVACIDADE = ["dados pessoais", "saude da tripulacao", "localizacao da tripulacao", "quem esta no"]
AVISO_HUMANO = "Sugestão automática: requer validação humana (engenharia/comando) antes de qualquer ação."


class Sessao:
    """Guarda o histórico da conversa e o último módulo citado (contexto entre perguntas)."""
    def __init__(self):
        self.historico = []
        self.ultimo_modulo = None


def guardrail(pedido_norm):
    """Barreira de segurança: bloqueia ações críticas, pedidos de dados pessoais e entradas vazias."""
    if not pedido_norm:
        return False, "pedido vazio."
    if any(p in pedido_norm for p in GUARDRAILS_ACOES):
        return False, "ação crítica fora do escopo. Os agentes apenas consultam e sugerem; nunca comandam equipamentos."
    if any(p in pedido_norm for p in GUARDRAILS_PRIVACIDADE):
        return False, "dados pessoais da tripulação são protegidos (privacidade). Apenas dados agregados são usados."
    return True, ""


def detectar_modulo(pedido_norm, sessao):
    for sigla, palavras in MODULO_PALAVRAS.items():
        if any(p in pedido_norm for p in palavras):
            sessao.ultimo_modulo = sigla
            return sigla
    return sessao.ultimo_modulo  # usa o contexto da sessão, se existir


# ---- Ferramentas determinísticas (funções que consultam o banco) ----
def tool_consumo(sigla=None):
    ini = ciclo_inicial_janela()
    sql = ("SELECT m.sigla, m.nome, AVG(l.consumo_kw), m.limite_consumo_kw FROM leituras l "
           "JOIN sensores s ON s.codigo = l.codigo_sensor JOIN modulos m ON m.id_modulo = s.id_modulo "
           "WHERE l.sensor_valido = 1 AND s.tipo = 'consumo' AND l.ciclo >= ? ")
    params = [ini]
    if sigla:
        sql += "AND m.sigla = ? "
        params.append(sigla)
    return consultar(sql + "GROUP BY m.sigla ORDER BY m.id_modulo", tuple(params))


def tool_alertas(sigla=None):
    alertas, _ = avaliar_leituras()
    return [a for a in alertas if sigla is None or a["sigla"] == sigla]


def tool_ambiente():
    ini = ciclo_inicial_janela()
    r = consultar("SELECT AVG(temperatura_c), MIN(temperatura_c), MAX(temperatura_c) FROM leituras WHERE ciclo >= ?", (ini,))
    return r[0] if r and r[0][0] is not None else None


# ---- Agentes especialistas ----
def agente_energia(pedido_norm, sessao, trace):
    sigla = detectar_modulo(pedido_norm, sessao)
    trace.append(f"[trace] ferramenta tool_consumo({sigla or 'colônia'})")
    dados = tool_consumo(sigla)
    if not dados:
        return {"resposta": "Não há leituras válidas recentes para responder."}
    linhas = [f"{nome}: média {media:.1f} kW (limite {lim:.0f} kW)" for _, nome, media, lim in dados]
    resposta = "Consumo médio recente -> " + "; ".join(linhas)
    alertas = tool_alertas(sigla)
    if alertas:  # handoff: encaminha para manutenção quando há alertas ativos
        trace.append(f"[trace] handoff energia -> manutencao ({len(alertas)} alerta(s) ativos)")
        return {"resposta": resposta, "handoff": "manutencao"}
    return {"resposta": resposta + ". Nenhum alerta ativo."}


def agente_ambiental(pedido_norm, sessao, trace):
    trace.append("[trace] ferramenta tool_ambiente()")
    amb = tool_ambiente()
    if amb is None:
        return {"resposta": "Não há leituras ambientais recentes."}
    media, minimo, maximo = amb
    resposta = f"Temperatura externa recente: média {media:.1f} °C, mínima {minimo:.1f} °C, máxima {maximo:.1f} °C."
    modelo = executar_regressao("HAB", verbose=False, salvar=False)
    if modelo:
        resposta += (f" Pelo modelo linear, cada 1 °C mais frio eleva o consumo da habitação em "
                     f"~{abs(modelo['b2'][1]):.2f} kW.")
    return {"resposta": resposta}


def agente_manutencao(pedido_norm, sessao, trace):
    sigla = detectar_modulo(pedido_norm, sessao)
    escopo = f"módulo {sigla} (contexto da conversa)" if sigla else "toda a colônia"
    trace.append(f"[trace] ferramenta tool_alertas({sigla or 'colônia'})")
    alertas = tool_alertas(sigla)
    if not alertas:
        return {"resposta": f"Escopo: {escopo}. Nenhum alerta ativo; nenhum chamado de manutenção sugerido."}
    p1 = sum(1 for a in alertas if a["prioridade"] == 1)
    p2 = len(alertas) - p1
    topo = montar_fila(alertas)[0][4]
    return {"resposta": (f"Escopo: {escopo}. {len(alertas)} alerta(s): {p1} de consumo alto em módulo crítico (P1) e "
                         f"{p2} de sensor inválido (P2). Mais urgente: {topo['descricao']}. "
                         f"Sugiro abrir chamado de manutenção.")}


def agente_conhecimento(pedido_norm, sessao, trace):
    trace.append("[trace] consulta à base de conhecimento (RAG simulado)")
    for chave, texto in BASE_CONHECIMENTO.items():
        if chave in pedido_norm:
            return {"resposta": f"Procedimento recuperado ({chave}) - {texto}"}
    return {"resposta": "Não encontrei procedimento correspondente na base de conhecimento."}


AGENTES = {"energia": agente_energia, "ambiental": agente_ambiental,
           "manutencao": agente_manutencao, "conhecimento": agente_conhecimento}


def triador(pedido_norm):
    """Agente triador: decide qual especialista atende o pedido (regras por palavras-chave)."""
    if any(p in pedido_norm for p in ["procedimento", "protocolo", "o que fazer", "como proceder", "como agir"]):
        return "conhecimento"
    if any(p in pedido_norm for p in ["manutencao", "falha", "sensor", "alerta", "defeito", "chamado"]):
        return "manutencao"
    if any(p in pedido_norm for p in ["energia", "consumo", "bateria", "carga", "kw", "solar"]):
        return "energia"
    if any(p in pedido_norm for p in ["temperatura", "ambiente", "clima", "frio", "poeira", "tempestade"]):
        return "ambiental"
    return None


def processar_pedido(pedido, sessao):
    trace = []
    pedido_norm = normalizar(pedido)
    sessao.historico.append(("tripulacao", pedido))
    print(f"\n> Pedido: {pedido}")
    ok, motivo = guardrail(pedido_norm)
    trace.append(f"[trace] guardrail -> {'aprovado' if ok else 'BLOQUEADO'}")
    if not ok:
        print(f"  [GUARDRAIL] Pedido bloqueado: {motivo}")
        sessao.historico.append(("sistema", "bloqueado"))
    else:
        agente = triador(pedido_norm)
        trace.append(f"[trace] triador -> {agente}")
        saltos = 0
        while agente and saltos < 3:  # limite de handoffs evita laços infinitos
            resultado = AGENTES[agente](pedido_norm, sessao, trace)
            print(f"  [agente {agente}] {resultado['resposta']}")
            sessao.historico.append((agente, resultado["resposta"]))
            agente = resultado.get("handoff")
            saltos += 1
        if saltos == 0:
            print("  [triador] Não entendi o pedido. Tente falar de energia, ambiente, manutenção ou procedimentos.")
        else:
            print(f"  [!] {AVISO_HUMANO}")
    for linha in trace:
        print("  " + linha)


def menu_agentes():
    sessao = Sessao()
    exemplos = ["Qual o consumo de energia da habitação?", "E a temperatura do ambiente?",
                "Preciso de manutenção nos sensores", "O que fazer em caso de tempestade de poeira?",
                "Abrir eclusa agora", "Quais os dados pessoais da tripulação?"]
    print("\n=== Agentes de IA (simulados) ===")
    print("Digite um pedido, 'demo' para rodar exemplos prontos ou Enter para voltar.")
    while True:
        pedido = pedir("\nPedido: ")
        if not pedido:
            print(f"Sessão encerrada ({len(sessao.historico)} mensagens no histórico).")
            return
        if normalizar(pedido) == "demo":
            for ex in exemplos:
                processar_pedido(ex, sessao)
        else:
            processar_pedido(pedido, sessao)


# ============================================================================
# 7. ARQUITETURA COMPUTACIONAL
# ============================================================================
def demonstrar_arquitetura():
    print("\n=== Arquitetura computacional do SIFSM ===")
    print("Onde cada parte do sistema rodaria:")
    print("  CPU (núcleos de uso geral): menu, CRUD, regra booleana, fila/pilha/grafo - tarefas sequenciais e com desvios.")
    print("  GPU/NPU/TPU: simulações em larga escala (muitos módulos/cenários) e inferência de modelos de IA - paralelismo.")
    print("  Computação heterogênea: cada tarefa vai para a unidade mais eficiente em energia (mais trabalho por joule).")
    print("  Memória: o gargalo costuma ser mover dados (memória -> processador), não calcular. Por isso processar")
    print("           perto do sensor (edge) e agregar antes de enviar reduz tráfego, latência e energia.")
    print("  Terra-Marte: o sinal leva de 3 a 22 minutos por sentido; a colônia precisa decidir localmente.")
    print("\nDemonstração prática: soma de quadrados de 1.000.000 de leituras")
    n = 1_000_000
    lista = [float(i % 100) * 0.5 for i in range(n)]
    t0 = time.perf_counter()
    s1 = 0.0
    for x in lista:  # um elemento por vez (sequencial)
        s1 += x * x
    t_loop = time.perf_counter() - t0
    arr = np.array(lista)
    t0 = time.perf_counter()
    s2 = float(np.dot(arr, arr))  # vetorizado: usa memória contígua e instruções SIMD
    t_vet = time.perf_counter() - t0
    print(f"  laço Python  : {t_loop * 1000:8.1f} ms  (resultado {s1:,.0f})")
    print(f"  NumPy vetor. : {t_vet * 1000:8.2f} ms  (resultado {s2:,.0f})")
    if t_vet > 0:
        print(f"  -> {t_loop / t_vet:.0f}x mais rápido. Menos tempo de CPU = menos energia gasta no mesmo trabalho.")


# ============================================================================
# 8. SUSTENTABILIDADE, CIDADANIA DIGITAL E IMPACTO SOCIOAMBIENTAL
# ============================================================================
def mostrar_reflexao():
    print("\n=== Sustentabilidade, cidadania digital e impacto socioambiental ===")
    topicos = [
        ("Integração de energia, dados e automação",
         "O sistema cruza consumo, temperatura e ocupação e aplica a regra de alerta; assim o gasto de energia deixa de ser\n"
         "   percebido tarde e passa a ser visto por módulo, permitindo cortar desperdício antes de faltar energia."),
        ("Simulação para reduzir desperdícios",
         "A simulação mostrou que uma tempestade de poeira esgota a bateria e que o modo economia reduz o déficit; testar\n"
         "   isso no modelo evita descobrir o problema com a colônia já sem energia."),
        ("Resiliência e autonomia",
         "O grafo de dependências revela que o módulo de energia sustenta todos os outros: é o ponto único de falha e\n"
         "   justifica reserva de bateria, redundância e decisão local (a latência Terra-Marte impede controle remoto)."),
        ("Cidadania digital e privacidade",
         "Ocupação é um dado sobre pessoas. O sistema usa apenas contagens agregadas por módulo, os agentes bloqueiam\n"
         "   pedidos de dados pessoais e toda recomendação fica registrada e explicável (transparência)."),
        ("IA, data centers e impacto ambiental",
         "Cada cálculo gasta energia, que em Marte é escassa. Por isso as regras e a regressão são leves, rodam na CPU local\n"
         "   e a IA só entra quando necessária; modelos grandes exigiriam hardware e energia que a colônia não tem."),
        ("Inovação com benefícios e riscos",
         "Automatizar decisões ajuda, mas um erro de sensor pode gerar alarme falso ou falsa segurança. Por isso há\n"
         "   tratamento de sensor inválido, guardrails e validação humana obrigatória nas recomendações."),
    ]
    for titulo, texto in topicos:
        print(f"\n* {titulo}\n   {texto}")


# ============================================================================
# 9. ANÁLISE INTEGRADA DA COLÔNIA (opção 13)
# ============================================================================
def analisar_sustentabilidade():
    print("\n" + "=" * 70)
    print("ANÁLISE DE SUSTENTABILIDADE DA COLÔNIA - AURORA SIGER")
    print("=" * 70)

    # 1) Consulta ao banco
    total, linhas = consumo_base_do_banco()
    ini = ciclo_inicial_janela()
    print(f"\n[1/6] Consumo médio por módulo (banco de dados, ciclos {ini}+):")
    for sigla, media in linhas:
        print(f"      {sigla}: {media:6.1f} kW")
    print(f"      TOTAL: {total:.1f} kW")
    if total <= 0:
        print("[ERRO] Não há leituras válidas para analisar. Cadastre leituras ou recarregue o CSV (opção 14).")
        return

    # 2) Regra booleana + fila de prioridade
    alertas, _ = avaliar_leituras()
    gravar_alertas(alertas)
    fila = montar_fila(alertas)
    print(f"\n[2/6] Regra booleana ALERTA = (consumo_alto E crítico) OU sensor inválido -> {len(alertas)} alerta(s)")
    for i in range(min(5, len(fila))):
        prio, _, _, _, a = heapq.heappop(fila)
        print(f"      {i + 1}. [P{prio}] {a['descricao']}")
    if len(fila) > 0:
        print(f"      ... e mais {len(fila)} alerta(s) (gravados na tabela alertas).")
    p1 = [a for a in alertas if a["prioridade"] == 1]

    # 3) Grafo de dependências
    print("\n[3/6] Grafo de dependências:")
    mods_alerta = sorted({a["sigla"] for a in alertas})
    for sigla in mods_alerta:
        dep = [m for m, _ in modulos_afetados(sigla)]
        print(f"      {sigla} em alerta -> dependentes: {', '.join(dep) if dep else 'nenhum (falha isolada)'}")
    print(f"      Ponto único de falha: ENE derruba {len(modulos_afetados('ENE'))} módulos.")

    # 4) Simulação
    print("\n[4/6] Simulação de 48 h com o consumo-base do banco:")
    cenarios = sim.rodar_cenarios(total)
    sim.imprimir_resumo(cenarios)
    print(f"      Gráfico: {sim.salvar_grafico(cenarios)}")
    normal, tempestade, economia = cenarios

    # 5) Regressão
    print("\n[5/6] Regressão linear do consumo da Habitação:")
    modelo = executar_regressao("HAB", verbose=False, salvar=True)
    previsao = None
    if modelo:
        b = modelo["b2"]
        lim = consultar("SELECT limite_consumo_kw FROM modulos WHERE sigla = 'HAB'")[0][0]
        previsao = b[0] + b[1] * (-85.0) + b[2] * 10
        print(f"      consumo = {b[0]:.1f} {b[1]:+.3f}*temperatura {b[2]:+.3f}*ocupação  "
              f"(R² teste {modelo['r2_multiplo']:.2f}, RMSE {modelo['rmse_multiplo']:.2f} kW)")
        print(f"      Previsão para noite fria (-85 °C, 10 pessoas): {previsao:.1f} kW (limite {lim:.0f} kW)")

    # 6) Recomendação
    print("\n[6/6] RECOMENDAÇÃO TÉCNICA")
    if normal["soc_minimo_pct"] < 20:
        print("      - Mesmo sem tempestade a bateria fica abaixo de 20%: gerar mais ou consumir menos já.")
    else:
        print(f"      - Operação normal segura: bateria nunca abaixo de {normal['soc_minimo_pct']:.0f}%.")
    if tempestade["deficit_kwh"] > 0:
        criticos = dict(consultar("SELECT sigla, critico FROM modulos"))
        nao_criticos = sum(media for sigla, media in linhas if not criticos.get(sigla, 1))
        share = nao_criticos / total
        print(f"      - Tempestade de poeira: déficit de {tempestade['deficit_kwh']:.0f} kWh. Um corte de 25% no consumo total "
              f"reduz para {economia['deficit_kwh']:.0f} kWh,")
        print("        o que ainda não basta: ampliar a reserva de bateria ou cortar mais cargas antes da tempestade.")
        if share > 0:
            print(f"      - Atenção: módulos não críticos são só {100 * share:.0f}% do consumo; cortar 25% do total exige "
                  f"desligar ~{100 * 0.25 / share:.0f}% dessas cargas.")
    if p1:
        print(f"      - {len(p1)} alerta(s) de consumo alto em módulo crítico: priorizar inspeção desses módulos.")
    if previsao is not None and modelo and previsao > lim:
        print("      - O modelo prevê consumo da habitação acima do limite em noites muito frias: revisar isolamento.")
    print("\n      Limitações: dados simulados, modelo de bateria simplificado, regressão com poucos pontos.")
    print(f"      {AVISO_HUMANO}")
    print("      Privacidade: apenas dados agregados por módulo foram usados (nenhum dado pessoal).")


# ============================================================================
# MENU PRINCIPAL
# ============================================================================
def mostrar_menu():
    print("\n" + "=" * 62)
    print("  SIFSM - Sistema Integrado de Futuro Sustentável em Marte")
    print("  Colônia Aurora Siger")
    print("=" * 62)
    print(" 1) Cadastrar sensor            8) Simulação da bateria (integral)")
    print(" 2) Listar sensores             9) Regressão linear")
    print(" 3) Alterar sensor             10) Agentes de IA")
    print(" 4) Excluir sensor             11) Arquitetura computacional")
    print(" 5) Leituras (CRUD)            12) Sustentabilidade e cidadania digital")
    print(" 6) Regra lógica / circuito    13) ANALISAR SUSTENTABILIDADE DA COLÔNIA")
    print(" 7) Estruturas de dados        14) Recarregar dados do CSV")
    print(" 0) Sair")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # evita erro de acentos em alguns terminais
    except (AttributeError, ValueError):
        pass
    inicializar_banco()
    if not banco_valido():
        print("\nNão foi possível iniciar o banco de dados. Confirme que script_banco.sql e "
              "dados_aurora_siger.csv estão na mesma pasta do codigo_fonte.py e execute novamente.")
        return
    acoes = {"1": cadastrar_sensor, "2": listar_sensores, "3": alterar_sensor, "4": excluir_sensor,
             "5": menu_leituras, "6": menu_regra, "7": menu_estruturas, "8": executar_simulacao,
             "9": menu_regressao, "10": menu_agentes, "11": demonstrar_arquitetura,
             "12": mostrar_reflexao, "13": analisar_sustentabilidade, "14": reiniciar_banco}
    while True:
        mostrar_menu()
        try:
            op = pedir("\nEscolha uma opção: ")
            if op == "0":
                break
            acao = acoes.get(op)
            if acao is None:
                print("Opção inválida. Digite um número do menu.")
                continue
            acao()
        except (EOFError, KeyboardInterrupt):
            print("\nEntrada encerrada.")
            break
        except Exception as e:  # rede de segurança: o sistema não deve "cair" por um erro inesperado
            print(f"[ERRO inesperado] {type(e).__name__}: {e}")
    print("\nSistema encerrado. Até a próxima, tripulação da Aurora Siger!")


if __name__ == "__main__":
    main()