"""
Futebol Tracker - identificação visual pela webcam.

O que este programa faz:
  1. Abre a webcam e mostra a imagem ao vivo.
  2. Detecta a BOLA em cada frame usando o YOLO (modelo de IA pronto).
  3. Detecta as PESSOAS/pernas usando o MediaPipe Pose.
  4. Conta EMBAIXADINHAS: cada vez que a bola faz um ciclo desce->sobe.
  5. Conta PASSES entre 2 jogadores sem a bola cair no chão.

Como funciona a lógica, em linguagem simples:
  - A bola tem uma posição (x, y) na tela em cada frame.
    Em imagem, y=0 é o TOPO da tela e y aumenta pra BAIXO.
  - Embaixadinha = a bola estava subindo e começou a descer (chegou no
    ponto mais alto de um toque). Contamos cada pico desses.
  - Passe = a bola sai da metade de um jogador e chega na metade do outro
    SEM tocar o chão no meio do caminho.

Controles:
  - Aperte 'q' para sair.
  - Aperte 'r' para zerar os contadores.
  - Aperte 'm' para alternar entre modo EMBAIXADINHA e modo PASSE.
  - Aperte 'c' para trocar de câmera (cicla entre as detectadas).

Escolha da câmera:
  - Ao iniciar, o programa lista as câmeras disponíveis e pergunta qual usar.
  - Você também pode passar o índice direto: python3 futebol_tracker.py 1
"""

import os
import sys
import json
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# O MediaPipe (esqueleto do corpo) é OPCIONAL - serve só para desenhar a pose.
# Em algumas versões/plataformas a API clássica "solutions" não vem disponível.
# Se não carregar, o programa continua funcionando normalmente sem o esqueleto,
# porque os contadores dependem apenas da bola detectada pelo YOLO.
try:
    import mediapipe as mp
    _ = mp.solutions.pose  # testa se a API clássica existe de verdade
    MEDIAPIPE_OK = True
except Exception:
    MEDIAPIPE_OK = False


# ---------------------------------------------------------------------------
# CONFIGURAÇÕES - você pode mexer nesses valores para calibrar
# ---------------------------------------------------------------------------

# Modelo YOLO. "yolo11m" (medium) detecta a bola muito melhor que o "nano"
# (testado no vídeo real: 85% vs 33% dos frames). Se ficar lento no seu
# computador, troque por "yolo11n.pt" (mais rápido, menos preciso).
MODELO_YOLO = "yolo11m.pt"

# Classe do YOLO que representa uma bola de esporte. No modelo COCO padrão,
# "sports ball" é a classe de número 32.
SPORTS_BALL_CLASS_ID = 32

# Confiança mínima (0 a 1) para aceitar uma detecção da bola.
# Baixamos para 0.15 porque o vídeo real mostrou que a bola em movimento
# costuma vir com confiança baixa; com o modelo medium isso recupera muitos
# frames sem gerar falsos positivos demais.
BALL_CONFIDENCE = 0.15

# "Memória" da bola: por quantos frames mantemos a última posição conhecida
# quando o YOLO perde a bola. Suaviza as falhas curtas de detecção.
MEMORIA_BOLA = 5

# Quantos frames de histórico da altura da bola guardamos para analisar
# o movimento (subindo/descendo).
HISTORY_LEN = 12

# Movimento vertical mínimo (em pixels) para considerar que a bola realmente
# subiu ou desceu, evitando contar tremores pequenos da detecção.
MIN_VERTICAL_MOVE = 15

# --- Detecção de QUEDA por "bola parada embaixo" ---
# Com câmera frontal, a bola no chão aparece na altura dos pés, não no rodapé.
# Por isso não usamos mais uma linha fixa: consideramos que a bola CAIU quando
# ela fica quase parada, na metade de baixo da imagem, por alguns frames.
#
# Fração da altura a partir da qual consideramos "parte de baixo" (0.55 = 45%
# de baixo da imagem). A bola precisa estar abaixo disso para contar queda.
ZONA_BAIXA_RATIO = 0.55
# Movimento máximo (em pixels) entre frames para considerar a bola "parada".
MOV_PARADA = 25
# Por quantos frames seguidos a bola precisa ficar parada embaixo para
# disparar a queda (com ~15 fps, 8 frames ≈ meio segundo).
FRAMES_PARADA_QUEDA = 8

# Zerar os contadores automaticamente quando a bola cair?
# Vale para os dois modos (embaixadinha e passe). Se preferir zerar só na mão,
# coloque False.
RESET_AO_CAIR = True

# Arquivo onde guardamos os recordes entre execuções (fica ao lado do script).
ARQUIVO_RECORDES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "recordes.json")

# Quantas tentativas recentes o gráfico mostra.
HISTORICO_MAX = 10


# ---------------------------------------------------------------------------
# CLASSE QUE CONTA EMBAIXADINHAS
# ---------------------------------------------------------------------------

class ContadorEmbaixadinhas:
    """
    Conta embaixadinhas observando a altura (y) da bola ao longo do tempo.

    Ideia: uma embaixadinha é um "quique" no pé. A bola sobe, chega num pico,
    e desce de novo. Cada vez que detectamos a transição de SUBINDO para
    DESCENDO (ou seja, um pico), contamos +1.
    """

    def __init__(self):
        self.count = 0
        # Guardamos as últimas alturas da bola (valor y do centro).
        self.alturas = deque(maxlen=HISTORY_LEN)
        # Estado atual do movimento: "subindo" ou "descendo".
        self.direcao = None
        # Guarda a última altura onde detectamos um pico, para exigir que a
        # bola desça um mínimo antes de contar o próximo (evita contagem dupla).
        self.ultima_altura_pico = None

    def atualizar(self, y_bola):
        """Recebe a altura y do centro da bola neste frame."""
        if y_bola is None:
            return  # Sem bola neste frame: não faz nada.

        self.alturas.append(y_bola)
        if len(self.alturas) < 3:
            return

        # Comparamos a altura atual com a de alguns frames atrás.
        # Lembre: y MENOR = mais ALTO na tela (subindo).
        y_antes = self.alturas[-3]
        y_agora = self.alturas[-1]
        delta = y_agora - y_antes  # positivo = desceu; negativo = subiu

        if delta < -MIN_VERTICAL_MOVE:
            nova_direcao = "subindo"
        elif delta > MIN_VERTICAL_MOVE:
            nova_direcao = "descendo"
        else:
            return  # movimento pequeno demais, ignora

        # Detectamos um PICO quando estávamos subindo e passamos a descer.
        if self.direcao == "subindo" and nova_direcao == "descendo":
            self.count += 1
            self.ultima_altura_pico = y_agora

        self.direcao = nova_direcao

    def reset(self):
        self.count = 0
        self.alturas.clear()
        self.direcao = None
        self.ultima_altura_pico = None


# ---------------------------------------------------------------------------
# CLASSE QUE CONTA PASSES ENTRE 2 JOGADORES
# ---------------------------------------------------------------------------

class ContadorPasses:
    """
    Conta passes entre o jogador da ESQUERDA e o da DIREITA da tela.

    Ideia simples: dividimos a tela ao meio. Sabemos em que lado a bola está.
    Quando a bola muda de um lado para o outro SEM ter tocado o chão no
    caminho, contamos +1 passe. Se ela toca o chão, cancelamos.
    """

    def __init__(self, largura_tela):
        self.count = 0
        self.meio_x = largura_tela / 2
        # Lado onde a bola estava na última vez: "esquerda" ou "direita".
        self.lado_atual = None
        # A bola tocou o chão desde a última troca de lado?
        self.tocou_chao = False

    def atualizar(self, x_bola, y_bola, linha_chao):
        if x_bola is None:
            return

        # A bola tocou o chão? (y grande = perto do rodapé da tela)
        if y_bola is not None and y_bola >= linha_chao:
            self.tocou_chao = True

        lado = "esquerda" if x_bola < self.meio_x else "direita"

        if self.lado_atual is None:
            self.lado_atual = lado
            return

        # A bola trocou de lado?
        if lado != self.lado_atual:
            if not self.tocou_chao:
                # Passe válido: cruzou o meio sem cair.
                self.count += 1
            # Reinicia o controle de chão para o próximo trajeto.
            self.tocou_chao = False
            self.lado_atual = lado

    def reset(self):
        self.count = 0
        self.lado_atual = None
        self.tocou_chao = False


# ---------------------------------------------------------------------------
# PLACAR: guarda RECORDE e ÚLTIMA contagem, por modo, e salva em arquivo
# ---------------------------------------------------------------------------

class Placar:
    """
    Mantém, para cada modo ("embaixadinha" e "passe"):
      - recorde: maior contagem já atingida (persiste entre execuções);
      - ultima:  a contagem da última tentativa encerrada (bola caiu/zerou).

    O recorde é salvo em disco (ARQUIVO_RECORDES) para não se perder quando
    você fecha o programa. A "última" é só da sessão atual.
    """

    def __init__(self):
        self.recorde = {"embaixadinha": 0, "passe": 0}
        self.ultima = {"embaixadinha": 0, "passe": 0}
        # Histórico das últimas tentativas de cada modo (para o gráfico).
        # deque com tamanho máximo: mantém só as N mais recentes.
        self.historico = {
            "embaixadinha": deque(maxlen=HISTORICO_MAX),
            "passe": deque(maxlen=HISTORICO_MAX),
        }
        self._carregar()

    def _carregar(self):
        """Lê os recordes salvos do arquivo, se existir."""
        try:
            with open(ARQUIVO_RECORDES, "r", encoding="utf-8") as f:
                dados = json.load(f)
            for modo in self.recorde:
                if isinstance(dados.get(modo), int):
                    self.recorde[modo] = dados[modo]
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass  # sem arquivo ainda, ou arquivo corrompido: começa do zero

    def _salvar(self):
        """Grava os recordes no arquivo (chamado quando bate um novo recorde)."""
        try:
            with open(ARQUIVO_RECORDES, "w", encoding="utf-8") as f:
                json.dump(self.recorde, f, indent=2)
        except OSError as e:
            print(f"Aviso: nao consegui salvar o recorde ({e}).")

    def encerrar_tentativa(self, modo, valor):
        """
        Chamado quando uma tentativa termina (bola caiu ou reset manual).
        Registra a 'última' e atualiza o 'recorde' se for o caso.
        Ignora valor 0 para não sujar a "última" com resets sem contagem.
        """
        if valor <= 0:
            return
        self.ultima[modo] = valor
        self.historico[modo].append(valor)  # guarda para o gráfico
        if valor > self.recorde[modo]:
            self.recorde[modo] = valor
            self._salvar()
            print(f"NOVO RECORDE de {modo}: {valor}!")
        else:
            print(f"Ultima tentativa de {modo}: {valor} "
                  f"(recorde: {self.recorde[modo]})")


# ---------------------------------------------------------------------------
# RASTREADOR DA BOLA: detecção + memória entre frames + detecção de queda
# ---------------------------------------------------------------------------

class RastreadorBola:
    """
    Cuida de tudo relacionado à posição da bola:
      - detecta a bola no frame (YOLO);
      - se o YOLO falhar por poucos frames, mantém a última posição (memória);
      - detecta QUEDA quando a bola fica parada na parte de baixo da imagem.

    A posição atual fica em self.x e self.y (ou None se realmente não há bola).
    """

    def __init__(self, modelo, altura_frame):
        self.modelo = modelo
        self.altura = altura_frame
        # Posição "efetiva" da bola (pode vir da memória).
        self.x = None
        self.y = None
        self.raio = 12
        # É uma posição real (YOLO viu agora) ou memória (estimada)?
        self.ao_vivo = False
        # Memória: quantos frames faz que não vemos a bola de verdade.
        self._frames_sem_ver = MEMORIA_BOLA + 1
        # Estado de queda: contador de frames parada embaixo.
        self._frames_parada = 0
        self._y_anterior = None
        # Trava: após uma queda, só rearmamos quando a bola voltar a subir.
        # Evita disparar "queda" repetidamente com a bola parada no chão.
        self._queda_travada = False

    def atualizar(self, frame):
        """Roda o YOLO no frame e atualiza a posição da bola (com memória)."""
        r = self.modelo(frame, verbose=False, conf=BALL_CONFIDENCE)[0]

        detec = None
        melhor_conf = 0.0
        for box in r.boxes:
            if int(box.cls[0]) == SPORTS_BALL_CLASS_ID:
                conf = float(box.conf[0])
                if conf > melhor_conf:
                    melhor_conf = conf
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    raio = int(max(x2 - x1, y2 - y1) / 2)
                    detec = (cx, cy, raio)

        if detec is not None:
            # Vimos a bola de verdade neste frame.
            self.x, self.y, self.raio = detec
            self.ao_vivo = True
            self._frames_sem_ver = 0
        else:
            # Não vimos. Se faz pouco tempo, mantém a última posição (memória).
            self._frames_sem_ver += 1
            self.ao_vivo = False
            if self._frames_sem_ver > MEMORIA_BOLA:
                self.x = self.y = None  # perdeu de vez

    def detectou_queda(self):
        """
        Retorna True UMA vez, no momento em que a bola é considerada caída.
        Critério: bola presente, na parte de baixo da imagem, e quase parada
        por FRAMES_PARADA_QUEDA frames seguidos.
        """
        y = self.y
        if y is None:
            self._frames_parada = 0
            self._y_anterior = None
            return False

        na_zona_baixa = y >= self.altura * ZONA_BAIXA_RATIO

        # Rearma a trava quando a bola volta para cima da zona baixa (subiu).
        if not na_zona_baixa:
            self._queda_travada = False

        parada = (self._y_anterior is not None
                  and abs(y - self._y_anterior) <= MOV_PARADA)
        self._y_anterior = y

        if na_zona_baixa and parada:
            self._frames_parada += 1
        else:
            self._frames_parada = 0

        if self._frames_parada >= FRAMES_PARADA_QUEDA and not self._queda_travada:
            self._frames_parada = 0
            self._queda_travada = True  # não dispara de novo até a bola subir
            return True
        return False

    def rearmar_apos_reset(self):
        """Chamado após um reset para não disparar a queda repetidamente."""
        self._frames_parada = 0
        self._queda_travada = True


# ---------------------------------------------------------------------------
# SELEÇÃO DE WEBCAM
# ---------------------------------------------------------------------------

def listar_cameras(max_indices=8):
    """
    Descobre quais câmeras existem tentando abrir os índices de 0 até
    max_indices-1. Retorna uma lista dos índices que funcionaram.

    Não existe uma forma 100% portátil de "listar câmeras" no OpenCV, então
    o jeito prático é tentar abrir cada índice e ver qual responde.
    """
    disponiveis = []
    for i in range(max_indices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()  # confirma que dá pra ler um frame de verdade
            if ok:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                disponiveis.append((i, w, h))
        cap.release()
    return disponiveis


def escolher_camera():
    """
    Fluxo de escolha da câmera:
      1. Se o usuário passou um índice na linha de comando, usa ele direto.
      2. Senão, lista as câmeras encontradas e pergunta qual usar.
    Retorna o índice escolhido (int) ou None se não houver câmera.
    """
    # 1) índice via linha de comando? ex: python3 futebol_tracker.py 1
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            print(f"Argumento '{sys.argv[1]}' nao e um numero de camera valido.")

    # 2) detectar e perguntar
    print("Procurando cameras disponiveis...")
    cams = listar_cameras()

    if not cams:
        print("Nenhuma camera encontrada.")
        return None

    if len(cams) == 1:
        idx = cams[0][0]
        print(f"Apenas uma camera encontrada (indice {idx}). Usando ela.")
        return idx

    print("\nCameras encontradas:")
    for idx, w, h in cams:
        print(f"  [{idx}] resolucao {w}x{h}")

    while True:
        escolha = input("\nDigite o numero da camera que quer usar: ").strip()
        try:
            idx = int(escolha)
            if any(idx == c[0] for c in cams):
                return idx
            print("Esse numero nao esta na lista. Tente de novo.")
        except ValueError:
            print("Digite apenas o numero. Tente de novo.")


def indices_camera_disponiveis(max_indices=8):
    """Versão enxuta usada pela tecla 'c' para ciclar entre câmeras ao vivo."""
    return [c[0] for c in listar_cameras(max_indices)]


# ---------------------------------------------------------------------------
# DESENHO DA INTERFACE (helpers para deixar bonito)
# ---------------------------------------------------------------------------

# Paleta de cores (OpenCV usa BGR, não RGB!).
COR_FUNDO_PAINEL = (28, 28, 30)      # cinza bem escuro
COR_DESTAQUE = (80, 220, 100)        # verde vibrante (número principal)
COR_TEXTO = (240, 240, 240)          # branco suave
COR_SECUNDARIA = (170, 170, 175)     # cinza claro (rótulos)
COR_DOURADO = (60, 200, 255)         # dourado/amarelo (recorde)
COR_VERMELHO = (60, 60, 235)         # vermelho (aviso de queda)


def painel_arredondado(img, p1, p2, cor, alpha=0.85, raio=18):
    """
    Desenha um retângulo com cantos arredondados e leve transparência, para
    servir de fundo dos textos. alpha controla a opacidade (1 = sólido).
    """
    x1, y1 = p1
    x2, y2 = p2
    overlay = img.copy()
    # Corpo do retângulo (duas faixas que se cruzam) + 4 círculos nos cantos.
    cv2.rectangle(overlay, (x1 + raio, y1), (x2 - raio, y2), cor, -1)
    cv2.rectangle(overlay, (x1, y1 + raio), (x2, y2 - raio), cor, -1)
    for cx, cy in [(x1 + raio, y1 + raio), (x2 - raio, y1 + raio),
                   (x1 + raio, y2 - raio), (x2 - raio, y2 - raio)]:
        cv2.circle(overlay, (cx, cy), raio, cor, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def texto_sombra(img, texto, pos, escala, cor, espessura, cor_sombra=(0, 0, 0)):
    """Escreve um texto com uma sombra sutil atrás, para dar contraste."""
    x, y = pos
    cv2.putText(img, texto, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX,
                escala, cor_sombra, espessura + 1, cv2.LINE_AA)
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                escala, cor, espessura, cv2.LINE_AA)


def texto_centralizado(img, texto, cx, y, escala, cor, espessura, sombra=True):
    """Escreve um texto centralizado horizontalmente em cx."""
    (tw, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, espessura)
    x = int(cx - tw / 2)
    if sombra:
        texto_sombra(img, texto, (x, y), escala, cor, espessura)
    else:
        cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    escala, cor, espessura, cv2.LINE_AA)


def desenhar_grafico(frame, historico, recorde, x, y, w, h):
    """
    Desenha um gráfico de BARRAS com as últimas tentativas.

    historico: lista/deque de números (ex: [5, 3, 8, 2, ...]) da mais antiga
               para a mais recente.
    x, y, w, h: posição e tamanho da área do gráfico (canto superior esquerdo).

    A altura de cada barra é proporcional ao maior valor do histórico, para o
    gráfico se ajustar sozinho. A barra mais recente fica destacada.
    """
    painel_arredondado(frame, (x, y), (x + w, y + h), COR_FUNDO_PAINEL,
                       alpha=0.85, raio=14)

    margem = 16
    titulo_h = 26
    texto_sombra(frame, "ULTIMAS", (x + margem, y + 20), 0.55, COR_SECUNDARIA, 1)

    dados = list(historico)
    if not dados:
        # Ainda não há tentativas: mensagem no lugar do gráfico.
        texto_centralizado(frame, "sem dados ainda", x + w // 2,
                           y + h // 2 + 6, 0.55, (110, 110, 115), 1, sombra=False)
        return

    area_x = x + margem
    area_y = y + titulo_h + 6
    area_w = w - 2 * margem
    area_h = h - titulo_h - margem - 14  # deixa espaço embaixo p/ os rótulos

    maximo = max(dados) if max(dados) > 0 else 1
    n = len(dados)
    # Largura de cada barra com um pequeno espaçamento entre elas.
    espaco = 6
    barra_w = max(6, int((area_w - espaco * (n - 1)) / n))

    for i, valor in enumerate(dados):
        bx = area_x + i * (barra_w + espaco)
        altura_barra = int((valor / maximo) * area_h)
        by = area_y + (area_h - altura_barra)
        # Barra mais recente (última) em destaque; as outras mais apagadas.
        eh_ultima = (i == n - 1)
        cor = COR_DESTAQUE if eh_ultima else (90, 140, 100)
        cv2.rectangle(frame, (bx, by), (bx + barra_w, area_y + area_h), cor, -1)
        # Valor em cima da barra
        texto_centralizado(frame, str(valor), bx + barra_w // 2, by - 4,
                           0.4, COR_TEXTO, 1, sombra=False)

    # Linha pontilhada do recorde, se couber no gráfico.
    if recorde > 0 and recorde <= maximo:
        ry = area_y + int((1 - recorde / maximo) * area_h)
        for lx in range(area_x, area_x + area_w, 10):
            cv2.line(frame, (lx, ry), (lx + 5, ry), COR_DOURADO, 1)


def desenhar_hud(frame, largura, altura, modo, valor_atual,
                 recorde, ultima, historico, cam_idx, bola_ok, frames_aviso_queda):
    """
    Desenha todo o painel de informações (HUD) sobre o frame:
      - painel principal com o número GRANDE do contador atual;
      - card lateral com RECORDE e ULTIMA;
      - status da câmera/bola no rodapé;
      - aviso "CAIU! ZERADO" quando aplicável.
    """
    titulo = "EMBAIXADINHAS" if modo == "embaixadinha" else "PASSES"

    # ---- Painel principal (canto superior esquerdo) ----
    pw, ph = 360, 150
    painel_arredondado(frame, (20, 20), (20 + pw, 20 + ph), COR_FUNDO_PAINEL)
    # Rótulo do modo
    texto_sombra(frame, titulo, (42, 58), 0.7, COR_SECUNDARIA, 2)
    # Número gigante
    texto_sombra(frame, str(valor_atual), (40, 140), 3.2, COR_DESTAQUE, 6)

    # ---- Card lateral: RECORDE e ULTIMA (canto superior direito) ----
    cw, ch = 220, 150
    cx0 = largura - cw - 20
    painel_arredondado(frame, (cx0, 20), (cx0 + cw, 20 + ch), COR_FUNDO_PAINEL)
    centro_card = cx0 + cw // 2

    texto_centralizado(frame, "RECORDE", centro_card, 52, 0.6, COR_SECUNDARIA, 1)
    texto_centralizado(frame, str(recorde), centro_card, 100, 1.5, COR_DOURADO, 3)
    # linha divisória
    cv2.line(frame, (cx0 + 20, 112), (cx0 + cw - 20, 112), (70, 70, 75), 1)
    texto_centralizado(frame, f"Ultima: {ultima}", centro_card, 145,
                       0.6, COR_TEXTO, 1)

    # ---- Rodapé: status da câmera/bola + atalhos ----
    barra_y = altura - 34
    painel_arredondado(frame, (20, barra_y), (largura - 20, altura - 12),
                       COR_FUNDO_PAINEL, alpha=0.7, raio=10)
    cor_bola = COR_DESTAQUE if bola_ok else COR_VERMELHO
    status = f"Cam {cam_idx}   Bola: {'OK' if bola_ok else '--'}"
    cv2.putText(frame, status, (36, altura - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_bola, 1, cv2.LINE_AA)
    atalhos = "q sair   r zerar   m modo   c camera"
    (tw, _), _ = cv2.getTextSize(atalhos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, atalhos, (largura - tw - 36, altura - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_SECUNDARIA, 1, cv2.LINE_AA)

    # ---- Gráfico das últimas tentativas (canto inferior esquerdo) ----
    g_w, g_h = 340, 150
    g_x = 20
    g_y = altura - 34 - 12 - g_h  # logo acima da barra de rodapé
    desenhar_grafico(frame, historico, recorde, g_x, g_y, g_w, g_h)

    # ---- Aviso de queda (grande, no centro) ----
    if frames_aviso_queda > 0:
        # Fundo semi-transparente atrás do aviso
        painel_arredondado(frame,
                           (largura // 2 - 220, altura // 2 - 55),
                           (largura // 2 + 220, altura // 2 + 25),
                           COR_VERMELHO, alpha=0.6, raio=16)
        texto_centralizado(frame, "CAIU!  ZERADO", largura // 2,
                           altura // 2, 1.4, (255, 255, 255), 3)


# ---------------------------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    print(f"Carregando modelo YOLO ({MODELO_YOLO})...")
    # Medium por padrão (detecta a bola bem melhor). Veja MODELO_YOLO no topo.
    modelo = YOLO(MODELO_YOLO)

    # MediaPipe Pose para desenhar o corpo (opcional, é mais visual).
    if MEDIAPIPE_OK:
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    else:
        pose = None
        print("Aviso: MediaPipe indisponivel - seguindo sem o esqueleto do corpo "
              "(os contadores funcionam normalmente).")

    # Escolhe qual webcam usar (pergunta ao usuário ou usa o índice do argumento).
    cam_idx = escolher_camera()
    if cam_idx is None:
        print("ERRO: nenhuma camera disponivel.")
        return

    cam = cv2.VideoCapture(cam_idx)
    if not cam.isOpened():
        print(f"ERRO: não consegui abrir a webcam de indice {cam_idx}.")
        return

    largura = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    altura = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    rastreador = RastreadorBola(modelo, altura)
    contador_emb = ContadorEmbaixadinhas()
    contador_passes = ContadorPasses(largura)
    placar = Placar()

    # Atalho: retorna a contagem atual do modo ativo.
    def contagem_atual(m):
        return contador_emb.count if m == "embaixadinha" else contador_passes.count

    # Modo inicial: "embaixadinha" ou "passe".
    modo = "embaixadinha"

    # Guarda por alguns frames a mensagem "CAIU! Zerado" para mostrar na tela.
    frames_aviso_queda = 0

    print(f"Rodando com a camera {cam_idx}! "
          "Teclas: q=sair  r=zerar  m=modo  c=trocar camera")

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        # Espelha a imagem (fica mais natural, como um espelho).
        frame = cv2.flip(frame, 1)

        # -------------------------------------------------------------------
        # 1) DETECTAR A BOLA (com memória entre frames)
        # -------------------------------------------------------------------
        rastreador.atualizar(frame)
        x_bola, y_bola = rastreador.x, rastreador.y

        # Desenha a bola: círculo cheio quando é detecção ao vivo, tracejado/
        # mais apagado quando é posição "de memória" (estimada).
        if x_bola is not None:
            if rastreador.ao_vivo:
                cv2.circle(frame, (x_bola, y_bola), rastreador.raio, (0, 165, 255), 3)
                cv2.circle(frame, (x_bola, y_bola), 4, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (x_bola, y_bola), rastreador.raio, (0, 120, 180), 1)

        # -------------------------------------------------------------------
        # 2) DETECTAR O CORPO (pose) - só para visual e futura evolução
        # -------------------------------------------------------------------
        if pose is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb)
            if pose_result.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

        # -------------------------------------------------------------------
        # 3) ATUALIZAR OS CONTADORES conforme o modo
        # -------------------------------------------------------------------
        # A "zona baixa" faz o papel de chão para o modo passe (câmera frontal).
        zona_baixa_y = int(altura * ZONA_BAIXA_RATIO)
        if modo == "embaixadinha":
            contador_emb.atualizar(y_bola)
        else:
            contador_passes.atualizar(x_bola, y_bola, zona_baixa_y)

        # -------------------------------------------------------------------
        # 3b) DETECTAR QUEDA E ZERAR (vale para os dois modos)
        # -------------------------------------------------------------------
        # Agora a queda é decidida pelo rastreador: bola quase parada na parte
        # de baixo da imagem por alguns frames (funciona com câmera frontal e
        # não depende de calibrar uma linha fixa).
        if RESET_AO_CAIR and rastreador.detectou_queda():
            placar.encerrar_tentativa(modo, contagem_atual(modo))
            contador_emb.reset()
            contador_passes.reset()
            rastreador.rearmar_apos_reset()
            frames_aviso_queda = 30  # mostra o aviso por ~30 frames
            print("Bola caiu -> contadores zerados.")

        # -------------------------------------------------------------------
        # 4) DESENHAR A INTERFACE (textos e linhas na tela)
        # -------------------------------------------------------------------
        # Linha da "zona baixa": abaixo dela, uma bola parada conta como queda.
        cv2.line(frame, (0, zona_baixa_y), (largura, zona_baixa_y), (0, 0, 200), 1)
        cv2.putText(frame, "zona de queda", (10, zona_baixa_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
        # Linha do meio (divisão dos jogadores) só no modo passe.
        if modo == "passe":
            cv2.line(frame, (largura // 2, 0), (largura // 2, altura), (255, 255, 0), 1)

        # Painel de informações (HUD) com números grandes, recorde e última.
        if frames_aviso_queda > 0:
            frames_aviso_queda -= 1
        desenhar_hud(
            frame, largura, altura, modo,
            valor_atual=contagem_atual(modo),
            recorde=placar.recorde[modo],
            ultima=placar.ultima[modo],
            historico=placar.historico[modo],
            cam_idx=cam_idx,
            bola_ok=(x_bola is not None),
            frames_aviso_queda=frames_aviso_queda,
        )

        cv2.imshow("Futebol Tracker", frame)

        # -------------------------------------------------------------------
        # 5) TECLADO
        # -------------------------------------------------------------------
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("r"):
            # Reset manual também conta como fim de tentativa (registra última/recorde).
            placar.encerrar_tentativa(modo, contagem_atual(modo))
            contador_emb.reset()
            contador_passes.reset()
            rastreador.rearmar_apos_reset()
            print("Contadores zerados.")
        elif tecla == ord("m"):
            modo = "passe" if modo == "embaixadinha" else "embaixadinha"
            print(f"Modo alterado para: {modo}")
        elif tecla == ord("c"):
            # Troca de câmera ao vivo: cicla para a próxima câmera detectada.
            indices = indices_camera_disponiveis()
            if len(indices) <= 1:
                print("So ha uma camera disponivel; nada para trocar.")
            else:
                pos = indices.index(cam_idx) if cam_idx in indices else -1
                novo_idx = indices[(pos + 1) % len(indices)]
                cam.release()
                nova = cv2.VideoCapture(novo_idx)
                if nova.isOpened():
                    cam = nova
                    cam_idx = novo_idx
                    # Recalcula dimensões, pois a nova câmera pode ter outra resolução.
                    largura = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                    altura = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                    rastreador.altura = altura
                    contador_passes.meio_x = largura / 2
                    print(f"Trocado para a camera {cam_idx} ({largura}x{altura}).")
                else:
                    # Se falhar, reabre a câmera anterior para não travar.
                    cam = cv2.VideoCapture(cam_idx)
                    print(f"Nao consegui abrir a camera {novo_idx}; mantendo a {cam_idx}.")

    cam.release()
    cv2.destroyAllWindows()
    if pose is not None:
        pose.close()


if __name__ == "__main__":
    main()
