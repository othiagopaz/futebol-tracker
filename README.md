# Futebol Tracker

Sistema de identificação visual pela webcam que conta **embaixadinhas** e
**passes entre 2 jogadores** sem a bola cair. Usa YOLO (detecção da bola) e,
opcionalmente, MediaPipe (esqueleto do corpo).

## Instalação (primeira vez)

Requer **Python 3.12** (o MediaPipe ainda não suporta 3.13+ de forma estável).

```bash
python3.12 -m venv .venv          # ou: uv venv --python 3.12 .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Os modelos YOLO (`yolo11m.pt` / `yolo11n.pt`) são baixados automaticamente na
primeira execução — não precisa baixá-los à mão.

## Como rodar

```bash
source .venv/bin/activate
python3 futebol_tracker.py
```

Ao iniciar, o programa **lista as câmeras disponíveis** e pergunta qual usar.
Conecte sua webcam externa antes de rodar para ela aparecer na lista.

Você também pode pular a pergunta passando o índice da câmera direto:

```bash
python3 futebol_tracker.py 1    # usa a câmera de índice 1
```

Na primeira vez, o macOS vai pedir permissão de acesso à câmera para o
terminal — aceite.

## Controles (com a janela do vídeo em foco)

- `q` — sair
- `r` — zerar os contadores
- `m` — alternar entre modo **Embaixadinha** e modo **Passe**
- `c` — trocar de câmera (cicla entre as detectadas)

## Como funciona

- **YOLO** detecta a bola (`sports ball`) em cada frame.
- **MediaPipe Pose** desenha o corpo das pessoas.
- **Embaixadinha**: conta cada pico do movimento da bola (sobe → desce).
- **Passe**: detecta as 2 pessoas na cena (YOLO) e associa a bola ao jogador
  mais próximo. Um passe é contado quando a bola troca de dono sem cair. Os
  jogadores aparecem marcados (verde = está com a bola).
- **Reset automático na queda**: sempre que a bola toca a linha do chão
  (vermelha), os contadores zeram e aparece "CAIU! ZERADO" na tela. Vale nos
  dois modos. Para desligar, coloque `RESET_AO_CAIR = False` no topo do script.
- **Recorde e última**: o card no canto direito mostra o **recorde** da sessão
  (maior contagem já feita) e a **última** tentativa. Cada modo tem seu próprio
  recorde. Uma tentativa "termina" quando a bola cai ou você aperta `r`.
- **Gráfico das últimas tentativas**: no canto inferior esquerdo, um gráfico de
  barras mostra as últimas 10 tentativas (a mais recente destacada em verde) e
  uma linha pontilhada dourada no valor do recorde. Ajuste quantas aparecem em
  `HISTORICO_MAX` no topo do script.
- **Queda por cor do chão**: a bola é considerada caída quando fica parada e o
  que está logo abaixo dela é o **concreto cinza** do chão (detectado por cor,
  não por uma linha fixa). Ignora paredes/teto claros e funciona com câmera
  frontal. A bola no chão fica marcada com um círculo vermelho.

## Recordes salvos

O recorde é gravado em `recordes.json` (ao lado do script) e sobrevive entre
execuções — abrir o programa amanhã mantém o recorde de hoje. A "última" é só
da sessão atual. Para apagar os recordes, delete o arquivo `recordes.json`.

## Calibração

Se contar demais ou de menos, ajuste no topo de `futebol_tracker.py`:

| Variável | O que faz |
|----------|-----------|
| `MODELO_YOLO` | `yolo11m.pt` (preciso) ou `yolo11n.pt` (leve/rápido) |
| `BALL_CONFIDENCE` | Confiança mínima para aceitar a bola (aumente se detectar coisas erradas) |
| `MEMORIA_BOLA` | Frames que mantém a última posição quando o YOLO perde a bola |
| `MIN_VERTICAL_MOVE` | Movimento mínimo para contar subida/descida (aumente se contar tremores) |
| `CHAO_S_MAX` / `CHAO_V_MIN` / `CHAO_V_MAX` | Faixa de cor do chão (concreto cinza) em HSV. Recalibre se o piso mudar |
| `CHAO_FRACAO_MIN` | Quanto da área abaixo da bola precisa ser chão para contar "no chão" |
| `MOV_PARADA` | Quão parada a bola precisa estar (px) para contar como caída |
| `FRAMES_PARADA_QUEDA` | Por quantos frames parada sobre o chão para disparar a queda |
| `ZONA_BAIXA_RATIO` | Só considera queda abaixo desta altura (evita paredes/teto) |

## Dicas para funcionar melhor

- Boa iluminação e fundo não muito bagunçado.
- Bola bem visível (cores contrastantes com o fundo ajudam o YOLO).
- No modo passe: um jogador de cada lado da câmera.
