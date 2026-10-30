# Educability Annotator

Anota corpora **JSONL, Parquet e datasets do Hugging Face Hub** com
[`gregassagraf/educability-norberto-mini-4class-v1`](https://huggingface.co/gregassagraf/educability-norberto-mini-4class-v1).
Preserva os registros, acrescenta classe e probabilidades, processa lotes limitados
e retoma partes confirmadas após interrupções. Funciona em CPU e foi preparado
para uma GPU, múltiplas GPUs e múltiplos nós com filesystem compartilhado.

O modelo é um classificador ordinal de educabilidade em português. Seus resultados
refletem supervisão sintética; a agregação de chunks implementada aqui não tem uma
validação pedagógica documental independente.

Na DGX do grupo (Slurm + Docker), e para corpora Parquet grandes com rótulos gravados à parte
(`python -m educability_annotator.sidecar`), veja [docs/dgx.md](docs/dgx.md).

## Instalação

Python **3.11 ou superior**. A validação local usa Python 3.11, macOS ARM64 e CPU.
Linux é o destino previsto para CUDA e servidores DGX/HGX.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]' -c constraints-tested-cpu.txt
educability-annotator --help
```

Com `uv`:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[test]' -c constraints-tested-cpu.txt
source .venv/bin/activate
```

Os comandos acima usam as versões de referência testadas em CPU. Em Linux,
instale antes o wheel CPU de PyTorch com `--index-url https://download.pytorch.org/whl/cpu`
se quiser evitar a distribuição CUDA padrão do PyPI.

Instale primeiro a distribuição de **PyTorch adequada à GPU e ao driver** quando
estiver em CUDA e omita as constraints de CPU. O pacote aceita PyTorch >=2.7.1,<3 e fixa Transformers 4.53.3,
versão usada na publicação do modelo. Veja a combinação efetivamente testada em
[docs/validation.md](docs/validation.md). A variante de PyTorch disponível para
macOS não serve como instalação CUDA.

O primeiro uso precisa de acesso ao Hugging Face para os arquivos públicos do
modelo. Textos são processados localmente; a ferramenta não os envia ao Hub.
`--offline` exige que os arquivos necessários já estejam no cache ou em uma
pasta local. `prepare` baixa configuração/tokenizer; os pesos são necessários
para `run`. Não há `trust_remote_code` nem carregamento de pesos pickle.

## Execução local

Crie um manifesto antes de executar. Escolha uma saída vazia, separada das entradas:

```bash
educability-annotator prepare \
  --input 'dados/*.parquet' \
  --output resultados \
  --text-column text \
  --max-document-tokens 5000

educability-annotator run \
  --output resultados \
  --device cpu \
  --torch-threads 4

educability-annotator status --output resultados
educability-annotator verify --output resultados --full-checksums
```

`--input` aceita arquivos, diretórios percorridos recursivamente e globs entre
aspas, inclusive uma lista mista de JSONL e Parquet. Caminhos duplicados são
eliminados. A coluna é um **nome literal de primeiro nível**, não uma expressão
para navegar em objetos. O formato de cada arquivo é mantido.

```bash
educability-annotator prepare \
  --input corpus/a.jsonl corpus/b.parquet 'outros/**/*.jsonl' \
  --output resultados-mistos --text-column conteudo
```

JSONL deve ser UTF-8, não comprimido, com um objeto por linha. Linhas em branco,
chaves JSON duplicadas, `NaN`, registros que não sejam objetos e arquivos
corrompidos são erros explícitos. CRLF e ausência de newline final são suportados.
Valores JSON originais, inclusive números de alta precisão, são preservados
pela inserção dos novos campos no objeto original.

## Tokens, chunks e probabilidades

O limite do modelo é **1.024 tokens por chunk, incluindo tokens especiais**.
O tokenizer publicado acrescenta dois tokens, restando 1.022 de conteúdo.

- Sem `--max-document-tokens`, classifica até **1.022 tokens de conteúdo**.
- `--max-document-tokens 5000` classifica os primeiros 5.000 tokens de conteúdo:
  quatro chunks de 1.022 e um de 912. Textos menores geram menos chunks.
- Não há sobreposição. O limite conta tokens do tokenizer do modelo, não palavras.
- O texto original permanece completo na saída. `truncated=true` indica que
  havia conteúdo além do limite escolhido.

Para cada chunk, aplica-se **uma única vez**
`softmax(logits.float() / temperature_calibration)`. A temperatura vem do
modelo; na revisão padrão é `1.138848624315927`.

As probabilidades documentais são a média das probabilidades dos chunks,
ponderada pela quantidade de tokens de conteúdo de cada um. Padding e tokens
especiais não entram no peso. A classe é a de maior probabilidade; empates
escolhem a menor classe. O score esperado é `1*p1 + 2*p2 + 3*p3 + 4*p4`.
Não se aplica uma segunda calibração à distribuição documental.

O mapeamento vem de `id2label`/`label2id`. O carregamento valida quatro classes
`EDUCABILITY_1` a `EDUCABILITY_4`, o limite e a temperatura.

## Campos de saída

Todos recebem o prefixo `educability_`, modificável com `prepare --prefix`.
Uma colisão com campos existentes é um erro; a ferramenta não os sobrescreve.

| Sufixo | Tipo Parquet | Significado |
|---|---|---|
| `class` | int8 | Classe prevista, 1–4 |
| `label` | string | Rótulo `EDUCABILITY_1` a `EDUCABILITY_4` |
| `p1`…`p4` | float64 | Probabilidades agregadas, na ordem das classes |
| `expected_score` | float64 | Score esperado contínuo |
| `tokens_used` | int64 | Tokens de conteúdo classificados |
| `chunk_count` | int64 | Quantidade de chunks classificados |
| `truncated` | bool | Conteúdo cortado pelo limite documental |
| `status` | string | `ok`, `missing_text`, `null_text`, `empty_text` ou `non_string_text` |

Para texto inválido, **todos os demais campos de anotação são nulos**. A linha e
seus dados originais são preservados. Uma coluna ausente em Parquet resulta em
`missing_text` para os registros daquele arquivo. Erros de modelo, leitura,
escrita ou infraestrutura interrompem a execução.

## Uma GPU e múltiplas GPUs

```bash
educability-annotator run --output resultados --device cuda \
  --batch-size 64 --max-batch-tokens 32768
```

```bash
torchrun --standalone --nproc-per-node=8 \
  -m educability_annotator run \
  --output resultados --device cuda \
  --batch-size 64 --max-batch-tokens 32768
```

Execute somente um desses exemplos por saída, ou acrescente `--resume` após
uma execução anterior. O rank local seleciona a GPU dentro de
`CUDA_VISIBLE_DEVICES`. `--dtype auto` usa BF16 em CUDA compatível e FP32 nas
demais situações; `--dtype float32` força a referência de precisão.

Para reproduzir a distribuição sem GPUs, use `--device cpu` e dois ou três
processos. Se a resolução de hostname/IPv6 local impedir o rendezvous no macOS,
use IPv4 explícito:

```bash
torchrun --nnodes=1 --nproc-per-node=3 \
  --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 \
  --rdzv-id=teste-local-001 --local-addr=127.0.0.1 \
  -m educability_annotator run --output resultados \
  --device cpu --torch-threads 1
```

Use um `--rdzv-id` novo em cada lançamento. Mais exemplos, incluindo dois nós e
Slurm, estão em [docs/cluster.md](docs/cluster.md).

## Retomada, verificação e ordem

```bash
educability-annotator run --output resultados --device cpu --resume
```

Depois de uma falha ou término forçado, **pare todos os processos da execução
anterior** antes de recuperar o bloqueio:

```bash
educability-annotator run --output resultados --device cpu \
  --resume --recover-lock --full-checksums
```

O mesmo argumento funciona sob `torchrun`, inclusive com outra quantidade de
processos. Uma parte confirmada não é reinferida. A parte ainda não confirmada
pode ser refeita. O padrão confirma aproximadamente a cada 4.096 registros,
podendo exceder esse número por até uma janela de leitura.

A retomada se aplica a `run`. Uma preparação/reparticionamento interrompida
deve ser refeita em uma saída vazia; ela não publica um manifesto parcial.

- Cada parte tem marcador de commit com localização na entrada, contagens e SHA-256.
- Arquivos temporários e partes sem marcador não são considerados resultados.
- Uma interrupção entre o marcador e o resumo é recuperada pelo marcador.
- A retomada rápida confere cobertura, existência/tamanho e metadados Parquet.
  `--full-checksums` também relê os bytes das partes para verificar SHA-256.
- `prepare --strict-fingerprints` registra SHA-256 das entradas; nesse caso
  `run/verify --full-checksums` também detecta alterações que preservem tamanho
  e mtime. Sem esse preparo, a identidade da entrada depende de caminho,
  tamanho e mtime, mesmo com `--full-checksums`.
- As entradas devem permanecer imutáveis durante toda a execução.
- Mudanças de modelo, tokenizer, limite, prefixo, precisão, backend de atenção
  ou versões registradas exigem outra saída. Ajustes de lote, prefetch,
  threads e quantidade de processos são permitidos na retomada.
- Partes confirmadas corrompidas causam erro. Não há reparo destrutivo automático:
  restaure a parte de uma cópia íntegra ou prepare uma nova saída.

`status` apresenta contagens confirmadas. Durante `run`, o log também apresenta
registros processados na tentativa atual, ainda que o checkpoint não tenha sido
publicado. `COMPLETE.json` só é criado após todos os ranks e tarefas terminarem.
O diretório `attempts/` guarda métricas e erros por tentativa/rank.
`status` consulta resumos e não substitui a verificação de integridade.

`verify` percorre os resultados, valida anotações, cobertura e contagens. Seus
códigos de saída são **0** para conclusão íntegra, **3** para resultado parcial
íntegro e **2** para erro. Interrupção por teclado retorna **130**.

A ordem canônica é a ordem de `tasks` em `manifest.json`, seguida pelo número
da parte e pela ordem dos registros. Arquivos de entrada são ordenados pelo
caminho absoluto. A ordem não é garantida pela leitura genérica de um diretório
Parquet: use o manifesto para reconstruí-la. As partes ficam em
`parts/<task-id>/<sequência>.parquet` ou `.jsonl`; leia **somente partes com
marcador `.commit.json` correspondente**. Não misture partes temporárias,
staging e metadados na leitura do corpus.

## Ajustes de desempenho

| Opção de `run` | Padrão | Efeito |
|---|---:|---|
| `--batch-size` | 32 | Máximo de chunks por inferência |
| `--max-batch-tokens` | 8192 | Limite de `tamanho do lote × comprimento após padding` |
| `--read-rows` | 256 | Registros por janela de leitura/tokenização |
| `--window-bytes` | 8388608 | Limite aproximado de bytes por janela JSONL |
| `--max-record-bytes` | 67108864 | Limite de uma linha JSONL; excesso interrompe |
| `--prefetch` | 2 | Profundidade de cada fila limitada |
| `--checkpoint-rows` | 4096 | Frequência de confirmação de partes |
| `--torch-threads` | 1 | Threads CPU por processo de inferência |
| `--arrow-threads` | 1 | Threads de leitura/CPU PyArrow por processo |
| `--attention` | sdpa | `sdpa` ou `eager`; não instala FlashAttention externo |

Chunks são agrupados por comprimento dentro da janela e reassociados aos
documentos antes da escrita. Em OOM CUDA, o lote é dividido e repetido; um chunk
que sozinho não caiba causa erro explícito. A lógica foi testada com falhas
simuladas; capacidade real de VRAM exige validação em GPU.

Parquet usa row groups como tarefas. Para um arquivo com um grupo enorme ou
poucos grupos para as GPUs disponíveis, prepare uma cópia reparticionada:

```bash
educability-annotator prepare --input grande.parquet --output reparticionado \
  --parquet-shard-rows 8192 --text-column text
```

Essa opção custa uma leitura/escrita adicional e espaço em `staging/`; a entrada
fica intacta. JSONL usa intervalos de aproximadamente 64 MiB por padrão,
configuráveis em `prepare --jsonl-task-bytes`.

A memória não depende do número total de registros, mas depende dos lotes,
filas, metadados de tarefas e do maior registro. O tokenizer precisa materializar
a codificação do texto individual para detectar truncamento. `--window-bytes`
se aplica ao JSONL; Parquet limita a janela por linhas e pode decodificar páginas
grandes. Reduza `--read-rows`/`--prefetch` em corpora com registros grandes.

## Testes e benchmark

```bash
python -m pytest -q

# Habilita o teste separado com pesos publicados (aproximadamente 277 MB).
EDUCABILITY_TEST_HUB=1 python -m pytest -m hub -q -s

# Modelo local já baixado, sem necessidade de outro download:
EDUCABILITY_TEST_HUB=1 EDUCABILITY_TEST_MODEL=/caminho/modelo \
  python -m pytest -m hub -q -s

python scripts/benchmark.py --rows 10000 --device cpu \
  --report benchmark.json
```

A suíte principal usa um ModernBERT pequeno criado localmente e inclui
lançadores `torchrun` reais, dois nós simulados em localhost, encerramento
forçado e retomada. O teste CUDA é automaticamente omitido sem GPU.
Veja [arquitetura](docs/architecture.md), [implantação](docs/cluster.md) e
[resultados de validação](docs/validation.md).

## Modelo e atribuição

Revisão padrão:
`6a114f1c8c43191137efe3e75f15e0d7af1ba292`.
O modelo deriva de **Itau-Unibanco/NorBERTo-base** e é distribuído sob
**CC BY-NC-SA 4.0**. Consulte a
[licença do modelo](https://huggingface.co/gregassagraf/educability-norberto-mini-4class-v1/blob/main/LICENSE)
e preserve sua atribuição. Este pacote não redistribui os pesos; os termos do
modelo continuam aplicáveis ao seu uso.


## Datasets do Hugging Face Hub

Instale o adaptador opcional (a instalação local continua independente de `datasets`):

```bash
python -m pip install -e '.[hub]'
# Para executar também os testes:
python -m pip install -e '.[hub,test]'
```

O leitor usa `datasets==4.4.1`, com `streaming=True`, sem shuffle. Aceita datasets
com arquivos vinculados a um commit do próprio repositório Hub e leitores
nativos da biblioteca; scripts remotos e fontes externas mutáveis são rejeitados.
Não materializa o corpus inteiro em disco ou memória.

```bash
educability-annotator prepare \
  --dataset cornell-movie-review-data/rotten_tomatoes \
  --dataset-split test \
  --text-column text \
  --output-format parquet \
  --output resultados-hub

educability-annotator run --output resultados-hub --device cpu
educability-annotator status --output resultados-hub
educability-annotator verify --output resultados-hub --full-checksums
```

O exemplo usa um dataset pequeno em inglês apenas para demonstrar o carregamento;
o classificador de educabilidade foi desenvolvido para português.
Use `--dataset-config nome` quando houver múltiplas configurações e
`--dataset-revision tag-ou-commit` para selecionar uma revisão. `prepare` resolve
a revisão para SHA e registra config, split, esquema e shards. O split é obrigatório;
se a config for omitida, vale a resolução padrão do próprio dataset. `--revision`
continua selecionando a revisão **do modelo**. `--input` e `--dataset` são exclusivos.

As saídas Hub são Parquet por padrão ou JSONL com `--output-format jsonl`.
São preservadas as colunas e os valores tabulares apresentados pelo leitor
`datasets`, incluindo listas, estruturas e nulos. Não se promete preservação
byte a byte dos arquivos remotos. JSONL rejeita tipos como bytes e timestamps
que não tenham representação JSON direta; use Parquet nesses casos. Não há
conversão silenciosa desses valores para strings. Mídia não é decodificada nem
baixada a partir de referências presentes nas células.

Para acesso privado ou gated, execute `hf auth login` no ambiente ou disponibilize
`HF_TOKEN` com acesso ao dataset. Cada processo/nó precisa de acesso autenticado.
Não passe tokens em argumentos da ferramenta: eles não são gravados em manifestos
nem checkpoints. Aceitar os termos de um dataset gated continua sendo uma ação
na conta do usuário. Os resultados ficam na saída local/compartilhada; nada é
publicado no Hub.

```bash
torchrun --standalone --nproc-per-node=8 \
  -m educability_annotator run --output resultados-hub --device cuda

# Depois de encerrar todos os processos de uma tentativa interrompida:
torchrun --standalone --nproc-per-node=4 \
  -m educability_annotator run --output resultados-hub --device cuda \
  --resume --recover-lock
```

Cada shard é uma tarefa estável, independente da quantidade de GPUs. Poucos shards
limitam o paralelismo e geram um aviso; não se baixa o corpus para reparticioná-lo.
O progresso apresenta tarefas e registros confirmados, sem estimar um total
inexistente. A retomada pode reler o shard corrente para alcançar o checkpoint,
mas não repete inferência de partes confirmadas. Mantenha a versão de `datasets`
registrada na execução.

`--offline`, `--strict-fingerprints` e `--parquet-shard-rows` não se aplicam a
entradas Hub. `--full-checksums` verifica integralmente as **partes de saída**;
a identidade remota é validada pela revisão, lista de arquivos, esquema e shards,
sem baixar toda a entrada para calcular hashes. `verify` requer acesso ao Hub e
explicita essa distinção no resultado.

O esquema é obtido dos metadados do leitor; se ausente, uma amostra de até 256
registros fixa o esquema. Novas colunas são erro. Buffers de leitura da biblioteca
e o maior registro individual também consomem memória; `--max-record-bytes` é
verificado após a decodificação do registro remoto, não antes do download.
