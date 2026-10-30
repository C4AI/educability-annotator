# Execução na DGX (Slurm + Docker)

Porte da ferramenta para a DGX H100 do grupo, no padrão dos projetos do pipeline. As regras são:
jobs só por Slurm, Python só em container, dados no NVMe e relatórios no NFS.

O porte acrescenta o modo de **rótulos laterais** (`python -m educability_annotator.sidecar`), feito
para corpora Parquet grandes. A CLI original (`prepare`/`run`/`verify`) continua igual. O plano
operacional, com decisões e números, fica em `Projetos/PLANO_EDUCABILIDADE.md`, fora do repositório.

## O modo de rótulos laterais

A semântica é a do `Classifier` original:
- chunks de 1.022 tokens de conteúdo, mais os tokens especiais, sem sobreposição;
- `softmax(logits / T)` uma vez por chunk, com a temperatura do modelo;
- média ponderada por tokens, com empate decidido pela menor classe;
- até 5 chunks por documento, por padrão.

O que muda é a execução:

- **Saída lateral:** em vez de reescrever o registro, grava só os rótulos. É um arquivo por arquivo de
  entrada, no mesmo caminho relativo e chaveado pelo ID (`doc_uid`). Cada arquivo tem marcador de
  commit (`.commit.json`, com SHA-256). As colunas são as da ferramenta, com o prefixo
  `educability_`: `class`, `label`, `p1`…`p4`, `expected_score`, `tokens_used`, `chunk_count`,
  `truncated` e `status`. Há mais duas: `max_chunk_score` e `first_chunk_score`.
- **Tokenização por prefixo:** só o texto que o modelo vai ver é tokenizado.
  - O corte cai no último espaço em branco antes de `max_document_tokens × 5` caracteres.
  - Os IDs saem idênticos aos do documento inteiro: os testes conferem com o tokenizer publicado, e
    o caso sem folga é refeito com o texto inteiro.
  - A tokenização roda em processos do DataLoader (`spawn`).
- **Inferência sem padding:** documentos inteiros vão em lotes por orçamento de tokens, com
  `cu_seqlens` e FlashAttention-2, que dá a janela local real do ModernBERT. O SDPA, com sublotes
  ordenados por tamanho, é o caminho de CPU e a alternativa ao FA2.
- **Tarefas dinâmicas:**
  - `prepare` cria o plano imutável (tarefas, custo e faixa de IDs pelos metadados do Parquet) e abre
    uma tentativa.
  - Os processos de todos os ranks tiram tarefas de um contador sob `flock`.
  - Para retomar, basta rodar de novo: o que tem commit não é refeito.
  - O contrato (precisão, atenção, versões) é fixo por diretório de saída.
- **Exclusões e inclusões:** `--exclude` e `--include` recebem Parquets com uma coluna de IDs, como
  as decisões do dedup. Cada tarefa lê só a faixa de IDs da sua parte.

```bash
python -m educability_annotator.sidecar prepare --input-root /input --pattern 'source=*/part-*.parquet' \
    --model /hf/.../snapshots/<rev> --offline --output /labels --dtype float16 --attention flash_attention_2
torchrun --standalone --nproc-per-node=8 -m educability_annotator.sidecar run --output /labels --loader-workers 27
python -m educability_annotator.sidecar verify --output /labels [--full]   # 0 completo, 3 parcial, 2 problema
python -m educability_annotator.sidecar status --output /labels
```

Os sbatch abaixo montam esses comandos; não é preciso chamá-los à mão.

## Pré-requisitos

- Acesso ao Slurm (partição `dgx`) e ao Docker do nó de computação.
- Rede para PyPI, `download.pytorch.org`, GitHub e Hugging Face no build e no primeiro check de
  ambiente. Depois disso a inferência roda offline, com o modelo no cache do NVMe.
- O `cpt-corpus` em `/output/$USER/cpt-corpus` para as rodadas no corpus.

## Configuração

```bash
cp config.env.template config.env   # e ajuste
```

| Variável | Para quê |
|---|---|
| `EDU_IMAGE`, `EDU_TOOLCHAIN_IMAGE` | Alvos `runtime` e `toolchain` do `docker/Dockerfile`. Mude a tag do runtime sempre que o lock ou o Dockerfile mudarem |
| `EDU_BASE_IMAGE`, `UV_VERSION` | Base CUDA (a mesma do Curator 1.4.0 do cpt-corpus, fixada por digest) e versão do uv |
| `FLASH_ATTN_*` | Versão e sha256 das duas variantes de ABI da wheel pré-compilada |
| `EDU_MODEL`, `EDU_MODEL_REVISION`, `EDU_MODEL_TAG` | Classificador, revisão fixada e nome do diretório de rótulos |
| `EDU_HF_HOME` | Cache do Hugging Face no NVMe |
| `EDU_GPU_POOL` | GPUs do nosso turno; os scripts nunca fixam o número de GPUs |
| `CPT_DATA_ROOT`, `CPT_VERSION` | Corpus de entrada |
| `EDU_LABELS_ROOT`, `EDU_PILOT_ROOT` | Rótulos (`labels/<versão>`) e piloto (`labels/<versão>/pilot`) |
| `EDU_EXCLUDE` | Decisões do dedup a pular na entrada `corpus` (padrão: exato e fuzzy) |
| `CPT_PROJECT_DIR`, `CPT_INGEST_CHECKS`, `CPT_BEFORE_AFTER` | Arquivos do cpt-corpus de que o relatório do piloto tira os tokens Qwen3 por fonte |

## Execução

Sempre da raiz do repositório; os logs vão para `logs/`.

```bash
# Ambiente
sbatch slurm/00_lock_deps.sbatch                 # lock das dependências, só quando requirements.gpu.in mudar
sbatch slurm/01_check_env.sbatch                 # imagem, modelo no cache, suíte em CPU, modelo publicado
sbatch --gres=gpu:1 slurm/01_check_env.sbatch    # + teste CUDA e FlashAttention-2 × SDPA no modelo real
sbatch slurm/02_tests.sbatch tests/test_sidecar.py   # pytest avulso (aceita os argumentos do pytest)

# Rodada no corpus final (nó exclusivo): partição kept do cpt-corpus, FA2/FP16
sbatch --exclusive --cpus-per-task=224 --mem=0 --export=ALL,ANNOTATE_INPUT=kept slurm/10_annotate.sbatch
# Retomar: submeter o mesmo comando (tarefas com commit não são refeitas)

# Relatórios
sbatch slurm/23_labels_report.sbatch             # cobertura (mantidos × rotulados) e distribuição por fonte
sbatch --export=ALL,COMPARE_REFERENCE=<dir>,COMPARE_CANDIDATE=<dir> slurm/22_compare_labels.sbatch
```

Variáveis do `10_annotate.sbatch` (por `--export`):

| Variável | Padrão | Efeito |
|---|---|---|
| `ANNOTATE_INPUT` | `corpus` | `corpus` (`unified` menos `EDU_EXCLUDE`), `kept` (`release/<versão>/kept`), `pilot` (amostra do piloto), `dir` (`ANNOTATE_INPUT_DIR`, com `ANNOTATE_OUTPUT` obrigatório) |
| `ANNOTATE_OUTPUT` | `${EDU_LABELS_ROOT}/${EDU_MODEL_TAG}` | Diretório dos rótulos. Outra precisão exige outro diretório |
| `EDU_DEVICE` | `cuda` | `cpu`: processos FP32 com `EDU_TORCH_THREADS` threads (`EDU_CPU_RANKS` processos), saída com sufixo `-cpu-fp32` |
| `EDU_DTYPE`, `EDU_ATTENTION` | `float16`, `flash_attention_2` (GPU) | Em CPU, `float32` e `sdpa` |
| `EDU_MAX_DOC_TOKENS` | 5110 | Tokens de conteúdo por documento (5 chunks) |
| `EDU_LOADER_WORKERS` | CPUs ÷ processos − 1 | Processos de tokenização por rank |
| `EDU_RANKS_PER_GPU` | 1 | Mais de um processo por GPU |
| `EDU_BATCH_TOKENS`, `EDU_READ_ROWS` | 65536, 4096 | Orçamento de tokens por lote; linhas por leitura |
| `EDU_PATTERN`, `EDU_SAMPLE_TASKS`, `EDU_SAMPLE_SEED` | todas as partes | Subconjunto das partes (testes) |

Piloto por fonte:

```bash
sbatch slurm/20_pilot_sample.sbatch                       # amostra uniforme por fonte, com o texto
sbatch --export=ALL,PILOT_SPLIT_ONLY=1,PILOT_SUBSAMPLE=10000 slurm/20_pilot_sample.sbatch   # reduz e reparte
sbatch --gres=none --cpus-per-task=64 --export=ALL,EDU_DEVICE=cpu,ANNOTATE_INPUT=pilot slurm/10_annotate.sbatch
sbatch slurm/21_pilot_report.sbatch                       # distribuição, scores da fonte, volume por limiar
sbatch slurm/11_bench_gpu.sbatch                          # vazão e paridade por configuração (1 GPU)
```

## Estrutura de saída

```text
${EDU_LABELS_ROOT}/${EDU_MODEL_TAG}/
├── plan.json, contract.json, attempt.json, tokenizer.json, claims.counter
├── attempts/<id>/rank-<r>.json                métricas por rank
└── source=<id>/part-*.parquet (+ .commit.json) rótulos, mesmo caminho da entrada
${EDU_PILOT_ROOT}/sample/, ${EDU_PILOT_ROOT}/labels/   amostra do piloto e rótulos dela
reports/env/<job>/     testes e envcheck
reports/runs/<job>/    plano, contrato, métricas e verify de cada rodada
reports/pilot/, reports/labels/   relatórios do piloto, de paridade e de cobertura
/output/$USER/hf-cache/           modelo na revisão fixada
```

## Números de referência (DGX H100)

- **FP16 × FP32** nos mesmos 287 mil documentos: 0,029% de trocas de classe, |Δ score esperado|
  p99 0,0013, tokenização idêntica.
- **Rodada no corpus final v1** (1.184.179.060 docs, 27.677 partes, nó exclusivo): 19h24 e `verify`
  íntegro.
  - **Cobertura:** 0 documentos sem rótulo.
  - **Gargalo:** a tokenização em CPU, no começo, com as partes de documentos longos.
- **Em CPU,** cada processo de 4 threads faz ~1.700 tokens/s em FP32. Serve para pilotos, não para o
  corpus.

## Integração com o pipeline

- **Entrada:** o `cpt-corpus`, de preferência a partição `release/<versão>/kept`, já deduplicada.
  Também aceita o `unified` com as decisões do dedup em `EDU_EXCLUDE`.
- **Saída:** rótulos chaveados por `doc_uid` em `labels/<versão>/`. Entram no release do cpt-corpus
  como arquivo à parte, ligado ao kept por `doc_uid`.

## Decisões de arquitetura

- **Imagem própria, não do NGC:**
  - o Transformers 4.53.3, com que o modelo foi publicado, e um FlashAttention-2 compatível com o
    torch ficam presos;
  - a base CUDA é a do Curator do cpt-corpus, e o Python 3.12 é o do Ubuntu 24.04;
  - o OpenSSL 3.0 não trava no buraco de PMTU da saída do cluster, e o `uv` usa o TLS do sistema.
- **Lock com hashes gerado no próprio container:** o `00_lock_deps` resolve o `requirements.gpu.in`
  no alvo `toolchain`, e o build instala só do lock, com `--require-hashes`.
- **FlashAttention-2 pré-compilado:** as duas variantes de ABI ficam fixadas por sha256, e o build
  escolhe pela flag do torch instalado.
- **Código montado do scratch:** todo job roda o código do momento da submissão.
  - Os containers rodam com o usuário do host (`--user`, com `USER` e `LOGNAME` exportados; sem
    isso, o torch quebra ao montar o cache do inductor).
  - Usam os núcleos do job (`--cpuset-cpus`) e só as GPUs do job (`--gpus "device=..."`, nunca
    `all`).
