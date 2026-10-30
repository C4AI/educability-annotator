# Execução na DGX (Slurm + Docker)

Porte da ferramenta para a DGX H100 do grupo, no padrão dos projetos do pipeline: jobs só por Slurm,
Python só em container, dados no NVMe, relatórios no NFS. O plano operacional, com estado, decisões e
critérios de aceite, fica em `Projetos/PLANO_EDUCABILIDADE.md`, fora do repositório.

## Pré-requisitos

- Acesso ao Slurm (partição `dgx`) e ao Docker do nó de computação.
- Rede para PyPI, `download.pytorch.org`, GitHub e Hugging Face no build e no primeiro check de
  ambiente. Depois disso a inferência roda offline, com o modelo no cache do NVMe.
- Para as fases de dados: o `cpt-corpus` em `/output/$USER/cpt-corpus`.

## Configuração

```bash
cp config.env.template config.env   # e ajuste
```

| Variável | Para quê |
|---|---|
| `EDU_IMAGE`, `EDU_TOOLCHAIN_IMAGE` | Alvos `runtime` e `toolchain` do `docker/Dockerfile`. Mude a tag do runtime sempre que o lock ou o Dockerfile mudarem |
| `EDU_BASE_IMAGE` | Base CUDA, a mesma do Curator 1.4.0 do cpt-corpus (já no nó), fixada por digest |
| `FLASH_ATTN_*` | Versão e sha256 das duas variantes de ABI da wheel pré-compilada |
| `EDU_MODEL`, `EDU_MODEL_REVISION` | Classificador e revisão fixada |
| `EDU_HF_HOME` | Cache do Hugging Face no NVMe |
| `EDU_GPU_POOL` | GPUs do nosso turno; os scripts nunca fixam o número de GPUs |
| `CPT_DATA_ROOT`, `EDU_LABELS_ROOT` | Corpus de entrada e rótulos de saída |

## Execução

Sempre da raiz do repositório; os logs vão para `logs/`:

```bash
# Lock das dependências: só quando docker/requirements.gpu.in mudar (o lock vai para o git)
sbatch slurm/00_lock_deps.sbatch

# Ambiente: imagem (se faltar), modelo no cache, suíte em CPU e modelo publicado contra a referência
sbatch slurm/01_check_env.sbatch

# Igual, mais o teste CUDA e FlashAttention-2 × SDPA no modelo real (no turno de GPU)
sbatch --gres=gpu:1 slurm/01_check_env.sbatch
```

## Estrutura de saída

```text
reports/env/<job>/           tests-cpu.xml, tests-hub.xml, tests-cuda.xml, envcheck_<job>.json
logs/                        saídas do Slurm (%x_%j.out/.err)
/output/$USER/hf-cache/      modelo na revisão fixada
/output/$USER/educability-annotator_<job>/   scratch do job, apagado no fim
```

Na FASE 2 do plano entram os rótulos laterais, em
`${EDU_LABELS_ROOT}/educability-<modelo>-<rev7>/source=<id>/part-*.parquet`.

## Integração com o pipeline

- **Entrada:** o `cpt-corpus`. É o `unified/v1` menos os documentos removidos pelo dedup e, depois da
  partição, o `release/v1/kept`.
- **Saída:** rótulos chaveados por `doc_uid`, que entram no release do `cpt-corpus`, como colunas
  ou como config separada.

## Decisões de arquitetura

- **Imagem própria, não do NGC.** Duas versões estão presas: o Transformers 4.53.3, com que o modelo
  foi publicado, e um FlashAttention-2 compatível com o torch. A base CUDA é a do Curator do
  `cpt-corpus`, que já está no nó, e o Python 3.12 é o do Ubuntu 24.04.
- **TLS.** O OpenSSL 3.0 do Ubuntu 24.04 não trava no buraco de PMTU da saída do cluster, e o `uv`
  usa o TLS do sistema (`UV_NATIVE_TLS`) pelo mesmo motivo.
- **Lock com hashes gerado no próprio container.** O `00_lock_deps` resolve o `requirements.gpu.in`
  no alvo `toolchain`, com o Python da imagem. O build instala só do lock, com `--require-hashes`;
  nada é instalado em tempo de execução.
- **FlashAttention-2 pré-compilado.** A wheel depende do ABI de C++ do torch. As duas variantes ficam
  fixadas por sha256, e o build escolhe pela flag do torch instalado.
- **Código montado do scratch.** Todo job roda o código do momento da submissão. Os containers rodam
  com o usuário do host (`--user`), nos núcleos que o Slurm deu ao job (`--cpuset-cpus`) e só com
  as GPUs do job (`--gpus "device=..."`, nunca `all`).
