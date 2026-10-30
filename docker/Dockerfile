# Imagem do educability-annotator na DGX (PLANO_EDUCABILIDADE.md, FASE 1). O código entra por
# montagem a partir do scratch do job, então todo job roda o código do momento da submissão.
#
# Base: a CUDA 12.9.1 devel do Curator 1.4.0, fixada por digest no config.env e já presente no nó por
# causa do cpt-corpus. Python 3.12 do Ubuntu 24.04. Rede: o OpenSSL 3.0 do Ubuntu não trava no buraco
# de PMTU da saída do cluster, e o uv usa o TLS do sistema (UV_NATIVE_TLS) pelo mesmo motivo.
#
# Alvos:
#   toolchain  Python + uv. O slurm/00_lock_deps.sbatch gera nele o docker/requirements.gpu.lock.
#   runtime    dependências do lock (com hashes) + flash-attn pré-compilado, fixado por sha256.
#              PyTorch 2.7.1 cu128 roda em H100 e B200; Transformers 4.53.3 é a versão da
#              publicação do modelo.
#
# Build (feito pelo lazy_build_edu dos sbatch, com os valores do config.env):
#   docker build -f docker/Dockerfile --target runtime -t educability-annotator:<tag> \
#       --build-arg BASE_IMAGE=... --build-arg FLASH_ATTN_SHA256_ABI_TRUE=... .
ARG BASE_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04
FROM ${BASE_IMAGE} AS toolchain

ARG UV_VERSION=0.12.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NATIVE_TLS=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:/opt/uv:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl python3.12 python3.12-venv python3.12-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/uv \
    && curl -fsSL "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
        | tar -xz -C /opt/uv --strip-components=1 \
    && uv --version

WORKDIR /work

FROM toolchain AS runtime

ARG FLASH_ATTN_VERSION=2.8.3
ARG FLASH_ATTN_SHA256_ABI_TRUE=
ARG FLASH_ATTN_SHA256_ABI_FALSE=

COPY docker/requirements.gpu.lock /tmp/requirements.gpu.lock
# Cache do uv montado pelo BuildKit: se o build cair no meio, os pacotes baixados ficam para a próxima.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python /usr/bin/python3.12 --seed \
    && uv pip install --require-hashes -r /tmp/requirements.gpu.lock

# O flash-attn tem de ter o ABI de C++ do torch instalado; as duas variantes ficam fixadas por sha256 e o
# build escolhe pela flag do próprio torch. Sem o sha256 da variante, o sha256sum -c falha e o build para.
RUN abi=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") \
    && torch_mm=$(python -c "import torch; print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))") \
    && if [ "${abi}" = TRUE ]; then sha="${FLASH_ATTN_SHA256_ABI_TRUE}"; else sha="${FLASH_ATTN_SHA256_ABI_FALSE}"; fi \
    && wheel="flash_attn-${FLASH_ATTN_VERSION}+cu12torch${torch_mm}cxx11abi${abi}-cp312-cp312-linux_x86_64.whl" \
    && curl -fsSL -o "/tmp/${wheel}" \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/$(echo "${wheel}" | sed 's/+/%2B/')" \
    && echo "${sha}  /tmp/${wheel}" | sha256sum -c - \
    && uv pip install --no-deps "/tmp/${wheel}" \
    && rm "/tmp/${wheel}"

# --user nos containers: o venv precisa ser legível por qualquer uid. O import do flash_attn fica para o
# check de ambiente com GPU (a extensão CUDA precisa do driver, que o build não tem).
RUN chmod -R a+rX /opt/venv \
    && python -c "import importlib.metadata as m; print({p: m.version(p) for p in ('torch', 'transformers', 'tokenizers', 'flash_attn', 'pyarrow', 'numpy')})"

ENV PYTHONPATH=/opt/educability-annotator/src
WORKDIR /work
