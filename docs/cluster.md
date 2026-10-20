# Implantação em DGX/HGX e múltiplos nós

## Contrato de infraestrutura

Use um processo por GPU e uma réplica completa do classificador por processo.
A quantidade de nós, placas HGX ou GPUs por servidor entra na topologia do
`torchrun`; o código de inferência é o mesmo.

Antes de ampliar a escala:

1. Instale uma distribuição PyTorch/CUDA compatível com os drivers e a geração
   exata das GPUs. Confira a [matriz de instalação do PyTorch](https://pytorch.org/get-started/locally/).
   A versão mínima do pacote não garante suporte a toda GPU futura.
2. Monte entradas e saída compartilhada nos **mesmos caminhos absolutos** em
   todos os nós. Confirme exclusividade de `mkdir`, renomeação atômica, coerência
   de leitura e durabilidade de `fsync` no filesystem usado.
3. Instale a mesma versão do pacote e das dependências em todos os nós.
4. Prepare o manifesto uma vez. Garanta mais tarefas do que GPUs; em Parquet,
   use `--parquet-shard-rows` se necessário.
5. Disponibilize os pesos no cache de cada nó antes do job, preferencialmente
   em SSD local. Com um modelo local, sua pasta precisa existir no mesmo caminho
   e ter os mesmos arquivos em todos os nós.
6. Libere a comunicação entre nós para o rendezvous do `torchrun`. O endereço
   anunciado pelo nó deve ser alcançável pelos demais.

O código não troca tensores entre GPUs, portanto não depende de NCCL, NVLink ou
InfiniBand para sincronização de inferência. A rede continua relevante para
acessar storage compartilhado, cache remoto e coordenação.

## Cache do modelo

Use a mesma revisão em todos os nós. O manifesto já fixa a revisão padrão.
Execute o download uma vez por nó, no mesmo ambiente/cache usado pelos workers:

```bash
export HF_HOME=/scratch/$USER/huggingface
hf download gregassagraf/educability-norberto-mini-4class-v1 \
  --revision 6a114f1c8c43191137efe3e75f15e0d7af1ba292 \
  --include '*.json' '*.safetensors'
```

Em seguida, `run --offline` evita dependência de rede externa durante o job.
A saída e as entradas continuam compartilhadas. Para entradas em SSD local,
elas precisam ter cópias idênticas nos caminhos registrados pelo manifesto;
a ferramenta não faz essa distribuição de arquivos.

## Dois nós, oito GPUs por nó

No nó coordenador, prepare a saída:

```bash
educability-annotator prepare \
  --input '/shared/corpus/*.parquet' \
  --output /shared/anotacoes/experimento-001 \
  --text-column text --max-document-tokens 5000
```

Em ambos os nós, use o mesmo ID novo e o endereço real do coordenador:

```bash
# Nó 0
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
  --rdzv-backend=c10d --rdzv-endpoint=10.0.0.10:29500 \
  --rdzv-id=educability-001 --local-addr=10.0.0.10 --max-restarts=0 \
  -m educability_annotator run \
  --output /shared/anotacoes/experimento-001 \
  --device cuda --dtype auto --offline \
  --batch-size 64 --max-batch-tokens 32768 \
  --torch-threads 2 --arrow-threads 2
```

```bash
# Nó 1
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=1 \
  --rdzv-backend=c10d --rdzv-endpoint=10.0.0.10:29500 \
  --rdzv-id=educability-001 --local-addr=10.0.0.11 --max-restarts=0 \
  -m educability_annotator run \
  --output /shared/anotacoes/experimento-001 \
  --device cuda --dtype auto --offline \
  --batch-size 64 --max-batch-tokens 32768 \
  --torch-threads 2 --arrow-threads 2
```

Os IPs são exemplos e devem ser substituídos pelos da rede do cluster.
Os mesmos argumentos semânticos devem ser usados em todos os ranks. Para
retomar, acrescente `--resume` em todos. Se houve falha, primeiro encerre todo
o job anterior; então acrescente também `--recover-lock`.

Use `--max-restarts=0`: esta versão oferece retomada explícita por checkpoints,
não reinício elástico automático no mesmo job. Alterar `WORLD_SIZE` durante
uma tentativa não é suportado; alterar o número de GPUs ao relançar é.

## Slurm

O arquivo [examples/slurm.sbatch](../examples/slurm.sbatch) fornece um template
de dois nós, oito GPUs por nó, com um lançador por nó.

```bash
export ANNOTATOR_ROOT=/shared/apps/educability-annotator
export ANNOTATOR_OUTPUT=/shared/anotacoes/experimento-001
sbatch examples/slurm.sbatch
```

Ajuste partição, tempo, conta, CPUs e recursos GPU às regras do cluster. O
template usa o hostname completo para anunciar o endereço local; substitua-o
pela interface correta se o cluster tiver redes separadas.

Para recuperação, após confirmar que o job anterior terminou:

```bash
RECOVER_ANNOTATOR_LOCK=1 sbatch examples/slurm.sbatch
```

O template sempre passa `--resume`, que também aceita uma saída preparada
ainda sem execução. `RECOVER_ANNOTATOR_LOCK=1` é uma decisão operacional
explícita e não deve ser habilitada enquanto outro job usar a saída.

## Dimensionamento e medição

Comece com uma GPU e uma amostra representativa dos comprimentos de texto.
Meça documentos/s, tokens/s, memória, tempo de leitura, tokenização, inferência
e escrita. Use `COMPLETE.json` e os registros em `attempts/`.

1. Aumente `--batch-size` e `--max-batch-tokens` até usar adequadamente a GPU,
   mantendo margem de VRAM. O limite considera padding.
2. Compare FP32 e BF16 em exemplos representativos antes de adotar BF16.
3. Ajuste `--read-rows` e `--prefetch`; mais dados em fila usam mais RAM.
4. Ajuste CPU/NUMA e threads por processo. Oito GPUs com oito processos
   multiplicam o consumo total de threads. Reserve CPU para tokenização e I/O.
5. Compare 1, 2, 4 e 8 GPUs no mesmo nó; depois dois ou mais nós.
6. Se a GPU ficar ociosa, identifique o estágio limitante antes de aumentar
   concorrência. Tokenização, leitura, compressão e metadados do filesystem
   podem saturar primeiro.
7. Aumente `--checkpoint-rows` se muitos arquivos pequenos sobrecarregarem o
   storage. Isso aumenta o trabalho que pode ser repetido após falha.
8. Reduza tarefas muito grandes para evitar uma cauda longa em um único rank.
   O escalonamento é estático por tentativa.

A soma dos tempos dos estágios pode exceder o tempo de parede, pois há
sobreposição. `write_seconds` mede escrita/finalização dos dados; hash e
publicação de metadados também consomem tempo de parede.

## O que permanece sem validação em hardware de destino

- Throughput, ocupação de GPU e quantidade ideal de chunks/tokens por lote.
- Comportamento real do mecanismo de recuperação de OOM e margem de VRAM.
- Desvios numéricos e possíveis mudanças de argmax com BF16.
- Saturação de CPU, NUMA, armazenamento, metadados e rede compartilhada.
- Eficiência de escala entre GPUs, nós e sistemas com múltiplas placas HGX.
- Durabilidade e visibilidade dos checkpoints sob falhas do storage/energia.

A simulação local testa divisão, exclusividade, coordenação e retomada.
Ela não mede essas propriedades de desempenho nem substitui um ensaio
controlado de falha no filesystem de produção.

## Entrada Hub em múltiplos nós

Instale `.[hub]` com `datasets==4.4.1` em todos os nós. Prepare o manifesto uma vez
na saída compartilhada e use os mesmos comandos multinó deste guia. Cada nó
precisa acessar o Hub e autenticar-se via seu cache de login ou `HF_TOKEN` fornecido
pelo mecanismo de secrets do cluster; não grave o token no script Slurm.

Entradas remotas dispensam caminhos locais compartilhados para o corpus, mas a
saída e o protocolo de coordenação continuam exigindo filesystem compartilhado.
Mantenha cache de modelo por nó. O número de shards limita a quantidade útil de
workers; os shards não são regravados. Custos desiguais podem deixar GPUs ociosas.
Meça separadamente rede/Hub, tokenização, inferência e escrita. Latência HTTP,
limites de requisições e banda de saída do Hub também afetam a escala. Retomar um
shard grande pode exigir releitura substancial mesmo sem repetir inferência.
