# Validação — 20 de outubro de 2026

## Ambiente e alcance

Validação local em **Apple M5, 32 GiB de RAM, macOS 27.0.1 ARM64**, usando CPU.
Nenhum servidor DGX/HGX, GPU NVIDIA, Slurm real ou filesystem de cluster esteve
disponível.

| Componente | Versão executada |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.7.1 |
| Transformers | 4.53.3 |
| Tokenizers | 0.21.4 |
| Hugging Face Hub | 0.36.2 |
| PyArrow | 26.0.0 |
| NumPy | 2.4.6 |
| pytest | 9.1.1 |

O ambiente completo dos componentes relevantes está em
[test-environment.json](test-environment.json).

## Testes automatizados

A suíte offline usa um tokenizer local e um `ModernBertForSequenceClassification`
pequeno, criado com seed fixa. Verifica:

- Preservação de JSONL Unicode, números de alta precisão e objetos aninhados.
- Preservação de tipos, campos internos de listas e metadados de Parquet.
- Particionamento JSONL por bytes, inclusive CRLF e ausência de newline final.
- Entradas vazias, textos inválidos, colisões de nomes e corrupção.
- Calibração, chunking sem perda/duplicação de tokens, limites e média ponderada.
- Inferência com limites de lote/tokens e restauração da associação documental.
- Falta de memória CUDA simulada, divisão de lotes e erro em chunk único.
- Distribuição disjunta, equivalência com dois e três processos CPU reais,
  mais workers que tarefas e dois lançadores simulando dois nós.
- Encerramento forçado de uma execução com dois processos e retomada com três.
- Falhas durante escrita, depois da renomeação e depois do marcador de commit.
- Retomada dentro de um row group com outra janela de leitura.
- Conflitos de posse da saída, assinaturas incompatíveis entre ranks e alterações
  de entradas, manifesto ou configuração.

Resultado final: **65 testes aprovados em 20,28 segundos e 2 omitidos**
(o teste Hub, executado separadamente abaixo, e o teste real CUDA).
Os resultados em formato JUnit ficam em [tests-cpu.xml](tests-cpu.xml).
O teste com pesos publicados é habilitado separadamente; o teste real de CUDA
é omitido na ausência de GPU. O workflow Linux/Python 3.11 e 3.13 está preparado,
mas não foi executado em GitHub Actions nesta entrega.

O rendezvous local padrão do PyTorch encontrou um problema de resolução
hostname/IPv6 neste Mac. Os testes distribuídos usam endereços IPv4 explícitos,
incluindo `--local-addr=127.0.0.1`; com isso executaram os lançadores reais.
Essa adaptação não substitui testes de rede entre servidores distintos.

## Comparação com o modelo publicado

Foi baixada e executada em CPU/FP32 a revisão:

`6a114f1c8c43191137efe3e75f15e0d7af1ba292`

O teste compara quatro textos, incluindo uma entrada truncada a 1.024 tokens,
com a referência calculada diretamente por:

```python
torch.softmax(model(**inputs).logits.float() / model.config.temperature_calibration, dim=-1)
```

Resultado: **1 teste aprovado**, diferença absoluta máxima nas probabilidades
de **4,17232513 × 10⁻⁷**, abaixo da tolerância usada (`atol=1e-6`, `rtol=1e-5`).
A comparação verifica equivalência de inferência; não reavalia a qualidade
pedagógica ou as métricas científicas do classificador.

Comando:

```bash
EDUCABILITY_TEST_HUB=1 \
EDUCABILITY_TEST_MODEL=.test-artifacts/published-model \
  .venv/bin/python -m pytest tests/test_hardware.py -m hub -q -s
```

## Ensaio incremental com pesos reais

Foi executado o pipeline completo de CPU/FP32 sobre **10.000 registros
sintéticos**, com **640.000 tokens de conteúdo**, usando o modelo publicado.
O arquivo JSONL tinha 2.968.890 bytes. Todos os registros foram posteriormente
verificados, incluindo checksums.

| Medida | Resultado |
|---|---:|
| Tempo de execução | 39,90 s |
| Registros por segundo | 250,63 |
| Tokens de conteúdo por segundo | 16.040,17 |
| Tempo acumulado de leitura | 0,154 s |
| Tempo acumulado de tokenização | 0,452 s |
| Tempo acumulado de inferência | 39,291 s |
| Tempo acumulado de escrita/finalização | 0,352 s |
| Pico de RSS | 559,03 MiB |
| Pico de RSS no trecho de 20% a 80% | 559,03 MiB |
| Pico de RSS no último quinto | 556,22 MiB |
| Amostras de memória | 722 |

Configuração: dois threads PyTorch, lotes de até 16 chunks, janelas de 32
registros, checkpoints a cada 128 registros e prefetch 2. O texto de cada
registro é sintético e repetitivo, resultando em 64 tokens de conteúdo.

O tempo reportado inclui inicialização da execução e inferência/escrita, mas
exclui geração do corpus, preparação inicial e a verificação posterior. Os
tempos por estágio se sobrepõem; não devem ser somados para estimar o tempo de
parede. Hash e publicação de metadados estão no tempo de parede.

O patamar observado de memória é compatível com as filas limitadas. Este ensaio
de aproximadamente 40 segundos **não é um teste de estresse prolongado**, nem
uma garantia para registros muito maiores ou outras distribuições de texto.
Não extrapole essa taxa para um corpus real ou GPUs.

Comando e dados completos:

```bash
.venv/bin/python scripts/benchmark.py \
  --model .test-artifacts/published-model \
  --rows 10000 --words 16 --report docs/benchmark-cpu.json
```

[benchmark-cpu.json](benchmark-cpu.json) contém as medições brutas.

### Ensaio ampliado de 100.000 registros

O mesmo pipeline foi executado por mais de sete minutos com dez vezes mais
registros, mantendo os parâmetros anteriores. Os **100.000 resultados foram
verificados integralmente**, com contagem, anotações e SHA-256.

| Medida | Resultado |
|---|---:|
| Registros/chunks | 100.000 |
| Tokens de conteúdo | 6.400.000 |
| Tamanho da entrada | 29.788.890 bytes |
| Tempo de execução | 426,77 s — 7 min 6,77 s |
| Registros por segundo | 234,32 |
| Tokens de conteúdo por segundo | 14.996,27 |
| Tempo acumulado de leitura | 1,658 s |
| Tempo acumulado de tokenização | 5,085 s |
| Tempo acumulado de inferência | 425,964 s |
| Tempo acumulado de escrita/finalização | 3,660 s |
| Pico de RSS | 559,28 MiB |
| Pico de RSS no trecho de 20% a 80% | 556,73 MiB |
| Pico de RSS no último quinto | 541,88 MiB |
| Amostras de memória | 7.821 |

Houve 782 partes confirmadas, com até 128 registros por parte. O pico foi
praticamente igual ao do ensaio de 10.000 registros, sem crescimento contínuo
observado no trecho final. Isso dá evidência local adicional para o uso limitado
de buffers; continua sendo uma execução sintética, em uma única máquina.

```bash
.venv/bin/python scripts/benchmark.py \
  --model .test-artifacts/published-model \
  --rows 100000 --words 16 --report docs/benchmark-cpu-100k.json
```

Medições completas: [benchmark-cpu-100k.json](benchmark-cpu-100k.json).

## Empacotamento e exemplo funcional

Foram verificados a construção e instalação isolada do wheel, a compatibilidade das dependências
instaladas, a compilação dos módulos Python e a sintaxe do template Slurm.
A CLI foi exercitada com `examples/sample.jsonl` e os pesos publicados, incluindo
textos nulos, vazios e ausentes.

## Validação ainda necessária

Exigem hardware e infraestrutura de destino: execução real CUDA/BF16,
dimensionamento de VRAM, recuperação de OOM real, ocupação das GPUs, eficiência
multinó, CPU/NUMA, throughput do storage compartilhado, rede e durabilidade sob
falhas do filesystem. O template Slurm não foi submetido a um cluster.

A agregação documental ponderada foi verificada matematicamente e em testes
de integridade. Sua validade como medida de educabilidade de documentos inteiros
requer avaliação científica específica.

## Extensão Hub — 20/10/2026

Implementada e validada no mesmo ambiente CPU macOS ARM64, com
`datasets==4.4.1` e `fsspec==2025.10.0`. O extra `[hub]` foi instalado no ambiente
local; a CLI e a construção do wheel foram verificadas. O workflow de CI passou
a instalar `[test,hub]`, mas não foi executado em um runner remoto nesta sessão.

Os testes novos usam o `IterableDataset` real com arquivos Parquet e JSON servidos
por HTTP local, além de fixtures em memória. Apenas o catálogo Hub é simulado
nesses testes. Cobrem seleção e pinagem, rejeição de origens externas, esquema,
campos aninhados, registros inválidos, dados vazios, JSON incompatível, estado do
iterador capturado antes do prefetch, corrupção de partes e interrupções antes e
depois da confirmação. Execuções reais com `torchrun` em dois/três processos CPU
foram comparadas; uma tentativa com dois processos foi encerrada e retomada com
três. A inferência também foi comparada ao pipeline de arquivos locais.

Foi executado separadamente o teste público opt-in com os **1.066 registros** do
split `test` de `cornell-movie-review-data/rotten_tomatoes`, revisão
`aa13bc287fa6fcab6daf52f0dfb9994269ffea28`: preparação, streaming, inferência,
publicação e verificação integral das partes passaram. Esse teste usa o
ModernBERT pequeno determinístico da suíte para validar a integração dos dados;
não é uma avaliação linguística nem benchmark do modelo de educabilidade.

```bash
# Suíte sem acesso externo (datasets usa somente as fixtures locais):
python -m pytest -q -m 'not hub and not cuda'

# Acrescentar a integração real com o dataset público:
EDUCABILITY_TEST_DATASET_HUB=1 python -m pytest -q
```

Acesso privado e falhas de autenticação foram **simulados**, sem uso de um dataset
privado real. A retomada evita repetir inferência confirmada, mas não promete
zero releitura de rede. Não foram medidos throughput CUDA, escala DGX/HGX,
limites de requisição do Hub, saturação da rede multinó ou latência de retomada
de shards grandes. Os benchmarks CPU locais anteriores não medem essas condições.

Resultado final desta extensão: **92 testes passaram e 2 foram omitidos em
85,58 segundos**, com `EDUCABILITY_TEST_DATASET_HUB=1`. Os omitidos são o teste
opt-in do modelo publicado (validado na etapa anterior deste projeto) e o teste
CUDA sem GPU disponível. Evidência: [tests-hub-cpu.xml](tests-hub-cpu.xml).
