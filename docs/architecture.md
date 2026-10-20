# Decisões arquiteturais

## Unidade de trabalho e memória

O manifesto imutável contém arquivos ordenados e tarefas com IDs estáveis.
Uma tarefa é um intervalo de linhas completas JSONL ou um row group Parquet.
Cada tarefa pode produzir várias partes/checkpoints. O usuário pode regravar
Parquet em grupos menores durante a preparação quando precisar de mais tarefas.
Nenhum leitor converte o corpus inteiro em lista, DataFrame ou tabela.

O fluxo por worker é:

1. Uma thread lê janelas e prepara os tokens/chunks.
2. O processo principal faz inferência e agrega probabilidades.
3. Uma thread escreve os resultados e publica checkpoints.

As duas filas têm profundidade limitada. O agrupamento por comprimento é local
à janela: não se ordena o corpus inteiro. Os dados Parquet seguem como Arrow;
somente a coluna textual vira objetos Python. Para JSONL, o texto da linha é
mantido para acrescentar campos sem converter números existentes em float.

O tamanho do maior registro continua relevante: tokenização não é streaming
dentro de uma string arbitrariamente grande. Parquet também possui buffers de
decodificação de páginas. Essas limitações são explícitas, e o benchmark mede
o fluxo real com dados sintéticos.

## Modelo e agregação

`model.py` valida a configuração, carrega pesos safetensors e usa somente a
implementação nativa do Transformers. O tokenizer é clonado para a thread de
preparação, separando seu estado do padding na thread de inferência.

Os chunks conservam a sequência dos tokens de conteúdo. O pós-processador do
tokenizer acrescenta os tokens especiais, evitando presumir IDs a partir de
campos da arquitetura. O limite `educability_max_length` prevalece sobre a
capacidade posicional maior da arquitetura.

A calibração ocorre em FP32, antes da média ponderada. A acumulação documental
usa floats Python de precisão dupla e a ordem original dos chunks. Alterar
lotes ou hardware ainda pode introduzir pequenas diferenças de ponto flutuante;
não se promete igualdade bit a bit. A regra documental é uma decisão desta
ferramenta, sem inferir validação documental a partir das métricas por chunk.

## Distribuição

O modelo cabe em uma GPU. A estratégia é replicar o modelo e dividir o corpus,
com um processo por GPU. Não há sincronização de gradientes nem comunicação de
tensores. `torchrun` lança e supervisiona os processos; a aplicação usa ranks
e pequenos arquivos de coordenação na saída compartilhada.

No início de cada tentativa, o rank 0 verifica o manifesto e os checkpoints,
elimina tarefas completas e distribui as restantes pela heurística de maior
custo primeiro para o rank de menor carga estimada. O custo usa bytes do
intervalo JSONL ou bytes descomprimidos do row group. Isso é uma estimativa,
não uma previsão exata do custo de tokenização/inferência.

As tarefas são exclusivas a um rank e a atribuição fica fixa nessa tentativa.
Não há redistribuição dinâmica ou recuperação de um rank isolado em execução.
Após falha, o job é encerrado e retomado, podendo mudar o número de processos.
Tarefas menores ajudam a reduzir desequilíbrio; não eliminam diferenças entre
textos, GPUs ou storage. Sem filesystem compartilhado, este protocolo não se
aplica.

Não se usam coletivas dentro do laço de dados: um rank sem tarefas pode encerrar
sem bloquear os que ainda processam. O rank 0 aguarda arquivos de conclusão
dos demais; `torchrun` deve encerrar o grupo se um worker morrer.

## Protocolo de publicação e retomada

Para cada parte:

1. Escrever `.tmp`, fechar o formato e executar `fsync`.
2. Renomear para o caminho definitivo e sincronizar o diretório.
3. Calcular SHA-256 e publicar atomicamente o marcador `.commit.json`.
4. Atualizar atomicamente o resumo derivado em `states/`.

O passo 3 é o ponto de confirmação. A retomada reconstrói a sequência pelos
marcadores, verificando intervalos contíguos e contratos. Um arquivo sem marcador
é órfão e pode ser refeito. Um marcador sem arquivo íntegro causa erro. Uma falha
entre os passos 3 e 4 não repete a inferência confirmada.

O cursor JSONL é um offset de byte em fronteira de linha. O cursor Parquet é a
posição dentro do row group; pode ser necessário decodificar e descartar linhas
anteriores para alcançar o cursor. O checkpoint evita repetir inferência, mas
não garante evitar toda releitura de Parquet.

Partes confirmadas não são modificadas. O tamanho de lote, janela e checkpoint
pode mudar ao retomar, porque as próximas partes começam no cursor confirmado,
sem depender do tamanho anterior dos lotes.

O contrato de execução registra manifesto, precisão, implementação de atenção
e versões relevantes. Um bloqueio exclusivo no diretório impede jobs
independentes de escreverem simultaneamente. Cada publicação verifica a posse
da saída. **A recuperação manual de bloqueio pressupõe que o job anterior já
parou**; ela não substitui o encerramento dos workers.

O protocolo exige `mkdir` exclusivo, `rename` atômico no mesmo filesystem,
visibilidade coerente e suporte a `fsync`. Não é um protocolo para S3 ou
filesystems que não ofereçam essas propriedades. A tolerância a perda de energia
também depende das garantias do armazenamento e precisa ser testada no destino.

## Organização e extensões

- `planning` e `data`: descoberta, reparticionamento, manifestos e formatos.
- `model` e `pipeline`: tokens, inferência, agregação e filas limitadas.
- `state`, `distributed` e `runner`: checkpoints, posse da saída e execução.
- `cli`: argumentos, códigos de saída e diagnósticos.

As dependências principais são PyTorch, Transformers, Hugging Face Hub e
PyArrow. Não há scheduler externo, banco de dados distribuído ou framework de
treinamento. Suporte direto a object storage, balanceamento dinâmico e aceleração
por outros runtimes são extensões futuras, não pressupostos ocultos da versão.

## Adaptador Hub

`hub.py` descobre o dataset e fornece janelas ao mesmo pipeline local. A origem
`source_kind=hub` é independente do campo `format`, que escolhe a saída. Manifestos
locais anteriores continuam válidos; a extensão acrescenta campos opcionais.
O manifesto Hub registra SHA resolvido, config, split, hash da lista ordenada de
arquivos, esquema Arrow serializado e número de shards. Não contém credenciais.
Os arquivos devem pertencer ao repositório e à revisão fixada.

Uma tarefa corresponde a `IterableDataset.shard(num_shards=N, index=i)`, com N
fixado na preparação. Sem estimativas por shard, o custo é unitário: distribui-se
a quantidade de shards, sem alegar balanceamento por volume. Não há descarte
modular de exemplos nem leitura integral repetida por cada rank.

O cursor Hub conta registros. O fim da tarefa é desconhecido até EOF. Cada janela
carrega uma cópia JSON independente de `state_dict()`, capturada antes de o
prefetch avançar. O escritor confirma o estado da última janela publicada junto
à parte, depois do rename atômico. `load_state_dict()` restaura a posição na
retomada; pode reler registros do shard atual, sem repetir inferência confirmada.
Se EOF chegar depois de um checkpoint cheio, publica-se uma parte vazia com
marcador terminal. Esse marcador é permitido somente como conclusão, sem avanço.

A saída Parquet usa o esquema do manifesto; a saída JSONL reutiliza o escritor
existente após serialização estrita das linhas. A inferência, agregação, filas,
controle de ranks e locks não são reimplementados. A versão de `datasets` integra
o contrato apenas para entradas Hub. Falhas de rede ou leitura interrompem a
execução, preservando os commits; não existe fallback para download integral.
