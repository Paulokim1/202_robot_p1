# Robótica Computacional 2020.2

[Mais orientações no README](./README.md)

## Prova P1 AI

**Você deve escolher somente 3 questões para fazer.**


Nome:_______________


Questões que fez:____________


Observações de avaliações nesta disciplina:

* Inicie a prova no Blackboard para a ferramenta de Proctoring iniciar. Só finalize o Blackboard quando enviar a prova via Github classroom
* Durante esta prova vamos registrar somente a tela, não a câmera nem microfone
* Ponha o nome no enunciado da prova no Github
* Tenha os repositórios https://github.com/Insper/robot202/ e https://github.com/arnaldojr/my_simulation atualizados em seu catkin_ws/src .
* Você pode consultar a internet ou qualquer material, mas não pode se comunicar com pessoas ou colegas a respeito da prova
* Faça commits e pushes frequentes no seu repositório (tem dicas [no final deste arquivo](./inst
rucoes_setup.md))
* Esteja conectado no Teams e pronto para receber calls do professor e da equipe. 
* Avisos importantes serão dados no chat da prova
* Permite-se consultar qualquer material online ou próprio. Não se pode compartilhar informações com colegas durante a prova
* Faça commits frequentes. O primeiro a enviar alguma ideia será considerado autor original
* A responsabilidade por ter o *setup* funcionando é de cada estudante
* Questões de esclarecimento geral podem ser perguntadas no chat do Teams
* Se você estiver em casa pode fazer pausas e falar com seus familiares, mas não pode receber ajuda na prova.
* É vedado colaborar ou pedir ajuda a colegas ou qualquer pessoa que conheça os assuntos avaliados nesta prova.


Existe algumas dicas de referência rápida de setup [instrucoes_setup.md](instrucoes_setup.md)

**Integridade Intelectual**

Se você tiver alguma evidência de fraude cometida nesta prova, [use este serviço de e-mail anônimo](https://www.guerrillamail.com/pt/compose)  para informar ao professor.  Ou [este formulário](https://forms.gle/JPhqjPmuKAHxmvwZ9)

# Setup 

Você precisa deste vídeo para a Questão 1: Salve na pasta Q1

[https://github.com/Insper/robot20/raw/master/media/dados.mp4](https://github.com/Insper/robot20/raw/master/media/dados.mp4)


# Questões


## Questão 1  (3.33 pontos)

Você deve fazer um programa que consegue ver as horas a partir do vídeo do relógio fornecido.

<img src="relogio.gif">

#### Orientações

Trabalhe no arquivo `q1/q1.py`. Este exercício **não precisa** de ROS. Portanto pode ser feito até em Mac ou Windows

Você vai notar que este programa roda o vídeo `relogio.mp4`. Baixe o vídeo [neste endereço](https://github.com/Insper/robot20/raw/master/media/relogio.mp4)


#### O que você deve fazer:

Escrever sobre a janela um número indicando o valor do dado presente na imagem.  O

Quando não houver nada seu programa não precisa escrever coisa alguma.  

|Resultado| Conceito| 
|---|---|
| Não executa | 0 |
| Segmenta ou filtra a imagem baseado em cores ou canais da imagem e produz output visual| 0.75|
| Identifica os ponteiros | 1.5 |
| Calcula o ângulod e cada ponteiro e os imprime ou mostra na tela| 2.6 |
| Informa as horas  | 3.33| 

Casos intermediários ou omissos da rubrica serão decididos pelo professor.


## Questão 2  (3.33 pontos)

<img src="animais_caixas.png">

Você deve fazer um programa que mostra se seu animal designado está dentro da caixa designada.

<img src="./q1_tabela.png" >


#### Orientações

Trabalhe no arquivo `q2/q2.py`. Este exercício **não precisa** de ROS. Portanto pode ser feito até em Mac ou Windows

Você vai notar que este programa roda o vídeo `animais_caixas.mp4`. Baixe o vídeo [neste endereço](https://github.com/Insper/robot20/raw/master/media/animais_caixas.mp4)


#### O que você deve fazer:

De acordo com seu animal e caixa designados, faça o seguinte.

Imprima a mensagem **DENTRO* na tela, sempre que o animal designado estiver 100% dentro da caixa designada.

Quando a condição acima não for verdadeira, seu programa não precisa fazer nada.  

|Resultado| Conceito| 
|---|---|
| Não executa | 0 |
| Usa a rede neural e verifica de alguma forma que o animal presente é o designado | 0.75 |
| Além do item acima, filtra a cor da caixa corretamente | 1.50 | 
| Detecta os limites do animal e da caixa, mas não imprime a mensagem correta | 2.4 | 
| Identifica sempre certo com output na tela  | 3.5 | 

Casos intermediários ou omissos da rubrica serão decididos pelo professor.



## Questão 3 (3.33 pontos)


**Atenção: ** 

Para fazer estra questão você precisa ter o `my_simulation` atualizado.

    cd ~/catkin_ws/src
    cd my_simulation
    git pull

Ou então se ainda não tiver:

    cd ~/catkin_ws/src
    git clone https://github.com/arnaldojr/my_simulation.git

Em seguida faça o [catkin_make](./instrucoes_setup.md). 


Para executar o cenário, faça:

    roslaunch my_simulation novas_formas.launch


Seu robô está num cenário como o que pode ser visto na figura: 

<img src="./formas.png"></img>

ge_1_launch.png" width=50%>


#### O que é para fazer

De acordo com a tabela abaixo, você tem uma cor preferencial.

<img src="./Q3_cores.png"></img>

Faça o robô girar até localizar o cubo da sua cor adequada. 

Quando o robô estiver centralizado no cubo, deve avançar em frente e parar quando estiver a uma distância de 1.5m do cubo. Esta distância deve ser controlada pelo laser. 


#### Detalhes de como rodar


O código para este exercício está em: `p1_202/scripts/Q3.py`

Para rodar, recomendamos que faça:

    roslaunch my_simulation novas_formas.launch

Depois:

    rosrun p1_202 Q3.py



|Resultado| Conceito| 
|---|---|
| Não executa | 0 |
| Consegue filtrar a cor certa| 0.75|
| Além de filtrar a cor, centraliza no cubo certo | 1.5|
| Consegue ler o laser e usar isso para o controle | 2.4 |
| Funciona perfeitamente | 3.33|


Casos intermediários ou omissos da rubrica serão decididos pelo professor.



## Questão 4 (3.33 pontos)


Seu robô está num cenário vazio.


    roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch



#### O que é para fazer


Crie uma função que aceita como parâmetro um valor n maior  que 3.

Faça o robô desenhar um polígono de n lados no chão. Cada lado do polígono deve ter 1.2m.  

Sabe-que que cada ângulo externo de um polígono de raio n deve ter $360/n$ . Por exemplo na figura a seguir cada ângulo externo do pentágono vale 72 graus.

<img src="./angulos_externos.png"></>


Você precisa usar a odometria para se assegurar que a orientação (ângulo) está correto mas *não precisa* para garantir o comprimento das arestas.

Seu código deve ser flexível, mas para testes adote $n=5$ fixo. 

Você pode assumir que o robô começa sempre em (0,0). No Gazebo aperte Ctrl R para fazê-lo voltar a esta posição.

#### Detalhes de como rodar


O código para este exercício está em: `p1_202/scripts/Q4.py`

Para rodar, recomendamos que faça:

    roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch

Depois:

    rosrun p1_202 Q4.py



|Resultado| Conceito| 
|---|---|
| Não executa | 0 |
| Consegue calcular o ângulo externo corretamente | 0.5 |
| Consegue fazer o polígono sem controlar pela odometria | 1.5 |
| Pega o vento da odometria mas o resultado ainda não é perfeito | 2.5|
| Funciona perfeitamente | 3.33|


Casos intermediários ou omissos da rubrica serão decididos pelo professor.
