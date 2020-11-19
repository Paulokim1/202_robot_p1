#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

# Esta solução pode ser vista em:
# https://youtu.be/BHu6OodU-iY


from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np
import math

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

# Arquivos necessários
# Baixe e salve na mesma pasta que este arquivo
# https://github.com/Insper/robot20/raw/master/media/relogio.mp4
video = "relogio.mp4"


### Inicio da solucao

c = (305,246)

def center_of_mass(mask):
    """ Retorna uma tupla (cx, cy) que desenha o centro do contorno"""
    M = cv2.moments(mask)
    # Usando a expressão do centróide definida em: https://en.wikipedia.org/wiki/Image_moment
    if M["m00"] == 0:
        M["m00"] = 1
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return [int(cX), int(cY)]

def crosshair(img, point, size, color):
    """ Desenha um crosshair centrado no point.
        point deve ser uma tupla (x,y)
        color é uma tupla R,G,B uint8
    """
    x,y = point
    cv2.line(img,(x - size,y),(x + size,y),color,5)
    cv2.line(img,(x,y - size),(x, y + size),color,5)


# c = (246, 305)
centro = c
h = 0 # horizontal
v = 1 # vertical

def deltas(ponteiro, centro):
    x = 0
    y = 1
    delta_x = ponteiro[x] - centro[x]
    delta_y = ponteiro[y] - centro[y]
    return delta_x, delta_y

def text_cv(bgr, pos, text, fontscale=1, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX    
    color=(255,255,255)    
    mensagem = "{}".format(text)
    cv2.putText(bgr, mensagem, pos, font, fontscale, color, thickness, cv2.LINE_AA) 
    

def horario_from_bgr(bgr, centro):
    x = 0
    y = 1
    cv2.circle(bgr, centro, 115, (255,0,0), -1)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    orange_min = np.array([3,50,50])
    orange_max = np.array([20, 255,255])

    mask = cv2.inRange(hsv, orange_min, orange_max)

    cm = center_of_mass(mask)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    delta = deltas(cm, centro)

    angulo = math.atan2(delta[y], delta[x])

    angulo+=(2.5*math.pi)

    angulo=angulo%(2*math.pi)

    graus = math.degrees(angulo)

    crosshair(mask_bgr, cm, 8, (0,0,255))

    crosshair(mask_bgr, centro, 200, (255,0,0))

    text_cv(mask_bgr, centro, str((graus/6)))

    return mask_bgr


### Fim da solucao



if __name__ == "__main__":

    # Inicializa a aquisição da webcam
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_FPS, 3)


    print("Se a janela com a imagem não aparecer em primeiro plano dê Alt-Tab")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            #print("Codigo de retorno FALSO - problema para capturar o frame")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            #sys.exit(0)

        # Our operations on the frame come here
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Vamos usar a função desenvolvida
        bgr = horario_from_bgr(frame, centro)

        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('Resultado', bgr)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


