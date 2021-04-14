#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np
from math import *

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

# Arquivos necessários
# Baixe e salve na mesma pasta que este arquivo
# https://github.com/Insper/robot20/raw/master/media/relogio.mp4
video = "relogio.mp4"

centro = (320,240)

def crosshair(img, point, size, color):

    x,y = point
    cv2.line(img,(x - size,y),(x + size,y),color,2)
    cv2.line(img,(x,y - size),(x, y + size),color,2)
    
def desenha_linha(img, pt1, pt2, cor, thick):
    cv2.line(img, pt1, pt2, cor, thick)
    

def segmenta_hora(bgr):
    hsv = cv2.cvtColor(bgr.copy(),cv2.COLOR_BGR2HSV)
    
    hsv1 = np.array([3,50,50])
    hsv2 = np.array([20,255,255])
    mask  = cv2.inRange(hsv, hsv1, hsv2)
    return mask

def segmenta_minuto(bgr):
    hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    
    hsv1 = np.array([110,50,50])
    hsv2 = np.array([160,255,255])
    mask  = cv2.inRange(hsv, hsv1, hsv2)
    return mask

def calcula_centro(mask):
    M = cv2.moments(mask)
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except:
        cX = 0
        cY = 0

    return (int(cX), int(cY))

def calcula_horario(centro, centro_horas, centro_minutos):
#     m_horas = (centro_horas[1] - centro[1])/(centro_horas[0] - centro[0])
#     teta_horas = atan(m_horas)
    
#     m_minutos = (centro_minutos[1] - centro[1])/(centro_minutos[0] - centro[0])
#     teta_minutos = atan(m_minutos
    
#     horas = int((12*teta_horas)/2*pi)
#     minutos = int((60*teta_minutos)/2*pi)

    m_horas = (centro_horas[1] - centro[1])/(centro_horas[0] - centro[0])
    ang_horas = atan(m_horas) 
    
    m_minutos = (centro_minutos[1] - centro[1])/(centro_minutos[0] - centro[0])
    ang_minutos
    
    # Verifica horas
    # Faixa das 3 horas
    if 12*pi/6 < ang_horas < pi/6:
        horas = 3
    
    return horas, minutos
    


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


        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


