#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

def load_mobilenet():
    """Não mude ou renomeie esta função
        Carrega o modelo e os parametros da MobileNet. Retorna a classe da rede.
    """
    proto = "./mobilenet_detection/MobileNetSSD_deploy.prototxt.txt" # descreve a arquitetura da rede
    model = "./mobilenet_detection/MobileNetSSD_deploy.caffemodel" # contém os pesos da rede em si
    net = cv2.dnn.readNetFromCaffe(proto, model)

    return net

# Carrega Rede
net = load_mobilenet()

# Classes da MobileNet
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

# Cor e confianca
CONFIDENCE = 0.7
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Funcao de deteccao
def detect(net, frame, CONFIDENCE, COLORS, CLASSES):
    """
        Recebe - uma imagem colorida BGR
        Devolve: objeto encontrado
    """
    image = frame.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    results = []
    
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence


        if confidence > CONFIDENCE:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            results.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY) ))

    # show the output image
    return image, results

def desenha_retang(img, posicao, cor_rgb):
    minx = posicao[0]
    miny = posicao[1]
    maxx = posicao[2]
    maxy = posicao[3]
    cv2.rectangle(img, (minx, miny), (maxx, maxy), cor_rgb)
    
def acha_dog_cat(img, resultados):
    animais = {}
    animais['dog'] = [0,0,0,0]
    animais['cat'] = [0,0,0,0]
    
    for resultado in resultados:
        if resultado[0] == 'cat':
            min_X = resultado[2][0]
            min_Y = resultado[2][1]
            max_X = resultado[3][0]
            max_Y = resultado[3][1]
            posicao = [min_X, min_Y, max_X, max_Y]
            animais['cat'] = posicao

        elif resultado[0] == 'dog':
            min_X = resultado[2][0]
            min_Y = resultado[2][1]
            max_X = resultado[3][0]
            max_Y = resultado[3][1]
            posicao = [min_X, min_Y, max_X, max_Y]
            animais['dog'] = posicao
    
    return animais

def acha_caixas(img):
    hsv_g = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    
    # Acha caixa verde
    hsv1_g = np.array([50 ,150 ,80 ])
    hsv2_g = np.array([80, 255, 255])
    mask_g  = cv2.inRange(hsv_g, hsv1_g, hsv2_g)
    # Calcula coordenadas da caixa verde
    positions_g = np.where(mask_g == 255)
    min_Y_g = np.min(positions_g[0])
    min_X_g = np.min(positions_g[1])
    max_Y_g = np.max(positions_g[0])
    max_X_g = np.max(positions_g[1])
    
    # Acha caixa azul
    hsv1_b = np.array([100, 50,50])
    hsv2_b = np.array([130, 255, 255])
    mask_b  = cv2.inRange(hsv_b, hsv1_b, hsv2_b)
    # Calcula coordenadas da caixa azul
    positions_b = np.where(mask_b == 255)
    min_Y_b = np.min(positions_b[0])
    min_X_b = np.min(positions_b[1])
    max_Y_b = np.max(positions_b[0])
    max_X_b = np.max(positions_b[1])
    
    caixa_g = (min_X_g, min_Y_g, max_X_g, max_Y_g)
    caixa_b = (min_X_b, min_Y_b, max_X_b, max_Y_b)
    
    return caixa_g, caixa_b
    
font = cv2.FONT_HERSHEY_SIMPLEX
def texto(img, a, p, color, font=font, width=2, size=1 ):
    cv2.putText(img, str(a), p, font,size,color,width,cv2.LINE_AA)

def verifica_dentro(animais, caixa_g, caixa_b, frame):
    dog = animais['dog']
    centroX_dog = ((dog[0] + dog[2])/2) 
    centroY_dog = ((dog[1] + dog[3])/2) 
    margemX_dog = centroX_dog - dog[0]
    margemY_dog = centroY_dog - dog[1]
    
    cat = animais['cat']
    centroX_cat = ((cat[0] + cat[2])/2) 
    centroY_cat = ((cat[1] + cat[3])/2) 
    margemX_cat = centroX_cat - cat[0]
    margemY_cat = centroY_cat - cat[1]
    
    # Checar se o dog esta dentro da caixa verde ou amarela
    # Caixa verde
    if (caixa_g[2] - margemX_dog > centroX_dog > caixa_g[0] + margemX_dog) and (caixa_g[3] - margemY_dog > centroY_dog > caixa_g[1] + margemY_dog):
        desenha_retang(frame, dog, [0,255,255])
        desenha_retang(frame, caixa_g, [0,255,255])
        x = dog[0]
        y = dog[1]
        msg = 'ESTA CONTIDO'
        texto(frame, msg, (x,y), color = (0,255,255))
    # Caixa azul
    if (caixa_b[2] - margemX_dog > centroX_dog > caixa_b[0] + margemX_dog) and (caixa_b[3] - margemY_dog > centroY_dog > caixa_b[1] + margemY_dog):
        desenha_retang(frame, dog, [0,255,255])
        desenha_retang(frame, caixa_b, [0,255,255])
        x = dog[0]
        y = dog[1]
        msg = 'ESTA CONTIDO'
        texto(frame, msg, (x,y), color = (0,255,255))
        
    # Checar se o cat esta dentro da caixa verde ou amarela
    # Caixa verde
    if (caixa_g[2] - margemX_cat > centroX_cat > caixa_g[0] + margemX_cat) and (caixa_g[3]- margemY_cat > centroY_cat > caixa_g[1] + margemY_cat):
        desenha_retang(frame, cat, [0,255,255])
        desenha_retang(frame, caixa_g, [0,255,255])
        x = cat[0]
        y = cat[1]
        msg = 'ESTA CONTIDO'
        texto(frame, msg, (x,y), color = (0,255,255))
    # Caixa azul
    if (caixa_b[2] - margemX_cat > centroX_cat > caixa_b[0] + margemX_cat) and (caixa_b[3] - margemY_cat > centroY_cat > caixa_b[1] +  margemY_cat):
        desenha_retang(frame, cat, [0,255,255])
        desenha_retang(frame, caixa_b, [0,255,255])
        x = cat[0]
        y = cat[1]
        msg = 'ESTA CONTIDO'
        texto(frame, msg, (x,y), color = (0,255,255))
        

# Arquivos necessários
# Baixe e salve na mesma pasta que este arquivo
# https://github.com/Insper/robot20/raw/master/media/animais_caixas.mp4
video = "animais_caixas.mp4"


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
        saida, resultados = detect(net, frame, CONFIDENCE, COLORS, CLASSES)

        # Our operations on the frame come here
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        animais = acha_dog_cat(frame, resultados)
        caixa_g, caixa_b = acha_caixas(frame)
        
        verifica_dentro(animais, caixa_g, caixa_b, frame)
        
        print(animais)
            
        
        
        


        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


