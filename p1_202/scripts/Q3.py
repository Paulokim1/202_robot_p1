#! /usr/bin/env python3
# -*- coding:utf-8 -*-

# Sugerimos rodar com:
# roslaunch turtlebot3_gazebo  turtlebot3_empty_world.launch 


from __future__ import print_function, division
import rospy
import numpy as np
import cv2
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Vector3
import math
import time
from tf import transformations


zero = Twist(Vector3(0,0,0),Vector3(0,0,0))

x = None
y = None

contador = 0
pula = 50


radianos = None

def recebe_odometria(data):
    global x
    global y
    global contador
    global radianos

    x = data.pose.pose.position.x
    y = data.pose.pose.position.y

    quat = data.pose.pose.orientation
    lista = [quat.x, quat.y, quat.z, quat.w]
    angulos = np.degrees(transformations.euler_from_quaternion(lista))  
    radianos =   transformations.euler_from_quaternion(lista)  

    if contador % pula == 0:
        print("Posicao (x,y)  ({:.2f} , {:.2f}) + angulo {:.2f}".format(x, y,angulos[2]))
    contador = contador + 1

def girar(ang, publisher):
    inicial = radianos[2]
    if inicial < 0: 
        inicial = inicial + 2*math.pi
    alvo = (inicial + ang)%(2*math.pi)
    atual = inicial
    print("inicial, alvo", inicial, alvo)

    w = 0.15

    vel = Twist(Vector3(0,0,0), Vector3(0,0,w))

    while(atual < alvo):
        publisher.publish(vel)
        rospy.sleep(0.05)
        atual = radianos[2]
        if atual < 0:
            atual = atual + 2*math.pi
    
    publisher.publish(zero)







def segmento(publisher):
    # anda 1.2 m
    v = 0.2
    delta_s = 1.2 
    t = delta_s/v

    vel = Twist(Vector3(v,0,0),Vector3(0,0,0))
    publisher.publish(vel)
    rospy.sleep(t)
    publisher.publish(zero)
    rospy.sleep(0.5)

def faz_poligono(n, publisher): 
    if n >= 3: 
        i = n 
        ang = 2*math.pi/n

        while (i > 0):
            segmento(publisher)
            girar(ang, publisher)
            i=i-1








if __name__=="__main__":

    rospy.init_node("q3")

    




    velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 3 )

    ref_odometria = rospy.Subscriber("/odom", Odometry, recebe_odometria)


    rospy.sleep(1.0) # contorna bugs de timing    

    n = 5

    while not rospy.is_shutdown():
        rospy.sleep(0.5) 
        faz_poligono(5, velocidade_saida)
