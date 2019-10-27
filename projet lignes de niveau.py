import autograd as ag
from autograd import numpy as np
import matplotlib.pyplot as plt
f = lambda x,y : x**2+y**2
g = lambda x,y : 2*(np.exp(-x**2-y**2)-np.exp(-(x-1)**2-(y-1)**2))


def grad_f(f,x,y): #calcul du gradient en (x,y)
    a,b=float(x),float(y)
    gradient=ag.grad
    return np.array([gradient(f,0)(a,b),gradient(f,1)(a,b)])



def find_seed(f,c=0,eps=2**(-26)): #dichotomie pour calculer le point de départ
    
    if (f(0,0)-c)*(f(0,1)-c)>0:
        return None
    
    else :
        a=0
        b=1
        delta=b-a
        
        while delta > eps :
            d=(a+b)/2
            
            if (f(0,a)-c)*(f(0,d)-c)>0:
                a=d
            
            else :
                b=d
            
            delta=b-a
        
        return (a+b)/2
    

def norme(vect):#calcul de distance
    return np.sqrt(vect[0]**2+vect[1]**2)

def calcul_suivant(f,x,y,delta,direc=1):#calcul du point approché suivant de la ligne de niveau à partir de (x,y)
    gfx, gfy=grad_f(f,x,y)
    ortho = np.array([-gfy,gfx])
    ortho = ortho*delta / norme(ortho)*direc
    
    return x+ortho[0],y+ortho[1]
    
    

def simple_contour(f,c=0,delta=0.01):#renvoie la liste des points d'une ligne de niveau de valeur c pour le carré unité
    x=0
    y=find_seed(f,c)
    lx=[]
    ly=[]
    
    if y==None :
        return [],[]
    
    else :
        x_test,y_test=calcul_suivant(f,x,y,delta)
        if x_test>=0 and x_test<=1 and y_test>=0 and y_test<=1 :#permet de choisr le bon sens de la ligne de niveau (pour entrer dans le carré souhaité)
            direc = 1
        else :
            direc = -1
        i=0
        while x>=0 and x<=1 and y>=0 and y<=1:
            lx.append(x)
            ly.append(y)
            x,y=calcul_suivant(f,x,y,delta,direc)
            i+=1
        return lx,ly

def f_carre(f,x1,x2,y1,y2):# transformation de la fonction en une fonction définie sur le carré unité
    nouv_f = lambda x,y : f(x1+x*(x2-x1),y1+y*(y2-y1))
    return nouv_f

def f_symetrie(nouv_f,bord):#fait une rotation de la fonction pour pouvoir partir du bord gauche
    if bord == 'g' :
        return nouv_f
    if bord == 'b' :
        return lambda x,y : nouv_f(y,x)
    if bord == 'd':
        return lambda x,y : nouv_f(1-x,y)
    if bord == 'h':
        return lambda x,y : nouv_f(1-y,1-x)

def f_transformee(f,x1,x2,y1,y2,bord):# renvoie la fonction après contraction/déplacement/rotation
    f_inter=f_carre(f,x1,x2,y1,y2)
    f_finale=f_symetrie(f_inter,bord)
    return f_finale

def contour_carre(f,c,delta,x1,x2,y1,y2):# renvoie les lignes de niveau sur un carré quelconque à partir de chacun des bords
    X,Y=[],[]
    for bord in {'g','d','b','h'}:
        f_transfo=f_transformee(f,x1,x2,y1,y2,bord)
        lx,ly=simple_contour(f_transfo,c,delta)
        if lx == []:
            continue
        lx,ly=np.array(lx),np.array(ly)
        if bord == 'b':
            lx,ly=ly,lx

        elif bord == 'd':
            lx=1-lx
        
        elif bord =='h':
            lx,ly=1-ly,1-lx
        
        lx = lx*(x2-x1)+x1
        ly = ly*(y2-y1)+y1
        lx,ly=list(lx),list(ly)
        X.append(lx)
        Y.append(ly)
    return X,Y



def contour(f,c=0.0,xc=[0.0,1.0],yc=[0.0,1.0],delta=0.01):#renvoie les lignes de niveau pour une rectangle quelconque à partir de subdivision en carrés élémentaires
    n=len(xc)
    m=len(yc)
    X,Y=[],[]
    for j in range(m-1):#ligne
        for i in range (n-1):#colonne
            x,y=contour_carre(f,c,delta,xc[i],xc[i+1],yc[j],yc[j+1])
            X+=x
            Y+=y

    return X,Y


#affichage
for i in [-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]:

    X,Y= contour(g,c=i,xc=[-2,-1,0,1,2],yc=[-2,-1,0,1,2],delta=0.01)

    for x,y in zip(X,Y):
        plt.plot(x,y,"r")
    
    

plt.show()