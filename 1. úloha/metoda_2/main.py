#Daniel Cífka A23N0100P
import sympy as simp
import matplotlib.pyplot as plt
import numpy as np


#Funkce pro nahrazení proměných v rovnici a výpočet maxima
def calculation_max(eq1):
    
    return  eq1.subs([(x,0),(y,0)])

def generate_data(max, size):
    
    #Generování hodnot x, Y a maxima
    return np.random.uniform(0, 1,size=size), np.random.uniform(0, 1,size=size), np.random.uniform(0, 1,size=size) * max

#Výpočet hodnot pomoc zamítací metody
def calculation_by_rejection_method(eq1, xi, yi, zi, size):
    pts_x = []
    pts_y = []
    mean_x = []
    mean_y = []
    var_x = []
    var_y = []
    cov = []
    pts_n = []
    n = 0
    
    
    for i in range(size):
        if (xi[i] + yi[i]) < 1: #Hodnota nesmí ýt vetší než 1
            if zi[i] <= eq1.subs([(x,xi[i]),(y,yi[i])]): #Vypočtená hodnota z funkce nesmí být vetší než maximum
                n += 1
                pts_n.append(n)
                pts_x.append(xi[i]) 
                pts_y.append(yi[i])
                mean_x.append(np.mean(pts_x)) #Uložení a výpočet hodnoty průměru X
                mean_y.append(np.mean(pts_y)) #Uložení a výpočet hodnoty průměru Y
                var_x.append(np.var(pts_x)) #Uložení a výpočet hodnoty rozptylu X
                var_y.append(np.var(pts_y)) #Uložení a výpočet hodnoty rozptylu Y
                cov.append(np.cov(pts_x, pts_y)[0][1]) #Uložení a výpočet hodnoty covariance
                
    return pts_x, pts_y, mean_x, mean_y, var_x, var_y, cov, pts_n

def print_result(pts_x, pts_y, mean_x, mean_y, var_x, var_y, cov, pts_n):
    
    print("Teoretická střední hodnota pro hodnotu Ex je 0,25 a průměr z dat pro hodnotu x je {:.4f}".format(np.mean(pts_x)))
    print("Teoretická střední hodnota pro hodnotu Ey je 0,25 a průměr z dat pro hodnotu y je {:.4f}".format(np.mean(pts_y)))
    print("var matice teoreticky vypočtená:")
    print([[0.0375,-0.0125],[-0.0125,0.0375]])
    print("Var matice získaná z vygenerovaných dat:")
    print(np.cov(pts_x, pts_y))
    
    plt.figure(figsize=[25,25])
    plt.tight_layout()
    
    plt.subplot(2, 3, 1)
    plt.scatter(pts_x, pts_y, c="blue")
    plt.title("Prostor Omega")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(2, 3, 2)
    plt.scatter(pts_n, mean_x, c="green")
    plt.title("Vývoj Ex")
    plt.xlabel("n")
    plt.ylabel("Ex")
    plt.subplot(2, 3, 3)
    plt.scatter(pts_n, mean_y, c="red")
    plt.title("Vývoj Ey")
    plt.xlabel("n")
    plt.ylabel("Ey")
    plt.subplot(2, 3, 4)
    plt.scatter(pts_n, var_x, c="orange")
    plt.title("Vývoj Varx")
    plt.xlabel("n")
    plt.ylabel("Varx")
    plt.subplot(2, 3, 5)
    plt.scatter(pts_n, var_y, c="brown")
    plt.title("Vývoj Vary")
    plt.xlabel("n")
    plt.ylabel("Vary")
    plt.subplot(2, 3, 6)
    plt.scatter(pts_n,cov, c="yellow")
    plt.title("Vývoj cov(x,y)")
    plt.xlabel("n")
    plt.ylabel("Cov(x,y)")
    plt.show()
    plt.figure(figsize=[50,50])
    plt.tight_layout()
    
    plt.subplot(2, 3, 1)
    plt.scatter(pts_x, pts_y, c="blue")
    plt.title("Prostor Omega")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(2, 3, 2)
    plt.scatter(pts_n, mean_x, c="green")
    plt.title("Vývoj Ex")
    plt.xlabel("n")
    plt.ylabel("Ex")
    plt.subplot(2, 3, 3)
    plt.scatter(pts_n, mean_y, c="red")
    plt.title("Vývoj Ey")
    plt.xlabel("n")
    plt.ylabel("Ey")
    plt.subplot(2, 3, 4)
    plt.scatter(pts_n, var_x, c="orange")
    plt.title("Vývoj Varx")
    plt.xlabel("n")
    plt.ylabel("Varx")
    plt.subplot(2, 3, 5)
    plt.scatter(pts_n, var_y, c="brown")
    plt.title("Vývoj Vary")
    plt.xlabel("n")
    plt.ylabel("Vary")
    plt.subplot(2, 3, 6)
    plt.scatter(pts_n,cov, c="yellow")
    plt.title("Vývoj cov(x,y)")
    plt.xlabel("n")
    plt.ylabel("Cov(x,y)")
    plt.savefig("graph.pdf")


if __name__ == '__main__':
    #Definování symbolů a proměnný potřebný pro výpočet hodnot
    x = simp.Symbol("x")
    y = simp.Symbol("y")
    pts_x = []
    pts_y = []
    mean_x = []
    mean_y = []
    var_x = []
    var_y = []
    cov = []
    pts_n = []
    size = 5000 #Definice počtu prvků
    
    fun = (1-x-y) #Definice základní funkce
    
    eq1 = simp.integrate(fun, (y, 0, 1-x)) #Integrace vnitřní funkci
    eq2 = simp.integrate(eq1, (x, 0, 1)) #Integrace vnější funkce

    c = 1 / eq2 #Výpočet koeficientu C
    eq1 = c*(fun) #Tvorba funkce s vypočteným koeficientem
    xi, yi, zi = generate_data(max=calculation_max(eq1=eq1), size=size) #Generování potřebných hodnot [x, y, maximální hodnotu]
    pts_x, pts_y, mean_x, mean_y, var_x, var_y, cov, pts_n = calculation_by_rejection_method(eq1=eq1, xi=xi, yi=yi, zi=zi, size=size) #Zjistění hodnot spadající do přísluného rozsahu omega
    print_result(pts_x=pts_x, pts_y=pts_y, mean_x=mean_x, mean_y=mean_y, var_x=var_x, var_y=var_y, cov=cov, pts_n=pts_n) #Výpis zsštěných a teoretický hodnot a vykreslení grafů
