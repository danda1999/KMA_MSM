#Jméno a příjmení: Daniel Cífka
#Studentské číslo: A23N0100P
#Předměk: KMA/SU úloha 2

#Importované knihovny
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import chi2

#Kostanty programu
FLOWER_NAME = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
SPECIFIC_PARAMETR = ["SepalLengthCm", "SepalWidthCm"]
ALPHA = 0.05
DF = 2

#Metoda pro načtení csv souboru s daty
def load_data(datase_name: str):
    
    return pd.read_csv(datase_name)

#Rozdělení načtených dat do skupin podle jejich jména
def load_specific_flower(data):
    
    return data.loc[data['Species'] == FLOWER_NAME[0]], data.loc[data['Species'] == FLOWER_NAME[1]], data.loc[data['Species'] == FLOWER_NAME[2]]

#Vykreslení a uložení grafu s načtenými hodnotai
def plot_of_dataset(Iris_sotasa, Iris_versicolor, Iris_virginica):
    
    ax = Iris_sotasa.plot.scatter(x=SPECIFIC_PARAMETR[0], y=SPECIFIC_PARAMETR[1], color="blue")
    Iris_versicolor.plot.scatter(x=SPECIFIC_PARAMETR[0], y=SPECIFIC_PARAMETR[1], ax=ax, color="red")
    Iris_virginica.plot.scatter(x=SPECIFIC_PARAMETR[0], y=SPECIFIC_PARAMETR[1], ax=ax, color="green")
    plt.savefig("dataset_graph.pdf")
    plt.show()

#Seznam vybraných načtených hodnot a uložení do pole
def list_of_parametr(data):
    
    x1 = data[SPECIFIC_PARAMETR[0]].values.tolist()
    x2 = data[SPECIFIC_PARAMETR[1]].values.tolist()
    return [x1, x2]

#Výpočet průměru z hodnot
def calculate_mean(data1, data2, data3):
    
    return np.array([np.mean(data1[0]), np.mean(data1[1])]), np.array([np.mean(data2[0]), np.mean(data2[1])]), np.array([np.mean(data3[0]), np.mean(data3[1])])

#Výpočet kovarianční matice
def calculate_cov(data1, data2, data3):
    
    return np.cov(data1[0], data1[1]), np.cov(data2[0], data2[1]), np.cov(data3[0], data3[1])

#@ykreslení transformovaných dat a kružnice do grafu
def plot_transformation(result, index):
    plt.subplot(1, 3, index)
    plt.scatter(x=result[0], y=result[1], color="blue")
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    radius = np.sqrt(chi2.ppf(1 - ALPHA, DF))
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)

    plt.plot(a, b, color="red")
    if index == 1:
        
        plt.title("Iris_sotasa")
        plt.ylabel(SPECIFIC_PARAMETR[1])
    elif index == 2:
        
        plt.title("Iris_versicolor")
    elif index == 3:
        
        plt.title("Iris_virginica")
    
    plt.xlabel(SPECIFIC_PARAMETR[0])
    
#Výpočet transformace původních dat do kružnice
def transformation_data(data, cov, mean, index):
    
    part_1 = np.linalg.pinv(sqrtm(cov))
    part_2 = ((data[0][:] - mean[0]), (data[1][:] - mean[1]))
    result = np.matmul(part_1, part_2)
    plot_transformation(result=result, index=index)
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    radius = np.sqrt(chi2.ppf(1 - ALPHA, DF))
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)
    result = [a, b]
    return result
#Výpočet opačné transformace na datech získaných z transformace
def transformation_data_back(data, cov, mean):
    
    help = np.matmul(sqrtm(cov), data)
    result = [(help[0][:] + mean[0]), (help[1][:] + mean[1])]
    return result

#Vykreslení elipsy ze zpetně transformovaných do původních dat 
def new_data_plot(data_old, data_new, index):
    if index == 1:
        plt.scatter(data_old[0], data_old[1], color='blue')
        plt.scatter(data_new[0][:], data_new[1][:], color='red')
    elif index == 2:
        plt.scatter(data_old[0], data_old[1], color='green')
        plt.scatter(data_new[0][:], data_new[1][:], color='brown')
    elif index == 3:
        plt.scatter(data_old[0], data_old[1], color='orange')
        plt.scatter(data_new[0][:], data_new[1][:], color='blue')

if __name__ == '__main__':
    
    data = load_data("Iris.csv")
    Iris_sotasa, Iris_versicolor, Iris_virginica = load_specific_flower(data=data)
    plot_of_dataset(Iris_sotasa=Iris_sotasa, Iris_versicolor=Iris_versicolor, Iris_virginica=Iris_virginica)
    Iris_sotasa_data = list_of_parametr(Iris_sotasa)
    Iris_versicolor_data = list_of_parametr(Iris_versicolor)
    Iris_virginica_data = list_of_parametr(Iris_virginica)
    Iris_sotasa_mean, Iris_versicolor_mean, Iris_virginica_mean = calculate_mean(Iris_sotasa_data, Iris_versicolor_data, Iris_virginica_data)
    Iris_sotasa_cov, Iris_versicolor_cov, Iris_virginica_cov = calculate_cov(Iris_sotasa_data, Iris_versicolor_data, Iris_virginica_data)
    Iris_sotasa_dataC = transformation_data(Iris_sotasa_data, Iris_sotasa_cov, Iris_sotasa_mean, 1)
    Iris_versicolor_dataC = transformation_data(Iris_versicolor_data, Iris_versicolor_cov, Iris_versicolor_mean, 2)
    Iris_virginica_dataC = transformation_data(Iris_virginica_data, Iris_virginica_cov, Iris_virginica_mean, 3)
    plt.savefig("Transformation_graph.pdf")
    plt.show()
    trb = transformation_data_back(Iris_sotasa_dataC, Iris_sotasa_cov, Iris_sotasa_mean)
    trb_1 = transformation_data_back(Iris_versicolor_dataC, Iris_versicolor_cov, Iris_versicolor_mean)
    trb_2 = transformation_data_back(Iris_virginica_dataC, Iris_virginica_cov, Iris_virginica_mean)
    new_data_plot(Iris_sotasa_data, trb, 1)
    new_data_plot(Iris_versicolor_data, trb_1, 2)
    new_data_plot(Iris_virginica_data, trb_2, 3)
    """plt.scatter(Iris_versicolor_data[0], Iris_versicolor_data[1], color='green')
    plt.scatter(trb_1[0][:], trb_1[1][:], color='brown')
    plt.scatter(Iris_virginica_data[0], Iris_virginica_data[1], color='orange')
    plt.scatter(trb_2[0][:], trb_2[1][:], color='blue')"""
    plt.xlabel("SepalLengthCm")
    plt.ylabel("SepalWidthCm")
    plt.savefig("Back_transformation_graph.pdf")
    plt.show()
    