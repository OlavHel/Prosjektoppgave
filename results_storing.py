import pandas as pd
import pickle

COMPLETE_FILE_NAME = "all_results"
RHOS = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

def initiate_data_storage(number_of_points,rhos):
    p_names = ["jeffrey", "arcsine", "uniform", "PC10-4"]
    est_names = ["E", "M", "FI2", "KL2","MAP"]
    eval_names = ["FIM", "MSE", "MAE", "KL"]

    total = [p_names[i // 5] + est_names[i % 5] for i in range(4 * 5)]
    total.append("Sample corr")
    total.append("Sample corr, var=1")
    total.append("Sample corr, var=1, trunc")
    total.append("MLE")

    data_dict = {}
    for rho in rhos:
        data_dict[rho] = pd.DataFrame(index=total,columns=eval_names)

    return data_dict


def load_data(number_of_points):
    return pickle.load(open(COMPLETE_FILE_NAME+str(number_of_points)+".p","rb"))

def add_data_point(data_frame,data_point,loss_name,est_name,rho):
    if est_name not in data_frame[rho].index.values:
        data_frame[rho]=data_frame[rho].append(pd.Series(name=est_name))
    data_frame[rho].loc[est_name][loss_name] = data_point
    return data_frame

def save_data(number_of_points,d_array,loss_names,est_names,rho):
    try:
        data = load_data(number_of_points=number_of_points)
    except:
        data = initiate_data_storage(number_of_points,RHOS)

    print(d_array)
    print(loss_names)
    print(est_names)
    for i in range(len(loss_names)):
        for j in range(len(est_names)):
            add_data_point(data,d_array[j,i],loss_names[i],est_names[j],rho)

    pickle.dump(data, open(COMPLETE_FILE_NAME+str(number_of_points)+".p","wb"))

    print("Successfully stored the data. A preview of the results are:")
    print(load_data(number_of_points))


if __name__ == "__main__":
    data = load_data(3)
#    for rho in data:
#        data[rho] = data[rho].drop(labels=["Sample corr var=1","Trunc sample corr"],axis=0)
    print(data)
#    pickle.dump(data, open(COMPLETE_FILE_NAME+str(10)+".p","wb"))