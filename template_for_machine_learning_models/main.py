import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pybaselines import Baseline
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

# Chart
class Chart:
    def __init__(self, dataset):
        self.dataset = dataset

    # draw 2D line chart
    def draw_2D_line_chart(self):        
        # configure the figure, title, xlabel, ylabel, etc.
        plt.figure(figsize=(10, 6))
        plt.title("Spectrogram")
        plt.xlabel("Raman Shift(cm⁻¹)")
        plt.ylabel("Intensity(a.u.)")
        plt.grid(linestyle="--", alpha=0.3)

        # draw line one by one
        for sample_label in self.dataset["sample_labels"]:
            X = self.dataset["input_features"]
            Y = self.dataset["inputs"].loc[sample_label]
            plt.plot(X, Y, label = sample_label)
        
        # add legend
        plt.legend(loc="upper right")
        plt.show()

# Model
class Model:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    
    # 1. data preprocess and feature engineering
    def data_preprocess_and_feature_engineering(self):
        # get and package dataset
        dataset = pd.read_excel(self.dataset_path, header=0, index_col="sample")
        dataset.columns = list(dataset.columns[:2]) + list(dataset.columns[2:].astype(float))
        input_features = dataset.columns[2:]
        inputs = dataset[input_features]
        output_features = dataset.columns[0:2]
        outputs = dataset[output_features]
        sample_labels = dataset.index.tolist()
        dataset = {
            "input_features" : input_features,
            "inputs" : inputs,
            "output_features" : output_features,
            "outputs" : outputs,
            "sample_labels" : sample_labels
        }
    
        # dataset before data preprocess and feature engineering
        print("\n-----------------------dataset before data preprocess and feature engineering---------------------------\n")
        print(dataset)

        # baselines correction
        corrected_spectra = []
        baseline_fitter = Baseline()
        for sample_label in dataset["sample_labels"]:
            spectrum = dataset["inputs"].loc[sample_label].values.reshape(-1, 1)  # 转换为列向量
            baselines, params = baseline_fitter.airpls(spectrum, 1e5)  # 调整参数 lam 和 p
            corrected_spectrum = spectrum.flatten() - baselines.flatten()
            corrected_spectra.append(corrected_spectrum)
        dataset["inputs"] = pd.DataFrame(
            corrected_spectra, 
            index = dataset["sample_labels"], 
            columns = dataset["input_features"]
        )

        # normalization
        scaler = StandardScaler()
        dataset["inputs"] = scaler.fit_transform(dataset["inputs"].values)
        dataset["inputs"] = pd.DataFrame(
            dataset["inputs"], 
            index = dataset["sample_labels"], 
            columns = dataset["input_features"]
        )

        # abstract critical feature peak
        mask = (
            ((dataset["input_features"] >= 1630) & (dataset["input_features"] <= 1665)) | 
            ((dataset["input_features"] >= 1440) & (dataset["input_features"] <= 1460)) |
            ((dataset["input_features"] >= 1290) & (dataset["input_features"] <= 1320)) | 
            ((dataset["input_features"] >= 850) & (dataset["input_features"] <= 880)) |
            ((dataset["input_features"] >= 1060) & (dataset["input_features"] <= 1100)) 
        )
        dataset["input_features"] = dataset["input_features"][mask]
        dataset["inputs"]  = dataset["inputs"].loc[:, mask]

        # dataset after data preprocess and feature engineering
        print("\n-----------------------dataset after data preprocess and feature engineering---------------------------\n")
        print(dataset)

        return dataset

    # 2. modeling
    def modeling(self):
        print("modeling....")


    # 3. split dataset to train set and test set
    def dataset_split_train_test(self):
        # get data after data preprocess and feature engineering
        dataset  = self.data_preprocess_and_feature_engineering()

        # dataset before spliting
        print("\n-----------------------dataset before spliting------------------------------------------------------\n")
        print(dataset)

        # split dataset to train set and test set
        inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
            dataset["inputs"], dataset["outputs"], 
            test_size=0.2, 
            random_state=42,
        )
        sample_labels_trian = inputs_train.index
        sample_labels_test = inputs_test.index

        # package train set
        train_set = {
            "input_features" : dataset["input_features"],
            "inputs" : inputs_train,
            "output_features" : dataset["output_features"],
            "outputs" : outputs_train,
            "sample_labels" : sample_labels_trian
        }

        # package test set
        test_set = {
            "input_features" : dataset["input_features"],
            "inputs" : inputs_test,
            "output_features" : dataset["output_features"],
            "outputs" : outputs_test,
            "sample_labels" : sample_labels_test
        }

        # dataset after spliting
        print("\n-----------------------train set--------------------------------------------------------------------\n")
        print(train_set)
        print("\n-----------------------test set---------------------------------------------------------------------\n")
        print(test_set)

        return train_set, test_set

    # 4. optimization and cross validation
    def optimization_and_validation(self):
        # get train set
        train_set, _ = self.dataset_split_train_test()

        # predicted results buffer
        olac_val = []
        olac_predict = []
        erac_val = []
        erac_predict = []
        
        # cross validation
        loo = LeaveOneOut()
        num = 0
        for train_index, val_index in loo.split(train_set["inputs"]):
            num = num + 1
            # split train set and validation set
            print(f"\n-------------------------第{num}次验证----------------------------------------------------------")
            print(f"\n train indices: {train_index}, test indices: {val_index}")
            inputs_train, inputs_val = train_set["inputs"].iloc[train_index], train_set["inputs"].iloc[val_index]
            outputs_train, outputs_val = train_set["outputs"].iloc[train_index], train_set["outputs"].iloc[val_index]

            # optimization
            model = MultiOutputRegressor(PLSRegression(n_components = 4))
            model.fit(inputs_train, outputs_train)

            # predict
            outputs_predict = model.predict(inputs_val)
            print("\n实际浓度: \n\n", outputs_val)
            print("\n预测浓度: \n\n", outputs_predict)

            # save predicted results
            olac_val.append(outputs_val["olac"])
            olac_predict.append(outputs_predict[:, 0])
            erac_val.append(outputs_val["erac"])
            erac_predict.append(outputs_predict[:, 1])  

        # evaluation   
        print("\n------------------------------留一法交叉验证评估得分: ---------------------------------------")
        print("\n 油酸浓度评估: ")
        print("\n R² Score: ", r2_score(olac_val, olac_predict))
        print("\n RMSE: ", np.sqrt(mean_squared_error(olac_val, olac_predict)))
        print("\n 芥酸浓度评估: ")
        print("\n R² Score: ", r2_score(erac_val, erac_predict))
        print("\n RMSE: ", np.sqrt(mean_squared_error(erac_val, erac_predict)))

        # optimization and save model
        model = MultiOutputRegressor(PLSRegression(n_components = 4))
        model.fit(train_set["inputs"], train_set["outputs"])
        dump(model, 'model.joblib')

    # test
    def test(self):
        # get test set
        _, test_set = self.dataset_split_train_test()
        
        # predict
        model = load('model.joblib')
        outputs_predict = model.predict(test_set["inputs"])
        print("\n实际浓度: \n\n", test_set["outputs"])
        print("\n预测浓度: \n\n", outputs_predict)

        # evaluation
        print("\n------------------------------测试集评估得分: -----------------------------------------------")
        print("\n 油酸浓度评估: ")
        print("\n R² Score: ", r2_score(test_set["outputs"]["olac"], outputs_predict[:, 0]))
        print("\n RMSE: ", np.sqrt(mean_squared_error(test_set["outputs"]["olac"], outputs_predict[:, 0])))
        print("\n 芥酸浓度评估: ")
        print("\n R² Score: ", r2_score(test_set["outputs"]["erac"], outputs_predict[:, 1]))
        print("\n RMSE: ", np.sqrt(mean_squared_error(test_set["outputs"]["erac"], outputs_predict[:, 1])))        


if __name__ == "__main__":

    model = Model("菜籽油_油酸_芥酸_NY6000_光谱数据_v3.0/dataset.xlsx")
    dataset = model.data_preprocess_and_feature_engineering()

    chart = Chart(dataset)
    chart.draw_2D_line_chart()
