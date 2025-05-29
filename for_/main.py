from altair import sample
from matplotlib.scale import scale_factory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pybaselines import Baseline
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load

# Chart
class Chart:
    def __init__(self, data):
        self.data = data

    # draw 2D line chart
    def draw_2D_line_chart(self):        
        # configure the figure, title, xlabel, ylabel, etc.
        plt.figure(figsize=(10, 6))
        plt.title("Spectrogram")
        plt.xlabel("Raman Shift(cm⁻¹)")
        plt.ylabel("Intensity(a.u.)")
        plt.grid(linestyle="--", alpha=0.3)

        # draw line one by one
        for sample_label in self.data["sample_labels"]:
            X = self.data["input_features"]
            Y = self.data["inputs"].loc[sample_label]
            plt.plot(X, Y, label = sample_label)
        
        # add legend
        plt.legend(loc="upper right")
        plt.show()

    # draw heatmap
    def draw_heatmap(self):
        plt.figure(figsize = (20, 20))
        plt.title("Cosine Similarity Heatmap")
        sns.heatmap(
            self.data,
            cmap = "YlGnBu",
            vmin = 0,
            vmax = 1,
            annot=True,     
            fmt=".2f",     
            annot_kws={"size": 10, "color": "black"} 
        )
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
        sample_weights = {label : 1.0 for label in sample_labels}
        dataset = {
            "input_features" : input_features,
            "inputs" : inputs,
            "output_features" : output_features,
            "outputs" : outputs,
            "sample_labels" : sample_labels,
            "sample_weights" : sample_weights
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

        # abstract critical feature peaks' integration
        mask_C2C = ((dataset["input_features"] >= 1630) & (dataset["input_features"] <= 1665))  #C=C伸缩振动 
        mask_CH2 = ((dataset["input_features"] >= 1440) & (dataset["input_features"] <= 1460))  #CH2伸缩振动 
        mask_C_H = ((dataset["input_features"] >= 1290) & (dataset["input_features"] <= 1320))  #CH2扭曲振动，芥酸更强
        mask_CH2N = ((dataset["input_features"] >= 850) & (dataset["input_features"] <= 880))   #CH2面外摇摆震动 
        mask_C_C = ((dataset["input_features"] >= 1060) & (dataset["input_features"] <= 1100))  #C-C伸缩振动，长链芥酸更强  
        I_C2C = dataset["inputs"].loc[:, mask_C2C].sum(axis=1).to_frame(name = 'I_C2C')
        I_CH2 = dataset["inputs"].loc[:, mask_CH2].sum(axis=1).to_frame(name = 'I_CH2')
        I_C_H = dataset["inputs"].loc[:, mask_C_H].sum(axis=1).to_frame(name = 'I_C_H')
        I_CH2N = dataset["inputs"].loc[:, mask_CH2N].sum(axis=1).to_frame(name = 'I_CH2N')
        I_C_C = dataset["inputs"].loc[:, mask_C_C].sum(axis=1).to_frame(name = 'I_C_C')
        dataset["inputs"] = pd.concat([I_C2C, I_CH2, I_C_H, I_CH2N, I_C_C], axis = 1)
        dataset["input_features"] = pd.Index(['I_C2C', 'I_CH2', 'I_C_H', 'I_CH2N', 'I_C_C'])

        # standardization
        scaler_inputs = StandardScaler()
        dataset["inputs"] = scaler_inputs.fit_transform(dataset["inputs"])
        dump(scaler_inputs, 'scaler_inputs.joblib')
        dataset["inputs"] = pd.DataFrame(
            dataset["inputs"], 
            index = dataset["sample_labels"], 
            columns = dataset["input_features"]
        )
        scaler_outputs = StandardScaler()
        dataset["outputs"] = scaler_outputs.fit_transform(dataset["outputs"])
        dump(scaler_outputs, 'scaler_outputs.joblib')
        dataset["outputs"] = pd.DataFrame(
            dataset["outputs"], 
            index = dataset["sample_labels"], 
            columns = dataset["output_features"]
        )

        # calculate sample's weights
        X = pd.concat([dataset["inputs"], dataset["outputs"]], axis = 1)
        cos_similarity_matrix = cosine_similarity(X)
        sample_cos_similarities = cos_similarity_matrix.mean(axis = 0)
        scaler = MinMaxScaler()
        weights = scaler.fit_transform(sample_cos_similarities.reshape(-1, 1)).flatten() 
        sample_weights = {label : weight for label, weight in zip(dataset["sample_labels"], weights)}
        dataset["sample_weights"] = sample_weights

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
            "sample_labels" : sample_labels_trian,
            "sample_weights" : dataset["sample_weights"]
        }

        # package test set
        test_set = {
            "input_features" : dataset["input_features"],
            "inputs" : inputs_test,
            "output_features" : dataset["output_features"],
            "outputs" : outputs_test,
            "sample_labels" : sample_labels_test,
            "sample_weights" : dataset["sample_weights"]
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
            model = RandomForestRegressor()
            model.fit(
                inputs_train, 
                outputs_train, 
                sample_weight = [train_set["sample_weights"][label] for label in inputs_train.index.to_list()]
            )

            # predict
            outputs_predict = model.predict(inputs_val)

            # inverse standardization/normalization
            scaler_outputs = load('scaler_outputs.joblib')
            outputs_val = scaler_outputs.inverse_transform(outputs_val)
            outputs_predict = scaler_outputs.inverse_transform(outputs_predict)
            print("\n实际浓度: \n\n", outputs_val)
            print("\n预测浓度: \n\n", outputs_predict)

            # save predicted results
            olac_val.append(outputs_val[:, 0])
            olac_predict.append(outputs_predict[:, 0])
            erac_val.append(outputs_val[:, 1])
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
        model = RandomForestRegressor()
        model.fit(
            train_set["inputs"], 
            train_set["outputs"], 
            sample_weight = [train_set["sample_weights"][label] for label in train_set["inputs"].index.to_list()]
        )
        dump(model, 'model.joblib')

    # test
    def test(self):
        # get test set
        _, test_set = self.dataset_split_train_test()
        
        # predict
        model = load('model.joblib')
        outputs_predict = model.predict(test_set["inputs"])

        # inverse standardization/normalization
        scaler_outputs = load('scaler_outputs.joblib')
        test_set["outputs"] = scaler_outputs.inverse_transform(test_set["outputs"])
        outputs_predict = scaler_outputs.inverse_transform(outputs_predict)
        print("\n实际浓度: \n\n", test_set["outputs"])
        print("\n预测浓度: \n\n", outputs_predict)

        # evaluation
        print("\n------------------------------测试集评估得分: -----------------------------------------------")
        print("\n 油酸浓度评估: ")
        print("\n R² Score: ", r2_score(test_set["outputs"][:, 0], outputs_predict[:, 0]))
        print("\n RMSE: ", np.sqrt(mean_squared_error(test_set["outputs"][:, 0], outputs_predict[:, 0])))
        print("\n 芥酸浓度评估: ")
        print("\n R² Score: ", r2_score(test_set["outputs"][:, 1], outputs_predict[:, 1]))
        print("\n RMSE: ", np.sqrt(mean_squared_error(test_set["outputs"][:, 1], outputs_predict[:, 1])))        


if __name__ == "__main__":

    model = Model("dataset.xlsx")
    data = model.test()

    #chart = Chart(data)
    #chart.draw_heatmap()