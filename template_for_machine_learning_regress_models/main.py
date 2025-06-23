from altair import sample
from matplotlib.scale import scale_factory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pybaselines import Baseline
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import minimize, least_squares
from joblib import dump, load
import pickle
from pathlib import Path

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
        #for sample_label in self.data["sample_labels"]:
        for sample_label in range(1, 5000):
            X = self.data["output_features"]
            Y = self.data["outputs"].loc[sample_label]
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

# Model and Optimizer
class ModelAndOptimizer:
    def __init__(self):
        ""
    
    def predict(self, inputs_val):
        ""
        
    def fit(self, inputs_train, outputs_train):
        ""
    
    def save(self, filepath):
        ""

    def load(cls, filepath):
        ""

# Modeling and Optimization
class ModelingAndOptimization:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    
    # 1. data preprocess and feature engineering
    def data_preprocess_and_feature_engineering(self):
        # get and package dataset
        dataset = pd.read_csv(self.dataset_path, header=0)   # pd.read_csv(dataset_path, header, names, index_col, dtype, skiprows)
        dataset.columns = list(dataset.columns[:5]) + list(dataset.columns[5:].astype(float))
        #dataset= dataset[dataset["method"] == "cAg@785"]  #  cAg@785
        input_features = dataset.columns[:5]
        inputs = dataset[input_features].astype(float)
        output_features = dataset.columns[5:]
        outputs = dataset[output_features].astype(float)
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
            spectrum = dataset["outputs"].loc[sample_label].values.reshape(-1, 1)  # 转换为列向量
            baselines, params = baseline_fitter.airpls(spectrum, 1e5)  # 调整参数 lam 和 p
            corrected_spectrum = spectrum.flatten() - baselines.flatten()
            corrected_spectra.append(corrected_spectrum)
        dataset["outputs"] = pd.DataFrame(
            corrected_spectra, 
            index = dataset["sample_labels"], 
            columns = dataset["output_features"]
        )

        # abstract critical feature peaks' integration
        mask_1 = ((dataset["output_features"] >= 20) & (dataset["output_features"] <= 110))     # 特征峰1
        mask_2 = ((dataset["output_features"] >= 200) & (dataset["output_features"] <= 300))    # 特征峰2
        mask_3 = ((dataset["output_features"] >= 310) & (dataset["output_features"] <= 450))    # 特征峰3
        mask_4 = ((dataset["output_features"] >= 450) & (dataset["output_features"] <= 530))    # 特征峰4
        mask_5 = ((dataset["output_features"] >= 590) & (dataset["output_features"] <= 610))    # 特征峰5
        mask_6 = ((dataset["output_features"] >= 700) & (dataset["output_features"] <= 730))    # 特征峰6
        I_1 = dataset["outputs"].loc[:, mask_1].sum(axis=1).to_frame(name = 'I_1')
        I_2 = dataset["outputs"].loc[:, mask_2].sum(axis=1).to_frame(name = 'I_2')
        I_3 = dataset["outputs"].loc[:, mask_3].sum(axis=1).to_frame(name = 'I_3')
        I_4 = dataset["outputs"].loc[:, mask_4].sum(axis=1).to_frame(name = 'I_4')
        I_5 = dataset["outputs"].loc[:, mask_5].sum(axis=1).to_frame(name = 'I_5')
        I_6 = dataset["outputs"].loc[:, mask_6].sum(axis=1).to_frame(name = 'I_6')
        dataset["outputs"] = pd.concat([I_1, I_2, I_3, I_4, I_5, I_6], axis = 1)
        dataset["output_features"] = ['I_1', 'I_2', 'I_3', 'I_4', 'I_5', 'I_6']

        # standardization
        scaler_inputs = MinMaxScaler()
        dataset["inputs"] = scaler_inputs.fit_transform(dataset["inputs"].values.reshape(-1, 1))
        dump(scaler_inputs, 'scaler_inputs.joblib')
        dataset["inputs"] = pd.DataFrame(
            dataset["inputs"], 
            index = dataset["sample_labels"], 
            columns = dataset["input_features"]
        )
        scaler_outputs = MinMaxScaler()
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
            
    # 2. split dataset to train set and test set
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

    # 3. cross validation
    def cross_validation(self):
        # get train set
        train_set, _ = self.dataset_split_train_test()
        
        # val results buffer
        true_values, pred_values = [], []

        # cross validation
        loo = LeaveOneOut()
        num = 0
        for train_index, val_index in loo.split(train_set["inputs"]):
            num = num + 1
            # split train set and validation set
            print(f"\n--------------------------------Validation No.{num}--------------------------------------------------")
            print(f"\n train indices: {train_index}, test indices: {val_index}")
            inputs_train, inputs_val = train_set["inputs"].iloc[train_index], train_set["inputs"].iloc[val_index]
            outputs_train, outputs_val = train_set["outputs"].iloc[train_index], train_set["outputs"].iloc[val_index]

            # modeling and optimization
            model = PLSRegression()
            model.fit(inputs_train, outputs_train)
            
            # predict
            inputs_pred = model.predict(outputs_val)
            
            # inverse standardization/normalization
            scaler_inputs = load('scaler_inputs.joblib')
            inputs_val = scaler_inputs.inverse_transform(inputs_val)
            inputs_pred = scaler_inputs.inverse_transform(inputs_pred)
            
            # save predicted results
            print(f"True: {inputs_val}, Pred: {inputs_pred}")
            true_values.append(inputs_val.to_numpy().flatten())
            pred_values.append(inputs_pred.flatten())


        # evaluation
        if len(true_values) > 0:
            # R² and RMSE
            print(f"\n-------------------------Cross Validation Results----------------------------------------------------------")
            print(f"R²: {r2_score(true_values, pred_values):.4f}")
            print(f"RMSE: {np.sqrt(mean_squared_error(true_values, pred_values)):.4f}")
            
            # visualization
            plt.scatter(true_values, pred_values)
            plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
            plt.xlabel('True Value')
            plt.ylabel('Predicted Value')
            plt.title('Cross Validation Results')
            plt.show()
        else:
            print("There are no valid prediction results for evaluation")

        # Fully modeling and optimization
        model = ModelAndOptimizer()
        model.fit(train_set["inputs"], train_set["outputs"])
        dump(model, 'model.joblib')
        print("The model has been saved as model.joblib")
        
    # 4. test
    def test(self):
        # get test set
        _, test_set = self.dataset_split_train_test()
        
        # load model
        model = load('model_pls.joblib')
        
        # predict
        inputs_test = test_set["inputs"].values  
        outputs_test = test_set["outputs"].values  
        outputs_pred = model.predict(inputs_test)
        
        # evaluation
        print("\n-----------------------------------Test Results------------------------------------------------------")
        evaluation_results = []
        for i in range(outputs_test.shape[1]):
            r2 = r2_score(outputs_test[:, i], outputs_pred[:, i])
            rmse = np.sqrt(mean_squared_error(outputs_test[:, i], outputs_pred[:, i]))
            evaluation_results.append((r2, rmse))
            print(f"\output {i+1}'s evaluation:")
            print(f"R² Score: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
        
        # visualization
        plt.figure(figsize=(12, 8))
        n_components = outputs_test.shape[1]
        cols = min(3, n_components)
        rows = (n_components - 1) // cols + 1
        for i in range(n_components):
            plt.subplot(rows, cols, i+1)
            plt.scatter(outputs_test[:, i], outputs_pred[:, i], alpha=0.6, label='Sample')
            plt.plot([outputs_test[:, i].min(), outputs_test[:, i].max()], 
                    [outputs_test[:, i].min(), outputs_test[:, i].max()], 
                    'r--', label='Ideal Line')
            slope, intercept = np.polyfit(outputs_test[:, i], outputs_pred[:, i], 1)
            plt.plot(outputs_test[:, i], slope*outputs_test[:, i] + intercept, 
                    'b-', label='Regression Line')
            plt.xlabel(f'true value {i+1}')
            plt.ylabel(f'pred value {i+1}')
            plt.title(f'Output {i+1} (R²={evaluation_results[i][0]:.2f})')
            plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    model = ModelingAndOptimization("dataset.csv")
    data = model.cross_validation()

    #chart = Chart(data)
    #chart.draw_2D_line_chart()
