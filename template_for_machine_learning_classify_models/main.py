from turtle import color
from altair import sample
from flask import redirect
from matplotlib.scale import scale_factory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pybaselines import Baseline
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import  confusion_matrix, precision_score, recall_score, f1_score
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
            cls = self.data["outputs"].loc[sample_label].values
            
            if  cls == 2020:
                plt.plot(X, Y, color = "red", label = cls)
            elif  cls == 2021:
                plt.plot(X, Y, color = "blue",  label = cls)
            elif  cls == 2022:
                plt.plot(X, Y, color = "yellow", label = cls)
            elif  cls == 2023:
                plt.plot(X, Y, color = "black", label = cls)
        
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
        dataset = pd.read_csv(self.dataset_path, header = None)  # header, names, index_col, datype, skiprows
        input_features = dataset.columns[2:]
        inputs = dataset[input_features]
        output_features = dataset.columns[0:1]
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


        # abstract critical feature peaks' integration

        # standardization


        # calculate sample's weights


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

        # predicted results buffer
        year_vals = []
        year_predicts = []
        
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

            # modeling and optimization
            model = MultiOutputClassifier(SVC())
            model.fit(
                inputs_train, 
                outputs_train, 
            )

            # predict
            outputs_predict = model.predict(inputs_val)

            # inverse standardization/normalization

            # save predicted results
            year_vals.append(outputs_val.values)
            year_predicts.append(outputs_predict)

        # validation set evaluation   
        print(year_vals)
        print(year_predicts)
        print("\n------------------------------留一法交叉验证评估得分: ---------------------------------------")
        print("\n 实际与预测结果: \n")
        for i in range(len(year_vals)):
            print(f"真实年份: {year_vals[i]}, 预测年份: {year_predicts[i]} \n")
        print("\n------------------------------------混淆矩阵：: -------------------------------------------\n")
        y_true = np.array(year_vals).squeeze() 
        y_pred = np.array(year_predicts).squeeze()  
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        print("\n----------------------------------混淆矩阵可视化: -----------------------------------------\n")
        plt.figure(figsize=(8, 6))  # visualization
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=np.unique(y_true), 
                    yticklabels=np.unique(y_true))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Single Output)')
        plt.show()
        print("\n-----------------------------------分类指标报告: -----------------------------------------\n")
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"加权平均查准率: {precision:.4f}")
        print(f"加权平均召回率: {recall:.4f}")
        print(f"加权平均F1分数: {f1:.4f}")
        
        
        # Fullly modeling and optimization
        model = MultiOutputClassifier(SVC())
        model.fit(
            train_set["inputs"], 
            train_set["outputs"], 
        )
        dump(model, 'model.joblib')

    # 4. test
    def test(self):
        # get test set
        _, test_set = self.dataset_split_train_test()
        
        # predict
        model = load('model.joblib')
        outputs_predict = model.predict(test_set["inputs"])

        # test set evaluation
        year_vals = test_set["outputs"]
        year_predicts = outputs_predict
        print(year_vals)
        print(year_predicts)
        print("\n------------------------------------混淆矩阵：: -------------------------------------------\n")
        y_true = np.array(year_vals).squeeze() 
        y_pred = np.array(year_predicts).squeeze()  
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        print("\n----------------------------------混淆矩阵可视化: -----------------------------------------\n")
        plt.figure(figsize=(8, 6))  # visualization
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=np.unique(y_true), 
                    yticklabels=np.unique(y_true))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Single Output)')
        plt.show()
        print("\n-----------------------------------分类指标报告: -----------------------------------------\n")
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"加权平均查准率: {precision:.4f}")
        print(f"加权平均召回率: {recall:.4f}")
        print(f"加权平均F1分数: {f1:.4f}")

if __name__ == "__main__":

    model = ModelingAndOptimization("dataset_iris.csv")
    data = model.cross_validation()

    #chart = Chart(data)
    #chart.draw_2D_line_chart()
