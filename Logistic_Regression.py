import pandas as pd
import numpy as np
import random
import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain

from sklearn.linear_model import LogisticRegression as _logr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.utils import resample
from biokit.viz import corrplot



class LogisticRegression:
    def __init__(self,data_x,data_y):
       self.dataset_x =np.array(data_x)
       self.expected=np.array(data_y).reshape(-1,1)
       self.thetas=[]
       self.hypothesis_matrix=[]
       #print(self.dataset_x[:,0])#all elements of first columns
       #print(self.dataset_x.shape[1])#number of columns

    def initial_coefficient(self):
        #print("initial_coefficient")
        self.thetas=np.array([0 for i in range((self.dataset_x.shape[1]))], dtype=np.float64).reshape(-1,1)
        #print(self.thetas)

    def cost_function(self):
         return -(self. expected*np.log(self.hypothesis_matrix)+ ((1 - self.expected) * np.log(1-self.hypothesis_matrix))).mean()

    def gradient_descent_logistic(self,alpha):
        old_thetas = self.thetas
        self.thetas = old_thetas - (alpha) * (np.matmul(self.dataset_x.T, (self.hypothesis_matrix - self.expected)) / self.dataset_x.shape[0])
        return self.thetas, old_thetas


    'Sigmoid function'
    def sigmod(self,z):
        return 1/(1+np.exp(-z))

    def fit_logisticregression(self,alpha,max_itreration):
        self.costfunct=[]
        print(alpha)
        for i in range(max_itreration):
           self.hypothesis_matrix=self.sigmod(np.array(np.matmul(self.dataset_x, self.thetas)))
           self.costfunct.append(self.cost_function())
           updated_theta,old_thetas=self.gradient_descent_logistic(alpha)
           convergence=True
           if np.isclose(updated_theta,old_thetas,rtol=0,atol=0).all():
               break

        return self.thetas,self.hypothesis_matrix,self.costfunct

    '''Regularization'''

    def gradient_descent_logistic_regularization(self,alpha,_lambda,regularization_type):
        old_thetas=self.thetas.copy()
        correction = np.matmul(self.dataset_x.T, (self.hypothesis_matrix - self.expected)) / self.dataset_x.shape[0]
        # self.thetas[0]=old_thetas[0]-(alpha)*(np.matmul(self.dataset_x.T , (self.hypothesis_matrix-self.expected))/self.dataset_x.shape[0])[0]
        if regularization_type=='lasso':
             penality=((_lambda*old_thetas)/self.dataset_x.shape[0]).item()
             penality = np.repeat(penality, self.thetas.shape[0]).reshape(-1,1)
             penality[0] = 0
             #self.thetas[1:,] = old_thetas[1:,] - (alpha) *( np.add((np.matmul(self.dataset_x.T, (self.hypothesis_matrix - self.expected)) / self.dataset_x.shape[0]),penality/self.dataset_x.shape[0]))[1:,]
        elif regularization_type=='Ridge':
            penality= _lambda * (np.matmul(old_thetas.T,old_thetas)/ self.dataset_x.shape[0])
            # avoid penalty on intercept
            penality = np.repeat(penality, self.thetas.shape[0]).reshape(-1, 1)
            penality[0] = 0
        #else:
        #    penality = 0
        self.thetas = old_thetas - (alpha) * (np.add(correction, penality))

        return self.thetas,old_thetas

    def fit_logisticregression_regularization(self,alpha,_lambda,max_itreration,regularization_type):

        for i in range(max_itreration):
           self.hypothesis_matrix=self.sigmod(np.array(np.matmul(self.dataset_x, self.thetas)))
           updated_theta,old_thetas=self.gradient_descent_logistic_regularization(alpha,_lambda,regularization_type)
           #print(updated_theta,old_thetas)
           convergence=True
           if np.isclose(updated_theta,old_thetas,rtol=0,atol=0).all():
               #print("iteration",i)
               break
           else:
               convergence=False
        return self.thetas,self.hypothesis_matrix

    'Predition on test data'
    def predit_logistic(self,test_data_x,modelcoefficient):
        Logistic_predited_y = self.sigmod(np.array(np.matmul(np.array(test_data_x), modelcoefficient)))
        return Logistic_predited_y


def Model_graph_plot(Model_probability,data_x,modelname):

    H_label_predict = Model_probability.copy()
    H_label_predict[H_label_predict >= 0.5] = int(1)
    H_label_predict[H_label_predict < 0.5] = int(0)
    H_label_predict = list(chain.from_iterable(H_label_predict))
    feature1=np.array(data_x)[:, 1]
    feature2=np.array(data_x)[:, 2]
    fig=plt.figure()
    ax=Axes3D(fig)
    #ax.plot3D(feature1, Model_probability)
    colors = 100 * ['orange', 'green', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    #for i in range(0,len(feature1)):
    c = [colors[int(H_label_predict[i])] for i in range(0,len(feature1))]
    ax.scatter3D(feature1,feature2,Model_probability,color=c, marker='x',depthshade=False,label=c)

    ax.set_xlabel("all_mcqs_avg_n20")
    ax.set_ylabel("all_NBME_avg_n4")
    ax.set_zlabel("ModelPrediction/sigmoidal trend")
    plt.title("Non linear classification boundary visualization graph for Model-"+ modelname)
    plt.legend
    plt.show()

def Model_costfunction_graph(costfunction,modelname,alpha):
    #Hypothesis_values= list(chain.from_iterable(H_Model))
    plt.plot(costfunction,label=alpha)
    plt.ylabel("Costfunction/gradient function")
    plt.xlabel("iterations")
    plt.legend()
    plt.title("Gradient descent cost function graph for model " + modelname)

def Model_residual_vs_fit_graph(H_model,E_model,modelname):
    #E_model observed value
    #H_model predicted value
    residual=E_model-H_model
    fitted=H_model
    #Hypothesis_values= list(chain
    # .from_iterable(H_Model))
    plt.scatter(fitted,residual,facecolors='none',edgecolors='C1')
    #plt.plot(fitted, residual)
    plt.axhline(y=0.0,linestyle='--',color='C0')
    plt.ylabel("Residual")
    plt.xlabel("Fitted values")
    plt.title("residual_vs_fit_graph Model " + modelname)
    plt.show()


def data_split(data):
    'split data into training set and test set'
    data_copy = data.copy()
    #print(data_copy.LEVEL.value_counts())

    train_data = data_copy.sample(frac=0.80,random_state=seed)
    #print(train_data)
    #print(train_data.LEVEL.value_counts())
    train_set = pd.get_dummies(train_data, prefix=['LEVEL'])

    test_data = data_copy.drop(train_set.index)
    #print(test_data)
    #print(test_data.LEVEL.value_counts())
    'create 4 different columns of LEVEL like LEVEL_A,LEVEL_B,LEVEL_C,LEVEL_D'
    test_set = pd.get_dummies(test_data, prefix=['LEVEL'])

    'storing expected level before categorizing them to A,B,C ,D '
    train_expected_data_y = np.array(train_data['LEVEL'].astype('category').cat.codes)
    test_expected_data_y = np.array(test_data['LEVEL'].astype('category').cat.codes)

    'training data set'
    train_set.insert(0, 'x0', 1)


    train_data_x = train_set[['x0', 'all_mcqs_avg_n20','all_NBME_avg_n4']]
    #train_data_x = train_set[['x0', 'all_mcqs_avg_n20','CBSE_01','CBSE_02']]
    #train_data_x = train_set[['x0', 'CBSE_01', 'CBSE_02']]
    #train_data_x = train_set[['x0', 'all_mcqs_avg_n20', 'CBSE_01']]
    #train_data_x = train_set[['x0', 'all_mcqs_avg_n20', 'CBSE_02']]
    train_data_y_A = train_set['LEVEL_A']
    train_data_y_B = train_set['LEVEL_B']
    train_data_y_C = train_set['LEVEL_C']
    train_data_y_D = train_set['LEVEL_D']

    'test data set'
    test_set.insert(0, 'x0', 1)
    test_data_x = test_set[['x0', 'all_mcqs_avg_n20','all_NBME_avg_n4']]
    #test_data_x = test_set[['x0', 'all_mcqs_avg_n20', 'CBSE_01', 'CBSE_02']]
    #test_data_x = test_set[['x0', 'CBSE_01', 'CBSE_02']]
    #test_data_x = test_set[['x0', 'all_mcqs_avg_n20', 'CBSE_01']]
    #test_data_x = test_set[['x0', 'all_mcqs_avg_n20', 'CBSE_02']]
    test_data_y_A = train_set['LEVEL_A']
    test_data_y_B = train_set['LEVEL_B']
    test_data_y_C = train_set['LEVEL_C']
    test_data_y_D = train_set['LEVEL_D']
    # test_expected_data_y=np.hstack((np.array(test_data_y_A),np.array(test_data_y_B),np.array(test_data_y_C),np.array(test_data_y_D)))

    # print(data_y_labelA.value_counts(),"\n",data_y_labelB.value_counts(),"\n",data_y_labelC.value_counts(),"\n",data_y_labelD.value_counts())

    return train_data_x,train_expected_data_y,train_data_y_A,train_data_y_B,train_data_y_C,train_data_y_D,test_data_x,test_expected_data_y

def train_model(train_data_x,train_data_y_Label,alpha,_lambda,Regularization,Regularization_type,iteration,seed):

    'Train the model'
    'Call to logistic regression '
    Linear_obj = LogisticRegression(train_data_x, train_data_y_Label)
    Linear_obj.initial_coefficient()
    if Regularization == False:
        coeff_Model, H_Model,cost_function = Linear_obj.fit_logisticregression(alpha, iteration)
    elif Regularization == True:
        cost_function=None
        coeff_Model, H_Model = Linear_obj.fit_logisticregression_regularization(alpha, _lambda, iteration,Regularization_type)
    H_Model = Linear_obj.predit_logistic(train_data_x, coeff_Model)
    #print(H_A,H_Model)

    return coeff_Model,H_Model,cost_function,Linear_obj



def Model_predit(obj,train_data_x,train_expected_data_y,test_data_x,test_expected_data_y,Coeff_A,Coeff_B,Coeff_C,Coeff_D):
    'Tool kit'
    logr = _logr().fit(train_data_x, train_expected_data_y)
    y_pred = logr.predict(test_data_x)
    sk_p,sk_R,sk_F,_=precision_recall_fscore_support(y_true=test_expected_data_y,y_pred=y_pred, average="macro")
    print("sklearn", precision_recall_fscore_support(y_true=test_expected_data_y,y_pred=y_pred, average="macro"))
    print(classification_report(y_true=test_expected_data_y,y_pred=y_pred))

    'Measure on Train data'
    print("train data")
    H_A = obj.predit_logistic(train_data_x, Coeff_A)
    H_B = obj.predit_logistic(train_data_x, Coeff_B)
    H_C = obj.predit_logistic(train_data_x, Coeff_C)
    H_D = obj.predit_logistic(train_data_x, Coeff_D)
    final_hypothesis_matrix = np.hstack((H_A, H_B, H_C, H_D))

    'Confusion matrix and precision and recall on test data'
    train_predicted_data_y = np.argmax(final_hypothesis_matrix, axis=1)

    print("Algorithm confusion matrix\n", confusion_matrix(train_expected_data_y, train_predicted_data_y))

    print("algorithm Precision,recall,f-score train data \n",
          precision_recall_fscore_support(y_true=train_expected_data_y, y_pred=train_predicted_data_y, average="macro"))

    print(classification_report(y_true=train_expected_data_y,y_pred=train_predicted_data_y))

    train_precision, train_recall, train_f_score, _ = precision_recall_fscore_support(y_true=train_expected_data_y,
                                                                    y_pred=train_predicted_data_y, average="macro")

    print("test data")
    'Measure on test data'
    H_A = obj.predit_logistic(test_data_x, Coeff_A)
    H_B = obj.predit_logistic(test_data_x, Coeff_B)
    H_C = obj.predit_logistic(test_data_x, Coeff_C)
    H_D = obj.predit_logistic(test_data_x, Coeff_D)
    final_hypothesis_matrix = np.hstack((H_A, H_B, H_C, H_D))

    'Confusion matrix and precision and recall on test data'
    test_predicted_data_y = np.argmax(final_hypothesis_matrix, axis=1)

    print("Algorithm confusion matrix\n",confusion_matrix(test_expected_data_y, test_predicted_data_y))

    print("algorithm Precision,recall,f-score test data \n",precision_recall_fscore_support(y_true=test_expected_data_y, y_pred=test_predicted_data_y,average="macro"))

    print(classification_report(y_true=test_expected_data_y,y_pred=test_predicted_data_y))

    precision,recall,f_score,_=precision_recall_fscore_support(y_true=test_expected_data_y, y_pred=test_predicted_data_y,average="macro")

    return precision,recall,f_score,train_precision,train_recall,train_f_score,sk_F



if __name__ == "__main__":
    seed = 123

    data_path="BSOM_DataSet_for_HW2.csv"
    __data_raw=pd.read_csv(data_path)
    print(__data_raw.shape)
    #5a feature set1
    __data = __data_raw[['all_mcqs_avg_n20','CBSE_01','CBSE_02','LEVEL']]
    # 5a feature set2
    #__data = __data_raw[['CBSE_01', 'CBSE_02', 'LEVEL']]
    # 5a feature set3
    #__data = __data_raw[['all_mcqs_avg_n20', 'CBSE_01', 'LEVEL']]
    # 5a feature set4
    #__data = __data_raw[['all_mcqs_avg_n20','CBSE_02', 'LEVEL']]
    #3a features
    #__data = __data_raw[["all_mcqs_avg_n20", "all_NBME_avg_n4",'LEVEL']]
    print(__data.describe())
    # corelatiodata=__data.corr()
    # c=corrplot.Corrplot(corelatiodata)
    # c.plot(colorbar=False,method="square",shrink=.9,rotation=45)
    # plt.show()


    __data = __data.dropna(subset=['LEVEL'])

    print(__data.shape)

    '''handling imbalance data over sampling'''
    count_levels=__data.LEVEL.value_counts()
    #print(count_levels[0])
    print(__data.LEVEL.value_counts())

    __data = __data.append(resample(__data[__data['LEVEL']=='A'], replace=True, n_samples=count_levels[0]-len(__data[__data['LEVEL']=='A']), random_state=seed), ignore_index=True)
    __data = __data.append(resample(__data[__data['LEVEL']=='D'], replace=True, n_samples=count_levels[0]-len(__data[__data['LEVEL']=='D']), random_state=seed), ignore_index=True)
    __data = __data.append(resample(__data[__data['LEVEL']=='C'], replace=True, n_samples=count_levels[0]-len(__data[__data['LEVEL']=='C']), random_state=seed), ignore_index=True)

    data_level = __data['LEVEL']
    __data=__data.drop(['LEVEL'],axis=1)
    'Z-normalization for scaling'
    data_scaled = (__data - __data.mean()) / __data.std()
    data_scaled=pd.concat([data_scaled,data_level],axis=1)
    # split data
    train_data_x, train_expected_data_y, train_data_y_A, train_data_y_B, train_data_y_C, train_data_y_D, test_data_x, test_expected_data_y = data_split(data_scaled)
    #train_data_x, train_expected_data_y, train_data_y_A, train_data_y_B, train_data_y_C, train_data_y_D, test_data_x, test_expected_data_y=data_split(__data)
    #Regularization = False or true 0 indicates no regularization '
    #Regularization_type = Ridge or lasso or none'
    #Logistic_scenario(data,alpha,lambda,regularization,regularizationtype)'
    #question 3-a  without scaling and regularization demostrating the model'
    print("**************"*5)
    print("Demonstrating Model without scaling and regularization")
    #Build model
    print("3a demostrating model")
    # alphas=[0.00001,0.000001,0.0001,0.0002,0.001]
    # #alphas=[0.001]
    # for alpha in alphas:
    #     coeff_A, H_A, cost_function_A, Linear_obj=  train_model(train_data_x, train_data_y_A, alpha=alpha, _lambda=None,Regularization=False, Regularization_type=None,iteration=1000, seed=seed)
    #     'Determining best parameters for the model based on cost function.The goal is to find the values of model parameters for which Cost Function return as small number as possible '
    #     Model_costfunction_graph(cost_function_A, "A",alpha)
    #     #Model_graph_plot(H_A, train_data_x, "A")
    #     #expected_A=np.array(train_data_y_A).reshape(-1,1)
    #     #Model_residual_vs_fit_graph(H_A,Linear_obj.expected,"A")
    #     print("Coefficient of Model A,",coeff_A)
    #
    # plt.show()
    # for alpha in alphas:
    #     coeff_B, H_B, cost_function_B, Linear_obj = train_model(train_data_x, train_data_y_B, alpha=alpha, _lambda=None,Regularization=False, Regularization_type=None,iteration=1000, seed=seed)
    #     Model_costfunction_graph(cost_function_B, "B",alpha)
    #     #Model_graph_plot(H_B, train_data_x, "B")
    #     #Model_residual_vs_fit_graph(H_B, Linear_obj.expected, "B")
    #     print("Coefficient of Model B,", coeff_B)
    #
    # plt.show()
    # for alpha in alphas:
    #     coeff_C, H_C, cost_function_C, Linear_obj = train_model(train_data_x, train_data_y_C, alpha=alpha, _lambda=None,Regularization=False, Regularization_type=None,iteration=1000, seed=seed)
    #     Model_costfunction_graph(cost_function_C, "C",alpha)
    #     #Model_graph_plot(H_C, train_data_x, "C")
    #     #Model_residual_vs_fit_graph(H_C, Linear_obj.expected, "C")
    #     print("Coefficient of Model C,", coeff_C)
    # plt.show()
    # for alpha in alphas:
    #     coeff_D, H_D, cost_function_D, Linear_obj = train_model(train_data_x, train_data_y_D, alpha=alpha, _lambda=None,Regularization=False, Regularization_type=None,iteration=1000, seed=seed)
    #     Model_costfunction_graph(cost_function_D, "D",alpha)
    #     #Model_graph_plot(H_D, train_data_x, "D")
    #     #Model_residual_vs_fit_graph(H_D, Linear_obj.expected, "D")
    #     print("Coefficient of Model D,", coeff_D)
    # plt.show()
    #

    #
    # print("3b evaluating performance on test data")
    # precision, recall, f_score, train_precision, train_recall, train_f_score=Model_predit(Linear_obj,train_data_x,train_expected_data_y,test_data_x,test_expected_data_y,coeff_A,coeff_B,coeff_C,coeff_D)

    # print("Question 4 b **************" * 5)
    print("Regularization without 5a scaling")
    fscore_values=[]
    precision_values=[]
    recall_values=[]

    train_fscore_values = []
    train_precision_values = []
    train_recall_values = []
    sk_f=[]
    __lambdas = [ 0.001,0.01, 0.1, 0,1,2,3,4,5,6,7,7.2,7.4,7.6,7.8,8,8.5,9,10]
    for __lambda in __lambdas:
        print("lambda",__lambda)
        coeff_A, H_A, cost_function_A, Linear_obj = train_model(train_data_x, train_data_y_A, alpha=0.001, _lambda=__lambda,
                                                                Regularization=True, Regularization_type='Ridge',
                                                                iteration=10000, seed=seed)
        coeff_B, H_B, cost_function_B, Linear_obj = train_model(train_data_x, train_data_y_B, alpha=0.001, _lambda=__lambda,
                                                                Regularization=True, Regularization_type='Ridge',
                                                                iteration=10000, seed=seed)
        coeff_C, H_C, cost_function_C, Linear_obj = train_model(train_data_x, train_data_y_C, alpha=0.001, _lambda=__lambda,
                                                                Regularization=True, Regularization_type='Ridge',
                                                                iteration=10000, seed=seed)
        coeff_D, H_D, cost_function_D, Linear_obj = train_model(train_data_x, train_data_y_D, alpha=0.001, _lambda=__lambda,
                                                                Regularization=True, Regularization_type='Ridge',
                                                                iteration=10000, seed=seed)
        precision, recall, f_score, train_precision, train_recall, train_f_score,sk_F = Model_predit(Linear_obj,
                                                                                                train_data_x,
                                                                                                train_expected_data_y,
                                                                                                test_data_x,
                                                                                                test_expected_data_y,
                                                                                                coeff_A, coeff_B,
                                                                                                coeff_C, coeff_D)

        sk_f.append(sk_F)
        fscore_values.append(f_score)
        precision_values.append(precision)
        recall_values.append(recall)
        train_fscore_values.append(train_f_score)
        train_precision_values.append(train_precision)
        train_recall_values.append(train_recall)
    print("fscore train 4th",train_fscore_values)
    print("precision train",train_precision_values)
    print("REcall train",train_recall_values)
    plt.plot([str(l) for l in __lambdas], train_fscore_values, label="F-score")
    #plt.plot([str(l) for l in __lambdas], train_precision_values, label="Precision")
    #plt.plot([str(l) for l in __lambdas], train_recall_values, label="Recall")
    plt.legend()
    plt.title("Model metrics on train data with and without regularization")
    plt.show()

    plt.plot([str(l) for l  in __lambdas], fscore_values,label="F-score")
    #plt.plot([str(l) for l in __lambdas], sk_f, label="sklearnF-score")
    #plt.plot([str(l) for l in __lambdas], precision_values,label="Precision")
    #plt.plot([str(l) for l in __lambdas], recall_values,label="Recall")
    plt.legend()
    plt.title("Model metrics on test data with and without regularization")
    plt.show()
    #
    #
    'Question 4-a feature scaling'
    data_level=__data['LEVEL']
    __data=__data.drop(['LEVEL'],axis=1)
    'Z-normalization for scaling'
    data_scaled = (__data - __data.mean()) / __data.std()
    data_scaled=pd.concat([data_scaled,data_level],axis=1)
    # split data
    train_data_x, train_expected_data_y, train_data_y_A, train_data_y_B, train_data_y_C, train_data_y_D, test_data_x, test_expected_data_y = data_split(data_scaled)
    print("**************" * 5)
    print("feature scaling without regularization")
    coeff_A, H_A, cost_function_A, Linear_obj = train_model(train_data_x, train_data_y_A, alpha=0.2, _lambda=None,
                                                            Regularization=False, Regularization_type=None,
                                                            iteration=750000, seed=seed)
    coeff_B, H_B, cost_function_B, Linear_obj = train_model(train_data_x, train_data_y_B, alpha=0.2, _lambda=None,
                                                            Regularization=False, Regularization_type=None,
                                                            iteration=390000, seed=seed)
    coeff_C, H_C, cost_function_C, Linear_obj = train_model(train_data_x, train_data_y_C, alpha=0.2, _lambda=None,
                                                            Regularization=False, Regularization_type=None,
                                                            iteration=400000, seed=seed)
    coeff_D, H_D, cost_function_D, Linear_obj = train_model(train_data_x, train_data_y_D, alpha=0.2, _lambda=None,
                                                            Regularization=False, Regularization_type=None,
                                                            iteration=800000, seed=seed)
    s_precision, s_recall, s_f_score, s_train_precision, s_train_recall, s_train_f_score = Model_predit(Linear_obj,
                                                                                            train_data_x,
                                                                                            train_expected_data_y,
                                                                                            test_data_x,
                                                                                            test_expected_data_y,
                                                                                            coeff_A, coeff_B,
                                                                                            coeff_C, coeff_D)

    data_type=("Raw_data_fscore","Scaled_data_fscore","Raw_data_Precision","Scaled_data_Precision","Raw_data_Recall","Scaled_data_Recall")
    data_type_length=np.arange(len(data_type))
    Metrics=(train_fscore_values[4],s_train_f_score,train_precision_values[4],s_train_precision,train_recall_values[4],s_train_recall)
    plt.bar(data_type_length,Metrics,align="center",alpha=0.5)
    plt.xticks(data_type_length,data_type)
    plt.ylabel('Scores/Metrics of Model on train data')
    plt.title("Model performance for Raw and scaled data")
    plt.show()

    print("**************" * 5)
    print("regularization With feature scaling")
    fscore_values = []
    precision_values = []
    recall_values = []

    train_fscore_values = []
    train_precision_values = []
    train_recall_values = []
    __lambdas = [0.0001]
    for __lambda in __lambdas:
        print("lambda", __lambda)
        coeff_A, H_A, cost_function_A, Linear_obj = train_model(train_data_x, train_data_y_A, alpha=0.2,
                                                                _lambda=__lambda,
                                                                Regularization=True, Regularization_type='Ridge',
                                                                iteration=750000, seed=seed)
        coeff_B, H_B, cost_function_B, Linear_obj = train_model(train_data_x, train_data_y_B, alpha=0.2,
                                                                _lambda=__lambda,
                                                                Regularization=True, Regularization_type='Ridge',
                                                                iteration=390000, seed=seed)
        coeff_C, H_C, cost_function_C, Linear_obj = train_model(train_data_x, train_data_y_C, alpha=0.2,
                                                                _lambda=__lambda,
                                                                Regularization=True, Regularization_type='Ridge',
                                                                iteration=400000, seed=seed)
        coeff_D, H_D, cost_function_D, Linear_obj = train_model(train_data_x, train_data_y_D, alpha=0.2,
                                                                _lambda=__lambda,
                                                                Regularization=True, Regularization_type='Ridge',
                                                                iteration=800000, seed=seed)
        RStest_precision, RStest_recall, RStest_f_score, RS_train_precision, RS_train_recall, RS_train_f_score = Model_predit(Linear_obj,
                                                                                                train_data_x,
                                                                                                train_expected_data_y,
                                                                                                test_data_x,
                                                                                                test_expected_data_y,
                                                                                                coeff_A, coeff_B,
                                                                                                coeff_C, coeff_D)




