import kfp
from kfp import dsl
from kfp import compiler
from typing import Dict,List
from kfp.dsl import Input,Output,Dataset,Model,component

#load dataset
@dsl.component(base_image="python:3.9")
def load_data(output_csv: Output[Dataset]):
    import subprocess
    subprocess.run(["pip","install","pandas","scikit-learn"],check=True)
    from sklearn.datasets import load_iris
    import pandas as pd 
    iris=load_iris()
    df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
    df['target']=iris.target

    df.to_csv(output_csv.path,index=False)

#preprocess data
@dsl.component(base_image="python:3.9")
def preprocess_data(input_csv: Input[Dataset],output_train: Output[Dataset],output_test: Output[Dataset],
                    output_ytrain: Output[Dataset],output_ytest: Output[Dataset]):
    import subprocess
    subprocess.run(["pip","install","pandas","scikit-learn"],check=True)
    import pandas as pd
    from sklearn.preprocessing import StandardScalar
    from sklearn.model_selection import train_test_split
    df=pd.read_csv(input_csv.path)

    print("dataset shape :",df.shape)
    print("missing values in a dataset :"df.isnull().sum())
    if df.isnull().values().any():
        print("missing values detected")
        df=df.dropna()
    features=df.drop(columns=['target'])
    target=df['target']
    #standardize features
    scalar=StandardScalar()
    scaled_features=scalar.fit_transform(features)

    #train-test split
    X_train,X_test,y_train,y_test=train_test_split(scaled_features,target,test_size=0.2,random_state=42)
    print("X_train :",X_train.shape, "X_test :",X_test.shape)
    print("y_train :",y_train.shape, "y_test :",y_test.shape)

    X_train_df=pd.DataFrame(X_train,columns=features.columns)
    X_train_df.to_csv(output_train.path,index=False)

    X_test_df=pd.DataFrame(X_test,columns=features.columns)
    X_test_df.to_csv(output_test.path,index=False)

    y_train_df=pd.DataFrame(y_train)
    y_train_df.to_csv(output_ytrain.path,index=False)

    y_test_df=pd.DataFrame(y_test)
    y_test_df.to_csv(output_ytest.path,index=False)

#train the model
@dsl.component(base_image="python:3.9")
def train_model(train_data: Input[Dataset],ytrain_data: Input[Dataset],model_output: Output[Model]):
    import subprocess
    subprocess.run(["pip","install","pandas","scikit-learn","joblib"],check=True)
    import pandas as pd
    import sklearn.linear_model import LogisticRegression
    from joblib import dump
    #load training data
    X_train=pd.read_csv(train_data.path)
    y_train_df=pd.read_csv(ytrain_data.path)
    model=LogisticRegression()
    model.fit(X_train,y_train)
    #save model
    dump(model,model_output.path)

#evaluate model
@dsl.component(base_image="python:3.9")
def evaluate_model(test_data: Input[Dataset],ytest_data: Input[Dataset],model: Input[Model],metrics_output: Output[Dataset]):
    import subprocess
    subprocess.run(["pip","install","pandas","scikit-learn","joblib"],check=True)
    import pandas as pd
    from sklearn.metrics import classification_report,confusion_matrix
    from joblib import load
    X_test=pd.read_csv(test_data.path)
    y_test=pd.read_csv(ytest_data.path)
    model=load(model.path)
    #predict
    y_pred=model.predict(X_test)
    report=classification_report(y_test,y_pred,output_dict=True)
    cm=confusion_matrix(y_test,y_pred)
    #save metrics
    metrics_path=metrics_output.path
    with open(metrics_path,'w')as f_in:
        f_in.write(str(report))

#define pipeline
@dsl.pipeline(name="ml_pipeline")
def ml_pipeline:
    load_op=load_data()
    preprocess_op=preprocess_data(input_csv=load_op.outputs["output_csv"])
    train_op=train_model(train_data=preprocess_op.outputs["output_train"],ytrain_data=preprocess_op.outputs["output_ytrain"])
    evaluate_op=evaluate_model(test_data=preprocess_op.outputs["output_test"],ytest_data=preprocess_op.outputs["output_ytest"],model=train_op.outputs["model_output"])


if __name__ == "main":
    compiler.Compiler().compile(pipeline_func=ml_pipeline,package_path="kubeflow_pipeline.yaml")








