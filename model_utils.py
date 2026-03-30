import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

# Heart Disease

@st.cache_data
def load_heart():
    df = pd.read_csv("heart.csv")
    X = df.drop("target", axis=1).copy()
    y = df["target"]
    X["age_group"] = pd.cut(X["age"], bins=[0,40,55,70,100], labels=[0,1,2,3]).astype(int)
    X["hr_age_ratio"] = X["thalach"] / X["age"]
    X["risk_score"] = (X["cp"] + X["exang"] + X["slope"]).astype(int)
    return df, X, y

@st.cache_resource
def train_heart_ml():
    _, X, y = load_heart()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xval = scaler.transform(X_val)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=4, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, subsample=0.8, random_state=42)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    ensemble = VotingClassifier(estimators=[("rf", rf), ("gb", gb), ("lr", lr)], voting="soft")
    ensemble.fit(Xtr, y_train)

    pred = ensemble.predict(Xval)
    proba = ensemble.predict_proba(Xval)[:, 1]
    acc = accuracy_score(y_val, pred)
    auc = roc_auc_score(y_val, proba)
    cm = confusion_matrix(y_val, pred)
    cr = classification_report(y_val, pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_val, proba)
    fi = np.mean([est.feature_importances_ for est in [
        ensemble.named_estimators_["rf"],
        ensemble.named_estimators_["gb"]
    ]], axis=0)

    return {
        "model": ensemble, "scaler": scaler,
        "X_val": X_val, "y_val": y_val,
        "acc": acc, "auc": auc, "cm": cm, "cr": cr,
        "fpr": fpr, "tpr": tpr,
        "feature_names": list(X.columns),
        "feature_importances": fi,
    }


@st.cache_resource
def train_heart_nn():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    _, X, y = load_heart()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xval = scaler.transform(X_val)

    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.Sequential([
        layers.Input(shape=(Xtr.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=15, restore_best_weights=True, mode="max"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]
    history = model.fit(Xtr, y_train, validation_data=(Xval, y_val), epochs=100, batch_size=32, callbacks=callbacks, verbose=0)

    pred_proba = model.predict(Xval, verbose=0).flatten()
    pred = (pred_proba >= 0.5).astype(int)
    acc = accuracy_score(y_val, pred)
    auc = roc_auc_score(y_val, pred_proba)
    cm = confusion_matrix(y_val, pred)
    cr = classification_report(y_val, pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_val, pred_proba)

    return {
        "model": model, "scaler": scaler,
        "X_val": X_val, "y_val": y_val,
        "acc": acc, "auc": auc, "cm": cm, "cr": cr,
        "fpr": fpr, "tpr": tpr,
        "history": history.history,
        "feature_names": list(X.columns),
    }

# Titanic

@st.cache_data
def load_titanic():
    df = pd.read_csv("Titanic-Dataset.csv")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Cabin"] = df["Cabin"].fillna("U").str[0]
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
    df["Title"] = df["Title"].replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs"})
    df["Title"] = df["Title"].map({"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Rare":4}).fillna(0)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Age*Class"] = df["Age"] * df["Pclass"]
    df.drop(["PassengerId","Name","Ticket"], axis=1, inplace=True)
    df["Sex"] = df["Sex"].map({"male":0,"female":1})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    df = pd.get_dummies(df, columns=["Cabin"], drop_first=True)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return df, X, y

@st.cache_resource
def train_titanic_ml():
    _, X, y = load_titanic()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    rf  = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_split=4, random_state=42)
    gb  = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    svm = SVC(C=1.5, kernel="rbf", probability=True, random_state=42)

    ensemble = VotingClassifier(estimators=[("rf",rf),("gb",gb),("svm",svm)], voting="soft")
    ensemble.fit(Xtr, y_train)

    pred = ensemble.predict(Xte)
    proba = ensemble.predict_proba(Xte)[:, 1]
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, proba)
    fi = np.mean([
        ensemble.named_estimators_["rf"].feature_importances_,
        ensemble.named_estimators_["gb"].feature_importances_,
    ], axis=0)

    return {
        "model": ensemble, "scaler": scaler,
        "X_test": X_test, "y_test": y_test,
        "acc": acc, "auc": auc, "cm": cm, "cr": cr,
        "fpr": fpr, "tpr": tpr,
        "feature_names": list(X.columns),
        "feature_importances": fi,
    }

@st.cache_resource
def train_titanic_nn():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    _, X, y = load_titanic()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.Sequential([
        keras.Input(shape=(Xtr.shape[1],)),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.1),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(5e-4), loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(Xtr, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)

    pred_proba = model.predict(Xte, verbose=0).flatten()
    pred = (pred_proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, pred_proba)
    cm = confusion_matrix(y_test, pred)
    cr = classification_report(y_test, pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, pred_proba)

    return {
        "model": model, "scaler": scaler,
        "X_test": X_test, "y_test": y_test,
        "acc": acc, "auc": auc, "cm": cm, "cr": cr,
        "fpr": fpr, "tpr": tpr,
        "history": history.history,
        "feature_names": list(X.columns),
    }
