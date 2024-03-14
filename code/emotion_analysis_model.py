import pycaret.classification as pc

# 모델 불러오기
def load_model(model):
    return pc.load_model(model_name=f"../model/{model}")


# "High" == 1, "Low" == 0
def predict_emotion(eeg_feature, model):
    predict = pc.predict_model(model, eeg_feature)
    return predict["Label"].value_counts(ascending=False)
