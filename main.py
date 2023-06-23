import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Creating flask deployment app and importing ML models
flask_app = Flask(__name__)
DT_model = pickle.load(open("DT_model.pkl", "rb"))
KNN_model = pickle.load(open("KNN_model.pkl", "rb"))
LR_model = pickle.load(open("LR_model.pkl", "rb"))
RF_model = pickle.load(open("RF_model.pkl", "rb"))
AB_model = pickle.load(open("AB_model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    
# defining the input dataset
    totalmatches = float(request.form['Total Matches'])
    totalrun = float(request.form['Total Run'])
    battingavg = float(request.form['Batting Average'])
    strikerate = float(request.form['Strike Rate'])
    wickets = float(request.form['Total Wickets'])
    wicket_per_match = round(wickets/totalmatches, 2)
    economy = float(request.form['Economy'])
    inForm = float(request.form['In Form?'])
    batting_impact = round((battingavg/45)*0.7, 2)
    bowling_impact = round((wicket_per_match/3)/(economy/5), 2)
    total_impact = batting_impact + bowling_impact
    if total_impact >= 0.7:
        key_player = 1
    else:
        key_player = 0
    

    features = [key_player, totalmatches, totalrun, battingavg, strikerate, wickets, wicket_per_match, 
                economy, batting_impact, bowling_impact, total_impact, inForm]
    input_data = np.array(features)
    input_data_reshaped = input_data.reshape(1, -1)
    
# output prediction and output generation
    prediction1 = DT_model.predict(input_data_reshaped)
    prediction2 = KNN_model.predict(input_data_reshaped)
    prediction3 = LR_model.predict(input_data_reshaped)
    prediction4 = RF_model.predict(input_data_reshaped)
    prediction5 = AB_model.predict(input_data_reshaped)

    if prediction1 == 0:
        output1 = "No"
    else:
        output1 = "Yes"

    if prediction2 == 0:
        output2 = "No"
    else:
        output2 = "Yes"

    if prediction3 == 0:
        output3 = "No"
    else:
        output3 = "Yes"
    
    if prediction4 == 0:
        output4 = "No"
    else:
        output4 = "Yes"

    if prediction5 == 0:
        output5 = "No"
    else:
        output5 = "Yes"

    output = (prediction1+prediction2+prediction3+prediction4+prediction5)/5*100

    return render_template('index.html', prediction_text='Percentage Of Becoming MVP: {}%'.format(output), 
                           decisionTree="Decision Tree Prediction: {}".format(output1),
                           KNN="KNearest Neighbor Prediction: {}".format(output2),
                           LR="Logistic Regression Prediction: {}".format(output3),
                           RF="Random Forest Prediction: {}".format(output4),
                           AB="AdaBoost Prediction: {}".format(output5)
                           )

if __name__ == "__main__":
    flask_app.run(debug=True)