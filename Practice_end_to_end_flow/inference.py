import pickle
import pandas as pd

def data_preprocessing(user_input):
  with open('artifacts/label_enc.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
  user_input_df = pd.DataFrame([user_input])
  user_input_df['city_encoded'] = label_encoder.transform(user_input_df["city"])
  user_input_df.drop('city', axis=1, inplace=True)
  return user_input_df

def inference(encoded_user_input):
  with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)
  prediction = model.predict(encoded_user_input)
  return prediction[0]

def predict(user_input):
   user_input_df = data_preprocessing(user_input)
   prediction = inference(user_input_df)
   return prediction



input_user = {"city": 'lakewood', "Second": 3.43, "Third": 4.56, "Fourth": 3.12}
final_output = predict(input_user)
print(final_output)

