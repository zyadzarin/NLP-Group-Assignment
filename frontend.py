import gradio as gr
import pickle
from preprocessing import preprocess_text
import pandas as pd

# paths to model and CountVectorizer files
model_path = "Model/random_forest_model.pkl"
vectorizer_path = "Model/countvector.pkl"

# load the trained model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# load the trained CountVectorizer
with open(vectorizer_path, "rb") as f:
    countvector = pickle.load(f)



def predict(news_headlines):
    d = {'headline': [news_headlines]}
    temp_df = pd.DataFrame(data=d)
    temp_df['headline'] = preprocess_text(temp_df['headline'])


    preprocessed_text = temp_df['headline'][0]

    news_headline_vector = countvector.transform(temp_df['headline'])
    predictions = model.predict(news_headline_vector)


    output = ''
    if predictions == 1:
        output = "Positive"
    elif predictions == 3:
        output = "Negative"
    else:
        output = "Neutral"



    return preprocessed_text, output

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="News Headline Here..."),
    outputs=["text", "text"]
)
demo.launch()

'''
SHORT TEST DATA

Tesla stock rises again, for record 12-day win streak
Twitter’s Stock Falls Further as Doubts Swirl Over Musk’s Takeover
Is Apple Stock a Buy Near $185?
'''
