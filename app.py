from flask import Flask,render_template, redirect, request, session,jsonify 
from flask_session import Session
import webbrowser
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
from automation import *
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
import simplejson
import os

result_df = pd.DataFrame()
upsampled_result_df=pd.DataFrame()
app = Flask(__name__)
app.secret_key = os.urandom(24)

run_with_ngrok(app)   #starts ngrok when the app is run
@app.route("/")
def home():
    return render_template('Index.html',content="Testing")

@app.route("/upload", methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      file_name=secure_filename(f.filename)
      f.save(file_name)
      df = read_data(file_name)
      session["filename"] = file_name
      return str(len(df)) + ' rows were processed successfully. The dataset has '+ str(df.shape[0]) +'rows and '+str(df.shape[1])+'columns.'

@app.route("/augment", methods = ['GET', 'POST'])
def augment_data():
  upsampled_result_df = pd.DataFrame()
  dat = session.get('filename')
  result_df = read_data('/content/'+dat)
  result_df.drop('Unnamed: 0',axis='columns', inplace=True)
  result_df.rename(columns = {'Data':'Date'}, inplace = True)
  result_df.rename(columns = {'Genre':'Gender'}, inplace = True)
  result_df.rename(columns = {'Employee or Third Party':'Employee Type'}, inplace = True)
  result_df.drop_duplicates(inplace=True)
  result_df['Accident_Level']=result_df.apply(lambda col: str(col['Accident Level']), axis=1)
  result_df=result_df[['Accident_Level','Description']]
  if request.method == 'POST':
    dict_result = simplejson.loads(request.get_data(as_text=True))
    session['dict_result'] = dict_result
    #print(dict_result)
    for key in dict_result:
      options=[key['className']]
      upsampled_df=augment_mydata(result_df.loc[result_df['Accident_Level'].isin(options)],
                                  float(key['syn_val'])/100,
                                  float(key['swap_val'])/100,
                                  float(key['rand_ins_val'])/100,
                                  float(key['rand_del_val'])/100,
                                  int(key['aug_val']),
                                  )
      upsampled_result_df = upsampled_result_df.append(upsampled_df)
      print('upsampling complete!')
    return 'The augmented dataset has '+str(upsampled_result_df.shape[0])+' rows and '+str(upsampled_result_df.shape[1])+'columns.'   

@app.route("/clean_dl_data", methods = ['GET', 'POST'])
def clean_dl_data():
  upsampled_result_df = pd.DataFrame()
  dat = session.get('filename')
  result_df = read_data('/content/'+dat)
  result_df.drop('Unnamed: 0',axis='columns', inplace=True)
  result_df.rename(columns = {'Data':'Date'}, inplace = True)
  result_df.rename(columns = {'Genre':'Gender'}, inplace = True)
  result_df.rename(columns = {'Employee or Third Party':'Employee Type'}, inplace = True)
  result_df.drop_duplicates(inplace=True)
  result_df['Accident_Level']=result_df.apply(lambda col: str(col['Accident Level']), axis=1)
  result_df=result_df[['Accident_Level','Description']]
  if request.method == 'POST':
    dict_result = session.get('dict_result')
    #print(dict_result)
    session['dict_result']=dict_result
    for key in dict_result:
      options=[key['className']]
      upsampled_df=augment_mydata(result_df.loc[result_df['Accident_Level'].isin(options)],
                                  float(key['syn_val'])/100,
                                  float(key['swap_val'])/100,
                                  float(key['rand_ins_val'])/100,
                                  float(key['rand_del_val'])/100,
                                  int(key['aug_val']),
                                  )
      upsampled_result_df = upsampled_result_df.append(upsampled_df)
    upsampled_result_df["Description_DL"] = upsampled_result_df["Description"].apply(lambda x: clean_DL_data1(x))
    return 'Data cleaned for deep learning!'         

@app.route("/clean_ml_data", methods = ['GET', 'POST'])
def clean_ml_data():
  upsampled_result_df = pd.DataFrame()
  dat = session.get('filename')
  print(dat)
  result_df = read_data('/content/'+dat)
  result_df.drop('Unnamed: 0',axis='columns', inplace=True)
  result_df.rename(columns = {'Data':'Date'}, inplace = True)
  result_df.rename(columns = {'Genre':'Gender'}, inplace = True)
  result_df.rename(columns = {'Employee or Third Party':'Employee Type'}, inplace = True)
  result_df.drop_duplicates(inplace=True)
  result_df['Accident_Level']=result_df.apply(lambda col: str(col['Accident Level']), axis=1)
  result_df=result_df[['Accident_Level','Description']]
  if request.method == 'POST':
    dict_result = session.get('dict_result')
    #print(dict_result)
    for key in dict_result:
      options=[key['className']]
      upsampled_df=augment_mydata(result_df.loc[result_df['Accident_Level'].isin(options)],
                                  float(key['syn_val'])/100,
                                  float(key['swap_val'])/100,
                                  float(key['rand_ins_val'])/100,
                                  float(key['rand_del_val'])/100,
                                  int(key['aug_val']),
                                  )
      upsampled_result_df = upsampled_result_df.append(upsampled_df)
    upsampled_result_df["Description_ML"] = upsampled_result_df["Description"].apply(lambda x: clean_data(x))
    return 'Data cleaned for machine learning!'   

@app.route("/load_ml_models", methods = ['GET', 'POST'])
def load_ml_models():
  upsampled_result_df = pd.DataFrame()
  dat = session.get('filename')
  print(dat)
  result_df = read_data('/content/'+dat)
  result_df.drop('Unnamed: 0',axis='columns', inplace=True)
  result_df.rename(columns = {'Data':'Date'}, inplace = True)
  result_df.rename(columns = {'Genre':'Gender'}, inplace = True)
  result_df.rename(columns = {'Employee or Third Party':'Employee Type'}, inplace = True)
  result_df.drop_duplicates(inplace=True)
  result_df['Accident_Level']=result_df.apply(lambda col: str(col['Accident Level']), axis=1)
  result_df=result_df[['Accident_Level','Description']]
  if request.method == 'POST':
    dict_result = session.get('dict_result')
    #print(dict_result)
    for key in dict_result:
      options=[key['className']]
      upsampled_df=augment_mydata(result_df.loc[result_df['Accident_Level'].isin(options)],
                                  float(key['syn_val'])/100,
                                  float(key['swap_val'])/100,
                                  float(key['rand_ins_val'])/100,
                                  float(key['rand_del_val'])/100,
                                  int(key['aug_val']),
                                  )
      upsampled_result_df = upsampled_result_df.append(upsampled_df)
    upsampled_result_df["Description_ML"] = upsampled_result_df["Description"].apply(lambda x: clean_data(x))
    num_features = int(simplejson.loads(request.get_data()))
    print(num_features)
    df = ml_models(upsampled_result_df,num_features)
    print('training complete!')
    return df.to_html(classes="table table-condensed table-responsive table-striped",table_id="tbl_model_list",border="None") #df.to_json(orient ='table')

ml_model=pickle.load(open('/content/finalised_model_SVC.pkl','rb'))
dl_model=load_model('/content/bidirectional_lstm_model.h5')

bot = ChatBot("Leah")
trainer = ListTrainer(bot)
for files in os.listdir('/content/english/'):
    data=open('/content/english/'+files,'r').readlines()
    trainer.train(data)

@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')    
    return str(bot.get_response(userText)) 

if __name__=='__main__':
  webbrowser.open_new('http://127.0.0.1:5000/')
app.debug=True
app.run()