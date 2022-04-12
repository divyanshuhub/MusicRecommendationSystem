from flask import Flask , render_template , request
import two
import os



app=Flask(__name__)
PEOPLE_FOLDER = os.path.join('static')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

def pro_name(str):
    result = ' '.join(elem.capitalize() for elem in str.split())
    return result



@app.route('/')
@app.route('/home')
def home():
    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'img\movielib2.jpg')
    return render_template('index.html')
@app.route('/predict',methods=["POST"])
def predict():
    #full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'img\movielib2.jpg')
    song = request.form.get('song')
    #year = request.form.get("year")
    #year = find_song_year(song)

    #songList = [{"name":pro_name(song), "year":int(year)}]

    pred = two.recommendations(song_name=song)
    print(pred)
    #pred =recommend_movies(movie=arr)
    return render_template('index.html', prediction_text ="Recommend{}".format(pred), data=pred, len=len(pred))


if __name__ == '__main__':
    app.run(debug=True , port='8000')
