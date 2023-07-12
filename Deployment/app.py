from flask import Flask,render_template,redirect,request

import Deployment_helper

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html")



@app.route('/',methods = ['POST'])
def captions():
    caption = None
    path = None
    res_dict = {}
    if request.method == 'POST':
        f = request.files['userfile']
        path = "./static/{}".format(f.filename)
        f.save(path)
        caption = Deployment_helper.caption_this_image(path)

        result_dict = {
            "image" : path,
            "caption" : caption,
        }

    return render_template("index.html",your_result = result_dict)  

if __name__ == '__main__':
    app.run(debug=True)

