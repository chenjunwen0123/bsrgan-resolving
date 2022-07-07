from flask import Flask, request, jsonify
from Recover import Recover

app = Flask(__name__)

host = 'http://127.0.0.1:5000'


@app.route('/upload', methods=['POST'])
def upload():
    imgFile = request.files.get('targetPhoto')
    imgName = 'static/requestPic/' + imgFile.filename
    imgFile.save(imgName)
    recover = Recover(imgName)
    ret = recover.recover().split('\\')
    link = host
    for item in ret:
        link = link + '/'+ item

    return jsonify(status=1, site=link)


if __name__ == '__main__':
    app.run(debug=True)
