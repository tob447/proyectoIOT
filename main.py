from logging import debug
from flask import Flask, request
from flask_restful import Api,Resource,request
from flask_cors import CORS
from Descarga import play_dowload
from Simulador import returnImageBase64,ejecutar

import numpy as np


app= Flask(__name__)
api=Api(app)
CORS(app)

class GetPrecisionGraphs(Resource):
    def post(self):
        play_dowload(request.json["link"])
        ejecutar()
    
        return returnImageBase64()
    

        





api.add_resource(GetPrecisionGraphs,"/callMethod")

if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=80,debug=True)
    app.run(debug=True)