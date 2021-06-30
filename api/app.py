
from fastapi import FastAPI
import pandas as pd
import joblib



app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint


@app.get("/predict")
def create_predict( acousticness,
                    danceability,
                    duration_ms,
                    energy,
                    explicit,
                    id,
                    instrumentalness,
                    key,
                    liveness,
                    loudness,
                    mode,
                    name,
                    release_date,
                    speechiness,
                    tempo,
                    valence,
                    artist):


    X = pd.DataFrame(dict(
        acousticness=[float(acousticness)],
        danceability=[float(danceability)],
        duration_ms=[int(duration_ms)],
        energy=[float(energy)],
        explicit=[int(explicit)],
        id=[str(id)],
        instrumentalness=[float(instrumentalness)],
        key=[int(key)],
        liveness=[float(liveness)],
        loudness=[float(loudness)],
        mode=[int(mode)],
        name=[str(name)],
        release_date=[str(release_date)],
        speechiness=[float(speechiness)],
        tempo=[float(tempo)],
        valence=[float(valence)],
        artist=[str(artist)]))



    # pipeline = get_model_from_gcp()
    pipeline = joblib.load('model.joblib')

    # make prediction
    results = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(
        artist = artist,
        name = name,
        prediction=pred)
