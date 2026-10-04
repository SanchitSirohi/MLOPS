"Behaviour tests for the trained model."

import pandas as pd


def make_order(distance_km=5.0, prep_time_min=20,
               traffic_level=2, rain=0):
    return pd.DataFrame([{
        "distance_km": distance_km,
        "prep_time_min": prep_time_min,
        "traffic_level": traffic_level,
        "rain": rain,
    }])


def test_rain_never_makes_delivery_faster(trained_model):
    dry = trained_model.predict(make_order(rain=0))[0]
    wet = trained_model.predict(make_order(rain=1))[0]
    assert wet >= dry


def test_further_is_never_faster(trained_model):
    near = trained_model.predict(make_order(distance_km=2))[0]
    far = trained_model.predict(make_order(distance_km=12))[0]
    assert far > near


def test_error_is_better_than_guessing(model_mae, baseline_mae):
    assert model_mae < baseline_mae


def test_heavier_traffic_is_never_faster(trained_model):
    # Same order, different traffic levels
    order_low = {
        "distance_km": 7,
        "prep_time_min": 25,
        "traffic_level": 1,
        "rain": 0,
    }

    order_high = {
        "distance_km": 7,
        "prep_time_min": 25,
        "traffic_level": 3,
        "rain": 0,
    }

    prediction_low = trained_model.predict([[
        order_low["distance_km"],
        order_low["prep_time_min"],
        order_low["traffic_level"],
        order_low["rain"],
    ]])[0]

    prediction_high = trained_model.predict([[
        order_high["distance_km"],
        order_high["prep_time_min"],
        order_high["traffic_level"],
        order_high["rain"],
    ]])[0]

    assert prediction_high >= prediction_low


def test_heavier_traffic_is_never_faster(trained_model):
    low_traffic = trained_model.predict(make_order(traffic_level=1))[0]
    high_traffic = trained_model.predict(make_order(traffic_level=3))[0]
    assert high_traffic >= low_traffic
