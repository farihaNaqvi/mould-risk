# mould-risk

Forecasting indoor mould risk in UK homes 1 to 3 days ahead, from room temperature and humidity sensors, outdoor weather and building physics.

Damp and mould in rented homes is a live problem in the UK. Awaab's Law now requires social landlords in England to act on hazards within fixed deadlines. Most landlords only hear about mould once a tenant reports it. This project asks whether cheap room sensors plus a weather forecast can flag the rooms heading into high-risk conditions a few days early, so that ventilation or heating advice can come before the mould does.

**Status:** pipeline, evaluation, reporting and a small API with a dashboard are written and tested. The full run on the 10 GB sensor archive is pending, so no headline numbers yet.

## Data

[IDEAL household energy dataset](https://datashare.ed.ac.uk/handle/10283/3647) (University of Edinburgh): 255 homes in Edinburgh, the Lothians and Fife, 2016 to 2018. Room sensors record temperature and humidity every 12 seconds. Home and room metadata (build era, wall type, room type) and occupant surveys come with it.

Outdoor weather is hourly ERA5 reanalysis from the [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api), one series per IDEAL weather location.

The data is not in this repo. IDEAL is licensed CC BY 4.0. Please cite:

> Pullinger, M., Kilgour, J., Goddard, N. et al. The IDEAL household energy dataset, electricity, gas, contextual sensor data and survey data for 255 UK homes. *Scientific Data* 8, 146 (2021). https://doi.org/10.1038/s41597-021-00921-y

## Method

1. **Surface humidity.** Mould grows on the coldest internal surface, not in room air. Surface temperature comes from indoor and outdoor temperature using the temperature factor fRsi (ISO 13788, design value 0.75). Surface RH is the room's vapour pressure over saturation at that temperature.
2. **Target.** Hours per room-day with surface RH at or above 80 %. A day with 6 or more such hours is a high-risk day. The VTT mould growth index (Hukka & Viitanen 1999) is tracked as a feature.
3. **Models**, for horizons of 1, 2 and 3 days:
   - persistence: tomorrow looks like today
   - physics: today's indoor moisture against the target day's outdoor temperature
   - LightGBM on recent room history, room and home features, and weather
   - a hybrid that also takes the physics estimate as a feature
4. **Weather scenarios.** Archived 2016-18 forecasts are not available, so every model is scored three ways:
   - *oracle*: actual target-day weather (an upper bound)
   - *forecast*: actual weather plus realistic forecast error that grows with lead time
   - *no foresight*: no target-day weather
5. **Evaluation.** 5-fold cross-validation grouped by home, so no home is in both training and test. Results are reported against persistence, which is hard to beat 1 to 3 days ahead:
   - MAE skill
   - F1 and recall on high-risk days
   - recall on onset days, when risk switches from low to high
   - how many days ahead each high-risk episode was flagged
   - 95 % intervals from a bootstrap over homes
6. **Checks.**
   - fRsi sensitivity sweep from 0.60 to 0.85.
   - Results split for below-median-income and pre-1919 homes. IDEAL has no tenure field, so these stand in for the housing Awaab's Law targets.
   - Comparison of predicted risk with the damp problems occupants reported in the survey.

## Running it

Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # .venv/bin/pip on Linux/macOS
```

Download the IDEAL files into one data folder. Point `MOULD_DATA` at it (default: `./data`):

```
$MOULD_DATA/
  raw/room_and_appliance_sensors.zip          # ~10 GB, leave zipped
  docs_extracted/                             # unzip metadata_and_surveys.zip here
    metadata/  surveys/
```

Then, with `PYTHONPATH=src`:

```bash
python -m mould_risk.weather       # hourly outdoor weather per location
python -m mould_risk.ingest        # stream the sensor zip into hourly room means
python -m mould_risk.dataset       # room-day table with targets and features
python -m mould_risk.evaluate      # cross-validation, results.csv, survey check
python -m mould_risk.report        # skill over persistence, bootstrap intervals, lead times
python -m mould_risk.sensitivity   # fRsi sweep (slow: six reduced cross-validation runs)
python -m mould_risk.serve fit     # train the served model, one per horizon
python -m mould_risk.serve run     # API and dashboard at http://127.0.0.1:8000
```

The API replays the IDEAL data: give a room and a date and get the forecast the model would have made that morning. Endpoints:

- `GET /rooms`: list of rooms
- `GET /rooms/{id}/forecast?date=`: forecast hours, risk band and advice for each of the next 3 days
- `GET /rooms/{id}/history`: observed hours per day
- `GET /homes/{id}/forecast?date=`: every room in a home

Interactive docs are at `/docs`, and `/` serves a dashboard. The served model is trained on every home, so its replayed scores are in-sample. The accuracy figures come from the cross-validation.

Outputs go to `$MOULD_DATA/processed/`. Tests run on synthetic data and need no download:

```bash
python -m pytest -q
```

## Limitations

- **The target is a physics proxy, not observed mould.** The only real-world check is the occupant survey, which is self-reported and at home level.
- **One fRsi for every wall.** Real cold spots vary, which is why the sensitivity sweep exists.
- **No real archived forecasts.** The forecast scenario adds assumed error to actual weather.
- **The sample is not the social housing stock.** It is Scottish, skews above median income and has no tenure data. Awaab's Law applies in England.

## Licence

Code: MIT, see [LICENSE](LICENSE). Data: IDEAL is CC BY 4.0 and is not redistributed here.
