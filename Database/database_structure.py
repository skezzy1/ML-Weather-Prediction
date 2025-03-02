from sqlalchemy import create_engine, ForeignKey, String, Integer, Column, Date, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid
import os
from dotenv import load_dotenv
load_dotenv()

Base = declarative_base()

class WeatherPrediction(Base):
    __tablename__ = "WeatherPrediction"
    RedcordID = Column(Integer, primary_key=True, autoincrement=True)
    Location = Column(String)
    DateTime = Column(DateTime, default=datetime.utcnow))
    Temperature_C = Column(Float)
    Humidity_pct = Column(Float)
    Precipitation_mm = Column(Float)
    Wind_Speed_kmh = Column(Float)
    will_rain = Column(Boolean)

    def __init__(self, RedcordID, Location, DateTime, Temperature_C, Humidity_pct, Wind_Speed_kmh):
        self.RedcordID = RedcordID
        self.Location = Location
        self.DateTime = DateTime
        self.Temperature_C = Temperature_C
        self.Humidity_pct = Humidity_pct
        self.Precipitation_mm = Precipitation_mm
        self.Wind_Speed_kmh = Wind_Speed_kmh

class WeatherActual(Base):
    __tablename__ = "ActualWeather"
    RedcordID = Column(Integer, primary_key=True, autoincrement=True)
    DateTime = Column(DateTime, default=datetime.utcnow))
    temperature_actual = Column(Float)
    humidity_actual = Column(Float)
    wind_speed_actual = Column(Float)
    precipitation_actual = Column(Float)
    will_rain_actual = Column(Boolean)

    def __init__(self, RedcordID, DateTime, temperature_actual, humidity_actual, wind_speed_actual, precipitation_actual, will_rain_actual):
        self.RedcordID = RedcordID
        self.DateTime = DateTime
        self.temperature_actual = temperature_actual
        self.humidity_actual = humidity_actual
        self.wind_speed_actual = wind_speed_actual
        self.precipitation_actual = precipitation_actual
        self.will_rain_actual = will_rain_actual

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()